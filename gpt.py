import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -----------------------

torch.manual_seed(1337)

# Load data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Use autocast for evaluation as well to stay consistent
            with torch.amp.autocast(device_type=('cuda' if 'cuda' in device else 'cpu'), dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    """ Optimized multi-head self-attention using FlashAttention kernels """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_head = num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        # Batch, Time, (3 * Heads * HeadSize)
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Efficient Flash Attention (SDPA)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )
        
        # Re-assemble heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    """ SwiGLU-style FeedForward (widely used in Llama/Modern GPTs) """
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(4 * n_embd * 2/3) # Standard dimension scaling for SwiGLU
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (SiLU(xW1) * xW3) W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight Tying: The embedding and output head share weights
        self.token_embedding_table.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Model Init & Compilation ---
model = GPTLanguageModel().to(device)
# torch.compile provides massive speedups on Linux/WSL2
if hasattr(torch, 'compile') and device == 'cuda':
    print("Compiling model...")
    model = torch.compile(model)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Use AdamW with Weight Decay for better generalization
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# Scaler for Mixed Precision training
scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

# --- Training Loop ---
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # Mixed Precision Forward Pass
    with torch.amp.autocast(device_type=('cuda' if 'cuda' in device else 'cpu'), dtype=torch.bfloat16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# --- Generation ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
