import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # how many independent sequences to process in parallel
block_size = 8 # max content length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337) # for reproducibility

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

# translate individual characters to integers
str_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_str = {i:ch for i,ch in enumerate(chars)}

# encoder: take a string and return a list of integers
encode = lambda s: [str_to_int[c] for c in s]
# decoder: take a list of integers and return a string
decode = lambda l: ''.join(int_to_str[i] for i in l) 

# wrap the encoded text into a data tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train on first 90%, rest is evaluation data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
data = torch.tensor(encode(text), dtype = torch.long)

def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad() # no back propagation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""
Bigram Language Module: Simplest neural network
- Predict the likelihood of a word in a sequence
  based on the preceding word.
"""
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # token embedding table, size vocabSize x vocabSize
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_table(idx) # (Batch, Time, Channel) tensor

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # Need to reshape logits for cross_entropy
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets) # how well are we predicting the next character

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        # take a (B,T) and make it a (B,T + 1) ... to max_new_tokens
        for _ in range(max_new_tokens):
            # get predictions, don't need loss
            logits, loss = self(idx)
            # focus on last time step 
            logits = logits[:,-1,:] # (B,C)
            # get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample fromm the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # concatenate to running sequence
            idx = torch.cat((idx,idx_next), dim = 1) # (B,T+1)
        return idx

model = Bigram(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
