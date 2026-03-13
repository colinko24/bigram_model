# NanoGPT

A high-performance, character-level Transformer based on the "Attention is All You Need" architecture. This project started with the baseline implementation from Andrej Karpathy's NanoGPT and was extended with modern architectural features and training optimizations used in state-of-the-art LLMs.

## 🛠 Features & Improvements

While the core logic remains a decoder-only Transformer, I have implemented several upgrades to move closer to modern standards (like Llama 3 and GPT-4).

### ⚡ Performance Optimizations
* **FlashAttention (SDPA):** I replaced the manual $O(T^2)$ attention mechanism with PyTorch’s `scaled_dot_product_attention`. This utilizes hardware-accelerated kernels, significantly reducing memory footprint and increasing speed.
* **Mixed Precision Training (AMP):** The training loop now utilizes `torch.amp` (Automatic Mixed Precision). By calculating in `bfloat16` or `float16`, the model achieves nearly **2x throughput** on modern GPUs.
* **Model Compilation:** Integrated `torch.compile` to fuse operations into optimized CUDA kernels, reducing Python overhead during the forward and backward passes.

### 🏗️ Architectural Upgrades
* **SwiGLU Activation:** Replaced standard ReLU with the **SwiGLU** gated activation function. This mechanism allows the model to learn more complex features more efficiently than traditional non-linearities.
* **Weight Tying:** The input embedding and output `lm_head` share the same weight matrix. This reduces total parameters by ~30% and enforces a stronger semantic relationship between input and output tokens.
* **Optimized Initialization:** Implemented a specific standard deviation ($\sigma = 0.02$) for weight initialization, which is critical for stabilizing the training of deeper networks.

---

## 📈 Comparison

| Feature | Original Tutorial | This Optimized Version |
| :--- | :--- | :--- |
| **Attention** | Manual Softmax/Masking | FlashAttention (SDPA) |
| **Precision** | Float32 | Mixed Precision (BF16/FP16) |
| **Activation** | ReLU | SwiGLU |
| **Parameter Efficiency** | Standard | Weight Tying |
| **Execution** | Eager Mode | Compiled (`torch.compile`) |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+ (Required for `torch.compile` and SDPA)
* NVIDIA GPU (Recommended for FlashAttention and AMP speedups)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/nanogpt-optimized.git](https://github.com/yourusername/nanogpt-optimized.git)
   cd nanogpt-optimized
