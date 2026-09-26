# GPT: Practical Engineering from Base Model to PPO RLHF

A modular, code-first implementation of a modern Large Language Model (LLM) built entirely from scratch in PyTorch. This project provides a step-by-step engineering roadmap—from foundational Transformer math and single decoder blocks to advanced modern optimizations (RoPE, GQA, MoE), instruction tuning (SFT), and post-training alignment (RLHF via PPO).

---

## 🌟 Overview & Key Features

* **Pure PyTorch Implementation**: Built from first principles without relying on high-level wrapper frameworks.
* **Modular Codebase**: Every major concept is isolated into standalone executable scripts with clear tensor shape tracing.
* **Modern LLM Architecture**: Progressively introduces industry-standard optimizations including RMSNorm, SwiGLU, Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), and Mixture of Experts (MoE).
* **Full Alignment Pipeline**: Implements the complete post-training lifecycle including Supervised Fine-Tuning (SFT), Reward Modeling (Bradley-Terry loss), and Reinforcement Learning from Human Feedback (PPO).
* **Python 3.12+ Ready**: Tested and fully compatible with Python 3.12+ and PyTorch 2.x.

---

## 🗂 Project Structure & Curriculum

The project is divided into 8 sequential parts mirroring the end-to-end development lifecycle of modern LLMs:

```text
llm-from-scratch/
├── part01_core_architecture/
│   ├── 01_attention_numpy.py    # Scaled dot-product attention in pure NumPy
│   ├── 02_pos_encoding.py       # Sinusoidal positional encodings (PyTorch buffer)
│   ├── 03_multihead.py          # Multi-Head Causal Self-Attention with QKV fusion
│   ├── 04_feed_forward.py       # Position-wise Feed-Forward Network with GELU
│   ├── 05_block.py              # Full Transformer Decoder Block assembly
│   ├── 06_visualize_utils.py    # Shape tracing and attention heatmap utilities
│   ├── 07_tests.py              # Unit tests and mathematical assertions
│   └── orchestrator.py          # Part 1 execution entry point
├── part02_tiny_llm_training/    # Tokenization, cross-entropy loss, and training loop
├── part03_modern_architecture/  # RMSNorm, SwiGLU, KV Cache, GQA, RoPE
├── part04_scaling_production/   # BPE, mixed precision (AMP), gradient accumulation
├── part05_mixture_of_experts/   # Sparse expert routing, gating, load balancing
├── part06_supervised_finetuning/# Instruction dataset formatting & SFT loss masking
├── part07_reward_modeling/      # Preference scoring & Bradley-Terry loss
└── part08_rlhf_ppo/             # Policy/value models, KL penalties, and PPO alignment
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
* **Python**: 3.10 – 3.12+ (Python 3.12.3 recommended)
* **PyTorch**: 2.2+ (CPU or CUDA-enabled GPU)

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
# Create virtual environment
python3.12 -m venv llm-env
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate

# Upgrade installer tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install torch numpy matplotlib tqdm
```

### 3. Running Part 1 Modules

You can execute each script independently to inspect tensor dimensions and intermediate operations:

```bash
# Run pure NumPy scaled dot-product attention
python part01_core_architecture/01_attention_numpy.py

# Run Sinusoidal Positional Encodings
python part01_core_architecture/02_pos_encoding.py

# Run Multi-Head Causal Self-Attention
python part01_core_architecture/03_multihead.py

# Run Position-Wise Feed-Forward Network
python part01_core_architecture/04_feed_forward.py
```

---

## 📚 Course Roadmap

1. **Part 1: Core Transformer Architecture** — Implement vanilla decoder blocks from scratch.
2. **Part 2: Training a Tiny LLM** — Byte-level tokenization, next-token prediction, and full training loops.
3. **Part 3: Modernizing the Architecture** — Upgrade with RMSNorm, SwiGLU, KV Cache, GQA, and RoPE.
4. **Part 4: Scaling Up & Production Engineering** — BPE, AMP, gradient accumulation, and checkpointing.
5. **Part 5: Mixture of Experts (MoE)** — Sparse routing, gating networks, and auxiliary load balancing.
6. **Part 6: Supervised Fine-Tuning (SFT)** — Prompt templates, causal loss masking, and instruction datasets.
7. **Part 7: Reward Modeling** — Preference scoring models with margin ranking losses.
8. **Part 8: RLHF via PPO** — Aligning models with human preferences using Proximal Policy Optimization.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
