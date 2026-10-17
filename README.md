# GPT — Building an LLM from Scratch: Transformer Architecture to PPO-Based RLHF

A modular, code-first journey into Large Language Models (LLMs), implemented in PyTorch from foundational mathematical operations to modern Transformer architectures, language model training, instruction tuning, reward modeling, and reinforcement learning from human feedback.

The goal is to understand not just *how to use an LLM*, but how its internal components work, how tensors flow through the architecture, how model parameters are learned, and how a pretrained language model can be adapted to follow instructions and align with human preferences.

> **Learning philosophy:** Understand the mathematics, implement the component, trace its tensor shapes, test its behavior, and then integrate it into the larger system.

---

## Table of Contents

- [Project Goals](#-project-goals)
- [Learning Roadmap](#-learning-roadmap)
- [Architecture and Engineering Concepts](#-architecture-and-engineering-concepts)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Modules](#-running-the-modules)
- [Implementation Philosophy](#-implementation-philosophy)
- [Current Progress](#-current-progress)
- [Future Directions](#-future-directions)
- [Technology Stack](#-technology-stack)
- [License](#-license)

---

## 🎯 Project Goals

This project is designed to build a practical understanding of LLM engineering through incremental implementation.

- **Build from first principles:** Implement core mathematical operations and neural network components instead of relying on high-level LLM frameworks.
- **Understand tensor transformations:** Trace dimensions through projections, attention heads, normalization, feed-forward networks, and complete Transformer blocks.
- **Progress from fundamentals to modern architectures:** Move from vanilla attention to RMSNorm, RoPE, Grouped-Query Attention (GQA), KV caching, and Mixture of Experts (MoE).
- **Train a language model:** Explore tokenization, next-token prediction, loss calculation, optimization, and autoregressive generation.
- **Understand post-training:** Implement instruction fine-tuning, preference-based reward modeling, and PPO-based reinforcement learning.
- **Develop engineering discipline:** Use modular Python scripts, reproducible experiments, shape assertions, and unit tests.

## 🗺️ Learning Roadmap

The project is organized into eight parts, progressing from foundational architecture to post-training alignment.

### Part 1 — Core Transformer Architecture

Understand the building blocks of a decoder-only Transformer.

- Scaled dot-product attention using NumPy
- Sinusoidal positional encoding
- Multi-head causal self-attention
- Query, Key, and Value projections
- Position-wise feed-forward networks with GELU
- Transformer decoder block assembly
- Tensor shape tracing and attention visualization
- Mathematical assertions and unit tests

**Outcome:** Understand how token representations flow through a basic Transformer decoder block.

### Part 2 — Training a Tiny LLM

Move from architecture to language modeling.

- Byte-level tokenization
- Token embeddings and positional information
- Dataset preparation and sequence chunking
- Autoregressive next-token prediction
- Cross-entropy loss
- AdamW optimization and training loops
- Greedy and sampling-based text generation

**Outcome:** Build and train a small language model and understand the relationship between logits, targets, loss, and generated tokens.

### Part 3 — Modern Transformer Architecture

Upgrade the baseline architecture with components used in modern LLMs.

- Root Mean Square Layer Normalization (RMSNorm)
- SwiGLU feed-forward networks
- Rotary Positional Embeddings (RoPE)
- Grouped-Query Attention (GQA)
- Key-Value (KV) caching for autoregressive decoding
- Modern decoder block assembly
- Integration tests and tensor shape validation

**Outcome:** Understand how modern architectural choices affect computation, parameterization, and inference efficiency.

### Part 4 — Scaling and Training Engineering

Explore techniques that make model training more practical as computational requirements increase.

- Subword tokenization with BPE
- Automatic mixed precision (AMP)
- Gradient accumulation
- Gradient clipping
- Learning-rate scheduling
- Checkpointing and checkpoint recovery
- Memory and throughput considerations

**Outcome:** Understand the engineering trade-offs involved in training larger models under limited hardware resources.

### Part 5 — Mixture of Experts (MoE)

Explore sparse conditional computation.

- Expert feed-forward networks
- Router and gating networks
- Top-k expert selection
- Token dispatch and output combination
- Load-balancing auxiliary losses
- Expert utilization and routing analysis

**Outcome:** Understand how a model can increase its total parameter count without activating every expert for every token.

### Part 6 — Supervised Fine-Tuning (SFT)

Adapt a pretrained language model to follow instructions.

- Instruction and response dataset formatting
- Chat templates and special tokens
- Causal language modeling objectives
- Prompt and response loss masking
- Fine-tuning loops and validation
- Evaluation of instruction-following behavior

**Outcome:** Understand how supervised examples teach a pretrained model to produce responses in a desired format.

### Part 7 — Reward Modeling

Learn how preference data can be used to train a reward model.

- Chosen and rejected response pairs
- Reward-model architecture
- Scalar preference scores
- Bradley–Terry preference modeling
- Pairwise ranking loss
- Preference evaluation and diagnostics

**Outcome:** Train a model that scores responses according to learned human preference data.

### Part 8 — RLHF with Proximal Policy Optimization (PPO)

Connect the policy model, reward model, and reinforcement learning objective.

- Policy and value models
- Response sampling and reward calculation
- Reference-model KL penalties
- Advantage estimation
- PPO clipped policy objective
- Value-function loss and entropy considerations
- Training stability and policy evaluation

**Outcome:** Understand the mechanics of PPO-based RLHF and how reward signals can guide a language model toward preferred behavior.

*Note: PPO-based RLHF is one alignment approach, not the only one. Other approaches, such as Direct Preference Optimization (DPO), may be explored separately.*

---

## 🧠 Architecture and Engineering Concepts

The implementation emphasizes understanding the internal mechanics of an LLM.

| Component | Core question |
|---|---|
| Embeddings | How are discrete tokens represented as vectors? |
| Attention | How does a token retrieve information from other tokens? |
| Multi-head attention | Why use multiple attention heads? |
| RMSNorm | How are hidden-state magnitudes normalized? |
| RoPE | How is positional information incorporated into attention? |
| GQA | How can multiple query heads share Key and Value heads? |
| KV cache | How can previous attention states be reused during generation? |
| SwiGLU | How does a gated feed-forward network transform representations? |
| MoE | How can only selected experts be activated for a token? |
| SFT | How does a model learn from instruction-response examples? |
| Reward modeling | How can preferences be represented as a learned score? |
| PPO | How can reward signals update a language model's policy? |

### Implementation workflow

For each major component, the intended workflow is:

1. Understand the mathematical operation.
2. Identify the input and output tensor shapes.
3. Implement the operation in PyTorch.
4. Run the module independently.
5. Validate shapes, numerical behavior, and edge cases.
6. Integrate it into the next architectural component.

---

## 📂 Project Structure

The intended long-term structure is:

```text
GPT/
├── README.md
├── .gitignore
│
├── Architecture/
│   ├── 1_attention_numpy.py
│   ├── 2_pos_encoding.py
│   ├── 3_multihead.py
│   ├── 4_feedforward_network.py
│   ├── 5_block.py
│   ├── 6_visualize_utils.py
│   ├── 7_tests.py
│   └── orchestrator.py
│
├── TinyLLM/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
│
├── Modern/
│   ├── rmsnorm.py
│   ├── swiglu.py
│   ├── rope.py
│   ├── gqa_kvcache.py
│   ├── block.py
│   ├── test.py
│   └── orchestrator.py
│
├── Scaling/
├── MoE/
├── SFT/
├── RewardModel/
└── RLHF_PPO/
```

*The structure above is a target organization. Directory names and filenames should be updated to match the modules actually present in the repository.*

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or newer; Python 3.12 is the development target.
- PyTorch 2.x.
- NumPy.
- Matplotlib for visualization.
- A CPU or compatible CUDA-enabled GPU.

### Environment setup

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd GPT
```

Create and activate a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Upgrade the package installer:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the dependencies:

```bash
python -m pip install torch numpy matplotlib tqdm
```

For GPU acceleration, install the PyTorch build appropriate for your CUDA environment using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Verify the environment

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## ▶️ Running the Modules

Run individual components to inspect their behavior and tensor transformations.

### Core architecture

```bash
python Architecture/1_attention_numpy.py
python Architecture/2_pos_encoding.py
python Architecture/3_multihead.py
python Architecture/4_feedforward_network.py
python Architecture/5_block.py
```

### Modern architecture

```bash
python Modern/rmsnorm.py
python Modern/swiglu.py
python Modern/rope.py
python Modern/gqa_kvcache.py
python Modern/block.py
python Modern/test.py
```

Run the modern architecture orchestrator:

```bash
python Modern/orchestrator.py
```

The orchestrator executes the listed modules sequentially and stops when a step fails.

*These commands assume the filenames and directory structure match the repository. Run each module from the project root, and update the commands if your actual filenames differ.*

---

## 🔬 Implementation Philosophy

This is a learning-focused engineering project, not a claim to reproduce a production-scale LLM.

The emphasis is on:

- **Clarity over abstraction:** Keep individual components understandable and inspectable.
- **Mathematics before optimization:** Establish correctness before adding performance improvements.
- **Explicit tensor shapes:** Make dimensional transformations visible and easy to verify.
- **Incremental integration:** Test components independently before assembling larger systems.
- **Reproducibility:** Document dependencies, configuration, and experiment settings.
- **Honest evaluation:** Distinguish functional correctness from training quality, benchmark performance, and production readiness.

The project uses small configurations where practical so that the mechanics can be explored on limited hardware.

---

## 📈 Current Progress

The repository is being developed incrementally.

- [x] Core Transformer components and decoder-block implementation
- [x] Tiny language model architecture and training experiments
- [x] RMSNorm
- [x] SwiGLU feed-forward network
- [x] RoPE
- [x] Grouped-Query Attention and KV cache
- [x] Modern Transformer block integration
- [ ] Complete and maintain unit-test coverage
- [ ] Scaling and training engineering
- [ ] Mixture of Experts
- [ ] Supervised Fine-Tuning
- [ ] Reward Modeling
- [ ] PPO-based RLHF

*Update this checklist as each component is implemented, tested, and integrated. Marking a module complete should mean more than having created its file.*

---

## 🧰 Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Numerical computing:** NumPy
- **Visualization:** Matplotlib
- **Testing:** Python assertions and unit tests
- **Version control:** Git

---

## 📄 License

This project is intended to be distributed under the MIT License. See [`LICENSE`](LICENSE) for the full license text.

---

## 🌱 Why This Project?

Modern LLMs combine ideas from linear algebra, probability, optimization, neural network architecture, and systems engineering. This project is an attempt to make those ideas tangible through implementation.

**The objective is not merely to run a language model. It is to understand the components, implement them, test them, and progressively connect them into a complete LLM training and alignment pipeline.**
