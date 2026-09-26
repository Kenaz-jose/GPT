import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attention_heatmaps(attn_weights, tokens=None, save_path="attention_heatmap.png"):
    """
    Plots and saves attention weight heatmaps across all heads.
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
        
    if attn_weights.ndim == 4:
        attn_weights = attn_weights[0]  # Take first batch sample
        
    n_head, T, _ = attn_weights.shape
    
    if tokens is None:
        tokens = [f"Token_{i}" for i in range(T)]
        
    cols = min(4, n_head)
    rows = (n_head + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2), squeeze=False)
    fig.suptitle("Multi-Head Causal Attention Heatmaps", fontsize=14, fontweight='bold')
    
    for h in range(n_head):
        r, c = h // cols, h % cols
        ax = axes[r, c]
        im = ax.imshow(attn_weights[h], cmap="Blues", vmin=0, vmax=1)
        
        ax.set_title(f"Head {h+1}", fontsize=11)
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        
        # Grid overlay for clarity
        ax.set_xticks(np.arange(T + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(T + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        
    # Hide unused subplots
    for h in range(n_head, rows * cols):
        r, c = h // cols, h % cols
        fig.delaxes(axes[r, c])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def trace_tensor_shapes(batch_size=2, seq_len=5, d_model=16, n_head=4):
    """
    Prints step-by-step tensor dimensions for Multi-Head Self-Attention.
    """
    head_dim = d_model // n_head
    print("=" * 60)
    print("Multi-Head Attention Tensor Shape Tracing")
    print("=" * 60)
    print(f"Hyperparameters: Batch={batch_size}, SeqLen={seq_len}, d_model={d_model}, n_head={n_head}, head_dim={head_dim}")
    print("-" * 60)
    print(f" 1. Input Tensor X:                     ({batch_size}, {seq_len}, {d_model})")
    print(f" 2. Unified Projection (c_attn):        ({batch_size}, {seq_len}, {3 * d_model})")
    print(f" 3. Chunk into Q, K, V:                  Each ({batch_size}, {seq_len}, {d_model})")
    print(f" 4. Reshape & Transpose for Multi-Head:  Each ({batch_size}, {n_head}, {seq_len}, {head_dim})")
    print(f" 5. Raw Scores (Q @ K^T):               ({batch_size}, {n_head}, {seq_len}, {seq_len})")
    print(f" 6. Causal Softmax Weights:              ({batch_size}, {n_head}, {seq_len}, {seq_len})")
    print(f" 7. Weighted Values (Weights @ V):       ({batch_size}, {n_head}, {seq_len}, {head_dim})")
    print(f" 8. Transpose & Merge Heads:            ({batch_size}, {seq_len}, {d_model})")
    print(f" 9. Output Projection (c_proj):         ({batch_size}, {seq_len}, {d_model})")
    print("=" * 60)