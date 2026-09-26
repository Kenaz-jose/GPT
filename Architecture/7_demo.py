import sys
import os
import torch
import torch.nn.functional as F
import math
import importlib

sys.path.extend([".", "/workspace/artifacts", "/workspace/out"])

viz_module = importlib.import_module("6_visualize")
mha_module = importlib.import_module("3_multihead_attention")

MultiHeadCausalAttention = mha_module.MultiHeadCausalAttention
trace_tensor_shapes = viz_module.trace_tensor_shapes
plot_attention_heatmaps = viz_module.plot_attention_heatmaps

def run_mha_demo():
    print("=" * 60)
    print("Part 1: Multi-Head Attention Shape Tracing & Heatmap Demo")
    print("=" * 60)

    batch_size = 1
    seq_len = 6
    d_model = 16
    n_head = 4
    head_dim = d_model // n_head

    # 1. Print Shape Trace Log
    trace_tensor_shapes(batch_size=batch_size, seq_len=seq_len, d_model=d_model, n_head=n_head)

    torch.manual_seed(42)
    mha = MultiHeadCausalAttention(d_model=d_model, n_head=n_head, dropout=0.0)

    sample_tokens = ["Language", "models", "learn", "patterns", "from", "data"]
    x = torch.randn(batch_size, seq_len, d_model)

    # 2. Extract Attention Weights & Plot
    B, T, C = x.size()
    qkv = mha.c_attn(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)

    image_path = "attention_heatmaps.png"
    plot_attention_heatmaps(attn_weights, tokens=sample_tokens, save_path=image_path)
    
    out = mha(x)
    print(f"\n[Verification] Output shape through standard forward pass: {out.shape}")
    print(f"[Verification] Heatmap saved to: {image_path}")

if __name__ == "__main__":
    run_mha_demo()
