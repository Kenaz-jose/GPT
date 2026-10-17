import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import importlib

rope_module = importlib.import_module("rope")
precompute_freqs_cis = rope_module.precompute_freqs_cis
apply_rotary_emb = rope_module.apply_rotary_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats KV heads to match the number of Query heads in Grouped-Query Attention (GQA).
    
    Args:
        x: Tensor of shape (batch_size, n_kv_head, seq_len, head_dim)
        n_rep: Expansion factor (n_head // n_kv_head)
    Returns:
        Tensor of shape (batch_size, n_head, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch_size, n_kv_head, seq_len, head_dim = x.shape
    return (
        x.unsqueeze(2)
        .expand(batch_size, n_kv_head, n_rep, seq_len, head_dim)
        .reshape(batch_size, n_kv_head * n_rep, seq_len, head_dim)
    )

class KVCache:
    """
    Key-Value Cache for autoregressive decoding efficiency.
    Stores past Key and Value states across generation steps.
    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_head: int, head_dim: int, device: str = "cpu"):
        self.k_cache = torch.zeros((max_batch_size, n_kv_head, max_seq_len, head_dim), device=device)
        self.v_cache = torch.zeros((max_batch_size, n_kv_head, max_seq_len, head_dim), device=device)

    def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache with new keys and values at start_pos and returns valid history up to current length.
        """
        bsz, n_kv_head, seq_len, head_dim = xk.shape
        self.k_cache[:bsz, :, start_pos : start_pos + seq_len, :] = xk
        self.v_cache[:bsz, :, start_pos : start_pos + seq_len, :] = xv
        
        keys = self.k_cache[:bsz, :, : start_pos + seq_len, :]
        values = self.v_cache[:bsz, :, : start_pos + seq_len, :]
        return keys, values

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with integrated RoPE and KV-Cache support.
    """
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_head % n_kv_head == 0, f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = d_model // n_head
        
        self.wq = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Projections
        xq = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos_seq = cos[start_pos : start_pos + T]
        sin_seq = sin[start_pos : start_pos + T]
        xq, xk = apply_rotary_emb(xq, xk, cos_seq, sin_seq)
        
        # KV Caching update
        if kv_cache is not None:
            xk, xv = kv_cache.update(start_pos, xk, xv)
            
        # Expand KV heads to match Q heads (GQA)
        keys = repeat_kv(xk, self.n_rep)     # (B, n_head, seq_len_kv, head_dim)
        values = repeat_kv(xv, self.n_rep)   # (B, n_head, seq_len_kv, head_dim)
        
        # Scaled dot-product attention
        scores = (xq @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Causal mask if seq_len > 1 (e.g. prompt prefill phase)
        if T > 1:
            seq_len_kv = keys.size(2)
            mask = torch.full((T, seq_len_kv), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=start_pos + 1)
            scores = scores + mask
            
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        output = scores @ values  # (B, n_head, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.wo(output)

def main():
    print("=" * 60)
    print("Part 3: Grouped-Query Attention (GQA) & KV-Cache")
    print("=" * 60)

    batch_size = 1
    d_model = 16
    n_head = 8
    n_kv_head = 2  # 4:1 ratio (GQA)
    max_seq_len = 32

    torch.manual_seed(42)

    # Precompute RoPE tables
    head_dim = d_model // n_head
    cos, sin = precompute_freqs_cis(dim=head_dim, end=max_seq_len)

    gqa = GroupedQueryAttention(d_model=d_model, n_head=n_head, n_kv_head=n_kv_head)
    kv_cache = KVCache(max_batch_size=batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head, head_dim=head_dim)

    print(f"\n[Step 1] Configuration:")
    print(f"  d_model: {d_model} | n_head: {n_head} | n_kv_head: {n_kv_head} (4x KV Memory Reduction)")

    # 1. Prefill Phase (Prompt length = 4)
    prompt_len = 4
    prompt_x = torch.randn(batch_size, prompt_len, d_model)
    out_prefill = gqa(prompt_x, start_pos=0, cos=cos, sin=sin, kv_cache=kv_cache)

    print(f"\n[Step 2] Prefill Phase (Prompt T={prompt_len}):")
    print(f"  Input Prompt shape:  {prompt_x.shape}")
    print(f"  Prefill Output shape: {out_prefill.shape}")
    print(f"  Cached Key length:    {kv_cache.k_cache.size(2)} (stored: {prompt_len} tokens)")

    # 2. Decoding Phase (Incremental step-by-step token generation)
    print("\n[Step 3] Decoding Phase (Incremental Step-by-Step Generation):")
    start_pos = prompt_len
    for step in range(3):
        token_x = torch.randn(batch_size, 1, d_model)  # Single token input (T=1)
        out_step = gqa(token_x, start_pos=start_pos + step, cos=cos, sin=sin, kv_cache=kv_cache)
        current_cached_tokens = start_pos + step + 1
        print(f"  Step {step + 1}: Input shape {token_x.shape} -> Output shape {out_step.shape} | Active Cache Tokens: {current_cached_tokens}")

if __name__ == "__main__":
    main()