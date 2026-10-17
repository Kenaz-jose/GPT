import sys
import os
import torch
import torch.nn as nn
import importlib

rmsnorm_module = importlib.import_module("rmsnorm")
swiglu_module = importlib.import_module("swiglu")
rope_module = importlib.import_module("rope")
gqa_module = importlib.import_module("gqa_kvcache")

RMSNorm = rmsnorm_module.RMSNorm
SwiGLU = swiglu_module.SwiGLU
precompute_freqs_cis = rope_module.precompute_freqs_cis
GroupedQueryAttention = gqa_module.GroupedQueryAttention
KVCache = gqa_module.KVCache

class ModernTransformerBlock(nn.Module):
    """
    Modern LLaMA-style Transformer Decoder Block.
    
    Combines:
      - Pre-RMSNorm
      - Grouped-Query Attention (GQA) with Rotary Positional Embeddings (RoPE)
      - SwiGLU Feed-Forward Network
      - Residual Highway Connections
    """
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim=d_model)
        self.attn = GroupedQueryAttention(d_model=d_model, n_head=n_head, n_kv_head=n_kv_head, dropout=dropout)
        
        self.ffn_norm = RMSNorm(dim=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            start_pos: Starting sequence position index for KV-Cache & RoPE
            cos, sin: Precomputed RoPE frequency tensors
            kv_cache: Optional KVCache instance
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-RMSNorm Attention
        h = x + self.attn(self.attn_norm(x), start_pos=start_pos, cos=cos, sin=sin, kv_cache=kv_cache)
        # Pre-RMSNorm SwiGLU FFN
        out = h + self.ffn(self.ffn_norm(h))
        return out

def main():
    print("=" * 60)
    print("Part 3: Modern LLaMA-Style Transformer Decoder Block")
    print("=" * 60)

    batch_size = 1
    d_model = 16
    n_head = 8
    n_kv_head = 2
    max_seq_len = 32

    torch.manual_seed(42)

    head_dim = d_model // n_head
    cos, sin = precompute_freqs_cis(dim=head_dim, end=max_seq_len)

    block = ModernTransformerBlock(d_model=d_model, n_head=n_head, n_kv_head=n_kv_head)
    kv_cache = KVCache(max_batch_size=batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head, head_dim=head_dim)

    print(f"\n[Step 1] Block Configuration:")
    print(f"  Norm: RMSNorm | Attention: GQA ({n_head}Q : {n_kv_head}KV) + RoPE | FFN: SwiGLU")

    # 1. Prompt Prefill Phase (T=4)
    prompt_len = 4
    x_prompt = torch.randn(batch_size, prompt_len, d_model)
    out_prompt = block(x_prompt, start_pos=0, cos=cos, sin=sin, kv_cache=kv_cache)

    print(f"\n[Step 2] Prefill Forward Pass:")
    print(f"  Input Prompt shape:  {x_prompt.shape}")
    print(f"  Output Prompt shape: {out_prompt.shape}")

    # 2. Incremental Decoding Step (T=1)
    x_step = torch.randn(batch_size, 1, d_model)
    out_step = block(x_step, start_pos=prompt_len, cos=cos, sin=sin, kv_cache=kv_cache)

    print(f"\n[Step 3] Autoregressive Generation Step (T=1):")
    print(f"  Input Token shape:   {x_step.shape}")
    print(f"  Output Token shape:  {out_step.shape}")

if __name__ == "__main__":
    main()