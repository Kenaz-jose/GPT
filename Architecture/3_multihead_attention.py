import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module built from scratch in PyTorch.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Unified projection for Query, Key, and Value
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, d_model

        # 1. Unified linear projection for Q, K, V -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape for multi-head attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply lower-triangular causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # 5. Softmax & Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Weighted values & combine heads -> (B, T, C)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Final output projection
        return self.resid_dropout(self.c_proj(out))