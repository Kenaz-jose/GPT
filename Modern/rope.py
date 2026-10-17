import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes frequency tensor for complex exponentials (cos and sin values) for Rotary Positional Embeddings (RoPE).
    
    Args:
        dim: Dimension per attention head (head_dim, must be even)
        end: Maximum sequence length to precompute
        theta: Base frequency scaling factor (default 10000.0)
    Returns:
        cos, sin tensors of shape (end, dim // 2)
    """
    assert dim % 2 == 0, "head_dim must be an even number for RoPE."
    # Frequencies: theta_i = 10000^(-2(i-1)/dim) for i in [1, dim/2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Outer product (end, dim // 2)
    freqs = torch.outer(t, freqs)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embeddings to Query and Key tensors.
    
    Args:
        xq: Query tensor of shape (batch_size, n_head, seq_len, head_dim)
        xk: Key tensor of shape (batch_size, n_kv_head, seq_len, head_dim)
        cos: Precomputed cos values sliced to seq_len: (seq_len, head_dim // 2)
        sin: Precomputed sin values sliced to seq_len: (seq_len, head_dim // 2)
    Returns:
        xq_out, xk_out with RoPE applied.
    """
    # Reshape xq and xk into pairs along head_dim: (B, n_head, T, head_dim // 2, 2)
    xq_shaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_shaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_r, xq_i = xq_shaped.unbind(-1)  # Real and imaginary components
    xk_r, xk_i = xk_shaped.unbind(-1)
    
    # Reshape cos and sin for broadcasting: (1, 1, seq_len, head_dim // 2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # 2D rotation matrix multiplication / complex multiplication:
    # [cos  -sin] [x_r]   [x_r * cos - x_i * sin]
    # [sin   cos] [x_i] = [x_r * sin + x_i * cos]
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    
    # Stack back real/imaginary components and flatten back to head_dim
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def main():
    print("=" * 60)
    print("Part 3: Rotary Positional Embeddings (RoPE) in PyTorch")
    print("=" * 60)

    batch_size = 2
    n_head = 4
    seq_len = 5
    head_dim = 8  # Dimension per head

    torch.manual_seed(42)

    # 1. Precompute frequencies
    cos, sin = precompute_freqs_cis(dim=head_dim, end=100)
    print(f"\n[Step 1] Precomputed cos/sin tables shape for max_len=100: {cos.shape}")

    # Slice cos/sin to current sequence length
    cos_seq = cos[:seq_len]
    sin_seq = sin[:seq_len]

    # Dummy Queries and Keys (B, n_head, T, head_dim)
    xq = torch.randn(batch_size, n_head, seq_len, head_dim)
    xk = torch.randn(batch_size, n_head, seq_len, head_dim)

    print(f"\n[Step 2] Input Query Tensor (xq) shape: {xq.shape}")
    print(f"         Input Key Tensor   (xk) shape: {xk.shape}")

    # Apply RoPE
    xq_rope, xk_rope = apply_rotary_emb(xq, xk, cos_seq, sin_seq)

    print(f"\n[Step 3] Output Query Tensor (xq_rope) shape: {xq_rope.shape}")
    print(f"         Output Key Tensor   (xk_rope) shape: {xk_rope.shape}")

    # Verify rotation property: Norms before and after rotation should be identical
    norm_before = xq.norm(dim=-1)
    norm_after = xq_rope.norm(dim=-1)
    print("\n[Step 4] Norm Preservation Check (Rotation preserves vector magnitude):")
    print("  Max norm difference:", (norm_before - norm_after).abs().max().item())

if __name__ == "__main__":
    main()