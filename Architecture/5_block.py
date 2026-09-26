import sys
import os
import torch
import torch.nn as nn
import importlib

# Include local directory and workspace artifacts path
sys.path.extend([".", "/workspace/artifacts", "/workspace/out"])

# Dynamic imports since filenames begin with numbers
multihead_module = importlib.import_module("3_multihead_attention")
feedforward_module = importlib.import_module("4_feedforward_network")

MultiHeadCausalAttention = multihead_module.MultiHeadCausalAttention
PositionWiseFeedForward = feedforward_module.PositionWiseFeedForward

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Decoder Block.
    
    Structure:
      x = x + SelfAttention(LayerNorm1(x))
      x = x + FeedForward(LayerNorm2(x))
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-LN architecture for improved gradient stability during deep network training
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def main():
    print("=" * 60)
    print("Part 1: Transformer Decoder Block (Pre-LN) in PyTorch")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_model = 16
    n_head = 4

    torch.manual_seed(42)

    block = TransformerBlock(d_model=d_model, n_head=n_head, dropout=0.1)

    # Dummy input tensor (B, T, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n[Step 1] Input Tensor shape: {x.shape} (B={batch_size}, T={seq_len}, C={d_model})")

    # Forward pass
    out = block(x)
    print(f"[Step 2] Output Tensor shape: {out.shape} (B={batch_size}, T={seq_len}, C={d_model})")

    print("\n[Step 3] Architecture Flow Verification:")
    print("  Input -> LayerNorm1 -> MultiHeadCausalAttention -> Add (Residual)")
    print("        -> LayerNorm2 -> PositionWiseFeedForward -> Add (Residual) -> Output")

if __name__ == "__main__":
    main()
