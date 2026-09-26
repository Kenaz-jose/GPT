import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (FFN) module.
    
    Expands d_model dimension by 4x and applies GELU activation:
    FFN(x) = Dropout(GELU(x @ W_1 + b_1) @ W_2 + b_2)
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        # Default to 4 * d_model as in standard Transformer architectures
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.c_fc = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def main():
    print("=" * 60)
    print("Part 1: Position-Wise Feed-Forward Network in PyTorch")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_model = 16
    d_ff = 4 * d_model  # 64

    torch.manual_seed(42)

    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)

    # Dummy input (B, T, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n[Step 1] Input Tensor shape: {x.shape} (B={batch_size}, T={seq_len}, C={d_model})")

    # Forward pass
    out = ffn(x)
    print(f"[Step 2] Output Tensor shape: {out.shape} (B={batch_size}, T={seq_len}, C={d_model})")
    
    print("\n[Step 3] Internal Dimension Flow Verification:")
    print(f"  Input:            {d_model}")
    print(f"  Expansion (c_fc):  {d_ff} (4x expansion)")
    print(f"  Activation:       GELU")
    print(f"  Projection (c_proj): {d_model}")

if __name__ == "__main__":
    main()
