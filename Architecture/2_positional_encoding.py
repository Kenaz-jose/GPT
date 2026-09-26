import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding module as described in 'Attention Is All You Need'.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) initialized with zeros
        pe = torch.zeros(max_len, d_model)
        
        # Position vector shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Term for scaling frequencies in log space: 10000^(-2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (2i), cosine to odd indices (2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register pe as a persistent buffer (part of module state, not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        # Add positional encoding up to current sequence length T
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def main():
    print("=" * 60)
    print("Part 1: Sinusoidal Positional Encoding in PyTorch")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_model = 8

    torch.manual_seed(42)

    # Instantiate module
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.0)
    
    # Input embeddings (B, T, C)
    x = torch.zeros(batch_size, seq_len, d_model)
    print(f"\n[Step 1] Input Embeddings (X) shape: {x.shape}")

    # Forward pass
    output = pos_encoder(x)
    print(f"[Step 2] Output with Positional Encoding added shape: {output.shape}")

    # Display encodings for the first 3 positions
    pe_buffer = pos_encoder.pe.squeeze(0)[:3, :]
    print("\n[Step 3] Positional Encodings for positions 0, 1, 2:")
    for pos in range(3):
        vals = [f"{v:.4f}" for v in pe_buffer[pos].tolist()]
        print(f"  Pos {pos}: [{', '.join(vals)}]")

if __name__ == "__main__":
    main()
