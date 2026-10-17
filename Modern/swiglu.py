import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Used in modern LLM architectures like LLaMA, PaLM, and Mistral.
    
    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            # LLaMA default: 2/3 * 4 * d_model rounded to nearest multiple of 64
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
            
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        gate = F.silu(self.w_gate(x))  # SiLU is Swish with beta=1
        up = self.w_up(x)
        return self.w_down(gate * up)

def main():
    print("=" * 60)
    print("Part 3: SwiGLU Feed-Forward Network (Gated Activation)")
    print("=" * 60)

    batch_size = 2
    seq_len = 4
    d_model = 16

    torch.manual_seed(42)

    swiglu = SwiGLU(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n[Step 1] Input Tensor shape: {x.shape} (B={batch_size}, T={seq_len}, C={d_model})")
    
    out = swiglu(x)
    print(f"[Step 2] Output Tensor shape: {out.shape} (B={batch_size}, T={seq_len}, C={d_model})")

    print("\n[Step 3] SwiGLU Projection Dimensions:")
    print(f"  Input Dimension (d_model):      {d_model}")
    print(f"  Hidden Dimension (d_ff):        {swiglu.w_gate.out_features} (LLaMA 8/3 scaling)")
    print(f"  Gate Projection (w_gate):       ({d_model} -> {swiglu.w_gate.out_features})")
    print(f"  Up Projection (w_up):           ({d_model} -> {swiglu.w_up.out_features})")
    print(f"  Down Projection (w_down):       ({swiglu.w_down.in_features} -> {d_model})")

if __name__ == "__main__":
    main()