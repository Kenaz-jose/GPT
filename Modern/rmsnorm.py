import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """    
    RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

def main():
    print("=" * 60)
    print("Part 3: RMSNorm (Root Mean Square Normalization)")
    print("=" * 60)

    batch_size = 2
    seq_len = 4
    d_model = 8

    torch.manual_seed(42)

    rmsnorm = RMSNorm(dim=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n[Step 1] Input Tensor shape: {x.shape}")
    print("Input Sample (first token):\n", x)

    out = rmsnorm(x)
    print(f"\n[Step 2] Output Tensor shape: {out.shape}")
    print("Normalized Output Sample (first token):\n", out)

    # RMS of output before weight scaling should be ~1.0
    rms_val = torch.sqrt(out.pow(2).mean(-1))
    print(f"\n[Step 3] Root Mean Square value per token (should be ~1.0):\n", rms_val)

if __name__ == "__main__":
    main()
