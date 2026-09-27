import sys
import os
import torch
import torch.nn as nn
import importlib

ARCH_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Architecture"
    )
)

# Tell Python where the Architecture modules are
sys.path.insert(0, ARCH_DIR)

pos_module = importlib.import_module("2_positional_encoding")
block_module = importlib.import_module("5_block")

PositionalEncoding = pos_module.PositionalEncoding
TransformerBlock = block_module.TransformerBlock

class TinyLLM(nn.Module):
    """
    A full Causal Decoder-Only Transformer Language Model.
    
    Structure:
      - Token Embedding Layer (vocab_size -> d_model)
      - Positional Encoding Layer
      - Stack of N TransformerBlocks
      - Final Layer Normalization
      - LM Head (d_model -> vocab_size) with optional weight tying
    """
    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int, max_len: int = 512, d_ff: int = None, dropout: float = 0.1, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Optional weight tying between token embedding and output projection
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
            
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of shape (batch_size, seq_len) with token indices.
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
        """
        B, T = idx.size()
        
        # Token embeddings + Positional Encoding
        x = self.tok_emb(idx) * (self.d_model ** 0.5)  # Scale embeddings
        x = self.pos_enc(x)
        
        # Pass through stacked Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def main():
    print("=" * 60)
    print("Part 2: TinyLLM Architecture Assembly in PyTorch")
    print("=" * 60)

    vocab_size = 256  # Byte vocabulary
    d_model = 16
    n_head = 4
    n_layer = 2
    batch_size = 2
    seq_len = 8

    torch.manual_seed(42)

    model = TinyLLM(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer, tie_weights=True)

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n[Step 1] Input Token IDs shape: {dummy_input.shape} (B={batch_size}, T={seq_len})")

    logits = model(dummy_input)
    print(f"[Step 2] Output Logits shape: {logits.shape} (B={batch_size}, T={seq_len}, Vocab={vocab_size})")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Step 3] Model Parameter Count:")
    print(f"  Total Parameters:     {num_params:,}")
    print(f"  Trainable Parameters: {num_trainable_params:,}")

if __name__ == "__main__":
    main()
