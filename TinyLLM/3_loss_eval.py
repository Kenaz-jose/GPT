import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import importlib

sys.path.extend([".", "/workspace/artifacts", "/workspace/out", "/workspace/scratch"])

llm_module = importlib.import_module("2_tiny_LLM")
data_module = importlib.import_module("1_dataset_dataloader")

TinyLLM = llm_module.TinyLLM
create_dataloader = data_module.create_dataloader

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Cross-Entropy Loss for Next-Token Prediction.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        targets: Tensor of shape (batch_size, seq_len)
    Returns:
        Scalar Cross-Entropy Loss
    """
    B, T, V = logits.size()
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    
    return F.cross_entropy(logits_flat, targets_flat)

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cpu"):
    """
    Evaluates average Cross-Entropy Loss and Perplexity across a DataLoader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = compute_loss(logits, y_batch)

        batch_tokens = x_batch.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    print("=" * 60)
    print("Part 2: Loss Calculation & Model Evaluation in PyTorch")
    print("=" * 60)

    sample_text = (
        "Large Language Models learn patterns from text through next-token prediction. "
        "Cross-entropy loss measures how well predicted probability distributions match ground truth target tokens."
    )

    vocab_size = 256
    d_model = 16
    n_head = 4
    n_layer = 2
    block_size = 16
    batch_size = 4

    torch.manual_seed(42)

    dataloader, tokenizer = create_dataloader(sample_text, block_size=block_size, batch_size=batch_size, shuffle=False)
    model = TinyLLM(vocab_size=vocab_size, d_model=d_model, n_head=n_head, n_layer=n_layer)

    x_batch, y_batch = next(iter(dataloader))
    logits = model(x_batch)

    loss = compute_loss(logits, y_batch)
    perplexity = math.exp(loss.item())

    print(f"\n[Step 1] Single Batch Loss Inspection:")
    print(f"  Logits shape:     {logits.shape}")
    print(f"  Targets shape:    {y_batch.shape}")
    print(f"  Initial Loss:     {loss.item():.4f}")
    print(f"  Initial Perplexity: {perplexity:.4f}")
    print(f"  Theoretical Random Loss (-ln(1/256)): {math.log(256):.4f}")

    avg_loss, eval_ppl = evaluate_model(model, dataloader)
    print(f"\n[Step 2] Full Dataset Evaluation:")
    print(f"  Average Loss:     {avg_loss:.4f}")
    print(f"  Perplexity:       {eval_ppl:.4f}")

if __name__ == "__main__":
    main()
