import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import importlib

sys.path.extend(".")

data_module = importlib.import_module("1_dataset_dataloader")
llm_module = importlib.import_module("2_tiny_LLM")
loss_module = importlib.import_module("3_loss_eval")

create_dataloader = data_module.create_dataloader
TinyLLM = llm_module.TinyLLM
compute_loss = loss_module.compute_loss
evaluate_model = loss_module.evaluate_model

def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    checkpoint_path: str = "tiny_llm_checkpoint.pt"
):
    """
    Trains TinyLLM using AdamW optimizer with gradient clipping and checkpoint saving.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print("\n--- Starting Training Loop ---")
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        total_epoch_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = compute_loss(logits, y_batch)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            total_epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Avg Loss: {avg_epoch_loss:.4f}")

    # Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': loss_history[-1],
        'num_epochs': num_epochs
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved successfully to: {checkpoint_path}")

    return loss_history

def main():
    print("=" * 60)
    print("Part 2: AdamW Training Loop with Gradient Clipping & Checkpointing")
    print("=" * 60)

    sample_text = (
        "Large Language Models learn patterns from text through next-token prediction. "
        "By training on text sequences, the model learns grammar, syntax, and knowledge. "
    ) * 4

    block_size = 16
    batch_size = 4
    num_epochs = 60

    torch.manual_seed(42)

    dataloader, tokenizer = create_dataloader(sample_text, block_size=block_size, batch_size=batch_size, shuffle=True)
    model = TinyLLM(vocab_size=256, d_model=32, n_head=4, n_layer=2)

    # Pre-training loss
    initial_loss, initial_ppl = evaluate_model(model, dataloader)
    print(f"\n[Before Training] Loss: {initial_loss:.4f} | Perplexity: {initial_ppl:.4f}")

    # Run training
    loss_history = train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=2e-3,
        max_grad_norm=1.0
    )

    # Post-training loss
    final_loss, final_ppl = evaluate_model(model, dataloader)
    print(f"\n[After Training]  Loss: {final_loss:.4f} | Perplexity: {final_ppl:.4f}")
    print(f"Loss reduction: {initial_loss:.4f} -> {final_loss:.4f} ({(1 - final_loss/initial_loss)*100:.1f}% reduction)")

if __name__ == "__main__":
    main()