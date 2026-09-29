import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

sys.path.extend(".")

data_module = importlib.import_module("1_dataset_dataloader")
llm_module = importlib.import_module("2_tiny_LLM")
train_module = importlib.import_module("4_training_loop")

ByteTokenizer = data_module.ByteTokenizer
create_dataloader = data_module.create_dataloader
TinyLLM = llm_module.TinyLLM
train_model = train_module.train_model

@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cpu"
) -> str:
    """
    Autoregressively generates text continuation given a prompt.
    
    Args:
        model: Trained TinyLLM instance
        tokenizer: ByteTokenizer instance
        prompt: Initial string prompt
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (>0.0 for diversity, 0.0 for greedy)
        top_k: If set, restricts sampling to the top k most probable logits
        device: Device to run generation on
    Returns:
        Generated full text string
    """
    model.eval()
    tokens = tokenizer.encode(prompt).tolist()
    
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to model's max positional encoding length if sequence grows too long
        idx_cond = idx[:, -512:] if idx.size(1) > 512 else idx
        
        # Forward pass to get logits for all tokens in context
        logits = model(idx_cond)
        
        # Pluck logits at the very last sequence position: shape (1, vocab_size)
        logits = logits[:, -1, :]

        if temperature == 0.0 or temperature is None:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Apply temperature scaling
            logits = logits / temperature

            # Optional Top-K Filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert to probabilities & sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append generated token ID to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode entire token sequence back to text
    return tokenizer.decode(idx[0].tolist())

def main():
    print("=" * 60)
    print("Part 2: Autoregressive Text Generation in PyTorch")
    print("=" * 60)

    sample_text = (
        "Large Language Models learn patterns from text through next-token prediction. "
        "By training on text sequences, neural networks generate human-like text word by word. "
    ) * 4

    block_size = 16
    batch_size = 4
    num_epochs = 50

    torch.manual_seed(42)

    dataloader, tokenizer = create_dataloader(sample_text, block_size=block_size, batch_size=batch_size, shuffle=True)
    model = TinyLLM(vocab_size=256, d_model=32, n_head=4, n_layer=2)

    print("\n[Step 1] Training TinyLLM on sample text...")
    train_model(model, dataloader, num_epochs=num_epochs, lr=2e-3, checkpoint_path="tiny_llm_gen_ckpt.pt")

    prompt = "Large Language"
    print(f"\n[Step 2] Input Prompt: '{prompt}'")

    print("\n--- Generating with Greedy Search (temperature=0.0) ---")
    greedy_text = generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.0)
    print("Output:", repr(greedy_text))

    print("\n--- Generating with Temperature Sampling (temperature=0.8, top_k=5) ---")
    sampled_text = generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, top_k=5)
    print("Output:", repr(sampled_text))

if __name__ == "__main__":
    main()