import torch
from torch.utils.data import Dataset, DataLoader

class ByteTokenizer:
    """
    A simple Byte-Level Tokenizer mapping text strings to UTF-8 byte integers (0-255).
    """
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        bytes_data = text.encode('utf-8')
        return torch.tensor(list(bytes_data), dtype=torch.long)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='replace')

class TextDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction language modeling.
    Yields input sequence x of length block_size and target sequence y shifted by 1 position.
    """
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

def create_dataloader(text: str, block_size: int, batch_size: int, shuffle: bool = True):
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)
    dataset = TextDataset(tokens, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, tokenizer

def main():
    print("=" * 60)
    print("Part 2: Byte Tokenization & DataLoader for Tiny LLM")
    print("=" * 60)

    sample_text = (
        "Large Language Models learn patterns from text through next-token prediction. "
        "By breaking text into bytes or subwords, neural networks process sequences efficiently."
    )

    block_size = 16   # Context window size (seq_len)
    batch_size = 4    # Number of sequences per batch

    dataloader, tokenizer = create_dataloader(sample_text, block_size=block_size, batch_size=batch_size)

    print(f"\n[Step 1] Original Sample Text length: {len(sample_text)} characters")
    print(f"[Step 2] Byte Vocabulary Size: {tokenizer.vocab_size}")

    # Fetch one batch
    for x_batch, y_batch in dataloader:
        print(f"\n[Step 3] Batch Shapes:")
        print(f"  Input Tensor (X) shape:  {x_batch.shape} (Batch={x_batch.shape}, ContextWindow={x_batch.shape})")
        print(f"  Target Tensor (Y) shape: {y_batch.shape} (Batch={y_batch.shape}, ContextWindow={y_batch.shape})")

        print("\n[Step 4] First Sequence Inspection:")
        print("  X (Input token IDs):  ", x_batch.tolist())
        print("  Y (Target token IDs): ", y_batch.tolist())
        print("  Decoded X Text:       ", repr(tokenizer.decode(x_batch[0])))
        print("  Decoded Y Text:       ", repr(tokenizer.decode(y_batch[0])))
        break

if __name__ == "__main__":
    main()
