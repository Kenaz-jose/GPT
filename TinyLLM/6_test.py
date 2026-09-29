import sys
import os
import torch
import torch.nn as nn
import unittest
import importlib

sys.path.extend(".")

data_module = importlib.import_module("1_dataset_dataloader")
llm_module = importlib.import_module("2_tiny_LLM")
loss_module = importlib.import_module("3_loss_eval")
gen_module = importlib.import_module("5_text_generation")

ByteTokenizer = data_module.ByteTokenizer
TextDataset = data_module.TextDataset
TinyLLM = llm_module.TinyLLM
compute_loss = loss_module.compute_loss
generate_text = gen_module.generate_text

class TestPart2TinyLLM(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = ByteTokenizer()
        self.vocab_size = 256
        self.d_model = 16
        self.n_head = 4
        self.n_layer = 2
        self.block_size = 8

    def test_01_tokenizer_roundtrip(self):
        text = "Hello LLM world!"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(text, decoded)

    def test_02_dataset_sequence_shift(self):
        text = "0123456789"
        tokens = self.tokenizer.encode(text)
        dataset = TextDataset(tokens, block_size=4)
        x, y = dataset[0]
        # Verify target sequence is shifted by 1 index
        torch.testing.assert_close(x[1:], y[:-1])

    def test_03_model_forward_logits_shape(self):
        model = TinyLLM(vocab_size=self.vocab_size, d_model=self.d_model, n_head=self.n_head, n_layer=self.n_layer)
        dummy_idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        logits = model(dummy_idx)
        self.assertEqual(logits.shape, (2, self.block_size, self.vocab_size))

    def test_04_loss_computation_gradients(self):
        model = TinyLLM(vocab_size=self.vocab_size, d_model=self.d_model, n_head=self.n_head, n_layer=self.n_layer)
        dummy_idx = torch.randint(0, self.vocab_size, (2, self.block_size))
        dummy_targets = torch.randint(0, self.vocab_size, (2, self.block_size))
        
        logits = model(dummy_idx)
        loss = compute_loss(logits, dummy_targets)
        loss.backward()

        self.assertIsNotNone(model.tok_emb.weight.grad)
        self.assertFalse(torch.isnan(loss).item())

    def test_05_autoregressive_generation(self):
        model = TinyLLM(vocab_size=self.vocab_size, d_model=self.d_model, n_head=self.n_head, n_layer=self.n_layer)
        prompt = "Test"
        gen_output = generate_text(model, self.tokenizer, prompt, max_new_tokens=10, temperature=0.0)
        self.assertTrue(gen_output.startswith(prompt))
        self.assertGreater(len(gen_output), len(prompt))

if __name__ == "__main__":
    unittest.main(verbosity=2)
