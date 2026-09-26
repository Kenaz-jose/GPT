import sys
import os
import torch
import torch.nn as nn
import unittest
import importlib

sys.path.extend([".", "/workspace/artifacts", "/workspace/out", "/workspace/scratch"])

pos_module = importlib.import_module("2_positional_encoding")
mha_module = importlib.import_module("3_multihead_attention")
ffn_module = importlib.import_module("4_feedforward_network")
block_module = importlib.import_module("5_block")

PositionalEncoding = pos_module.PositionalEncoding
MultiHeadCausalAttention = mha_module.MultiHeadCausalAttention
PositionWiseFeedForward = ffn_module.PositionWiseFeedForward
TransformerBlock = block_module.TransformerBlock

class TestPart1Transformer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 16
        self.n_head = 4

    def test_01_positional_encoding(self):
        pe = PositionalEncoding(d_model=self.d_model, max_len=100, dropout=0.0)
        x = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        out = pe(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertTrue(hasattr(pe, 'pe'))
        self.assertFalse(pe.pe.requires_grad)

    def test_02_causal_masking_isolation(self):
        """
        Crucial Causal Test: Changing future tokens (t > 1) MUST NOT change 
        the output at token position 1.
        """
        mha = MultiHeadCausalAttention(d_model=self.d_model, n_head=self.n_head, dropout=0.0)
        mha.eval()

        x1 = torch.randn(1, 4, self.d_model)
        x2 = x1.clone()
        # Mutate future tokens (positions 2 and 3)
        x2[0, 2:, :] = x2[0, 2:, :] + 10.0

        with torch.no_grad():
            out1 = mha(x1)
            out2 = mha(x2)

        # Output at position 0 and 1 must be identical across x1 and x2
        torch.testing.assert_close(out1[0, :2, :], out2[0, :2, :], rtol=1e-5, atol=1e-5)

    def test_03_feed_forward_shape(self):
        ffn = PositionWiseFeedForward(d_model=self.d_model, dropout=0.0)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = ffn(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_04_transformer_block_gradient_flow(self):
        block = TransformerBlock(d_model=self.d_model, n_head=self.n_head, dropout=0.1)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        out = block(x)
        
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())

if __name__ == "__main__":
    unittest.main(verbosity=2)