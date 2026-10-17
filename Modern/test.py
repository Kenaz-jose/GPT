import sys
import os
import torch
import torch.nn as nn
import unittest
import importlib

rmsnorm_module = importlib.import_module("rmsnorm")
swiglu_module = importlib.import_module("swiglu")
rope_module = importlib.import_module("rope")
gqa_module = importlib.import_module("gqa_kvcache")
block_module = importlib.import_module("block")

RMSNorm = rmsnorm_module.RMSNorm
SwiGLU = swiglu_module.SwiGLU
precompute_freqs_cis = rope_module.precompute_freqs_cis
apply_rotary_emb = rope_module.apply_rotary_emb
GroupedQueryAttention = gqa_module.GroupedQueryAttention
KVCache = gqa_module.KVCache
ModernTransformerBlock = block_module.ModernTransformerBlock

class TestPart3ModernArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 4
        self.d_model = 16
        self.n_head = 4
        self.n_kv_head = 2
        self.head_dim = self.d_model // self.n_head

    def test_01_rmsnorm(self):
        norm = RMSNorm(dim=self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = norm(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        rms_val = torch.sqrt(out.pow(2).mean(-1))
        torch.testing.assert_close(rms_val, torch.ones_like(rms_val), rtol=1e-3, atol=1e-3)

    def test_02_swiglu_shape(self):
        swiglu = SwiGLU(d_model=self.d_model)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = swiglu(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_03_rope_norm_preservation(self):
        cos, sin = precompute_freqs_cis(dim=self.head_dim, end=32)
        xq = torch.randn(self.batch_size, self.n_head, self.seq_len, self.head_dim)
        xk = torch.randn(self.batch_size, self.n_kv_head, self.seq_len, self.head_dim)
        
        xq_rope, xk_rope = apply_rotary_emb(xq, xk, cos[:self.seq_len], sin[:self.seq_len])
        
        torch.testing.assert_close(xq.norm(dim=-1), xq_rope.norm(dim=-1), rtol=1e-4, atol=1e-4)

    def test_04_gqa_kvcache_incremental_step(self):
        gqa = GroupedQueryAttention(d_model=self.d_model, n_head=self.n_head, n_kv_head=self.n_kv_head)
        kv_cache = KVCache(max_batch_size=1, max_seq_len=16, n_kv_head=self.n_kv_head, head_dim=self.head_dim)
        cos, sin = precompute_freqs_cis(dim=self.head_dim, end=16)

        x_prompt = torch.randn(1, 2, self.d_model)
        out_prompt = gqa(x_prompt, start_pos=0, cos=cos, sin=sin, kv_cache=kv_cache)
        self.assertEqual(out_prompt.shape, (1, 2, self.d_model))

        x_step = torch.randn(1, 1, self.d_model)
        out_step = gqa(x_step, start_pos=2, cos=cos, sin=sin, kv_cache=kv_cache)
        self.assertEqual(out_step.shape, (1, 1, self.d_model))

    def test_05_modern_block_forward(self):
        block = ModernTransformerBlock(d_model=self.d_model, n_head=self.n_head, n_kv_head=self.n_kv_head)
        cos, sin = precompute_freqs_cis(dim=self.head_dim, end=16)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        
        out = block(x, start_pos=0, cos=cos, sin=sin)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

if __name__ == "__main__":
    unittest.main(verbosity=2)