import unittest
import torch
from nexus5b import NEXUS5B, NexusConfig, ByteTokenizer, SafeTaskController


class NexusCoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NexusConfig(vocab_size=259, d_model=24, n_layers=2,
                               n_experts=8, top_k=2, expert_hidden=48,
                               max_seq_len=32)
        self.model = NEXUS5B(self.cfg)

    def test_logits_and_loss(self):
        ids = torch.randint(0, 259, (2, 10))
        out = self.model(ids, labels=ids)
        self.assertEqual(tuple(out["logits"].shape), (2, 10, 259))
        self.assertTrue(torch.isfinite(out["loss"]).item())

    def test_no_attention_modules(self):
        forbidden = (torch.nn.MultiheadAttention, torch.nn.Transformer,
                     torch.nn.TransformerEncoder)
        self.assertFalse(any(isinstance(m, forbidden) for m in self.model.modules()))

    def test_byte_tokenizer_roundtrip(self):
        tok = ByteTokenizer()
        text = "NEXUS नमस्ते"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_tool_requires_confirmation(self):
        tools = SafeTaskController()
        tools.register("sum_values", lambda a, b: a + b)
        with self.assertRaises(PermissionError):
            tools.dispatch("sum_values", {"a": 2, "b": 3})
        self.assertEqual(tools.dispatch("sum_values", {"a": 2, "b": 3},
                                        user_confirmed=True), 5)


if __name__ == "__main__":
    unittest.main()
