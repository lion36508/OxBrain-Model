"""NEXUS-5B text-first research core.

This is an executable, attention-free recurrent/MoE prototype and runtime boundary.
It is NOT a pretrained 5B model and does not claim bit-exact RWKV-7 compatibility.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class NexusConfig:
    vocab_size: int = 259             # UTF-8 byte IDs: 0..255 plus BOS/EOS/PAD
    d_model: int = 256
    n_layers: int = 6
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 512
    dropout: float = 0.0
    max_seq_len: int = 2048
    target_parameters: int = 5_000_000_000  # design target, not allocated by default

    def validate(self) -> None:
        if self.vocab_size < 259:
            raise ValueError("vocab_size must be >=259 for the byte tokenizer")
        if min(self.d_model, self.n_layers, self.n_experts, self.top_k,
               self.expert_hidden, self.max_seq_len) <= 0:
            raise ValueError("All model dimensions must be positive")
        if self.top_k > self.n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer. IDs 0..255 are byte values."""
    BOS, EOS, PAD = 256, 257, 258

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        return ([self.BOS] if add_bos else []) + ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(i for i in ids if 0 <= i <= 255)
        return data.decode("utf-8", errors="replace")


class TernaryLinear(nn.Module):
    """BitNet-b1.58-style ternary weight STE; not Microsoft's official kernel."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def quantized_weight(self) -> Tensor:
        w = self.weight
        scale = w.detach().abs().mean().clamp_min(1e-5)
        q = torch.round(w / scale).clamp(-1, 1) * scale
        return w + (q - w).detach()

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)


class RecurrentMix(nn.Module):
    """Causal per-channel recurrent state update; contains no attention."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.key = TernaryLinear(dim, dim, bias=False)
        self.value = TernaryLinear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = TernaryLinear(dim, dim, bias=False)
        self.decay_logit = nn.Parameter(torch.full((dim,), 2.0))
        self.gate = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        k = torch.tanh(self.key(z))
        v = self.value(z)
        r = torch.sigmoid(self.receptance(z))
        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        state = torch.zeros_like(k[:, 0])
        outputs = []
        for t in range(k.shape[1]):
            state = decay * state + (1.0 - decay) * (k[:, t] * v[:, t])
            outputs.append(r[:, t] * state)
        y = torch.stack(outputs, dim=1)
        return x + self.output(y) * torch.sigmoid(self.gate(z))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up_gate = TernaryLinear(dim, hidden * 2)
        self.down = TernaryLinear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.up_gate(self.norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class SparseMoE(nn.Module):
    """Top-k token routing across eight learned experts by default."""
    def __init__(self, dim: int, hidden: int, experts: int, top_k: int):
        super().__init__()
        self.n_experts, self.top_k = experts, top_k
        self.router = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                TernaryLinear(dim, hidden * 2),
                nn.GLU(dim=-1),
                TernaryLinear(hidden, dim),
            ) for _ in range(experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        probs = self.router(flat).softmax(dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(top_i == expert_id)
            if token_idx.numel() == 0:
                continue
            contribution = expert(flat.index_select(0, token_idx))
            contribution = contribution * top_p[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, contribution)
        # Switch-style balancing diagnostic/loss, not a forced expert assignment.
        importance = probs.mean(dim=0)
        load = F.one_hot(top_i, self.n_experts).float().sum(dim=(0, 1))
        load = load / load.sum().clamp_min(1.0)
        balance_loss = self.n_experts * torch.sum(importance * load)
        return out.reshape(b, t, d), balance_loss


class NexusBlock(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.recurrent = RecurrentMix(cfg.d_model)
        self.moe = SparseMoE(cfg.d_model, cfg.expert_hidden,
                             cfg.n_experts, cfg.top_k)
        self.channel = SwiGLU(cfg.d_model, cfg.expert_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.recurrent(x)
        moe_out, aux = self.moe(x)
        x = self.dropout(x + moe_out)
        x = self.channel(x)
        return x, aux


class NEXUS5B(nn.Module):
    """Text-only recurrent language model. No CNN, vision path or self-attention."""
    def __init__(self, cfg: NexusConfig | None = None):
        super().__init__()
        self.cfg = cfg or NexusConfig()
        self.cfg.validate()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([NexusBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.final_norm = nn.LayerNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = self.embedding(input_ids)
        aux_losses = []
        for block in self.blocks:
            x, aux = block(x)
            aux_losses.append(aux)
        logits = self.lm_head(self.final_norm(x))
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        result = {"logits": logits, "aux_loss": aux_loss}
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 labels.reshape(-1), ignore_index=ByteTokenizer.PAD)
            result["loss"] = ce + 0.01 * aux_loss
        return result

    def parameter_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total_parameters": total, "target_parameters": self.cfg.target_parameters,
                "target_gap": self.cfg.target_parameters - total}


class SQLiteMemory:
    """Persistent exact-text memory; retrieval is lexical, not semantic vector search."""
    def __init__(self, db_path: str | Path = "nexus_memory.sqlite3"):
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL, content TEXT NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category)")

    def add(self, content: str, category: str = "general") -> int:
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("INSERT INTO memories(category, content) VALUES (?, ?)",
                              (category[:64], content))
            return int(cur.lastrowid)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [w for w in query.split() if len(w) > 1][:8]
        if not terms:
            return []
        where = " OR ".join("content LIKE ?" for _ in terms)
        args = [f"%{term[:64]}%" for term in terms] + [max(1, min(limit, 20))]
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"SELECT id, created_at, category, content FROM memories WHERE {where} "
                "ORDER BY id DESC LIMIT ?", args).fetchall()
        return [dict(row) for row in rows]


class SafeTaskController:
    """Registry/planning boundary only. It never runs shell, Python or filesystem commands."""
    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        if not name.isidentifier():
            raise ValueError("tool name must be a Python identifier")
        self._tools[name] = handler

    def available_tools(self) -> list[str]:
        return sorted(self._tools)

    def dispatch(self, name: str, arguments: dict[str, Any], *,
                 user_confirmed: bool = False) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool is not registered: {name}")
        if not user_confirmed:
            raise PermissionError("Explicit user confirmation is required before tool execution")
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a dictionary")
        return self._tools[name](**arguments)


def smoke_test() -> None:
    torch.manual_seed(7)
    cfg = NexusConfig(vocab_size=259, d_model=32, n_layers=2,
                      n_experts=8, top_k=2, expert_hidden=64, max_seq_len=32)
    model = NEXUS5B(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        result = model(ids, labels=ids)
    assert result["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(result["loss"]).item()
    assert torch.isfinite(result["aux_loss"]).item()
    # Architectural guard: this model must not silently acquire attention modules.
    forbidden = [m for m in model.modules()
                 if isinstance(m, (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder))]
    assert not forbidden, "Attention/Transformer module detected"
    print(json.dumps({"status": "PASS", "shape": list(result["logits"].shape),
                      "loss": float(result["loss"]), **model.parameter_report()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS-5B text-first research core")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", action="store_true")
    args = parser.parse_args()
    if args.config:
        print(json.dumps(asdict(NexusConfig()), indent=2))
    if args.smoke:
        smoke_test()
    if not args.config and not args.smoke:
        parser.print_help()


if __name__ == "__main__":
    main()
