import os
import gc
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from tinygrad import Tensor, nn, Context
from tinygrad.nn.state import get_state_dict, load_state_dict

ROOT = Path.home() / "OxBrain"
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights_2bit_moe.npz"
CONFIG = ROOT / "config_2bit_moe.json"

# Q-Mini-MoE Mobile CPU Configuration
CONTEXT = 24
D_MODEL = 32
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
NUM_EXPERTS = 4
BATCH = 2
MAX_VOCAB = 128

SEED_TEXT = "Hello. I am OxBrain Q-Mini-MoE.\n"

def clean_workspace(full_wipe=False):
    removed = []
    for f in [WEIGHTS, CONFIG, ROOT / "weights.npz", ROOT / "config.json", ROOT / "weights.tmp.npz"]:
        if f.exists():
            f.unlink()
            removed.append(f.name)
    if full_wipe and DATA.exists():
        shutil.rmtree(DATA)
        DATA.mkdir(parents=True, exist_ok=True)
        removed.append("data/*")
    gc.collect()
    print("Cleaned:", ", ".join(removed) if removed else "Already clean.")

def get_text():
    files = sorted(DATA.glob("*.txt"))
    if not files:
        (DATA / "starter.txt").write_text(SEED_TEXT, encoding="utf-8")
        files = [DATA / "starter.txt"]

    # 1GB क्रैश से बचने के लिए रैंडम फाइल से सिर्फ 5MB डेटा लोड करना
    file = random.choice(files)
    try:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(5 * 1024 * 1024) 
    except OSError as exc:
        print("File error:", file.name, exc)
        text = SEED_TEXT

    if len(text) < CONTEXT + 2:
        text = SEED_TEXT
    return text

def make_vocab(text):
    # अगर पुरानी कन्फिग मौजूद है, तो उसी डिक्शनरी का इस्तेमाल करें ताकि ट्रेनिंग रिज्यूम हो सके
    if CONFIG.exists():
        try:
            config = json.loads(CONFIG.read_text(encoding="utf-8"))
            vocab = config.get("vocab")
            if vocab:
                mapping = {char: i for i, char in enumerate(vocab)}
                return vocab, mapping
        except Exception:
            pass

    counts = Counter(text)
    chars = [
        char for char, _ in counts.most_common(MAX_VOCAB - 2)
        if char not in ("\0", "\ufffd")
    ]
    vocab = ["\0", "\ufffd"] + chars
    mapping = {char: i for i, char in enumerate(vocab)}
    return vocab, mapping

def one_hot(ids, vocab_size):
    batch, length = ids.shape
    result = np.zeros((batch, length, vocab_size), dtype=np.float32)
    rows = np.arange(batch)[:, None]
    cols = np.arange(length)[None, :]
    result[rows, cols, ids] = 1.0
    return result

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x):
        norm = x * (x.square().mean(axis=-1, keepdim=True) + self.eps).rsqrt()
        return norm * self.weight

class BitLinear2Bit:
    def __init__(self, in_features, out_features):
        self.weight = Tensor.kaiming_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features)

    def quantize_weights_2bit(self, w):
        scale = w.abs().mean() + 1e-8
        w_norm = (w / scale).clip(-1.0, 1.0)
        pos_mask = (w_norm >= 0.0).cast(w.dtype)
        sign = pos_mask * 2.0 - 1.0
        high_mag = (w_norm.abs() >= 0.75).cast(w.dtype)
        q_levels = sign * (0.5 + 0.5 * high_mag)
        w_q = w_norm + (q_levels - w_norm).detach()
        return w_q, scale

    def __call__(self, x):
        x_scale = x.abs().max(axis=-1, keepdim=True) + 1e-8
        x_norm = x / x_scale
        x_int = (x_norm * 127.0).round().clip(-128.0, 127.0) / 127.0
        x_q = x_norm + (x_int - x_norm).detach()
        w_q, w_scale = self.quantize_weights_2bit(self.weight)
        out = x_q.dot(w_q.transpose())
        return out * (w_scale * x_scale) + self.bias

class Expert2Bit:
    def __init__(self, dim):
        self.w1 = BitLinear2Bit(dim, dim * 2)
        self.w2 = BitLinear2Bit(dim * 2, dim)

    def __call__(self, x):
        return self.w2(self.w1(x).relu())

class MoELayer2Bit:
    def __init__(self, dim, num_experts):
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = [Expert2Bit(dim) for _ in range(num_experts)]

    def __call__(self, x):
        gate_probs = self.router(x).softmax(axis=-1)
        out = Tensor.zeros(*x.shape)
        for i, expert in enumerate(self.experts):
            gate = gate_probs[:, :, i:i+1]
            out = out + gate * expert(x)
        return out

class OxBrainQMiniMoE:
    def __init__(self, vocab_size):
        self.token_embedding = Tensor.kaiming_uniform(vocab_size, D_MODEL)
        self.position_embedding = Tensor.kaiming_uniform(CONTEXT, D_MODEL)
        self.norm1 = RMSNorm(D_MODEL)
        self.qkv = BitLinear2Bit(D_MODEL, D_MODEL * 3)
        self.attention_output = BitLinear2Bit(D_MODEL, D_MODEL)
        self.norm2 = RMSNorm(D_MODEL)
        self.moe = MoELayer2Bit(D_MODEL, NUM_EXPERTS)
        self.norm_final = RMSNorm(D_MODEL)
        self.output = BitLinear2Bit(D_MODEL, vocab_size)

    def __call__(self, x):
        batch, length, vocab_size = x.shape
        h = x.dot(self.token_embedding) + self.position_embedding.reshape(1, CONTEXT, D_MODEL)
        h_norm = self.norm1(h)
        qkv = self.qkv(h_norm).reshape(batch, CONTEXT, 3, HEADS, HEAD_DIM)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        scores = q.matmul(k.transpose(2, 3)) / (HEAD_DIM ** 0.5)
        mask = Tensor.ones(CONTEXT, CONTEXT).triu(1) * -1.0e9
        scores = scores + mask.reshape(1, 1, CONTEXT, CONTEXT)
        attention = scores.softmax(axis=-1)
        attended = attention.matmul(v).transpose(1, 2).reshape(batch, CONTEXT, D_MODEL)
        h = h + self.attention_output(attended)
        h = h + self.moe(self.norm2(h))
        return self.output(self.norm_final(h))

def load_weights(model, vocab):
    if not WEIGHTS.exists() or not CONFIG.exists():
        return False
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        if config.get("vocab") != vocab or config.get("d_model") != D_MODEL:
            return False
        with np.load(WEIGHTS, allow_pickle=False) as saved:
            state = {name: Tensor(saved[name].copy()) for name in saved.files}
        load_state_dict(model, state)
        return True
    except Exception as exc:
        print("Checkpoint load skipped:", exc)
        return False

def save_weights(model, vocab, steps):
    state = get_state_dict(model)
    arrays = {name: tensor.numpy() for name, tensor in state.items()}
    temporary = ROOT / "weights.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, WEIGHTS)
    CONFIG.write_text(
        json.dumps({
            "vocab": vocab,
            "context": CONTEXT,
            "d_model": D_MODEL,
            "heads": HEADS,
            "experts": NUM_EXPERTS,
            "quantization": "2-bit",
            "steps": steps
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def train(steps, clean_first=False):
    if clean_first:
        clean_workspace(full_wipe=False)

    print("\n========== OXBRAIN Q-MINI-MOE ==========")
    print("Device        : Mobile ARM CPU (Safe RAM Mode)")
    print("Quantization  : 2-Bit [-1.0, -0.5, +0.5, +1.0]")
    print("========================================")

    text = get_text()
    vocab, mapping = make_vocab(text)
    ids = np.asarray([mapping.get(char, 1) for char in text], dtype=np.int32)

    model = OxBrainQMiniMoE(len(vocab))
    resumed = load_weights(model, vocab)

    params = nn.state.get_parameters(model)
    print(f"Vocab Size    : {len(vocab)}")
    print(f"Resumed       : {resumed}")
    print(f"Target Steps  : {steps}\n")

    optimizer = nn.optim.Adam(params, lr=0.003)
    rng = np.random.default_rng()
    offsets = np.arange(CONTEXT)

    with Context(TRAINING=1):
        for step in range(1, steps + 1):
            starts = rng.integers(CONTEXT, len(ids) - CONTEXT - 1, size=BATCH)
            x_ids = ids[starts[:, None] + offsets]
            y_ids = ids[starts[:, None] + offsets + 1]

            x = Tensor(one_hot(x_ids, len(vocab)))
            y = Tensor(y_ids)

            optimizer.zero_grad()
            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)
            loss.backward()
            optimizer.step()

            if step == 1 or step % 10 == 0 or step == steps:
                print(f"Step {step:4d}/{steps} | Loss: {float(loss.numpy()):.4f}")

            if step % 25 == 0:
                gc.collect()

    save_weights(model, vocab, steps)
    print("\nTraining complete. Saved checkpoint:", WEIGHTS)

def chat(max_new):
    if not WEIGHTS.exists() or not CONFIG.exists():
        print("No trained model found. Run: python oxbrain.py train")
        return

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    vocab = config["vocab"]
    mapping = {char: i for i, char in enumerate(vocab)}

    model = OxBrainQMiniMoE(len(vocab))
    if not load_weights(model, vocab):
        print("Checkpoint incompatible. Run with --clean.")
        return

    print("\n========== Q-MINI-MOE CHAT ==========")
    print("Type /exit to quit | /clean to reset")
    print("=====================================")

    rng = np.random.default_rng()

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("/exit", "exit", "quit"):
            break
        if prompt.lower() == "/clean":
            clean_workspace()
            break
        if not prompt:
            continue

        history = [mapping.get(char, 1) for char in prompt][-CONTEXT:]
        history = [0] * (CONTEXT - len(history)) + history
        generated = []

        for _ in range(max_new):
            ids = np.asarray([history[-CONTEXT:]], dtype=np.int32)
            x = Tensor(one_hot(ids, len(vocab)))

            logits = model(x).numpy()[0, -1].astype(np.float64)
            logits /= 0.8
            logits -= np.max(logits)

            probs = np.exp(logits)
            probs[0] = 0

            top_k = min(8, len(vocab) - 1)
            top = np.argpartition(probs, -top_k)[-top_k:]

            filtered = np.zeros_like(probs)
            filtered[top] = probs[top]
            total = filtered.sum()

            if not np.isfinite(total) or total <= 0:
                break

            filtered /= total
            next_id = int(rng.choice(len(vocab), p=filtered))
            history.append(next_id)

            char = vocab[next_id]
            if char not in ("\0", "\ufffd"):
                generated.append(char)

            if char == "\n" and len(generated) > 25:
                break

        print("OxBrain>", "".join(generated))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--steps", type=int, default=60)
    train_p.add_argument("--clean", action="store_true")
    chat_p = sub.add_parser("chat")
    chat_p.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    if args.command == "train":
        train(args.steps, clean_first=args.clean)
    elif args.command == "chat":
        chat(args.max_new)

if __name__ == "__main__":
    main()
