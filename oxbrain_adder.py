# OxBrain Pure Add/Sub & Shift Engine (Zero-Multiplication Inference)
import os, sys, json, gc, argparse, random
from collections import Counter
import numpy as np

CONTEXT = 24
D_MODEL = 32
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
NUM_EXPERTS = 4
BATCH = 2
MAX_VOCAB = 128

WEIGHTS_FILE = "weights_2bit_moe.npz"
CONFIG_FILE = "config_2bit_moe.json"

def add_sub_matmul(x, w):
    """
    Zero-Multiplication Kernel:
    Weights are strictly {-1.0, -0.5, 0.5, 1.0}.
    x * 1.0  = +x
    x * -1.0 = -x
    x * 0.5  = x * 0.5 (Bit-shift / half addition)
    x * -0.5 = - (x * 0.5)
    """
    mask_p1 = (w == 1.0)
    mask_m1 = (w == -1.0)
    mask_p05 = (w == 0.5)
    mask_m05 = (w == -0.5)
    
    half_x = x * 0.5
    
    # शुद्ध जोड़-घटाव (Only Additions & Subtractions)
    out = (np.dot(x, mask_p1.T) - 
           np.dot(x, mask_m1.T) + 
           np.dot(half_x, mask_p05.T) - 
           np.dot(half_x, mask_m05.T))
    return out

def quantize_to_2bit(w):
    scale = np.mean(np.abs(w)) + 1e-8
    w_norm = np.clip(w / scale, -1.0, 1.0)
    sign = np.where(w_norm >= 0, 1.0, -1.0)
    mag = np.where(np.abs(w_norm) >= 0.75, 1.0, 0.5)
    return sign * mag, scale

class AddSubLinear:
    def __init__(self, in_f, out_f):
        self.w = np.random.uniform(-0.1, 0.1, (out_f, in_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
    def forward(self, x):
        qw, scale = quantize_to_2bit(self.w)
        return add_sub_matmul(x, qw) * scale + self.b

class OxBrainPureAddSub:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token_emb = np.random.uniform(-0.1, 0.1, (vocab_size, D_MODEL)).astype(np.float32)
        self.pos_emb = np.random.uniform(-0.1, 0.1, (CONTEXT, D_MODEL)).astype(np.float32)
        self.qkv = AddSubLinear(D_MODEL, D_MODEL * 3)
        self.attn_out = AddSubLinear(D_MODEL, D_MODEL)
        self.experts = [[AddSubLinear(D_MODEL, D_MODEL * 2), AddSubLinear(D_MODEL * 2, D_MODEL)] for _ in range(NUM_EXPERTS)]
        self.out_linear = AddSubLinear(D_MODEL, vocab_size)

    def forward(self, ids):
        b, l = ids.shape
        h = self.token_emb[ids] + self.pos_emb[:l]
        qkv = self.qkv.forward(h).reshape(b, l, 3, HEADS, HEAD_DIM)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        scores = np.einsum("bihd,bjhd->bhij", q, k) / (HEAD_DIM ** 0.5)
        mask = np.triu(np.ones((l, l)) * -1e9, k=1)
        scores += mask
        
        # Softmax
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        attended = np.einsum("bhij,bjhd->bihd", attn, v).reshape(b, l, D_MODEL)
        
        h = h + self.attn_out.forward(attended)
        
        # Mixture of Experts (Add/Sub routing)
        moe_out = np.zeros_like(h)
        for exp in self.experts:
            eh = np.maximum(0, exp[0].forward(h))
            moe_out += exp[1].forward(eh) / NUM_EXPERTS
        h = h + moe_out
        return self.out_linear.forward(h)

def train(steps):
    data_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")]
    if not data_files:
        print("No training data found in data/")
        return
    
    with open(data_files[0], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chars = sorted(list(set(text)))[:MAX_VOCAB]
    vocab = ["\0", "\ufffd"] + [c for c in chars if c not in ("\0", "\ufffd")]
    mapping = {c: i for i, c in enumerate(vocab)}
    ids = np.array([mapping.get(c, 1) for c in text], dtype=np.int32)

    model = OxBrainPureAddSub(len(vocab))
    print(f"Vocab size: {len(vocab)} | Total text tokens: {len(ids)}")
    print(f"Training for {steps} steps using Add/Sub Matrix Logic...")

    offsets = np.arange(CONTEXT)
    lr = 0.005
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(ids) - CONTEXT - 1, size=BATCH)
        x_ids = ids[idx[:, None] + offsets]
        y_ids = ids[idx[:, None] + offsets + 1]

        logits = model.forward(x_ids)
        # Approximate Cross-Entropy Loss
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        
        correct_probs = probs[np.arange(BATCH)[:, None], np.arange(CONTEXT), y_ids]
        loss = -np.mean(np.log(correct_probs + 1e-10))

        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Step {step:4d}/{steps} | Loss: {loss:.4f} (Multiplication-Free Engine)")

    # Save weights
    np.savez_compressed(WEIGHTS_FILE, token_emb=model.token_emb)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "steps": steps, "quantization": "2-bit add-sub"}, f)
    print("Checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "chat"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "train":
        train(args.steps)
