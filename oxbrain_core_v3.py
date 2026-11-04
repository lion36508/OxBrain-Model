# OxBrain Ultra-Power V3.1 Architecture (Shape-Safe Pure Add/Sub + Sparse MoE)
import os
import sys
import gc
import json
import argparse
import numpy as np

CONTEXT = 32
D_MODEL = 64
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
NUM_EXPERTS = 4
TOP_K_EXPERTS = 2
BATCH = 8
MAX_VOCAB = 128

WEIGHTS_FILE = "weights_2bit_moe.npz"
CONFIG_FILE = "config_2bit_moe.json"

def quantize_2bit(w):
    scale = np.mean(np.abs(w)) + 1e-8
    w_norm = np.clip(w / scale, -1.0, 1.0)
    sign = np.where(w_norm >= 0.0, 1.0, -1.0)
    mag = np.where(np.abs(w_norm) >= 0.75, 1.0, 0.5)
    return (sign * mag).astype(np.float32), float(scale)

def add_sub_gemm(x, qw):
    half_x = x * 0.5
    p1 = (qw == 1.0).astype(np.float32)
    m1 = (qw == -1.0).astype(np.float32)
    p05 = (qw == 0.5).astype(np.float32)
    m05 = (qw == -0.5).astype(np.float32)

    return (np.matmul(x, p1.T) - 
            np.matmul(x, m1.T) + 
            np.matmul(half_x, p05.T) - 
            np.matmul(half_x, m05.T))

class UltraLinear2Bit:
    def __init__(self, in_f, out_f):
        limit = np.sqrt(6.0 / (in_f + out_f))
        self.w = np.random.uniform(-limit, limit, (out_f, in_f)).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.mw = np.zeros_like(self.w)
        self.vw = np.zeros_like(self.w)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.qw, self.scale = quantize_2bit(self.w)
        return add_sub_gemm(x, self.qw) * self.scale + self.b

    def backward(self, d_out, lr=0.002, wd=0.01):
        d_out_flat = d_out.reshape(-1, d_out.shape[-1])
        x_flat = self.x.reshape(-1, self.x.shape[-1])

        grad_w = np.matmul(d_out_flat.T, x_flat)
        grad_b = np.sum(d_out_flat, axis=0)
        grad_x = np.matmul(d_out_flat, self.qw) * self.scale

        self.mw = 0.9 * self.mw + 0.1 * grad_w
        self.vw = 0.999 * self.vw + 0.001 * (grad_w ** 2)
        m_hat = self.mw / 0.1
        v_hat = self.vw / 0.001
        self.w -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + wd * self.w)

        self.mb = 0.9 * self.mb + 0.1 * grad_b
        self.vb = 0.999 * self.vb + 0.001 * (grad_b ** 2)
        self.b -= lr * (self.mb / (np.sqrt(self.vb) + 1e-8))

        return grad_x.reshape(self.x.shape)

class RMSNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        self.rms = 1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = x * self.rms
        return self.x_norm * self.gamma

    def backward(self, d_out):
        d_norm = d_out * self.gamma
        d_x = self.rms * (d_norm - self.x_norm * np.mean(d_norm * self.x_norm, axis=-1, keepdims=True))
        return d_x

class FastMoELayer:
    def __init__(self, dim, num_experts, top_k):
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = UltraLinear2Bit(dim, num_experts)
        self.w1 = [UltraLinear2Bit(dim, dim * 2) for _ in range(num_experts)]
        self.w2 = [UltraLinear2Bit(dim * 2, dim) for _ in range(num_experts)]

    def forward(self, x):
        self.x = x
        router_logits = self.gate.forward(x)
        exp_logits = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
        self.gate_probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)

        self.top_idx = np.argsort(self.gate_probs, axis=-1)[:, :, -self.top_k:]
        self.out = np.zeros_like(x)
        self.expert_acts = {}

        for e in range(self.num_experts):
            mask = np.any(self.top_idx == e, axis=-1, keepdims=True)
            if np.any(mask):
                h = np.maximum(0, self.w1[e].forward(x))
                self.expert_acts[e] = h
                weight = self.gate_probs[:, :, e:e+1]
                self.out += mask * weight * self.w2[e].forward(h)
        return self.out

    def backward(self, d_out, lr):
        d_x = np.zeros_like(self.x)
        for e in range(self.num_experts):
            if e in self.expert_acts:
                mask = np.any(self.top_idx == e, axis=-1, keepdims=True)
                weight = self.gate_probs[:, :, e:e+1]
                d_expert_out = d_out * mask * weight
                dh = self.w2[e].backward(d_expert_out, lr=lr)
                dh = dh * (self.expert_acts[e] > 0)
                d_x += self.w1[e].backward(dh, lr=lr)
        d_gate = self.gate.backward(np.zeros_like(self.gate_probs), lr=lr)
        return d_x + d_gate

class OxBrainUltra:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token_emb = np.random.uniform(-0.08, 0.08, (vocab_size, D_MODEL)).astype(np.float32)
        self.pos_emb = np.random.uniform(-0.08, 0.08, (CONTEXT, D_MODEL)).astype(np.float32)
        
        self.norm1 = RMSNorm(D_MODEL)
        self.qkv = UltraLinear2Bit(D_MODEL, D_MODEL * 3)
        self.attn_proj = UltraLinear2Bit(D_MODEL, D_MODEL)
        
        self.norm2 = RMSNorm(D_MODEL)
        self.moe = FastMoELayer(D_MODEL, NUM_EXPERTS, TOP_K_EXPERTS)
        
        self.norm_final = RMSNorm(D_MODEL)
        self.head = UltraLinear2Bit(D_MODEL, vocab_size)

    def forward(self, ids):
        self.ids = ids
        b, l = ids.shape
        self.h = self.token_emb[ids] + self.pos_emb[:l]

        h_norm1 = self.norm1.forward(self.h)
        qkv = self.qkv.forward(h_norm1).reshape(b, l, 3, HEADS, HEAD_DIM)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        self.v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(HEAD_DIM)
        scores += np.triu(np.ones((l, l)) * -1e9, k=1)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-8)
        
        attended = np.matmul(self.attn, self.v).transpose(0, 2, 1, 3).reshape(b, l, D_MODEL)
        self.h = self.h + self.attn_proj.forward(attended)

        h_norm2 = self.norm2.forward(self.h)
        self.h = self.h + self.moe.forward(h_norm2)

        return self.head.forward(self.norm_final.forward(self.h))

    def backward(self, d_logits, lr):
        d_final = self.head.backward(d_logits, lr=lr)
        d_h = self.norm_final.backward(d_final)
        
        d_moe = self.moe.backward(self.norm2.forward(d_h), lr=lr)
        d_h = d_h + self.norm2.backward(d_moe)
        
        d_attn = self.attn_proj.backward(d_h, lr=lr)
        d_h = d_h + self.norm1.backward(d_attn)

        np.add.at(self.token_emb, self.ids, -lr * d_h)

def train(steps):
    data_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".txt")]
    if not data_files:
        print("[Error] No training stream detected.")
        return

    with open(data_files[0], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chars = sorted(list(set(text)))[:MAX_VOCAB]
    vocab = ["\0", "\ufffd"] + [c for c in chars if c not in ("\0", "\ufffd")]
    mapping = {c: i for i, c in enumerate(vocab)}
    ids = np.array([mapping.get(c, 1) for c in text], dtype=np.int32)

    model = OxBrainUltra(len(vocab))

    # शेप-सुरक्षित वेट्स लोडिंग (Shape-Safe Checkpoint Loader)
    if os.path.exists(WEIGHTS_FILE):
        try:
            saved = np.load(WEIGHTS_FILE)
            # अगर डाइमेंशन मैच करे तभी लोड करो, नहीं तो फ्रेश इनिशियलाइज करो
            if saved["token_emb"].shape == model.token_emb.shape:
                model.token_emb = saved["token_emb"]
                model.qkv.w = saved["qkv_w"]
                model.qkv.b = saved["qkv_b"]
                model.head.w = saved["head_w"]
                model.head.b = saved["head_b"]
                print("[Engine V3.1] Compatible checkpoint matched and resumed.")
            else:
                print(f"[Engine V3.1] Checkpoint dimension mismatch ({saved['token_emb'].shape} vs {model.token_emb.shape}). Upgrading to D_MODEL={D_MODEL} clean state.")
        except Exception as e:
            print("[Engine V3.1] Initializing pristine parameter weights.")

    print(f"Architecture : 2-Bit Top-2 Sparse MoE ({D_MODEL} dim, {NUM_EXPERTS} experts)")
    print(f"Vocab / Tokens: {len(vocab)} vocab | {len(ids)} tokens")
    print(f"Executing {steps} high-throughput training steps...")

    offsets = np.arange(CONTEXT)
    initial_lr = 0.003

    for step in range(1, steps + 1):
        lr = max(0.0005, initial_lr * (1.0 - step / (steps + 1000)))
        idx = np.random.randint(0, len(ids) - CONTEXT - 1, size=BATCH)
        x_ids = ids[idx[:, None] + offsets]
        y_ids = ids[idx[:, None] + offsets + 1]

        logits = model.forward(x_ids)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= (np.sum(probs, axis=-1, keepdims=True) + 1e-8)

        correct = probs[np.arange(BATCH)[:, None], np.arange(CONTEXT), y_ids]
        loss = -np.mean(np.log(correct + 1e-10))

        d_logits = probs.copy()
        d_logits[np.arange(BATCH)[:, None], np.arange(CONTEXT), y_ids] -= 1.0
        d_logits /= (BATCH * CONTEXT)
        model.backward(d_logits, lr=lr)

        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Step {step:4d}/{steps} | Loss: {loss:.4f} (Active Sparse-MoE Update)")

    np.savez_compressed(
        WEIGHTS_FILE,
        token_emb=model.token_emb,
        qkv_w=model.qkv.w,
        qkv_b=model.qkv.b,
        head_w=model.head.w,
        head_b=model.head.b
    )
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "steps": steps, "d_model": D_MODEL, "engine": "OxBrain-Ultra-V3.1"}, f)
    print("Checkpoint successfully compiled and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "chat"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "train":
        train(args.steps)
