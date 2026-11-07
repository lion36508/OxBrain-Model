# OxBrain / NEXUS-5B

## NEXUS-5B text-first core

The repository now contains an initial executable NEXUS text-intelligence core in
`nexus5b.py`, with regression tests in `test_nexus5b.py`.

### Scope
- UTF-8 byte tokenizer (lossless byte-level round-trip).
- Causal recurrent state processing; no self-attention or Transformer modules.
- Eight learned experts with top-2 router selection.
- Ternary straight-through weight quantization research layer.
- SQLite persistent exact-text memory (lexical matching, not semantic vector search).
- Tool registry with explicit confirmation before any registered handler executes.

### Important engineering status
This is a runnable small-scale research implementation, **not a trained 5-billion-parameter model**.
The 5B value is retained as a target budget in the configuration; actual instantiated parameter
count is printed by `parameter_report()`. Do not infer model capability from the target number.
The recurrent block is a custom causal recurrent design; it is **not claimed to be a faithful
RWKV-7 implementation**, and the ternary layer is not Microsoft's official BitNet runtime.
No vision, image generation, CNN, GAN, VAE, GNN, or self-attention path is included.

### Run
Install PyTorch, then run from the repository root:

```bash
python -m pip install torch
python nexus5b.py --smoke
python -m unittest -v test_nexus5b.py
```

Existing OxBrain training data and checkpoints are preserved. The legacy files are not silently
rewritten or loaded by the NEXUS core because their tensor shapes and attention-based architecture
are incompatible with this recurrent model.
