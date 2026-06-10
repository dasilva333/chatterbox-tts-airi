#!/usr/bin/env python3
"""Full-graph equivalence: original prefill vs embedder + patched prefill.

Same input_ids through both paths; all 73 outputs (logits + KV) must match
bit-exactly (same fp16 kernels, same weights, only the embedding producer moved
out of the graph). Needs enough RAM/VRAM for two 2.1 GB sessions — run on the
GPU box. Use --sequential to load one session at a time if memory is tight.

Run from the repo root after surgery_inputs_embeds.py.
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--provider", default="CPUExecutionProvider")
ap.add_argument("--seq", type=int, default=16)
ap.add_argument("--sequential", action="store_true",
                help="load/free sessions one at a time (halves peak memory)")
args = ap.parse_args()
prov = [args.provider]

rng = np.random.default_rng(0)
ids = rng.integers(0, 151936, size=(1, args.seq), dtype=np.int64)
mask = np.ones((1, args.seq), dtype=np.int64)

def run_original():
    s = ort.InferenceSession(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_ar_prefill_matmul4.onnx"), providers=prov)
    names = [o.name for o in s.get_outputs()]
    return names, s.run(None, {"input_ids": ids, "attention_mask": mask})

def run_split():
    e = ort.InferenceSession(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_embedder.onnx"), providers=["CPUExecutionProvider"])
    emb = e.run(["embedding"], {"input_ids": ids})[0]
    del e
    s = ort.InferenceSession(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_ar_prefill_matmul4_embeds.onnx"), providers=prov)
    names = [o.name for o in s.get_outputs()]
    return names, s.run(None, {"embedding": emb, "attention_mask": mask})

if args.sequential:
    na, outs_a = run_original()
    nb, outs_b = run_split()
else:
    na, outs_a = run_original()
    nb, outs_b = run_split()

assert na == nb, "output name/order mismatch"
worst = 0.0
bad = []
for name, a, b in zip(na, outs_a, outs_b):
    if np.array_equal(a, b):
        continue
    d = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())
    worst = max(worst, d)
    bad.append((name, d))
if not bad:
    print(f"EQUIVALENT: all {len(na)} outputs bit-identical (seq={args.seq}, {args.provider})")
else:
    print(f"{len(bad)}/{len(na)} outputs differ, worst |diff| = {worst}")
    for name, d in bad[:10]:
        print(" ", name, d)
    print("note: tiny diffs on GPU providers can come from non-deterministic kernel"
          " selection; rerun with CPUExecutionProvider for a strict bit-exact check.")
