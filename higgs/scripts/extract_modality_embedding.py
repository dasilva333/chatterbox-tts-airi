#!/usr/bin/env python3
"""Extract the fused multi-codebook audio embedding/head table to a .npy sidecar.

model.modality_embedding.weight [8208, 2560] fp16, where 8208 = 8 codebooks x 1026
(1024 codes + BOC 1024 + EOC 1025). Tied embedding/head: the prefill graph uses it
as the output head (Gemm node_linear_253 -> logits [1, 8, 1026]).

Run from the repo root (the directory containing models/).
"""
import numpy as np
import onnx

SRC = "models/higgs_audio_v3_ar_prefill_matmul4.onnx"
OUT = "refs/modality_embedding_fp16.npy"

m = onnx.load(SRC, load_external_data=False)
init = next(i for i in m.graph.initializer if i.name == "model.modality_embedding.weight")
kv = {e.key: e.value for e in init.external_data}
off, ln = int(kv["offset"]), int(kv["length"])
assert ln == 8208 * 2560 * 2, ln
with open("models/" + kv["location"], "rb") as f:
    f.seek(off)
    tbl = np.frombuffer(f.read(ln), dtype=np.float16).reshape(8208, 2560)
np.save(OUT, tbl)
print("saved", OUT, tbl.shape, tbl.dtype)
