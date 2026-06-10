#!/usr/bin/env python3
"""Low-RAM seam verification (CPU, ~300 MB).

1. Runs models/higgs_audio_v3_embedder.onnx under onnxruntime on probe tokens.
2. Independently dequantizes the same rows by reading raw int4 + fp16 scales
   straight from the .onnx.data byte ranges (signed int4, low nibble first,
   block=128 along the hidden dim).
3. Asserts bit-exact match, then shape-infers the patched prefill graph.

Run from the repo root after surgery_inputs_embeds.py.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

V, D, BLK = 151936, 2560, 128

m = onnx.load(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_ar_prefill_matmul4.onnx"), load_external_data=False)
ext = {}
for init in m.graph.initializer:
    if init.name in ("model.backbone.lm_head.weight_Q4", "model.backbone.lm_head.weight_scales"):
        kv = {e.key: e.value for e in init.external_data}
        ext[init.name] = (kv["location"], int(kv["offset"]), int(kv["length"]))
loc, off_q4, len_q4 = ext["model.backbone.lm_head.weight_Q4"]
_, off_sc, len_sc = ext["model.backbone.lm_head.weight_scales"]
assert len_q4 == V * D // 2 and len_sc == V * (D // BLK) * 2

ids = np.array([[0, 1, 7, 1024, 1025, 9906, 100257, 151935]], dtype=np.int64)

rows = []
with open("models/" + loc, "rb") as f:
    for tid in ids[0]:
        f.seek(off_q4 + int(tid) * (D // 2))
        packed = np.frombuffer(f.read(D // 2), dtype=np.uint8)
        f.seek(off_sc + int(tid) * (D // BLK) * 2)
        scales = np.frombuffer(f.read((D // BLK) * 2), dtype=np.float16)
        lo = (packed & 0x0F).astype(np.int8)
        hi = (packed >> 4).astype(np.int8)
        lo[lo > 7] -= 16
        hi[hi > 7] -= 16
        q = np.empty(D, dtype=np.int8)
        q[0::2], q[1::2] = lo, hi
        rows.append((q.reshape(D // BLK, BLK).astype(np.float32)
                     * scales[:, None].astype(np.float32)).reshape(D))
manual = np.stack(rows).astype(np.float16)

sess = ort.InferenceSession(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_embedder.onnx"), providers=["CPUExecutionProvider"])
out = sess.run(["embedding"], {"input_ids": ids})[0][0]
diff = float(np.abs(out.astype(np.float32) - manual.astype(np.float32)).max())
print("embedder vs raw dequant: max |diff| =", diff)
assert diff == 0.0, "embedder mismatch — do not proceed"

p = onnx.load(str(Path(__file__).resolve().parent.parent / "models" / "higgs_audio_v3_ar_prefill_matmul4_embeds.onnx"), load_external_data=False)
onnx.checker.check_model(p)
pi = onnx.shape_inference.infer_shapes(p)
vi = {v.name: v for v in pi.graph.value_info}
for t in ("add_330", "val_0", "arange"):
    tt = vi[t].type.tensor_type
    print(t, "->", onnx.TensorProto.DataType.Name(tt.elem_type),
          [d.dim_param or d.dim_value for d in tt.shape.dim])
assert [x.name for x in pi.graph.input] == ["embedding", "attention_mask"]
print("SEAM OK")
