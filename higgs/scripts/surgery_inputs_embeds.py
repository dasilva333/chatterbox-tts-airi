#!/usr/bin/env python3
"""ONNX surgery: cut the token-embedding gather out of the Higgs AR prefill graph.

Produces, next to models/higgs_audio_v3_ar_prefill_matmul4.onnx:
  1. models/higgs_audio_v3_ar_prefill_matmul4_embeds.onnx
     - graph input `embedding` FLOAT16 [1, seq, 2560] replaces input_ids
     - node_embedding_Q4 (GatherBlockQuantized) removed
     - node_Shape_0 re-pointed to `embedding` (start=1,end=2 still yields [seq])
     - lm_head Q4 table + scales dropped from the proto (the .onnx.data file is
       NOT modified; both new models keep referencing it by offset)
  2. models/higgs_audio_v3_embedder.onnx
     - tiny model: input_ids -> GatherBlockQuantized -> embedding
     - reuses the original external-data byte ranges (zero weight copying)

Zero-shot pipeline after surgery: embedder(input_ids) -> patched_prefill(embedding, attention_mask)
Cloning pipeline:               embedder(text ids) + audio overlay rows -> patched_prefill(...)

Run from the repo root (the directory containing models/).
"""
import copy
import sys
from collections import defaultdict

import onnx
from onnx import TensorProto, helper

SRC = "models/higgs_audio_v3_ar_prefill_matmul4.onnx"
DST_PREFILL = "models/higgs_audio_v3_ar_prefill_matmul4_embeds.onnx"
DST_EMBEDDER = "models/higgs_audio_v3_embedder.onnx"

EMB_NODE = "node_embedding_Q4"
SHAPE_NODE = "node_Shape_0"
W_Q4 = "model.backbone.lm_head.weight_Q4"
W_SCALES = "model.backbone.lm_head.weight_scales"

m = onnx.load(SRC, load_external_data=False)
g = m.graph

cons = defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n.name)

# safety: the surgery assumes these exact facts; abort loudly if upstream re-exports change them
assert sorted(cons["input_ids"]) == sorted([SHAPE_NODE, EMB_NODE]), cons["input_ids"]
assert cons[W_Q4] == [EMB_NODE] and cons[W_SCALES] == [EMB_NODE]
emb_node = next(n for n in g.node if n.name == EMB_NODE)
shape_node = next(n for n in g.node if n.name == SHAPE_NODE)
attrs = {a.name: a.i for a in shape_node.attribute}
assert attrs == {"start": 1, "end": 2}, attrs
assert len(cons["embedding"]) == 2

# ---- embedder model (built BEFORE mutating g) ----
emb_inits = [copy.deepcopy(next(i for i in g.initializer if i.name == name))
             for name in (W_Q4, W_SCALES)]
embedder_graph = helper.make_graph(
    nodes=[copy.deepcopy(emb_node)],
    name="higgs_embedder",
    inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq"])],
    outputs=[helper.make_tensor_value_info("embedding", TensorProto.FLOAT16, [1, "seq", 2560])],
    initializer=emb_inits,
)
embedder = helper.make_model(embedder_graph, opset_imports=list(m.opset_import))
embedder.ir_version = m.ir_version
onnx.save(embedder, DST_EMBEDDER)

# ---- patched prefill ----
g.node.remove(emb_node)
shape_node.input[0] = "embedding"
del g.input[:]  # rebuild: embedding replaces input_ids; attention_mask kept
g.input.extend([
    helper.make_tensor_value_info("embedding", TensorProto.FLOAT16, [1, "seq", 2560]),
    helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "seq"]),
])
for name in (W_Q4, W_SCALES):
    g.initializer.remove(next(i for i in g.initializer if i.name == name))
# drop stale value_info for the tensor that is now a graph input
for vi in [v for v in g.value_info if v.name == "embedding"]:
    g.value_info.remove(vi)

onnx.save(m, DST_PREFILL)
onnx.checker.check_model(DST_PREFILL)
print("OK:", DST_EMBEDDER, "and", DST_PREFILL)
print("note: both reference the original higgs_audio_v3_ar_prefill_matmul4.onnx.data — keep it in models/")
