# ONNX Surgery Probe Addendum

## What was verified

I ran a cheap graph probe against the local Higgs ONNX prefill model:

- `higgs_audio_v3_ar_prefill_matmul4.onnx`

The probe confirmed the prefill graph has a real, identifiable embedding entry point.

### Graph inputs

The prefill graph exposes only:

- `input_ids`
- `attention_mask`

### Embedding path

The first token-embedding node is:

- `node_embedding_Q4`
- op type: `GatherBlockQuantized`

Its inputs are:

- `model.backbone.lm_head.weight_Q4`
- `input_ids`
- `model.backbone.lm_head.weight_scales`

Its output is:

- `embedding`

### Immediate downstream consumers of `embedding`

The `embedding` tensor fans out into only a small number of nodes near the graph entry:

- `Cast: node__to_copy_5`
- `Add: node_add_330`

That is the key structural result.

## What this means

This is not a generic “missing file” situation anymore.

The ONNX graph does have an explicit embedding path, but it is quantized and routed through `GatherBlockQuantized` rather than a plain token embedding `Gather`.

That means the clone surgery idea is still plausible:

- replace the token-embedding entry surface with `inputs_embeds`
- preserve attention mask and downstream transformer behavior
- compute the reference-audio overlay outside the graph

It is still not guaranteed to work, but it is no longer speculative.

## Important caution

This probe does **not** prove the surgery is complete.

It only proves:

- there is a concrete embedding subgraph to inspect
- the graph is not already opaque at the input edge
- the next question is whether the `embedding -> Cast/Add -> transformer` path can be cleanly rewired

The biggest remaining unknowns are:

- whether positional handling depends on `input_ids` in a way that is hard to replace
- whether the `GatherBlockQuantized` output is the only tensor feeding the transformer entry
- whether a patched graph can still preserve all outputs needed by decode/KV-cache behavior

## Practical next step for Fable

Ask it to inspect the actual ONNX graph and determine:

1. whether `node_add_330` is the final entry tensor into the transformer block
2. whether `embedding` can be replaced by a new graph input named `inputs_embeds`
3. whether any position or shape logic would break if `input_ids` is removed from the embedding path

## Bottom line

The probe result is a qualified green light for further ONNX surgery work.

It is enough to justify trying the cut.
It is not enough to assume the cut will succeed without inspecting the full downstream fan-out.
