# ONNX Surgery Results — inputs_embeds Cut

Date: 2026-06-11. Status: **surgery performed, seam verified bit-exact. Green light.**

## Answers to the addendum questions

**1. Is `node_add_330` the final entry into the transformer block?**
Yes, structurally. `embedding` (FLOAT16 `[1, seq, 2560]`) has exactly two consumers, the standard pre-norm residual entry:

- `Cast node__to_copy_5` → fp32 → RMSNorm (Pow/ReduceMean/Sqrt/Reciprocal/Mul) → layer-0 attention input
- `Add node_add_330` (`embedding + linear_3`) → first residual add

So `embedding` itself is the canonical hidden-states entry tensor h0. No other tensor feeds the transformer entry.

**2. Can `embedding` be replaced by a graph input?**
Yes. Done by reusing the tensor name `embedding` as the new graph input — zero consumer rewiring needed.

**3. Does position/shape logic break without `input_ids`?**
No. `input_ids` had exactly 2 consumers:

- `node_embedding_Q4` (`GatherBlockQuantized`, com.microsoft, block=128) — the cut target
- `node_Shape_0` (`Shape`, start=1, end=2 → emits `[seq]`) → feeds `arange` (RoPE positions 0..seq-1) and 4 KV-reshape Concats

Positions derive from *sequence length only*, never from token values. `node_Shape_0` was re-pointed to `embedding`; with start=1,end=2 it reads dim 1 of `[1, seq, 2560]` — identical result. `attention_mask` (1 consumer, bias mask path) is untouched and independent.

## What was produced

| File | What |
|---|---|
| `ar/higgs_audio_v3_ar_prefill_matmul4_embeds.onnx` | Patched prefill. Inputs: `embedding` FP16 `[1,seq,2560]`, `attention_mask`. All 73 outputs (logits + KV) intact. Passes onnx.checker + shape inference. |
| `ar/higgs_audio_v3_embedder.onnx` | 624-byte model: `input_ids → GatherBlockQuantized → embedding`. References the **same** byte ranges in the original `.onnx.data`. |
| `ar/modality_embedding_fp16.npy` | Fused audio table `[8208, 2560]` fp16 extracted from the graph. 8208 = 8 codebooks × 1026 (1024 codes + BOC 1024 + EOC 1025). |
| `surgery_inputs_embeds.py` | Reproducible surgery script with hard asserts on every structural assumption. |

Both new models reference `higgs_audio_v3_ar_prefill_matmul4.onnx.data` by offset — do not move/rename it, no weights were copied.

## Verification done here (CPU sandbox, no GPU)

- Embedder run under onnxruntime vs independent numpy dequant reading raw int4+scales bytes from the `.data` file: **max |diff| = 0.0** across 8 probe tokens spanning the vocab.
- Patched graph: onnx.checker clean; shape inference propagates `add_330 [1,seq,2560]`, `val_0 [1]`, `arange [seq]` correctly.
- NOT yet done (needs your GPU box): full prefill session equivalence — `prefill(input_ids)` vs `embedder(input_ids) → patched(embedding)` must produce identical logits/KV. Sandbox has 3 GB RAM; the 2.1 GB graph doesn't fit.

## Key discovery for cloning

The **fused audio embedding table is the output head** (`Gemm node_linear_253` against `model.modality_embedding.weight`, logits reshaped to `[1, 8, 1026]`) — embeddings and head are tied, and the table is in the prefill graph unquantized fp16. Combined with the Q4 text table (extractable via the embedder), everything needed to build the cloning overlay externally is now on disk:

```
text rows   = embedder(input_ids)                          # bit-exact, verified
audio rows  = sum_c modality_embedding[c*1026 + code_delayed[t,c]]   # [T+7, 2560]
embedding   = concat per SGLang prompt layout, audio rows at -100 placeholder spans
→ patched prefill(embedding, attention_mask)
```

## Next steps (GPU box)

1. Zero-shot equivalence: same `input_ids` through old vs split path; assert logits/KV bit-identical (both fp16, same kernels — should be exact).
2. Confirm one open question: whether the fused-embed overlay in SGLang sums per-codebook rows exactly as `modeling.py` does (offset `c*1026`, sum over 8) and whether a scale factor is applied — read `modeling.py` lines around the fused embedding before wiring.
3. Build the delayed-codes → overlay → prompt assembly runner (the cloning contract code from the voice-cloning patch already does delay + placeholders).
4. First clone attempt with a `reference_codes` fixture before touching raw-audio encoding.
