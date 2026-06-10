# Higgs ONNX Surgery Patch — inputs_embeds Cut

Converts the zero-shot-only Higgs AR prefill graph into an embedding-injectable one,
opening the external voice-cloning overlay path. No weights are modified or copied:
generated models reference the original `higgs_audio_v3_ar_prefill_matmul4.onnx.data`
by byte offset.

## What it does

```
before:  input_ids ──GatherBlockQuantized──> embedding ──> transformer
after:   embedding (graph input) ──────────────────────> transformer
         input_ids ──GatherBlockQuantized──> embedding    (separate 624-byte embedder.onnx)
```

`node_Shape_0` (seq length for RoPE arange + KV reshapes) is re-pointed from
`input_ids` to `embedding` — same attrs (start=1,end=2) read the same `seq` dim.
`attention_mask` untouched. All 73 outputs (logits + KV cache) preserved.

## Requirements

- the Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX snapshot on disk (`ar/` with both `.onnx` + `.onnx.data`)
- `pip install onnx onnxruntime numpy` (onnxruntime-gpu for the equivalence test on GPU)

## Apply

From the snapshot root (the directory containing `ar/`):

```bash
python surgery_inputs_embeds.py        # writes ar/higgs_audio_v3_ar_prefill_matmul4_embeds.onnx
                                       #        ar/higgs_audio_v3_embedder.onnx
python extract_modality_embedding.py   # writes ar/modality_embedding_fp16.npy  [8208, 2560]
python verify_seam.py                  # CPU, low-RAM: embedder bit-exactness + shape inference
python verify_equivalence.py --provider CUDAExecutionProvider
                                       # GPU/big-RAM: original vs embedder+patched, all 73 outputs
```

The surgery script hard-asserts every structural assumption (consumer sets, Shape attrs,
external-data lengths) and aborts loudly if the upstream export ever changes.

## Verified status (2026-06-11)

- Surgery applied cleanly; onnx.checker + shape inference pass.
- Embedder vs independent raw int4+scales dequant from the `.data` file: max |diff| = 0.0.
- `verify_equivalence.py` not yet run (needs >3 GB RAM / GPU) — run it first on your box.

## Cloning overlay sketch (next step after equivalence passes)

```
text rows  = embedder(input_ids)                                       # fp16 [n, 2560]
audio rows = sum over c of modality_embedding[c*1026 + delayed[t, c]]  # fp16 [T+7, 2560]
embedding  = rows per SGLang prompt layout:
             <|tts|> <|ref_text|> ref_text <|ref_audio|> [audio rows] <|text|> text <|audio|>
out        = patched_prefill(embedding, attention_mask=ones)
```

Before wiring: confirm against sglang-omni `modeling.py` whether the fused embed
is a plain per-codebook sum (offset c*1026) or applies any scaling. The table here
is the tied output head (`node_linear_253`), extracted unquantized fp16.

See `onnx-surgery-results.md` for the full graph findings.
