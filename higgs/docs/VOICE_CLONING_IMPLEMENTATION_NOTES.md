# Higgs ONNX Voice Cloning Implementation Notes

This patch replaces the old diagnostic-only `test_higgs_onnx_voice_cloning_poc.py` with a real runner for the SGLang Higgs voice-cloning contract.

## What the new script implements

- Builds the SGLang Higgs prompt layout:
  - zero-shot: `<|tts|> <|text|> target_text <|audio|>`
  - cloning with transcript: `<|tts|> <|ref_text|> reference_text <|ref_audio|> [-100]... <|text|> target_text <|audio|>`
  - cloning without transcript: `<|tts|> <|ref_audio|> [-100]... <|text|> target_text <|audio|>`
- Loads pre-encoded `reference_codes` in JSON, NPY, NPZ, CSV, or TXT form.
- Accepts raw pre-delay `[T, 8]` codes, matching the SGLang API, and applies the Higgs delay pattern to produce `[T+7, 8]` rows.
- Optionally encodes `--reference-audio` through SGLang's `HiggsAudioCodec` when a full Higgs checkpoint is available through `--codec-model-path`.
- Detects whether the prefill ONNX export exposes a clone-conditioning input such as `reference_codes_delayed`, `ref_codes_delayed`, or `input_embeds`.
- Refuses to silently perform fake cloning if the loaded ONNX export is zero-shot only. Use `--allow-zero-shot-fallback` only for benchmarking the existing path.

## Current blocker with the supplied ONNX artifacts

The current `higgs_audio_v3_ar_prefill_matmul4.onnx` export exposes only `input_ids` and `attention_mask`. SGLang voice cloning requires prefill-time embedding overlay at `-100` placeholder positions. That means the current ONNX export cannot consume the reference voice codes even after they are correctly encoded and delayed.

To make this script perform true ONNX voice cloning, provide one of the following missing artifacts:

1. A prefill ONNX export that accepts `reference_codes_delayed` / `ref_codes_delayed` and overlays the fused multi-codebook embeddings internally.
2. A prefill ONNX export that accepts already-built `input_embeds`, plus an embedding extractor for the Higgs text and fused audio embeddings.
3. A sidecar full SGLang/PyTorch Higgs runtime for the cloning path, while keeping the current ONNX path for zero-shot.

## Example commands

Dry-run the prompt contract using pre-encoded reference codes:

```bash
python test_higgs_onnx_voice_cloning_poc.py \
  --text "Have a nice day and enjoy south california sunshine." \
  --reference-codes ref_codes.json \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --dry-run-contract
```

Attempt real ONNX cloning with a clone-capable export:

```bash
python test_higgs_onnx_voice_cloning_poc.py \
  --provider CUDAExecutionProvider \
  --text "Have a nice day and enjoy south california sunshine." \
  --reference-codes ref_codes.json \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --out clone.wav
```

Encode reference audio to reusable raw `[T,8]` codes, if `sglang_omni` and a full Higgs checkpoint are available:

```bash
python test_higgs_onnx_voice_cloning_poc.py \
  --reference-audio voices/ivy.wav \
  --codec-model-path /path/to/full/higgs-audio-v3-tts-4b \
  --save-reference-codes ivy_codes.npy \
  --encode-reference-only \
  --i-have-rights-to-this-voice
```
