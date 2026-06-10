# Higgs Codec Research

Date: 2026-06-10

This note captures the current codec-side research state for Higgs Audio v3 and why the real male-voice encode path is still not resolved.

## What We Confirmed

The full Higgs checkpoint in `higgs/bosonai/` does contain the codec subtree needed for reference-audio encoding.

Checkpoint facts:

- `model_type`: `higgs_multimodal_qwen3`
- `architectures`: `["HiggsMultimodalQwen3ForConditionalGeneration"]`
- `audio_encoder_config.num_codebooks = 8`
- `audio_encoder_config.out_dim = 2560`
- `audio_encoder_config.use_delay_pattern = True`
- `audio_token_id = -100`

The safetensors key space includes:

- `tied.embedding.modality_embeddings.0.embedding.weight`
- `tied.embedding.modality_embeddings.0.model.acoustic_encoder.*`
- `tied.embedding.modality_embeddings.0.model.acoustic_decoder.*`
- `tied.embedding.modality_embeddings.0.model.quantizer.*`
- `tied.embedding.modality_embeddings.0.model.semantic_model.*`

That means the codec is not missing. It is present in the checkpoint and structurally broad enough to support the encode/decode path.

## What Failed

The local vendored codec implementation currently loaded from `sglang_reference/sglang_omni/models/higgs_tts/_vendored/` is the wrong version for this checkpoint.

The mismatch is not just a missing file or a bad download.

Observed mismatch:

- vendored codec config is Higgs Audio V2 shaped
- checkpoint is Higgs Audio v3 shaped
- the codec load fails on multiple audio encoder / quantizer tensor shapes

The vendored config currently encodes:

- `n_codebooks = 9`
- `sample_rate = 16000`
- `codebook_dim = 64`

The Higgs v3 checkpoint expects a different layout:

- `num_codebooks = 8`
- `sample_rate = 24000`
- `hidden_size/out_dim = 2560`

So the blocker is now clearly a version/layout mismatch, not a missing weight blob.

## Current Interpretation

The evidence points to one of these explanations:

1. The vendored codec class is from the wrong generation of Higgs codec implementation.
2. The checkpoint contains a newer Higgs codec layout than the vendored model expects.
3. A small adapter layer is needed to map the checkpoint tensors into the expected encode/decode interface.

At this point, the most useful next step is not more ONNX surgery.
It is to find the codec implementation that matches Higgs v3 or adapt the current loader to the v3 layout.

## Why This Matters

We already have:

- working zero-shot ONNX synthesis
- patched `inputs_embeds` surgery
- a clone-capable runner built around the SGLang Higgs contract
- warm benchmark data on CUDA

What we do not yet have is a real male reference sample encoded into `[T, 8]` codes from the full checkpoint.

That final step remains blocked until the codec implementation and checkpoint layout agree.

## Files To Reopen First

- `higgs/scripts/audio_codec.py`
- `higgs/bosonai/config.json`
- `higgs/bosonai/model.safetensors.index.json`
- `higgs/bosonai/model.safetensors`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/_vendored/higgs_audio_v2_tokenizer_config.json`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/_vendored/higgs_audio_v2_tokenizer_hf.py`

## Bottom Line

The codec weights are present.
The current codec loader is not the right one.
The remaining work is to align the loader/class with Higgs v3 so a real voice sample can be encoded and fed into the already-working clone pipeline.
