# Higgs ONNX Requirements

## Goal

Run the currently working **Higgs Audio v3 ONNX zero-shot TTS** path locally from this repository.

This document is only about the ONNX path that is already proven to synthesize audio. It is not about NVFP4 or voice cloning.

## Required model artifacts

These files are currently required in [`higgs/`](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs):

- `higgs_audio_v3_ar_prefill_matmul4.onnx`
- `higgs_audio_v3_ar_prefill_matmul4.onnx.data`
- `higgs_audio_v3_ar_decode_matmul4.onnx`
- `higgs_audio_v3_ar_decode_matmul4.onnx.data`
- `higgs_audio_v3_vocoder_decode.onnx`
- `higgs_audio_v3_vocoder_decode.onnx.data`

Optional alternative vocoder:

- `higgs_audio_v3_vocoder_decode_matmul4.onnx`
- `higgs_audio_v3_vocoder_decode_matmul4.onnx.data`

Tokenizer assets currently used by the working Python runner:

- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `chat_template.jinja`

## What each artifact does

- `ar_prefill_matmul4`: converts tokenized text prompt into first-step logits plus KV cache
- `ar_decode_matmul4`: autoregressive continuation over 8 delayed codebooks
- `vocoder_decode`: converts `[B, 8, T]` audio codebooks into waveform
- `tokenizer.*`: local text tokenization without requiring Hugging Face network access

## Python environment

Current repo-local environment that worked:

- Python `3.11` venv at [`venv/`](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/venv)
- `transformers`
- `soundfile`
- `huggingface_hub`
- `onnxruntime-gpu`
- `fastapi`
- `uvicorn`

## ONNX Runtime

Current preferred runtime:

- `onnxruntime-gpu==1.26.0`

Confirmed available providers after install:

- `TensorrtExecutionProvider`
- `CUDAExecutionProvider`
- `CPUExecutionProvider`

The benchmarked path used:

- `CUDAExecutionProvider`

## Current scripts

Working local scripts:

- [test_higgs_onnx_poc.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/test_higgs_onnx_poc.py)
- [test_higgs_onnx_voice_cloning_poc.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/test_higgs_onnx_voice_cloning_poc.py)
- [higgs_onnx_benchmark_server.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/higgs_onnx_benchmark_server.py)

## Important limitation

The currently available ONNX bundle is **zero-shot only**.

The exposed graph inputs are:

- prefill: `input_ids`, `attention_mask`
- decode: `codes`, `position_ids`, KV-cache tensors
- vocoder: `audio_codes`

There is no reference waveform input, no speaker embedding input, and no reference-code input in the available ONNX graphs.

## Current best-known runtime configuration

Best practical config observed so far:

- AR: `matmul4`
- vocoder: `fp32`
- provider: `CUDAExecutionProvider`

Reason:

- AR dominates latency
- fp32 vocoder cost is already very small on GPU
- fp32 vocoder produced better end-to-end warm RTF than the int4 vocoder in the direct comparison run
