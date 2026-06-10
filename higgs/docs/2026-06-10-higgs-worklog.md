# Higgs Audio v3 Worklog

Date: 2026-06-10

This document records the Higgs Audio v3 progress made today, including what was proven, what failed, and what remains unresolved at the current last mile.

## Executive Summary

The main breakthrough today was that the Higgs Audio v3 ONNX path is real, runnable, and extensible enough to support cloned-reference conditioning through graph surgery.

What we proved:

- zero-shot Higgs ONNX synthesis works locally
- CUDA execution works
- warm-model benchmarking works and is much more representative than cold-load timing
- the prefill ONNX graph can be surgically patched to expose `inputs_embeds`
- the external embedding builder can feed that patched graph successfully
- the reference-code cloning contract used by SGLang is real and recoverable from the local SGLang source
- the full Higgs checkpoint does contain the codec weights under `tied.embedding.modality_embeddings.0.model.*`

What remains unresolved:

- the real male-voice encode path is still blocked by a codec/class mismatch between the vendored codec implementation and the downloaded full checkpoint
- the current warm clone demo is still based on synthetic reference codes, not a real voice sample
- the clone path is runnable, but the quality/latency story for the real reference-voice route still needs to be finished

## What We Started With

At the start of the day, the working assumption was that the ONNX repo only exposed a vocoder and that a missing generator artifact or a missing public runtime might be the blocker. That turned out to be too conservative.

The local evidence eventually showed that:

- the prefill graph exists and is usable
- the AR decode graph exists and is usable
- the vocoder graph exists and is usable
- the browser-space / local ONNX artifacts contain the missing runtime glue that the README alone did not describe

This was the key shift: the repo README under-described the actual usable artifact set.

## Zero-Shot Baseline

We first proved that the ONNX stack can synthesize audio end-to-end without any voice cloning.

Working artifacts:

- `higgs_audio_v3_ar_prefill_matmul4.onnx`
- `higgs_audio_v3_ar_decode_matmul4.onnx`
- `higgs_audio_v3_vocoder_decode.onnx`
- `higgs_audio_v3_vocoder_decode_matmul4.onnx`
- `tokenizer.json`
- `tokenizer_config.json`

Result:

- the pipeline produced a valid WAV file
- the output was audible and recognizable
- the demo confirmed the local Higgs ONNX path is not hypothetical

This eliminated the initial idea that the repo was too incomplete to run locally.

## CUDA Recovery

The benchmark path initially ran through CPU and showed very poor RTF because it was effectively timing the wrong thing and because the CPU backend was not the target path.

We then recovered CUDA execution by ensuring the shell could locate the required Torch/CUDA DLLs:

- `C:\Users\h4rdc\anaconda3\envs\qwen_env\Lib\site-packages\torch\lib`

After prepending that path to `PATH`, `onnxruntime-gpu` was able to use `CUDAExecutionProvider`.

This was important because it established that the GPU path is live on this machine, not just the CPU fallback.

## Warm vs Cold Benchmarking

The first timing numbers were misleading because they mixed model loading and warmup into the measurement.

To fix that, a dedicated warm benchmark harness was added:

- `higgs_onnx_warm_benchmark.py`

The warm benchmark does three things in sequence:

1. load the model into memory once
2. run a short throwaway warmup generation
3. run the real measured generation and report its RTF separately

Observed result on CUDA with the FP32 vocoder:

- warmup: `audio=0.36s`, `total=1.85s`, `RTF=5.129`
- benchmark: `audio=7.40s`, `AR=12.67s`, `vocoder=1.55s`, `total=14.21s`, `RTF=1.921`

Interpretation:

- the warm path is much more meaningful than the earlier cold timing
- the benchmark still is not sub-1 RTF in that run
- the major cost is the AR path, not the vocoder
- the vocoder cost is measurable but not the dominant issue

This was a useful calibration step because it gave a more realistic performance profile for burst-style usage.

## Benchmark Server Findings

We also created and used a small benchmark server around the same engine.

That server showed:

- CPU execution was much slower and not representative of the target setup
- CUDA execution was materially better
- the `fp32` vocoder was more useful than the `matmul4` vocoder in practical timing, even though the `matmul4` vocoder is slightly smaller/faster on raw vocoder time

The important conclusion is that the best current configuration is:

- `matmul4` AR graphs
- `fp32` vocoder
- `CUDAExecutionProvider`

That combination gave the best balance of speed and fidelity for the current benchmark work.

## ONNX Graph Surgery Breakthrough

This was the major technical breakthrough of the day.

The prefill graph originally exposed only:

- `input_ids`
- `attention_mask`

The working claim from the SGLang reference was that Higgs cloning uses a prefill-time overlay of delayed reference-code embeddings into placeholder positions. The local ONNX prefill graph looked like it might not have a usable interface for that.

That turned out to be wrong in the important sense:

- the prefill graph has an identifiable embedding entry path
- that path is not a dead end
- it can be patched to expose `inputs_embeds`

We then produced a surgically patched ONNX prefill model:

- `higgs_audio_v3_ar_prefill_inputs_embeds.onnx`

The surgery succeeded structurally and ONNX Runtime accepted the patched graph.

The important finding was that the embedding entry node is not a generic plain-text `Gather` in this export. It is a quantized embedding entry node:

- `GatherBlockQuantized`
- node name: `node_embedding_Q4`

This still gave us a clean enough seam to continue.

## External Embedding Builder

Once the prefill graph could accept `inputs_embeds`, the next step was to build the embeddings externally.

That led to an external builder path:

- extract the embedding output from the original prefill graph
- build the same embedding tensor outside the patched graph
- insert the reference-code overlay externally
- feed the patched `inputs_embeds` graph directly

That builder worked.

This proved that the surgery path was not just a theoretical graph edit; the patched graph could be driven with externally prepared embeddings.

## Clone Demo Path

With the graph surgery and embedding builder in place, the clone demo path became runnable end-to-end.

The clone-capable demo produced a valid WAV:

- `higgs_clone_demo.wav`

Later, a CUDA-backed variant also produced a valid WAV:

- `higgs_clone_demo_cuda.wav`

At that point, the pipeline proved:

- patched prefill graph
- external embedding generation
- AR decode
- vocoder decode
- WAV output

That was the second major milestone of the day.

## Reference-Code Contract From SGLang

Local SGLang Higgs source was critical here.

The key reference files showed that the real Higgs cloning contract is:

- `references`
- `reference_codes`
- `reference_text`
- delayed reference codes
- placeholder tokens at `-100`
- fused multi-codebook embeddings overlaid at placeholder positions during prefill

The relevant local files that made this clear were:

- `higgs/sglang_reference/sglang_omni/models/higgs_tts/stages.py`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/model_runner.py`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/text_tokenizer.py`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/utils.py`
- `higgs/sglang_reference/sglang_omni/models/higgs_tts/modeling.py`

This was important because it confirmed that the clone path is not a vague speaker-embedding trick. It is a deterministic delayed-codebook conditioning flow.

## Full Checkpoint and Codec Findings

We eventually downloaded the full Higgs checkpoint locally into:

- `higgs/bosonai/`

The checkpoint files include:

- `config.json`
- `model.safetensors`
- `model.safetensors.index.json`
- `tokenizer.json`
- `tokenizer_config.json`

The checkpoint layout confirmed:

- `model_type: higgs_multimodal_qwen3`
- `architectures: ["HiggsMultimodalQwen3ForConditionalGeneration"]`
- `audio_encoder_config.num_codebooks = 8`
- `audio_token_id = -100`
- `text_config.hidden_size = 2560`
- `text_config.num_hidden_layers = 36`

The safetensors key scan showed:

- 929 total keys
- 1 key under `tied.embedding.modality_embeddings.0.embedding.weight`
- 528 keys under `tied.embedding.modality_embeddings.0.model.*`
- 11 keys under `body.layers.0.*`
- no separate top-level `codec` or `audio` key namespace

That established that the codec is present in the full checkpoint and nested under the expected `tied.embedding.modality_embeddings.0.model.*` prefix.

That was an important correction to earlier uncertainty.

## Codec Mismatch Blocker

Even though the codec weights are present in the full checkpoint, the vendored codec class still did not load cleanly against the downloaded checkpoint.

What happened:

- the vendored codec implementation was able to locate the codec prefix conceptually
- but the checkpoint shapes did not match the codec class layout expected by the vendored model
- the failure was structural, not a simple missing-file problem

That means:

- the full checkpoint definitely contains the codec
- but the local vendored codec implementation appears to be out of sync with the checkpoint version/layout

This is the current last-mile blocker for the real male-voice encode step.

## NVFP4 Investigation

We also spent time on the NVFP4 artifact:

- `Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4`

The key findings were:

- the artifact is not a drop-in runtime
- the weight layout does not match SGLang’s default `modelopt_fp4` loader format
- the local safetensors keys use fields like:
  - `*.packed_nvfp4`
  - `*.scales_e4m3`
- SGLang’s FP4 loader expects a different shape contract:
  - `weight`
  - `input_scale`
  - `weight_scale`
  - `weight_scale_2`

We verified that the local NVFP4 checkpoint does contain the expected body weights, but it does not expose the codec weights in a directly usable form for the encoding path we needed.

This is why NVFP4 did not become the final short path.

## What Did Not Work

These were the important dead ends and partial failures:

- assuming the ONNX repo README described the full runtime surface accurately
- trying to treat NVFP4 as a direct replacement for the full checkpoint when the codec/encoder path was still needed
- assuming warm timing could be inferred from cold start numbers
- trying to use CPU timing as the primary signal for an AR-heavy sequential model
- trying to force the vendored codec implementation to load the downloaded full checkpoint without first resolving version/layout mismatches
- treating the `matmul4` vocoder as a meaningful source of total latency improvement

These failures were useful because they narrowed the problem:

- the pipeline exists
- the surgery exists
- the benchmark exists
- the real blocker is now codec compatibility for the human-voice encode step

## Current Best Read

The most accurate current summary is:

1. The ONNX clone-capable path is real.
2. The warm CUDA benchmark is real.
3. The clone pipeline can already run using synthetic reference codes.
4. The full Higgs checkpoint contains the codec weights.
5. The remaining blocker is making the codec implementation line up with the downloaded checkpoint so that a real male sample from `voices/` can be encoded into `[T, 8]` reference codes.

That means the project is no longer blocked on architecture or graph surgery.
It is blocked on one of:

- codec version mismatch
- checkpoint layout mismatch
- or a small adapter/shim layer around the codec loader

## Files Created Today

Selected files created or updated in the Higgs folder today include:

- `higgs_onnx_warm_benchmark.py`
- `higgs_audio_v3_ar_prefill_inputs_embeds.onnx`
- `higgs_audio_v3_ar_prefill_with_embedding_output.onnx`
- `run_higgs_inputs_embeds_voice_demo.py`
- `onnx_inputs_embeds_surgery.py`
- `onnx_inputs_embeds_builder.py`
- `onnx-surgery-cleanroom-leonid.md`
- `onnx-surgery-probe-addendum.md`
- `codex-resident-expert.md`
- `strategy-nvfp4-sglang-load-path.md`
- `strategy-higgs-sglang-crossport.md`
- `status-nvfp4-lockout.md`
- `requirements-onnx.md`
- `benchmark-analysis-report.md`

## Final Note

The important point from today is not just that something worked. It is that the main technical hypothesis changed:

- we are not waiting on a mythical missing ONNX export anymore
- we are not blocked on the existence of the clone contract
- we are blocked on the practical final mile of matching the codec implementation to the full checkpoint so a real voice sample can be encoded and pushed through the already-working clone pipeline

That is a much better place to be than where the day started.
