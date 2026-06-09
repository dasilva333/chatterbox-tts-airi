# Strategy: NVFP4 Through the Existing SGLang Higgs Stack

## Core reframing

With the SGLang Higgs source files in hand, the problem is less mysterious than it first looked.

The right framing is:

**The SGLang Higgs cloning/runtime path mostly already exists in Python. The main NVFP4 question is not whether we know the classes anymore. The main question is whether the NVFP4 checkpoint can be loaded into those classes through the SGLang quantization and weight-loading path without custom glue.**

This is a materially better position than “we need to reverse-engineer the whole model.”

## What `stages.py` proves

[stages.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/stages.py) makes the runtime pipeline explicit:

```text
preprocessing -> audio_encoder -> tts_engine -> vocoder
```

It also makes the cloning mechanism explicit:

- raw reference audio can be encoded into discrete Higgs codec codes
- client-supplied `reference_codes` are supported
- delayed reference codes are built with `apply_delay_pattern`
- the TTS engine consumes `reference_codes_delayed`
- the model runner overlays fused multi-codebook embeddings at `-100` placeholder positions during prefill

That means the cloning path is conceptually understood now.

## What this changes

The scary part is no longer “how does Higgs voice cloning even work?”

The scary part is now narrower:

**Can the NVFP4 checkpoint be loaded into the existing Higgs + Qwen3 SGLang runtime correctly?**

That is a weight-loading and quantization-path problem.

## Core Higgs files that matter

These files now exist locally under [sglang_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/sglang_reference):

### Higgs-specific runtime

- `sglang_omni/models/higgs_tts/__init__.py`
- `sglang_omni/models/higgs_tts/audio_codec.py`
- `sglang_omni/models/higgs_tts/config.py`
- `sglang_omni/models/higgs_tts/hf_config.py`
- `sglang_omni/models/higgs_tts/model.py`
- `sglang_omni/models/higgs_tts/model_runner.py`
- `sglang_omni/models/higgs_tts/modeling.py`
- `sglang_omni/models/higgs_tts/payload_types.py`
- `sglang_omni/models/higgs_tts/request_builders.py`
- `sglang_omni/models/higgs_tts/sampler.py`
- `sglang_omni/models/higgs_tts/text_tokenizer.py`
- `sglang_omni/models/higgs_tts/utils.py`
- `sglang_omni/models/higgs_tts/vocoder_scheduler.py`
- `sglang_omni/models/higgs_tts/weight_loader.py`
- `sglang_omni/models/higgs_tts/_vendored/higgs_audio_v2_tokenizer_hf.py`
- `sglang_omni/models/higgs_tts/_vendored/higgs_audio_v2_tokenizer_config.json`

### Supporting SGLang-Omni runtime

- `sglang_omni/model_runner/base.py`
- `sglang_omni/models/tts_streaming.py`
- `sglang_omni/preprocessing/cache_key.py`
- `sglang_omni/scheduling/bootstrap.py`
- `sglang_omni/scheduling/omni_scheduler.py`
- `sglang_omni/scheduling/simple_scheduler.py`
- `sglang_omni/scheduling/stage_cache.py`
- `sglang_omni/scheduling/threaded_simple_scheduler.py`

One requested file did **not** land:

- `sglang_omni/scheduling/sglang_backend.py`

That path returned `404` from the raw GitHub URL used during fetch, so either the file moved or the import path changed upstream.

## Why these files matter

### `model.py`

Defines the Higgs TTS model wrapper around the Qwen3 backbone and ties in:

- fused multimodal embedding/head
- delayed multi-codebook generation
- weight loading

### `model_runner.py`

This is one of the most important files.

It is the place where:

- prompt prefill is prepared
- `reference_codes_delayed` are turned into fused embeddings
- those embeddings are overlaid at `-100` placeholder positions

This is the strongest direct explanation of the cloning path.

### `sampler.py`

Implements the multi-codebook generation state machine:

- delay count
- EOC countdown
- generation done behavior
- last generated codes

### `request_builders.py`

Bridges `HiggsTtsState` into scheduler/runtime requests and ensures reference-conditioned requests are keyed correctly.

### `text_tokenizer.py`

Critical for prompt construction:

- Higgs prompt layout
- audio placeholder ids
- reference-text / reference-audio prompt shape

### `weight_loader.py`

Likely one of the most important files for the NVFP4 pivot.

This is the first place to inspect if the checkpoint fails to load, because it sits at the seam between:

- checkpoint tensor names
- Higgs multimodal wrapper modules
- Qwen3 backbone parameter routing

## What the NVFP4 metadata says

These files now exist locally under [nvfp4_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/nvfp4_reference):

- `config.json`
- `quantization_config.json`
- `functional_test_report.json`
- `tensor_manifest.json`
- `tokenizer_config.json`

One important caveat:

- `tokenizer.json` downloaded only as a Git LFS pointer file, not as the actual `11.4 MB` tokenizer blob

So if the NVFP4 route needs the repo-local tokenizer, that still needs a real blob fetch path rather than the raw pointer file.

### What `quantization_config.json` confirms

The NVFP4 artifact says:

- base model: `bosonai/higgs-audio-v3-tts-4b`
- format: `nvfp4`
- quantized scope: `body.layers.* 2D transformer weights only`
- preserved scope: `codec/vocoder, fused modality embedding/head, norms, biases, non-2D tensors`
- quantized parameter fraction: about `0.7805`

This is encouraging.

It means the artifact likely preserves the non-body Higgs-specific pieces needed for cloning:

- codec/vocoder
- fused multimodal embedding/head
- non-2D support tensors

So cloning is probably not blocked by the conceptual model architecture being stripped away.

### What `functional_test_report.json` suggests

The report validates quantized tensor behavior for sampled `body.layers.*` weights and shows the repo author did numerical checks on those projections.

That suggests the NVFP4 artifact is serious as a quantized backbone artifact, not a random partial dump.

## Most likely NVFP4 failure points

Given what we now know, the likely breakpoints are:

1. `weight_loader.py` may not understand the NVFP4 checkpoint’s tensor names or manifest conventions.
2. The SGLang quantized Qwen3 loader may expect a quantization config shape that does not match the Reza2kn repo cleanly.
3. The Higgs wrapper may pass quantization config through correctly, but the remapping of `body.layers.*` tensors into the Qwen3 modules may still fail.
4. The runtime may load the transformer body but fail to resolve codec/vocoder assets from the NVFP4 repo layout.
5. The model registration/config path may not associate the repo config cleanly with the Higgs TTS class.

## What is now *not* the main blocker

The cloning path itself is no longer the main unknown.

We now have strong source evidence that the cloning contract is:

- reference audio or pre-encoded reference codes
- optional but recommended reference transcript
- delay pattern
- prompt placeholders
- fused embedding overlay during prefill

So the hard part is not “invent cloning.”

It is:

**Get the NVFP4 checkpoint loaded into the already-understood Higgs runtime.**

## Practical pivot plan

### Step 1

Attempt to instantiate the NVFP4 repo through the SGLang Higgs stack as-is.

Goal:

- identify whether the first failure is config registration, quantization parsing, weight remapping, or codec extraction

### Step 2

If it fails, inspect these first:

- `weight_loader.py`
- `model.py`
- `config.py`
- NVFP4 `config.json`
- NVFP4 `quantization_config.json`
- NVFP4 `tensor_manifest.json`

### Step 3

Determine whether the failure is:

- small mapping patch
- missing tokenizer/blob fetch
- missing runtime quantization compatibility
- or broader unsupported NVFP4 path

## Bottom-line bet

Current best bet:

**Voice cloning can likely work with NVFP4 if the NVFP4 checkpoint can be loaded into `HiggsTTSModel` with the SGLang quantized Qwen3 backbone.**

The cloning mechanism itself no longer looks like the scary part.

The scary part is the **NVFP4 weight-loading seam**.

## Local reference locations

Fetched local reference trees:

- [sglang_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/sglang_reference)
- [nvfp4_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/nvfp4_reference)
