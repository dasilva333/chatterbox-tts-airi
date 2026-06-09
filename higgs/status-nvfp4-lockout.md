# NVFP4 Status Quo

## Short version

The `Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4` route is still the most attractive path for an `8 GB` GPU, but it is not currently a drop-in runtime in this repo.

We are effectively locked out from multiple directions at once:

- official supported path is the full model, which is too heavy for the target hardware/performance goal
- native Windows `sglang-omni` is a bad fit operationally
- public NVFP4 artifact is incomplete as a standalone runtime
- current ONNX bundle proves zero-shot browser/local inference, but not cloning and not NVFP4 loading

## The browser-space lead that changed the situation

The most important practical lead ended up being the browser ONNX work around:

- `Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX`
- `Reza2kn/Higgs-Audio-v3-4bit-Browser-Space`

That route mattered because it turned out not to be a dead-end vocoder-only curiosity. It exposed the only currently viable local path that actually got us to working Higgs audio in this repo:

- local tokenizer
- ONNX AR prefill
- ONNX AR decode
- ONNX vocoder

It still does not solve NVFP4 or cloning, but it was the breakthrough that converted “locked out everywhere” into “zero-shot local Higgs is real and benchmarkable.”

## Why NVFP4 is attractive

- model artifact size is much smaller than the full `~10 GB` stack
- target hardware is `8 GB` VRAM
- avoiding paging/offload is critical if the goal is `RTF < 1`

This makes a compact quantized route strategically correct even if it is harder to integrate.

## What the public NVFP4 artifact says

The NVFP4 model card states:

- it is a **transformer-body quantized artifact**
- it is **not a complete drop-in runtime**
- integration is through **SGLang-Omni or a custom loader**

That is the key constraint.

## Why the official path does not solve the target problem

The official Higgs path in `sglang-omni` targets the original full model, not the NVFP4 artifact.

That gives:

- a supported runtime story
- but not a memory-footprint story compatible with the current performance target on `8 GB`

Docker or SGLang can solve packaging and serving. They do not change the underlying model footprint enough to make the full model a good fit here.

## Why the ONNX route does not yet solve NVFP4

Current local ONNX artifacts prove:

- AR prefill exists
- AR decode exists
- vocoder exists
- zero-shot synthesis works

Current local ONNX artifacts do **not** prove:

- NVFP4 loading
- reference-audio conditioning
- end-to-end voice cloning

The available ONNX graphs are already exported runnable graphs. They do not provide a general NVFP4 loader story for the original PyTorch checkpoint.

## Current lockout map

### 1. Full model

Blocked by:

- likely poor fit on `8 GB`
- real-time target would be compromised by memory pressure/offload

### 2. SGLang-Omni native on Windows

Blocked by:

- Linux-first runtime assumptions
- higher operational friction than desired
- still points at the full model for the supported path

### 3. NVFP4 as plain Transformers/PyTorch

Blocked by:

- public artifact is not a finished runtime
- would require custom loading/dequantization or runtime glue

### 4. ONNX local/browser route

Blocked by:

- currently zero-shot only
- no public reference-conditioning path in the local artifact set

## What is proven vs unproven

Proven:

- Higgs ONNX zero-shot TTS can run locally
- warm CUDA RTF can go below real time with the current ONNX bundle

Unproven:

- NVFP4 can be used as a clean local runtime in this repo without custom glue
- Higgs voice cloning is possible with the current ONNX artifact set

## Practical conclusion

NVFP4 remains the right strategic direction for constrained local hardware, but there is still no clean, supported, low-effort path to use it here.

Right now:

- ONNX zero-shot is the working local path
- NVFP4 is the still-desired but not-yet-landed path

That is the true status quo.
