# Strategy: Cross-Port Higgs v3 Voice Cloning from SGLang

## Goal

Treat the already-working **Higgs Audio v3 implementation inside SGLang-Omni** as the primary reference system for voice cloning.

This document is intentionally narrow:

- no MOSS framing
- no browser-first framing
- no AIRI architecture discussion except where needed for eventual landing

The question here is simple:

**How does SGLang Higgs v3 do voice cloning today, and what would it take to cross-port that path into our local setup?**

## Why this is the right reference

We already know:

- current local ONNX Higgs artifacts can do zero-shot synthesis
- current local ONNX Higgs artifacts do **not** expose any obvious reference-audio conditioning inputs
- SGLang-Omni already has a working Higgs v3 runtime, including voice cloning behavior

So the most defensible next step is not guessing. It is reading the working implementation that already solves the problem.

## Known local limitation

Current local Higgs ONNX graphs only expose:

- prefill: `input_ids`, `attention_mask`
- decode: `codes`, `position_ids`, KV-cache tensors
- vocoder: `audio_codes`

That means the missing piece is upstream of the current ONNX export.

So the core investigation target is:

**Where does reference voice information enter the Higgs pipeline before or during generation?**

## Primary source to study

Use the Higgs runtime inside `sglang-omni` as the source of truth.

The important code areas are the Higgs-specific runtime files referenced by the SGLang Higgs documentation / DeepWiki page:

- `model.py`
- `model_runner.py`
- `sampler.py`
- `stages.py`
- `audio_codec.py`
- `request_builders.py`

These are the files most likely to answer:

- how reference audio is loaded
- how reference audio is encoded
- whether reference codes are injected into the prompt
- whether reference text is also used
- whether the generator itself consumes extra conditioned inputs not present in the current ONNX export

## Questions the code review must answer

### 1. What is the exact cloning contract?

We need the concrete interface, not a guess.

Specifically:

- Does cloning require only reference audio?
- Does it optionally require reference transcript text?
- Is the reference audio turned into discrete codec codes?
- Are those codes inserted as prompt content?
- Are there placeholder audio spans in the prompt?

### 2. Where is conditioning applied?

Possible locations:

- before AR generation, by constructing a richer prompt
- inside the model inputs, via hidden conditioning tensors
- after prompt assembly, via a separate embedding path

This is the most important architectural fork.

If conditioning is only prompt assembly plus codec encoding, the current ONNX route may be salvageable.

If conditioning depends on extra tensors or separate model branches not exported to ONNX, then the missing export surface is larger.

### 3. What artifacts are implicitly assumed?

We need to identify whether SGLang relies on components we do not currently have in local form, such as:

- a reference audio encoder
- a tokenizer/codec stage not yet exported
- an additional Higgs-specific preprocessing graph
- runtime-only helper logic around delayed codebooks

### 4. What is model logic vs runtime glue?

Separate:

- essential model behavior
- SGLang serving infrastructure

This matters because we do **not** want to port unnecessary framework machinery.

The target is the smallest possible functional subset needed for:

- local reference-audio conditioning
- local AR generation
- local vocoder decode

## Expected possible outcomes

## Outcome A: Cloning is mostly prompt-level conditioning

Best case.

This would mean:

- reference waveform is encoded into codec codes
- those codes are inserted into the prompt or prompt embeddings
- the ONNX AR graphs may already be enough if we can reproduce the same prompt structure locally

If this is true, the cross-port path is relatively clean:

1. reproduce the reference-audio encode step
2. reproduce the prompt-building logic
3. feed the resulting token/code layout into the existing ONNX runner

## Outcome B: Cloning requires extra hidden inputs not present in local ONNX

This would mean the current ONNX export is incomplete for cloning.

If true, the cross-port path becomes:

1. identify the missing tensors or branches
2. identify whether they already exist in unpublished ONNX artifacts or only in PyTorch/SGLang
3. decide whether to export them or stay on a PyTorch-sidecar route for cloning

## Outcome C: Cloning relies on a codec/reference path that is not yet published

This is the pessimistic but realistic possibility.

If true, then we will still gain something important:

- a precise explanation of what is missing
- a concrete artifact wish-list
- a real boundary between “working local zero-shot” and “not yet possible local cloning”

## Cross-port strategy if Outcome A is true

If SGLang Higgs cloning turns out to be mostly:

- encode reference audio
- build conditioned prompt
- run normal generation

then the porting plan is:

1. replicate reference-audio preprocessing locally
2. replicate the reference-code insertion logic locally
3. update `test_higgs_onnx_poc.py` into a clone-capable runner
4. validate output by comparing zero-shot vs conditioned outputs on the same text
5. only then consider Chatterbox-side integration

This is the cleanest possible path.

## Cross-port strategy if Outcome B is true

If cloning depends on additional inputs that current ONNX graphs do not expose:

1. document the missing input contract exactly
2. search for matching unpublished/local/browser artifacts
3. if none exist, decide whether to:
   - export missing ONNX graphs
   - or use a Python-sidecar hybrid path for cloning only

In that case, the current ONNX route remains valuable for zero-shot, but not sufficient for full cloning.

## What not to do

- Do not assume the browser-space UI proves cloning is implemented.
- Do not assume the `voice` dropdown in the current browser demo corresponds to real reference conditioning.
- Do not start by bolting random voice inputs into the current ONNX runner.
- Do not treat SGLang serving internals as the thing to port wholesale.

The job is to extract the **conditioning contract**, not the entire server stack.

## Deliverable of this investigation

The output of the SGLang code-reading pass should be a short technical memo answering:

1. What exact inputs are required for Higgs v3 cloning?
2. At what point do they enter the pipeline?
3. Which of those inputs are representable with the ONNX artifacts we already have?
4. Which artifact or export is still missing?
5. Is a clean local cross-port feasible right now?

## Recommendation

Use SGLang Higgs v3 as the canonical cloning reference.

The next serious technical step is not more benchmarking and not more speculative architecture talk. It is a source-level extraction of the Higgs cloning path from the SGLang implementation so we can determine whether our local ONNX setup is:

- one missing preprocessing step away from cloning
- or fundamentally missing export surface
