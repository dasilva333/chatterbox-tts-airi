# Strategy Notes: Cross-Comparing MOSS and SGLang Voice Cloning Paths

## Goal

Use adjacent implementations to understand what is missing for Higgs voice cloning and how we might port or reconstruct the required path inside this repo.

This is not a claim that MOSS and Higgs are directly compatible. They are not. The point is to use them as structural references.

## Core problem

Current local Higgs ONNX artifacts expose:

- text token input
- AR codebook generation
- vocoder decode

They do not expose:

- reference waveform conditioning
- speaker embedding input
- reference audio code input

So the missing piece is not “how to decode audio.” It is “how does a reference voice enter the graph.”

## Why compare against MOSS

MOSS-style voice cloning stacks are useful because they usually make the conditioning path explicit:

- speaker encoder or reference encoder
- conditioning vector or conditioning tokens
- talker/generator that consumes that conditioning
- downstream decoder/vocoder

That makes them a good structural reference even if the weights and exact graph contracts differ.

## Why compare against SGLang Higgs

The SGLang Higgs runtime is valuable because it is the best public evidence that the complete Higgs pipeline exists in working form.

Specifically, it should reveal:

- how prompt construction works when reference audio is present
- whether reference audio is turned into discrete codec codes, embeddings, or both
- where the conditioning enters the generator
- what preprocessing is required before generation

## Likely comparison targets

### 1. SGLang Higgs TTS runtime

Target questions:

- how reference audio is encoded
- whether encoded reference codes are inserted into the prompt
- whether the model uses placeholder token spans for reference audio
- whether there is separate reference text usage

Most important likely files:

- Higgs request builder
- Higgs stages
- Higgs audio codec handling
- Higgs sampler/model runner

### 2. MOSS voice-cloning runtime

Target questions:

- where the speaker/reference conditioning is computed
- what tensor shape gets passed into the generator
- whether the conditioning is global, token-level, or frame-level
- how ONNX exports externalize that path

### 3. Current local Higgs ONNX browser/runtime code

Target questions:

- which parts were already exported
- whether any hidden assumption exists that the missing conditioning was intentionally omitted from the export
- whether the current AR graph could accept a conditioned variant with only a changed prompt layout

## Possible strategies

## Strategy A: Find missing Higgs ONNX conditioning artifacts

Best-case path.

Look for:

- reference audio encoder ONNX
- reference-code preprocessor/export
- conditioned prefill graph variant

Why this is best:

- minimal architecture rewriting
- keeps the current proven ONNX local path intact

Risk:

- those artifacts may simply not be published

## Strategy B: Reconstruct Higgs conditioning from SGLang source

If SGLang reveals that conditioning is prompt-level discrete-code insertion rather than a hidden side network, this becomes much more tractable.

What to look for:

- reference waveform -> codec code conversion
- delay-pattern handling for reference codes
- prompt placeholder insertion semantics

If conditioning is just “encode reference audio to codes, then splice codes into the prompt before `<|audio|>` generation,” the current ONNX route may be recoverable with an external pre-step.

Risk:

- conditioning may still depend on unpublished model-side export details

## Strategy C: Use MOSS as architectural reference only

If Higgs conditioning requires a separate reference encoder export, MOSS can help answer:

- how to package a reference encoder for local use
- how to keep that path sidecar-friendly
- how to separate reference conditioning from the main autoregressive loop

This does not produce compatibility by itself. It only informs implementation design.

## Strategy D: Hybrid sidecar design

If the missing Higgs conditioning cannot be exported cleanly right now, use a split approach:

- local ONNX Higgs zero-shot for fast default TTS
- existing clone-capable engine for explicit voice cloning requests

That is operationally pragmatic even if architecturally inelegant.

## Recommended investigation order

1. Read SGLang Higgs source to determine the true reference-audio injection path.
2. Check whether that path depends on assets not present in the current ONNX export.
3. Compare the missing path with how MOSS exposes cloning components.
4. Decide whether the missing Higgs piece is:
   - a missing export artifact
   - a small prompt-building pre-step
   - or a larger unpublished subsystem

## What success would look like

A successful outcome would be one of these:

- a newly found Higgs ONNX reference-conditioning artifact set
- proof that current ONNX AR graphs can be conditioned externally with encoded reference codes
- a clear engineering spec for a custom preconditioning step modeled after the SGLang reference path

## Recommendation

Do not start by porting MOSS code into Higgs directly.

Start by extracting the **contract** from SGLang Higgs:

- what goes in
- in what format
- at what point in the prompt/generation pipeline

Then use MOSS only as a reference for how to package the missing conditioning path in a clean local runtime.
