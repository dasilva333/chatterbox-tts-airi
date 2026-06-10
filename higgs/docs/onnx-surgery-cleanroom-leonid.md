# Higgs ONNX Surgery Cleanroom for Leonid

## Purpose

This is the clean-room setup note for the ONNX surgery approach to Higgs Audio v3 voice cloning.

The goal is not to hunt for missing model artifacts.
The goal is to inspect the existing Higgs ONNX prefill graph and determine whether it can be surgically modified so the model accepts `inputs_embeds` instead of only `input_ids`.

If that cut is viable, then external reference-audio conditioning can be built outside the graph and injected into the prefill input surface.

## What We Already Know

- Zero-shot Higgs ONNX works locally.
- The current public ONNX graph interface does not expose any reference-audio conditioning input.
- SGLang Higgs cloning does use reference conditioning.
- SGLang Higgs cloning is built from:
  - reference audio or reference codes
  - delayed 8-codebook rows
  - `-100` placeholder spans
  - fused multi-codebook embedding overlay during prefill

The surgery idea is:

```text
current:
  input_ids -> embedding gather -> transformer prefill

surgical target:
  inputs_embeds -> transformer prefill
```

Then outside ONNX:

```text
text tokens + reference codebook overlays -> inputs_embeds
```

## Files To Download

Leonid should collect these files into a local clean-room workspace.
The Higgs-side files live in the repo:

- `https://github.com/dasilva333/chatterbox-tts-airi/tree/main/higgs`

The large model artifacts live in the Hugging Face repo:

- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX`

### Required ONNX artifacts

- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/ar/higgs_audio_v3_ar_prefill_matmul4.onnx`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/ar/higgs_audio_v3_ar_prefill_matmul4.onnx.data`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/ar/higgs_audio_v3_ar_decode_matmul4.onnx`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/ar/higgs_audio_v3_ar_decode_matmul4.onnx.data`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/higgs_audio_v3_vocoder_decode.onnx`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/higgs_audio_v3_vocoder_decode.onnx.data`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/higgs_audio_v3_vocoder_decode_matmul4.onnx`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/higgs_audio_v3_vocoder_decode_matmul4.onnx.data`

### Required tokenizer and config files

- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/tokenizer.json`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/tokenizer_config.json`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/config.json`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/chat_template.jinja`

### Required local reference docs and code

- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/test_higgs_onnx_poc.py`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/test_higgs_onnx_voice_cloning_poc%20(1).py`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/stages.py`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/higgs_tts.md`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/strategy-higgs-sglang-crossport.md`
- `https://github.com/dasilva333/chatterbox-tts-airi/blob/main/higgs/strategy-nvfp4-sglang-load-path.md`

### Optional but useful reference material

- `https://github.com/sgl-project/sglang-omni`
- `https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4`

## Scope Of Work

The scope is intentionally narrow.

### In scope

1. Inspect `higgs_audio_v3_ar_prefill_matmul4.onnx` for a usable embedding cut point.
2. Determine whether the graph still contains a recognizable token embedding `Gather`.
3. Determine whether the output of that embedding path can be replaced with a new graph input named `inputs_embeds`.
4. Preserve the rest of the prefill graph, including:
   - attention mask handling
   - positional handling
   - KV-cache outputs
   - logits output
5. Build an external embedding overlay path that can combine:
   - normal text token embeddings
   - delayed reference-code embeddings
   - placeholder spans
6. Prove the patched graph can still run the zero-shot path before attempting cloning.

### Out of scope

- Do not search for missing model files first.
- Do not spend time on NVFP4 loader work for this task.
- Do not try to port the whole SGLang runtime.
- Do not assume the browser demo UI implies the graph already supports cloning.
- Do not hardcode voice-clone logic into the existing zero-shot runner without first proving the graph surgery point exists.

## Exact Questions To Answer

1. Does the prefill ONNX graph contain a `Gather`-style token embedding node fed by `input_ids`?
2. What initializer or table feeds that node?
3. What is the name and shape of the tensor that leaves the embedding path?
4. Can that tensor be replaced by an external `inputs_embeds` graph input?
5. Are there positional or matmul4-specific rewrites that would break a naive cut?
6. If the cut is feasible, what is the smallest patched graph that still runs?

## Cheap Falsification Probe

Use this kind of inspection first after the files are on disk:

```python
import onnx

model = onnx.load("higgs_audio_v3_ar_prefill_matmul4.onnx", load_external_data=True)

print("inputs:")
for x in model.graph.input:
    print(x.name)

print("initializers:")
for init in model.graph.initializer:
    if "embed" in init.name.lower() or "wte" in init.name.lower():
        print(init.name, init.dims)

print("gather nodes:")
for node in model.graph.node:
    if node.op_type == "Gather":
        print(node.name, node.input, node.output)
```

The first useful answer is not “can it clone.”
The first useful answer is:

```text
is the embedding path cuttable?
```

## Recommended Work Plan

1. Download the files above into a clean local folder.
2. Load the prefill graph and inspect graph inputs, initializers, and gather nodes.
3. Decide whether `inputs_embeds` surgery is viable.
4. If viable, patch the prefill graph and write a tiny wrapper that builds external embeddings.
5. Only after that, wire in reference-code overlay logic and clone validation.

## Success Criteria

This task succeeds if we can answer one of the following with evidence:

- `yes, the prefill graph can be cut to inputs_embeds`
- `no, the graph layout is too fused or rewritten to support a clean surgery`

Either answer is useful.

The failure mode we want to avoid is spending tokens on speculation when the graph structure can be checked directly.

## Practical Note

If Leonid wants to work locally, he should make a private scratch repo or fork and keep the large `.onnx.data` files out of Git history.

The clean-room experiment can be entirely local and disposable.
