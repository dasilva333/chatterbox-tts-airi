Built a first-pass NVFP4 bridge package here:

[Download the NVFP4 SGLang bridge patch](sandbox:/mnt/data/higgs_nvfp4_sglang_patch.zip)
[Open `higgs_nvfp4_bridge.py`](sandbox:/mnt/data/higgs_nvfp4_sglang_patch/higgs_nvfp4_bridge.py)
[Open the README](sandbox:/mnt/data/higgs_nvfp4_sglang_patch/README_NVFP4_BRIDGE.md)
[Open the `stages.py` patch](sandbox:/mnt/data/higgs_nvfp4_sglang_patch/patches/stages_nvfp4.patch)

What I made is **not another ONNX attempt**. It pivots to the SGLang-Omni Higgs stack, because your `stages.py` confirms that the working cloning pipeline is already there: `preprocessing → audio_encoder → tts_engine → vocoder`, with reference codes delayed and overlaid as fused multi-codebook embeddings at `-100` placeholder positions during prefill.  The strategy doc also correctly narrowed the issue to whether the NVFP4 checkpoint can load through the existing Higgs/Qwen3 SGLang runtime, not whether we still need to invent cloning. 

The bridge does four concrete things:

1. **Prepares the NVFP4 repo for SGLang’s ModelOpt FP4 loader** by adding SGLang-style `quantization_config` into `config.json`, writing an `hf_quant_config.json` sidecar, and adding Qwen3 packed-module metadata for `q/k/v` and `gate/up`.

2. **Fixes common snapshot problems**: the HF file listing shows the real model payload is `quantized.safetensors` at about 2.04 GB and `tokenizer.json` is an 11.4 MB Xet/LFS file, so the bridge detects a pointer-style tokenizer and replaces it from your working Higgs tokenizer or downloads the real blob. ([Hugging Face][1])

3. **Patches the Higgs SGLang overlay**: I copied the Higgs reference files into `sglang_overlay/` and patched `stages.py` to add the missing `DEFAULT_MAX_CONCURRENCY`, detect prepared NVFP4 checkpoints, and pass a `modelopt_fp4` quantization hint into SGLang’s server args.

4. **Adds preflight checks** so the next failure is precise instead of vague. SGLang’s ModelOpt FP4 path expects serialized NVFP4 tensors with `input_scale`, packed `weight`, `weight_scale`, and `weight_scale_2`; my script inspects the actual safetensors header and tells us whether the downloaded file matches that layout. ([GitHub][2])

The reason this bridge is necessary is that Reza’s NVFP4 artifact describes itself as a transformer-body quantized artifact, not a complete drop-in runtime, while preserving the codec/vocoder/fused embedding/head pieces and requiring SGLang-Omni or a custom loader for runtime integration. ([Hugging Face][3]) The metadata you uploaded matches that: its `quantization_config.json` is descriptive, not the SGLang-native ModelOpt config shape. SGLang’s ModelOpt FP4 config parser wants `quant_algo=NVFP4` plus a group size and exclusion metadata. ([Hugging Face][4])

Basic usage from the patch directory:

```bash
python higgs_nvfp4_bridge.py install-overlay --dry-run
python higgs_nvfp4_bridge.py install-overlay
```

Then prepare the model:

```bash
python higgs_nvfp4_bridge.py prepare \
  --source Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4 \
  --output ./higgs_nvfp4_prepared \
  --tokenizer-source ./higgs/tokenizer.json \
  --preflight
```

Or prepare an already-downloaded local copy:

```bash
python higgs_nvfp4_bridge.py prepare \
  --model-path ./nvfp4_reference \
  --tokenizer-source ./higgs/tokenizer.json \
  --preflight
```

Then run the strict preflight once the full 2 GB safetensors payload is present:

```bash
python higgs_nvfp4_bridge.py preflight \
  --model-path ./higgs_nvfp4_prepared \
  --strict
```

Start serving with the Marlin FP4 backend first on non-Blackwell cards:

```bash
HIGGS_NVFP4_FP4_GEMM_BACKEND=marlin \
python higgs_nvfp4_bridge.py serve \
  --model-path ./higgs_nvfp4_prepared \
  --port 8000
```

SGLang’s docs say FP4/NVFP4 GEMM backend selection is supported through `--fp4-gemm-backend`; the patched stage also reads `HIGGS_NVFP4_FP4_GEMM_BACKEND` so the backend hint can flow into the SGLang engine. ([SGLang Documentation][5])

Zero-shot smoke test:

```bash
python higgs_nvfp4_bridge.py speech \
  --text "Hello from Higgs NVFP4." \
  --out zero_shot.wav
```

Voice-clone smoke test:

```bash
python higgs_nvfp4_bridge.py speech \
  --text "Have a nice day and enjoy south California sunshine." \
  --reference-audio ./voices/reference.wav \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --out clone.wav
```

The script intentionally requires `--i-have-rights-to-this-voice` when a reference voice is supplied.

What I could test here: Python syntax, metadata preparation, tokenizer-pointer detection/replacement, and preflight behavior. The metadata-only `nvfp4_reference` folder from your zip prepared successfully, then correctly failed on the missing `.safetensors` payload:

```text
[OK] config.json: model_type='higgs_multimodal_qwen3'
[OK] SGLang quantization_config: quant_algo='NVFP4', group_size=16
[OK] tokenizer.json: materialized
[FAIL] safetensors file: none found
      fix: download quantized.safetensors/model.safetensors from the NVFP4 repo
```

I could not run end-to-end SGLang inference in this sandbox because the environment does not have the CUDA/SGLang runtime or the actual 2.04 GB `quantized.safetensors` payload.

The remaining make-or-break test is the safetensors header check. If the real `quantized.safetensors` contains ModelOpt-style scale tensors, this bridge has a good shot at loading through SGLang. If preflight reports `ModelOpt FP4 scale tensors: found 0`, then the next patch needs to be a **Reza-specific loader/dequant shim**, because SGLang’s native `modelopt_fp4` loader will be expecting serialized scale tensors that the artifact does not expose.

[1]: https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4/tree/main "https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4/tree/main"
[2]: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/layers/quantization/modelopt_quant.py "https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/layers/quantization/modelopt_quant.py"
[3]: https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4 "https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4"
[4]: https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4/raw/main/quantization_config.json "https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4/raw/main/quantization_config.json"
[5]: https://sgl-project.github.io/advanced_features/quantization.html "https://sgl-project.github.io/advanced_features/quantization.html"
