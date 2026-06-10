# Higgs Audio v3 NVFP4 SGLang bridge

This patch is a practical bridge for trying `Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4` through the existing SGLang-Omni Higgs runtime instead of the zero-shot-only ONNX path.

It does **not** try to force voice cloning through the current ONNX graphs. The SGLang Higgs path already knows how to do cloning: reference audio or `reference_codes` are converted into delayed 8-codebook rows, prompt placeholders are built, and the model runner overlays fused audio-code embeddings during prefill. The remaining NVFP4 problem is the loader/runtime seam.

## Contents

- `higgs_nvfp4_bridge.py` — CLI tool for preparing the NVFP4 repo, installing the Higgs overlay, launching the server, and sending smoke-test speech requests.
- `sglang_overlay/` — Higgs-specific SGLang-Omni reference files plus a patched `stages.py`.

The patched `stages.py` adds:

- `DEFAULT_MAX_CONCURRENCY`, which the supplied `config.py` expects.
- Detection for prepared NVFP4 checkpoints.
- A SGLang ModelOpt FP4 quantization hint when the model has been prepared.
- Optional FP4 GEMM backend selection via `HIGGS_NVFP4_FP4_GEMM_BACKEND`.

## Why this bridge exists

The NVFP4 repo metadata is descriptive, not directly the same as SGLang's native ModelOpt FP4 config. This bridge writes SGLang-style metadata into `config.json` and `hf_quant_config.json`, fixes common snapshot issues, then runs preflight checks against the actual safetensors header.

The important preflight check is whether the real downloaded safetensors file contains ModelOpt-style FP4 tensors such as:

- `weight`
- `input_scale`
- `weight_scale`
- `weight_scale_2`

If the actual 2 GB `quantized.safetensors` has no scale-like keys, SGLang's native `modelopt_fp4` loader will probably still reject it. In that case, the next required work is a Reza-specific loader/dequant shim or a re-export into SGLang ModelOpt FP4 format.

## Recommended environment

Use Linux or WSL2 with a CUDA-capable PyTorch/SGLang install. Native Windows is likely to be painful for SGLang-Omni.

Install your normal SGLang-Omni stack first. Exact versions move quickly, so prefer the versions known to work in your repo, or install from current SGLang/SGLang-Omni source.

Minimum Python packages used by the bridge itself:

```bash
pip install huggingface_hub requests safetensors
```

The server path additionally needs your working `sglang`, `sglang-omni`, `torch`, `torchaudio`, tokenizer, and CUDA runtime stack.

## Step 1: install the Higgs overlay

From the patch directory:

```bash
python higgs_nvfp4_bridge.py install-overlay --dry-run
python higgs_nvfp4_bridge.py install-overlay
```

The installer copies the overlay into the active `sglang_omni` package and writes `*.bak` backups next to overwritten files.

## Step 2: prepare the NVFP4 model directory

Best case, let the bridge download the full HF snapshot:

```bash
python higgs_nvfp4_bridge.py prepare \
  --source Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4 \
  --output ./higgs_nvfp4_prepared \
  --tokenizer-source ./higgs/tokenizer.json \
  --preflight
```

For an already-downloaded local copy:

```bash
python higgs_nvfp4_bridge.py prepare \
  --model-path ./nvfp4_reference \
  --tokenizer-source ./higgs/tokenizer.json \
  --preflight
```

`--tokenizer-source` can point either to a real `tokenizer.json` file or a directory containing one. This matters because the metadata-only zip contained a Git-LFS pointer for `tokenizer.json`, not the real 11.4 MB tokenizer blob.

## Step 3: run preflight again after the full 2 GB safetensors file is present

```bash
python higgs_nvfp4_bridge.py preflight --model-path ./higgs_nvfp4_prepared --strict
```

Expected failures and meanings:

- `safetensors file: none found` — you only have metadata, not the actual `quantized.safetensors` payload.
- `tokenizer.json: missing or Git-LFS pointer` — provide a real tokenizer from the working Higgs/ONNX bundle or allow HF download.
- `ModelOpt FP4 scale tensors: found 0` — the artifact is not in SGLang's native serialized ModelOpt FP4 layout. A custom loader/dequant shim is still needed.

## Step 4: serve

On Ampere/Ada/Hopper, start with Marlin for the FP4 GEMM backend:

```bash
HIGGS_NVFP4_FP4_GEMM_BACKEND=marlin \
python higgs_nvfp4_bridge.py serve \
  --model-path ./higgs_nvfp4_prepared \
  --port 8000
```

On Blackwell, native/CUTLASS/FlashInfer paths may be viable depending on your installed SGLang stack.

## Step 5: smoke-test zero-shot speech

```bash
python higgs_nvfp4_bridge.py speech \
  --text "Hello from Higgs NVFP4." \
  --out zero_shot.wav
```

## Step 6: smoke-test cloning

Only use a reference voice you own or have permission to clone:

```bash
python higgs_nvfp4_bridge.py speech \
  --text "Have a nice day and enjoy south California sunshine." \
  --reference-audio ./voices/reference.wav \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --out clone.wav
```

## What was tested here

In this sandbox I could test the metadata preparation path and Python syntax. I could not run full SGLang/NVFP4 inference because the sandbox does not have the CUDA/SGLang runtime or the 2 GB `quantized.safetensors` payload.

The bridge was tested against the metadata-only `nvfp4_reference` folder from your zip. It successfully:

- wrote SGLang-style ModelOpt FP4 quantization metadata,
- copied a real tokenizer from the local Higgs folder when supplied,
- and correctly failed preflight on the missing safetensors payload.

