# Codex Resident Expert

Created: 2026-06-09T13:23:28.2660867-04:00

## Session identifiers

- `CODEX_THREAD_ID`: `019eaa16-9886-76c3-a76b-f39e237f8dd6`

## Repo anchor

- Current commit during note creation: `bf1be070068c2443e278115378964b096e053c3c`

## Notes

- I did not find a separate exposed `agent_id` environment variable in this session.
- The most useful resumable identifier exposed to the process is `CODEX_THREAD_ID`.
- Higgs NVFP4 status at this point: the real `quantized.safetensors` is downloaded locally, but the payload uses `*.packed_nvfp4` plus `*.scales_e4m3` tensors rather than SGLang's expected `weight` / `input_scale` / `weight_scale` / `weight_scale_2` layout, so a loader shim is still needed.

## Current state

- Local zero-shot Higgs ONNX pipeline works.
- Local ONNX voice cloning does not work with the current public ONNX graph interface.
- The working ONNX path is useful for zero-shot serving and benchmarking, but not for true cloning.
- The SGLang Higgs path is the correct reference for cloning because it already contains:
  - preprocessing
  - audio encoder
  - TTS engine
  - vocoder
  - delayed reference-code handling
  - fused embedding overlay at `-100` placeholder positions during prefill

## What was verified most recently

- `higgs_nvfp4_bridge.py` can prepare the local NVFP4 snapshot:
  - patch `config.json`
  - write `hf_quant_config.json`
  - replace the LFS-pointer tokenizer with the real local `tokenizer.json`
- The real `quantized.safetensors` was downloaded into:
  - [nvfp4_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/nvfp4_reference)
- Strict preflight against the real checkpoint succeeded on existence checks and failed on loader-format expectations.

## Real NVFP4 checkpoint findings

- The payload is no longer hypothetical. The real file is present locally.
- The safetensors header shows quantized tensors named like:
  - `body.layers.0.self_attn.q_proj.weight.packed_nvfp4`
  - `body.layers.0.self_attn.q_proj.weight.scales_e4m3`
- The checkpoint does not expose SGLang ModelOpt FP4 tensors in the names/layout currently expected by the installed `sglang` loader.
- The installed `sglang` `ModelOptFp4LinearMethod` expects a serialized structure shaped around:
  - `weight`
  - `input_scale`
  - `weight_scale`
  - `weight_scale_2`
- So the current blocker is not missing cloning logic, and not missing model files. The blocker is the weight-loading seam.

## Important conclusion

- This is not "we are locked out from all angles" anymore.
- This is now a narrower technical problem:
  - determine whether Reza's `packed_nvfp4` plus `scales_e4m3` layout can be mapped into SGLang's FP4 loader expectations
  - or add a Reza-specific loader/dequant shim for Higgs/Qwen3 body layers

## Environment findings

- `sglang` is installed in the repo venv.
- `sglang_omni` is not currently installed in the repo venv.
- That means even if the loader issue were solved, a full local SGLang-Omni Higgs serve path still needs the Omni runtime installed in the target environment.
- This was not the first blocker. The first blocker is still the checkpoint tensor layout mismatch.

## Highest-value next steps

1. Inspect the installed `sglang` FP4 loader and determine whether it can be patched to consume:
   - `*.packed_nvfp4` as packed FP4 weights
   - `*.scales_e4m3` as block scales
2. Determine whether missing `input_scale` and `weight_scale_2`:
   - exist implicitly elsewhere in the Reza export,
   - can be derived,
   - can be defaulted safely,
   - or require a different export path.
3. Inspect whether `weight_loader.py` in the Higgs overlay is the clean insertion point for remapping these tensors before they reach SGLang quantized linear modules.
4. If the loader path looks too invasive, test a fallback approach:
   - offline convert the Reza NVFP4 checkpoint into SGLang-native ModelOpt FP4 tensor names/layout
   - then load the converted checkpoint through unmodified SGLang.
5. Only after the weight format issue is resolved:
   - install `sglang_omni`
   - install the overlay
   - try a true voice-cloning smoke test through the SGLang Higgs runtime

## Lower-value paths to avoid repeating

- Do not go back to the full unquantized official Higgs model for this machine if the goal is fast local inference on 8 GB VRAM.
- Do not spend more time trying to force true cloning through the current public ONNX graph interface without a new export surface.
- Do not assume the ONNX repo README is fully current; actual artifact inspection mattered more than the README text.

## Useful local files for resume

- [higgs_nvfp4_bridge.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/higgs_nvfp4_bridge.py)
- [chatgpt-pro-attempt-2.md](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/chatgpt-pro-attempt-2.md)
- [stages_nvfp4.patch](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/stages_nvfp4.patch)
- [strategy-nvfp4-sglang-load-path.md](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/strategy-nvfp4-sglang-load-path.md)
- [strategy-higgs-sglang-crossport.md](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/strategy-higgs-sglang-crossport.md)
- [stages.py](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/stages.py)
- [higgs_tts.md](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/higgs_tts.md)
- [nvfp4_reference](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/nvfp4_reference)

## Best short summary

- Zero-shot ONNX Higgs: working.
- ONNX voice cloning: not available with current public graph interface.
- SGLang Higgs cloning path: understood.
- NVFP4 checkpoint: downloaded and inspected.
- Remaining problem: adapt Reza's `packed_nvfp4` plus `scales_e4m3` format to SGLang's FP4 runtime expectations.
