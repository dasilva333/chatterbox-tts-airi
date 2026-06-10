Implemented the voice-cloning POC replacement and packaged it here:

[Download the patch zip](sandbox:/mnt/data/higgs_voice_cloning_patch.zip)
[Download the updated script directly](sandbox:/mnt/data/higgs_voice_cloning_patch/test_higgs_onnx_voice_cloning_poc.py)
[Download the unified diff](sandbox:/mnt/data/higgs_voice_cloning_poc.patch)

What I changed: the old `test_higgs_onnx_voice_cloning_poc.py` was only a diagnostic script. I replaced it with a runner that implements the **actual SGLang Higgs cloning contract**: reference transcript support, `reference_codes` loading, raw `[T, 8]` code validation, Higgs delay-pattern application to `[T+7, 8]`, `-100` placeholder prompt construction, optional SGLang codec encoding for raw reference audio, and hard detection of whether the ONNX prefill graph can actually consume the reference conditioning.

The important finding is that the current ONNX artifacts still cannot perform true cloning by themselves. SGLang’s prompt builder uses this layout for cloning: `<|tts|> <|ref_text|> reference text <|ref_audio|> [-100]... <|text|> target text <|audio|>`, where the number of `-100` placeholders must match the delayed reference-code row count. ([GitHub][1]) During prefill, SGLang does not treat those as ordinary token IDs; its model runner replaces the placeholder positions with fused audio-code embeddings built from `reference_codes_delayed`. ([GitHub][2]) The delay pattern itself shifts codebook `c` by `c` frames and pads with `BOC_ID=1024` / `EOC_ID=1025`. ([GitHub][3]) The fused audio embedding sums per-codebook embeddings after offsetting each codebook into one contiguous `[N*V, D]` table. ([GitHub][4])

That matches your docs: the API supports `references` with `audio_path` and `text`, plus a high-throughput `reference_codes` shortcut shaped `[T, 8]`.  But the supplied local ONNX bundle exposes only `input_ids` / `attention_mask` for prefill, `codes` / `position_ids` / KV cache for decode, and `audio_codes` for the vocoder, with no reference waveform, speaker embedding, or reference-code input. 

So the script now **does not fake voice cloning**. If you pass reference audio/codes against the current zero-shot-only ONNX export, it will fail with a detailed actionable error instead of silently generating zero-shot audio and calling it cloned. You can explicitly use `--allow-zero-shot-fallback` for benchmarking, but that is intentionally opt-in.

Example dry-run with pre-encoded reference codes:

```bash
python test_higgs_onnx_voice_cloning_poc.py \
  --text "Have a nice day and enjoy south california sunshine." \
  --reference-codes ref_codes.json \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --dry-run-contract
```

Example real-cloning attempt, which will work only once the prefill ONNX export exposes a conditioning surface such as `reference_codes_delayed`:

```bash
python test_higgs_onnx_voice_cloning_poc.py \
  --provider CUDAExecutionProvider \
  --text "Have a nice day and enjoy south california sunshine." \
  --reference-codes ref_codes.json \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --i-have-rights-to-this-voice \
  --out clone.wav
```

The missing artifact I’d ask you to get next is one of these:

1. **Best:** a Higgs AR prefill ONNX export that accepts `reference_codes_delayed` or `ref_codes_delayed` and performs the placeholder embedding overlay internally.
2. **Also workable:** a Higgs AR prefill ONNX export that accepts `input_embeds` / `inputs_embeds`, plus a way to extract/build the Higgs text embeddings and fused audio-code embeddings locally.
3. **Sidecar fallback:** the full SGLang/PyTorch Higgs runtime files and checkpoint assets for cloning only, while keeping the existing ONNX path for zero-shot.

I also included `VOICE_CLONING_IMPLEMENTATION_NOTES.md` in the zip with the exact blocker and example commands.

I could not run end-to-end ONNX inference inside this sandbox because `onnxruntime` and `transformers` are not installed here, and the extracted zip did not include the large `.onnx.data` external weight files that your requirements doc says are required. I did run a Python syntax compile on the replacement script successfully.

[1]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/text_tokenizer.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/text_tokenizer.py"
[2]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/model_runner.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/model_runner.py"
[3]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/utils.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/utils.py"
[4]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/modeling.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/higgs_tts/modeling.py"
