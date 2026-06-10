# Higgs Benchmark Analysis Report

## Scope

This report summarizes only the measurements directly observed during the current local investigation.

It covers:

- local ONNX Higgs zero-shot TTS
- CPU vs CUDA behavior
- fp32 vocoder vs int4 vocoder
- warm-model and burst-style interpretation

It does not cover voice cloning quality because the current ONNX artifacts do not expose reference-audio conditioning.

## Confirmed working path

Confirmed working end-to-end path:

- local tokenizer
- `ar_prefill_matmul4`
- `ar_decode_matmul4`
- local vocoder decode
- saved WAV output

Two example output files generated during testing:

- [higgs_hello_world.wav](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/higgs_hello_world.wav)
- [higgs_male_voice_test.wav](C:/Users/h4rdc/Documents/Github/coding-agent/chatterbox/higgs/higgs_male_voice_test.wav)

The second filename was only a disambiguation label. It was not a true male-clone output.

## Important caveat

All current performance results are for **zero-shot Higgs ONNX**.

No current benchmark here measures reference-clone inference.

## CPU result

Warm benchmark on CPU:

- roughly `7.4s` audio
- roughly `36.88s` total in one measured run
- `RTF ~5.33`

Conclusion:

- CPU path is functional but operationally poor
- not suitable for the intended AIRI-sidecar usage target

## CUDA result

Warm benchmark on CUDA with `matmul4` AR + `fp32` vocoder:

- benchmark sentence audio length: `7.4s`
- warm total time: `6.09s`
- warm `RTF ~0.823`

This is the strongest result observed in the whole investigation.

Conclusion:

- the local ONNX path is viable on GPU
- sub-1 warm RTF is already possible in at least one realistic test case

## fp32 vs int4 vocoder comparison

Same benchmark sentence, same CUDA provider.

### fp32 vocoder

- cold `RTF ~0.939`
- warm `RTF ~0.823`
- warm vocoder time: about `0.0317s`

### int4 vocoder

- cold `RTF ~0.991`
- warm `RTF ~1.038`
- warm vocoder time: about `0.0266s`

### Interpretation

- vocoder cost is tiny in both cases
- AR dominates total latency
- int4 vocoder saves only a few milliseconds
- fp32 vocoder produced the better total end-to-end warm result

Recommendation:

- keep using `fp32` vocoder unless a larger repeated sample shows a different trend

## Burst profile results

The burst profile ran each case 5 times:

- run 1 = cold-ish
- runs 2-5 = warm burst approximation

### Short case

- audio: about `3.92s`
- warm mean `RTF ~1.313`

### Medium case

- audio: about `7.4s`
- warm mean `RTF ~1.334`

### Long case

- audio: about `7.4s`
- warm mean `RTF ~1.367`

### Interpretation

Strengths:

- repeated warm requests are stable
- no catastrophic burst collapse
- GPU vocoder overhead is negligible

Weaknesses:

- short bursty replies in the `~4s` range are still above real-time in the burst sweep
- AR generation remains the dominant bottleneck
- longer prompt text did not automatically translate to longer output in the current setup, suggesting output-length behavior is bounded by current sampling/termination behavior rather than text length alone

## Why some numbers vary

Observed RTF varies because:

- stochastic sampling changes emitted frame count
- current termination behavior changes output length
- AR stage dominates, so small generation differences move total RTF noticeably

This is why a single good warm run and the broader burst sweep can disagree somewhat without indicating a tooling error.

## Recommendation

Current best practical recommendation:

- Use `CUDAExecutionProvider`
- Use `matmul4` AR graphs
- Use `fp32` vocoder
- Treat this as a **zero-shot Higgs engine**

Decision guidance:

- If you need a working sidecar soon, integrate this zero-shot GPU path first
- If you need voice cloning, do not overclaim. That capability is not proven with the current ONNX bundle
- If you want further speed gains, focus on the AR stage, not the vocoder
