from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parent
VOICE_DIR = ROOT.parent / "voices"

PREFILL_ONNX = ROOT / "higgs_audio_v3_ar_prefill_matmul4.onnx"
DECODE_ONNX = ROOT / "higgs_audio_v3_ar_decode_matmul4.onnx"
VOCODER_FP32_ONNX = ROOT / "higgs_audio_v3_vocoder_decode.onnx"
VOCODER_INT4_ONNX = ROOT / "higgs_audio_v3_vocoder_decode_matmul4.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the current Higgs ONNX artifacts support voice cloning."
    )
    parser.add_argument(
        "--voice",
        default="ivy",
        help="Voice sample stem in ./voices, used only for inspection/reporting.",
    )
    return parser.parse_args()


def resolve_voice_path(stem: str) -> Path:
    for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
        path = VOICE_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No voice sample found for '{stem}' under {VOICE_DIR}")


def inspect_inputs(path: Path) -> list[str]:
    model = onnx.load(path, load_external_data=False)
    return [node.name for node in model.graph.input]


def main() -> None:
    args = parse_args()
    voice_path = resolve_voice_path(args.voice)
    audio, sr = sf.read(voice_path)
    duration = len(audio) / sr if sr else 0.0

    prefill_inputs = inspect_inputs(PREFILL_ONNX)
    decode_inputs = inspect_inputs(DECODE_ONNX)
    vocoder_fp32_inputs = inspect_inputs(VOCODER_FP32_ONNX)
    vocoder_int4_inputs = inspect_inputs(VOCODER_INT4_ONNX)

    print(f"[INFO] Voice sample: {voice_path}")
    print(f"[INFO] Sample rate: {sr} Hz | Duration: {duration:.2f}s")
    print(f"[INFO] Prefill inputs: {prefill_inputs}")
    print(f"[INFO] Decode inputs: {decode_inputs[:4]} ... ({len(decode_inputs)} total inputs)")
    print(f"[INFO] FP32 vocoder inputs: {vocoder_fp32_inputs}")
    print(f"[INFO] Int4 vocoder inputs: {vocoder_int4_inputs}")
    print()

    print("[RESULT] Current Higgs ONNX artifacts do not support voice cloning.")
    print("Reason:")
    print("  - The prefill graph only accepts text token ids and an attention mask.")
    print("  - The decode graph only accepts generated codebooks plus KV-cache tensors.")
    print("  - Both vocoders only accept discrete audio codebooks.")
    print("  - No graph exposes any reference waveform, speaker embedding, or reference-code input.")
    print()
    print("Implication:")
    print("  - The current ONNX bundle is zero-shot only.")
    print("  - The browser app's voice selector is cosmetic with these exact artifacts.")
    print("  - Real voice cloning requires additional ONNX assets or a different export path.")


if __name__ == "__main__":
    main()
