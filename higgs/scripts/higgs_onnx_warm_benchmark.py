from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
TOKENIZER_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"

PREFILL_ONNX = MODELS_DIR / "higgs_audio_v3_ar_prefill_matmul4.onnx"
DECODE_ONNX = MODELS_DIR / "higgs_audio_v3_ar_decode_matmul4.onnx"
VOCODER_FP32_ONNX = MODELS_DIR / "higgs_audio_v3_vocoder_decode.onnx"
VOCODER_INT4_ONNX = MODELS_DIR / "higgs_audio_v3_vocoder_decode_matmul4.onnx"

SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8
CODEC_CODEBOOK_SIZE = 1024
BOC_ID = 1024
EOC_ID = 1025
TTS_ID = 151667
TEXT_ID = 151672
AUDIO_ID = 151670

DEFAULT_WARMUP_TEXT = "hi."
DEFAULT_BENCH_TEXT = "Hello world, this practical benchmark sentence helps us measure warm Higgs speech speed accurately today."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm Higgs ONNX benchmark")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--vocoder", choices=("fp32", "int4"), default="fp32")
    parser.add_argument("--warmup-text", default=DEFAULT_WARMUP_TEXT)
    parser.add_argument("--warmup-max-steps", type=int, default=16)
    parser.add_argument("--benchmark-text", default=DEFAULT_BENCH_TEXT)
    parser.add_argument("--benchmark-max-steps", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "audio" / "higgs_warm_benchmark.wav"))
    return parser.parse_args()


def ensure_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    providers = [provider]
    available = ort.get_available_providers()
    if provider not in available:
        print(f"[WARN] Provider {provider} unavailable. Falling back to CPUExecutionProvider.")
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def sample_codebooks(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> list[int]:
    out: list[int] = []
    for cb in range(NUM_CODEBOOKS):
        scores = logits[0, cb].astype(np.float64)
        best = int(scores.argmax())
        if temperature <= 1e-5 or top_k == 1:
            out.append(best)
            continue
        order = np.argsort(scores)[::-1]
        if 0 < top_k < len(order):
            order = order[:top_k]
        kept_scores = scores[order]
        kept_scores = (kept_scores - kept_scores.max()) / temperature
        probs = np.exp(kept_scores)
        probs /= probs.sum()
        if 0.0 < top_p < 1.0:
            cdf = np.cumsum(probs)
            cutoff = int(np.searchsorted(cdf, top_p, side="left")) + 1
            order = order[:cutoff]
            probs = probs[:cutoff]
            probs /= probs.sum()
        chosen = int(rng.choice(order, p=probs))
        out.append(chosen)
    return out


def apply_delay_mask(codes: list[int], step: int) -> list[int]:
    if step < NUM_CODEBOOKS:
        next_cb = step + 1
        for cb in range(next_cb, NUM_CODEBOOKS):
            codes[cb] = BOC_ID
    return codes


def extract_past(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for layer in range(36):
        feeds[f"past_{layer}_key"] = outputs[f"present_{layer}_key"]
        feeds[f"past_{layer}_value"] = outputs[f"present_{layer}_value"]
    return feeds


def delayed_to_codec_rows(rows: list[list[int]]) -> list[list[int]]:
    total = len(rows) - (NUM_CODEBOOKS - 1)
    frames: list[list[int]] = []
    for t in range(total):
        row: list[int] = []
        valid = True
        for cb in range(NUM_CODEBOOKS):
            code = rows[t + cb][cb]
            if code < 0 or code >= CODEC_CODEBOOK_SIZE:
                valid = False
                break
            row.append(code)
        if not valid:
            break
        frames.append(row)
    return frames


def to_codes_tensor(codes: list[int]) -> np.ndarray:
    return np.asarray(codes, dtype=np.int64).reshape(1, NUM_CODEBOOKS)


class HiggsOnnxEngine:
    def __init__(self, provider: str, vocoder_kind: str):
        vocoder_onnx = VOCODER_FP32_ONNX if vocoder_kind == "fp32" else VOCODER_INT4_ONNX
        ensure_files([PREFILL_ONNX, DECODE_ONNX, vocoder_onnx])
        self.provider = provider
        self.vocoder_kind = vocoder_kind
        print(f"[INFO] Loading tokenizer from {TOKENIZER_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        print(f"[INFO] Loading prefill session: {PREFILL_ONNX.name}")
        self.prefill_sess = make_session(PREFILL_ONNX, provider)
        print(f"[INFO] Loading decode session: {DECODE_ONNX.name}")
        self.decode_sess = make_session(DECODE_ONNX, provider)
        print(f"[INFO] Loading vocoder session: {vocoder_onnx.name}")
        self.vocoder_sess = make_session(vocoder_onnx, provider)

    def build_prompt_ids(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(text.strip(), add_special_tokens=False)
        ids = [TTS_ID, TEXT_ID, *encoded["input_ids"], AUDIO_ID]
        return np.asarray(ids, dtype=np.int64)

    def synthesize(
        self,
        text: str,
        *,
        max_steps: int = 192,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int = 7,
    ) -> tuple[np.ndarray, dict[str, float]]:
        input_ids = self.build_prompt_ids(text)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        rng = np.random.default_rng(seed)

        ar_start = time.perf_counter()
        outputs = self.prefill_sess.run(
            None,
            {
                "input_ids": input_ids.reshape(1, -1),
                "attention_mask": attention_mask.reshape(1, -1),
            },
        )
        outputs_dict = dict(zip((o.name for o in self.prefill_sess.get_outputs()), outputs))
        delay_count = 0
        eoc_countdown: int | None = None
        done = False
        codes = sample_codebooks(outputs_dict["logits"], temperature, top_p, top_k, rng)
        codes = apply_delay_mask(codes, delay_count)
        delay_count += 1
        rows = [codes[:]]
        past = extract_past(outputs_dict)

        for step in range(1, max_steps):
            if done:
                break
            outputs = self.decode_sess.run(
                None,
                {
                    "codes": to_codes_tensor(codes),
                    "position_ids": np.asarray([[input_ids.shape[0] + step - 1]], dtype=np.int64),
                    **past,
                },
            )
            outputs_dict = dict(zip((o.name for o in self.decode_sess.get_outputs()), outputs))
            codes = sample_codebooks(outputs_dict["logits"], temperature, top_p, top_k, rng)
            if delay_count < NUM_CODEBOOKS:
                codes = apply_delay_mask(codes, delay_count)
                delay_count += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    done = True
            elif codes[0] == EOC_ID:
                eoc_countdown = 0 if NUM_CODEBOOKS <= 2 else NUM_CODEBOOKS - 2
            rows.append(codes[:])
            past = extract_past(outputs_dict)

        codec_rows = delayed_to_codec_rows(rows)
        if not codec_rows:
            raise RuntimeError("AR path produced no complete codec frames.")
        ar_dt = time.perf_counter() - ar_start

        t = len(codec_rows)
        flat = np.zeros((1, NUM_CODEBOOKS, t), dtype=np.int64)
        for frame_idx, row in enumerate(codec_rows):
            for codebook_idx, code in enumerate(row):
                flat[0, codebook_idx, frame_idx] = code

        voc_start = time.perf_counter()
        voc_outputs = self.vocoder_sess.run(None, {"audio_codes": flat})
        voc_dt = time.perf_counter() - voc_start
        wav = np.asarray(voc_outputs[0][0, 0], dtype=np.float32)
        audio_seconds = wav.shape[0] / SAMPLE_RATE
        total_dt = ar_dt + voc_dt
        metrics = {
            "prompt_tokens": float(input_ids.shape[0]),
            "codec_frames": float(len(codec_rows)),
            "audio_seconds": audio_seconds,
            "ar_seconds": ar_dt,
            "vocoder_seconds": voc_dt,
            "total_seconds": total_dt,
            "rtf": total_dt / audio_seconds if audio_seconds > 0 else math.inf,
        }
        return wav, metrics


def main() -> None:
    args = parse_args()
    engine = HiggsOnnxEngine(args.provider, args.vocoder)

    print("[INFO] Running warmup generation (discarded)...")
    _warm_wav, warmup_metrics = engine.synthesize(
        args.warmup_text,
        max_steps=args.warmup_max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(
        "[INFO] warmup "
        f"audio={warmup_metrics['audio_seconds']:.2f}s "
        f"total={warmup_metrics['total_seconds']:.2f}s "
        f"rtf={warmup_metrics['rtf']:.3f}"
    )

    print("[INFO] Running measured benchmark generation...")
    wav, metrics = engine.synthesize(
        args.benchmark_text,
        max_steps=args.benchmark_max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    out_path = Path(args.out)
    sf.write(out_path, wav, SAMPLE_RATE)
    print(f"[INFO] saved {out_path}")
    print(
        "[INFO] benchmark "
        f"audio={metrics['audio_seconds']:.2f}s "
        f"ar={metrics['ar_seconds']:.2f}s "
        f"vocoder={metrics['vocoder_seconds']:.2f}s "
        f"total={metrics['total_seconds']:.2f}s "
        f"rtf={metrics['rtf']:.3f}"
    )


if __name__ == "__main__":
    main()
