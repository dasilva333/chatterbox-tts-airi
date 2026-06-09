from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT

PREFILL_ONNX = ROOT / "higgs_audio_v3_ar_prefill_matmul4.onnx"
DECODE_ONNX = ROOT / "higgs_audio_v3_ar_decode_matmul4.onnx"
VOCODER_FP32_ONNX = ROOT / "higgs_audio_v3_vocoder_decode.onnx"
VOCODER_INT4_ONNX = ROOT / "higgs_audio_v3_vocoder_decode_matmul4.onnx"

SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1026
CODEC_CODEBOOK_SIZE = 1024
BOC_ID = 1024
EOC_ID = 1025

TTS_ID = 151667
TEXT_ID = 151672
AUDIO_ID = 151670


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Higgs ONNX hello-world proof of concept")
    parser.add_argument("--text", default="hello world")
    parser.add_argument("--out", default=str(ROOT / "higgs_hello_world.wav"))
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--vocoder",
        choices=("fp32", "int4"),
        default="fp32",
        help="Use the fp32 or matmul4 vocoder.",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime provider, e.g. CPUExecutionProvider or CUDAExecutionProvider.",
    )
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


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def build_prompt_ids(tokenizer: AutoTokenizer, text: str) -> np.ndarray:
    encoded = tokenizer(text.strip(), add_special_tokens=False)
    ids = [TTS_ID, TEXT_ID, *encoded["input_ids"], AUDIO_ID]
    return np.asarray(ids, dtype=np.int64)


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


def synthesize_codes(
    prefill_sess: ort.InferenceSession,
    decode_sess: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[list[int]]:
    rng = np.random.default_rng(args.seed)

    outputs = prefill_sess.run(
        None,
        {
            "input_ids": input_ids.reshape(1, -1),
            "attention_mask": attention_mask.reshape(1, -1),
        },
    )
    output_names = [o.name for o in prefill_sess.get_outputs()]
    outputs_dict = dict(zip(output_names, outputs))

    delay_count = 0
    eoc_countdown: int | None = None
    done = False

    codes = sample_codebooks(
        outputs_dict["logits"], args.temperature, args.top_p, args.top_k, rng
    )
    codes = apply_delay_mask(codes, delay_count)
    delay_count += 1
    rows = [codes[:]]
    past = extract_past(outputs_dict)

    for step in range(1, args.max_steps):
        if done:
            break
        feeds = {
            "codes": to_codes_tensor(codes),
            "position_ids": np.asarray([[input_ids.shape[0] + step - 1]], dtype=np.int64),
            **past,
        }
        outputs = decode_sess.run(None, feeds)
        output_names = [o.name for o in decode_sess.get_outputs()]
        outputs_dict = dict(zip(output_names, outputs))
        codes = sample_codebooks(
            outputs_dict["logits"], args.temperature, args.top_p, args.top_k, rng
        )
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
        if step % 8 == 0:
            print(f"[INFO] AR step {step}/{args.max_steps}")
    return delayed_to_codec_rows(rows)


def decode_audio(vocoder_sess: ort.InferenceSession, codec_rows: list[list[int]]) -> np.ndarray:
    t = len(codec_rows)
    flat = np.zeros((1, NUM_CODEBOOKS, t), dtype=np.int64)
    for frame_idx, row in enumerate(codec_rows):
        for codebook_idx, code in enumerate(row):
            flat[0, codebook_idx, frame_idx] = code
    outputs = vocoder_sess.run(None, {"audio_codes": flat})
    wav = outputs[0][0, 0]
    return np.asarray(wav, dtype=np.float32)


def main() -> None:
    args = parse_args()
    vocoder_onnx = VOCODER_FP32_ONNX if args.vocoder == "fp32" else VOCODER_INT4_ONNX
    ensure_files([PREFILL_ONNX, DECODE_ONNX, vocoder_onnx])

    print(f"[INFO] Loading tokenizer from {TOKENIZER_DIR}")
    tokenizer = load_tokenizer()
    input_ids = build_prompt_ids(tokenizer, args.text)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    print(f"[INFO] Prompt token count: {input_ids.shape[0]}")

    print(f"[INFO] Loading prefill session: {PREFILL_ONNX.name}")
    prefill_sess = make_session(PREFILL_ONNX, args.provider)
    print(f"[INFO] Loading decode session: {DECODE_ONNX.name}")
    decode_sess = make_session(DECODE_ONNX, args.provider)
    print(f"[INFO] Loading vocoder session: {vocoder_onnx.name}")
    vocoder_sess = make_session(vocoder_onnx, args.provider)

    t0 = time.perf_counter()
    codec_rows = synthesize_codes(prefill_sess, decode_sess, input_ids, attention_mask, args)
    if not codec_rows:
        raise RuntimeError("AR path produced no complete codec frames.")
    ar_dt = time.perf_counter() - t0
    print(f"[INFO] Generated {len(codec_rows)} codec frames in {ar_dt:.2f}s")

    v0 = time.perf_counter()
    wav = decode_audio(vocoder_sess, codec_rows)
    voc_dt = time.perf_counter() - v0
    out_path = Path(args.out)
    sf.write(out_path, wav, SAMPLE_RATE)
    audio_seconds = wav.shape[0] / SAMPLE_RATE
    rtf = (ar_dt + voc_dt) / audio_seconds if audio_seconds > 0 else math.inf
    print(f"[INFO] Saved audio to {out_path}")
    print(
        f"[INFO] audio={audio_seconds:.2f}s ar={ar_dt:.2f}s vocoder={voc_dt:.2f}s total_rtf={rtf:.3f}"
    )


if __name__ == "__main__":
    main()
