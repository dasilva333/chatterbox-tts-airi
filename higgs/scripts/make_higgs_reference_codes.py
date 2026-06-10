from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
TOKENIZER_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
REFS_DIR = PROJECT_ROOT / "refs"

PREFILL_ONNX = MODELS_DIR / "higgs_audio_v3_ar_prefill_matmul4.onnx"
DECODE_ONNX = MODELS_DIR / "higgs_audio_v3_ar_decode_matmul4.onnx"

NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 1026
CODEC_CODEBOOK_SIZE = 1024
BOC_ID = 1024
EOC_ID = 1025
TTS_ID = 151667
TEXT_ID = 151672
AUDIO_ID = 151670


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a Higgs reference-code fixture from a zero-shot prompt")
    p.add_argument("--text", default="hello world")
    p.add_argument("--out", default=str(REFS_DIR / "synthetic_reference_codes.json"))
    p.add_argument("--provider", default="CUDAExecutionProvider")
    p.add_argument("--max-steps", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
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
        kept_scores = (scores[order] - scores[order].max()) / temperature
        probs = np.exp(kept_scores)
        probs /= probs.sum()
        if 0.0 < top_p < 1.0:
            cdf = np.cumsum(probs)
            cutoff = int(np.searchsorted(cdf, top_p, side="left")) + 1
            order = order[:cutoff]
            probs = probs[:cutoff]
            probs /= probs.sum()
        out.append(int(rng.choice(order, p=probs)))
    return out


def apply_delay_mask(codes: list[int], step: int) -> list[int]:
    if step < NUM_CODEBOOKS:
        for cb in range(step + 1, NUM_CODEBOOKS):
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


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()
    input_ids = build_prompt_ids(tokenizer, args.text)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    rng = np.random.default_rng(args.seed)

    prefill = make_session(PREFILL_ONNX, args.provider)
    decode = make_session(DECODE_ONNX, args.provider)

    outputs = prefill.run(None, {"input_ids": input_ids.reshape(1, -1), "attention_mask": attention_mask.reshape(1, -1)})
    out_map = dict(zip([o.name for o in prefill.get_outputs()], outputs))

    delay_count = 0
    eoc_countdown: int | None = None
    done = False
    codes = sample_codebooks(out_map["logits"], args.temperature, args.top_p, args.top_k, rng)
    codes = apply_delay_mask(codes, delay_count)
    delay_count += 1
    rows = [codes[:]]
    past = extract_past(out_map)

    for step in range(1, args.max_steps):
        if done:
            break
        outputs = decode.run(
            None,
            {
                "codes": to_codes_tensor(codes),
                "position_ids": np.asarray([[input_ids.shape[0] + step - 1]], dtype=np.int64),
                **past,
            },
        )
        out_map = dict(zip([o.name for o in decode.get_outputs()], outputs))
        codes = sample_codebooks(out_map["logits"], args.temperature, args.top_p, args.top_k, rng)
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
        past = extract_past(out_map)

    codec_rows = delayed_to_codec_rows(rows)
    if not codec_rows:
        raise RuntimeError("No codec rows produced")

    fixture = {
        "reference_codes": codec_rows,
        "source_text": args.text,
        "frames": len(codec_rows),
        "num_codebooks": NUM_CODEBOOKS,
        "sample_rate": 24000,
        "note": "Generated locally from the working zero-shot Higgs ONNX path; suitable as a reference-code fixture for the clone-capable runner.",
    }
    Path(args.out).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"frames={len(codec_rows)} audio_seconds={len(codec_rows)/25:.2f}")


if __name__ == "__main__":
    main()
