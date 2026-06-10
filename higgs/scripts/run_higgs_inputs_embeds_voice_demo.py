from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import helper, shape_inference
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT

PREFILL_ONNX = ROOT / "higgs_audio_v3_ar_prefill_matmul4.onnx"
PATCHED_PREFILL = ROOT / "higgs_audio_v3_ar_prefill_inputs_embeds.onnx"
EXPOSED_PREFILL = ROOT / "higgs_audio_v3_ar_prefill_with_embedding_output.onnx"
DECODE_ONNX = ROOT / "higgs_audio_v3_ar_decode_matmul4.onnx"
VOCODER_ONNX = ROOT / "higgs_audio_v3_vocoder_decode.onnx"

SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8
VOCAB_SIZE = 1026
BOC_ID = 1024
EOC_ID = 1025
AUDIO_PLACEHOLDER_ID = -100

TEXT_SPECIALS = {
    "<|tts|>": 151667,
    "<|audio|>": 151670,
    "<|text|>": 151672,
    "<|ref_audio|>": 151679,
    "<|ref_text|>": 151680,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Higgs ONNX voice demo via patched inputs_embeds prefill")
    p.add_argument("--reference-codes", default=str(ROOT / "synthetic_reference_codes.json"))
    p.add_argument("--reference-text", default=None)
    p.add_argument("--text", default="Hello world, this is a local Higgs voice demo.")
    p.add_argument("--out", default=str(ROOT / "higgs_voice_demo.wav"))
    p.add_argument("--max-steps", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--provider", default="CUDAExecutionProvider")
    return p.parse_args()


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def get_special_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
    vocab = dict(tokenizer.get_added_vocab())
    out = dict(TEXT_SPECIALS)
    out["tts"] = int(vocab.get("<|tts|>", TEXT_SPECIALS["<|tts|>"]))
    out["audio"] = int(vocab.get("<|audio|>", TEXT_SPECIALS["<|audio|>"]))
    out["text"] = int(vocab.get("<|text|>", TEXT_SPECIALS["<|text|>"]))
    out["ref_audio"] = int(vocab.get("<|ref_audio|>", TEXT_SPECIALS["<|ref_audio|>"]))
    out["ref_text"] = int(vocab.get("<|ref_text|>", TEXT_SPECIALS["<|ref_text|>"]))
    return out


def apply_delay_pattern(codes_tn: np.ndarray) -> np.ndarray:
    if codes_tn.ndim != 2:
        raise ValueError(f"expected [T, N], got {codes_tn.shape}")
    t, n = codes_tn.shape
    out = np.full((t + n - 1, n), EOC_ID, dtype=np.int64)
    for c in range(n):
        out[:c, c] = BOC_ID
        out[c : c + t, c] = codes_tn[:, c]
    return out


def load_reference_codes(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".json":
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded = loaded.get("reference_codes", loaded.get("codes", loaded))
        arr = np.asarray(loaded, dtype=np.int64)
    elif p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        raise ValueError(f"Unsupported reference-codes file: {path}")
    if arr.ndim != 2 or arr.shape[1] != NUM_CODEBOOKS:
        raise ValueError(f"Expected [T, {NUM_CODEBOOKS}] reference codes, got {arr.shape}")
    return arr


def build_prompt_ids(tokenizer: AutoTokenizer, specials: dict[str, int], prompt_text: str, num_ref_tokens: int, reference_text: str | None) -> np.ndarray:
    ids = [specials["tts"]]
    if reference_text and num_ref_tokens > 0:
        ids.append(specials["ref_text"])
        ids.extend(tokenizer.encode(reference_text, add_special_tokens=False))
    if num_ref_tokens > 0:
        ids.append(specials["ref_audio"])
        ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
    ids.append(specials["text"])
    ids.extend(tokenizer.encode(prompt_text, add_special_tokens=False))
    ids.append(specials["audio"])
    return np.asarray(ids, dtype=np.int64)


def patch_prefill_to_expose_embedding(src: Path, dst: Path) -> Path:
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph
    if any(o.name == "embedding" for o in graph.output):
        onnx.save(model, str(dst))
        return dst
    embedding_vi = next((vi for vi in graph.value_info if vi.name == "embedding"), None)
    if embedding_vi is None:
        raise RuntimeError("Could not find embedding value_info in prefill graph.")
    dims = []
    for dim in embedding_vi.type.tensor_type.shape.dim:
        dims.append(dim.dim_param if dim.dim_param else (dim.dim_value if dim.dim_value != 0 else None))
    graph.output.append(helper.make_tensor_value_info("embedding", embedding_vi.type.tensor_type.elem_type, dims))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, str(dst))
    return dst


def extract_initializer(model_path: Path, name: str) -> np.ndarray:
    model = onnx.load(str(model_path), load_external_data=True)
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(f"initializer not found: {name}")


def fused_reference_embeddings(delayed_codes: np.ndarray, fused_weight: np.ndarray) -> np.ndarray:
    offsets = (np.arange(NUM_CODEBOOKS, dtype=np.int64) * VOCAB_SIZE)[None, :]
    fused_ids = delayed_codes.astype(np.int64) + offsets
    return fused_weight[fused_ids].sum(axis=-2)


def sample_codebooks(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    for cb in range(NUM_CODEBOOKS):
        scores = logits[0, cb].astype(np.float64)
        best = int(scores.argmax())
        if temperature <= 1e-5 or top_k == 1:
            out.append(best)
            continue
        scores = scores / max(temperature, 1e-5)
        order = np.argsort(scores)[::-1]
        if 0 < top_k < len(order):
            order = order[:top_k]
        probs = np.exp(scores[order] - scores[order].max())
        probs /= probs.sum()
        if 0 < top_p < 1.0:
            cdf = np.cumsum(probs)
            keep = cdf <= top_p
            if not np.any(keep):
                keep[0] = True
            order = order[keep]
            probs = probs[: len(order)]
            probs /= probs.sum()
        out.append(int(rng.choice(order, p=probs)))
    return out


def apply_generation_delay_mask(codes: list[int], delay_count: int) -> list[int]:
    masked = codes[:]
    for i in range(min(delay_count, NUM_CODEBOOKS)):
        masked[i] = BOC_ID
    return masked


def to_codes_tensor(codes: list[int]) -> np.ndarray:
    return np.asarray(codes, dtype=np.int64).reshape(1, NUM_CODEBOOKS)


def delayed_to_codec_rows(rows: list[list[int]]) -> list[list[int]]:
    codec: list[list[int]] = []
    total = len(rows) - (NUM_CODEBOOKS - 1)
    for frame in range(total):
        row: list[int] = []
        valid = True
        for cb in range(NUM_CODEBOOKS):
            code = int(rows[frame + cb][cb])
            if code < 0 or code >= VOCAB_SIZE - 2:
                valid = False
                break
            row.append(code)
        if not valid:
            continue
        codec.append(row)
    return codec


def extract_past(outputs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    past: dict[str, np.ndarray] = {}
    for name, value in outputs_dict.items():
        if name.startswith("past_"):
            past[name] = value
        elif name.startswith("present_"):
            past["past_" + name[len("present_"):]] = value
    return past


def build_inputs_embeds(
    exposed_prefill: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    prompt_ids: np.ndarray,
    placeholder_mask: np.ndarray,
    delayed_reference_codes: np.ndarray,
    fused_weight: np.ndarray,
) -> np.ndarray:
    safe_input_ids = np.where(prompt_ids == AUDIO_PLACEHOLDER_ID, TEXT_SPECIALS["<|audio|>"], prompt_ids).astype(np.int64)
    outputs = exposed_prefill.run(None, {"input_ids": safe_input_ids.reshape(1, -1), "attention_mask": np.ones_like(safe_input_ids, dtype=np.int64).reshape(1, -1)})
    out_names = [o.name for o in exposed_prefill.get_outputs()]
    out_map = dict(zip(out_names, outputs))
    embeds = out_map["embedding"].astype(np.float16, copy=True)
    ref_embeds = fused_reference_embeddings(delayed_reference_codes, fused_weight).astype(np.float16)
    idx = np.flatnonzero(placeholder_mask)
    if len(idx) != len(ref_embeds):
        raise ValueError(f"placeholder count {len(idx)} != delayed reference rows {len(ref_embeds)}")
    embeds[0, idx, :] = ref_embeds
    return embeds


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()
    specials = get_special_ids(tokenizer)

    raw_reference_codes = load_reference_codes(args.reference_codes)
    delayed_reference_codes = apply_delay_pattern(raw_reference_codes)
    reference_text = args.reference_text

    prompt_ids = build_prompt_ids(tokenizer, specials, args.text, len(delayed_reference_codes), reference_text)
    placeholder_mask = prompt_ids == AUDIO_PLACEHOLDER_ID

    ensure = [PREFILL_ONNX, DECODE_ONNX, VOCODER_ONNX]
    missing = [str(p) for p in ensure if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    patch_prefill_to_expose_embedding(PREFILL_ONNX, EXPOSED_PREFILL)
    exposed_prefill = make_session(EXPOSED_PREFILL, args.provider)
    prefill_sess = make_session(PATCHED_PREFILL, args.provider)
    decode_sess = make_session(DECODE_ONNX, args.provider)
    vocoder_sess = make_session(VOCODER_ONNX, args.provider)

    fused_weight = extract_initializer(PREFILL_ONNX, "model.modality_embedding.weight")
    inputs_embeds = build_inputs_embeds(
        exposed_prefill,
        tokenizer,
        prompt_ids,
        placeholder_mask,
        delayed_reference_codes,
        fused_weight,
    )

    prefill_feed = {
        "input_ids": np.where(prompt_ids == AUDIO_PLACEHOLDER_ID, specials["audio"], prompt_ids).astype(np.int64).reshape(1, -1),
        "attention_mask": np.ones_like(prompt_ids, dtype=np.int64).reshape(1, -1),
        "inputs_embeds": inputs_embeds,
    }

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    outputs = prefill_sess.run(None, prefill_feed)
    out_map = dict(zip((o.name for o in prefill_sess.get_outputs()), outputs))
    delay_count = 0
    eoc_countdown: int | None = None
    done = False

    codes = sample_codebooks(out_map["logits"], args.temperature, args.top_p, args.top_k, rng)
    codes = apply_generation_delay_mask(codes, delay_count)
    delay_count += 1
    rows = [codes[:]]
    past = extract_past(out_map)

    for step in range(1, args.max_steps):
        if done:
            break
        feeds = {
            "codes": to_codes_tensor(codes),
            "position_ids": np.asarray([[prompt_ids.shape[0] + step - 1]], dtype=np.int64),
            **past,
        }
        outputs = decode_sess.run(None, feeds)
        out_map = dict(zip((o.name for o in decode_sess.get_outputs()), outputs))
        codes = sample_codebooks(out_map["logits"], args.temperature, args.top_p, args.top_k, rng)
        if delay_count < NUM_CODEBOOKS:
            codes = apply_generation_delay_mask(codes, delay_count)
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
    flat = np.zeros((1, NUM_CODEBOOKS, len(codec_rows)), dtype=np.int64)
    for i, row in enumerate(codec_rows):
        for j, code in enumerate(row):
            flat[0, j, i] = code
    wav = vocoder_sess.run(None, {"audio_codes": flat})[0][0, 0].astype(np.float32)
    ar_dt = time.perf_counter() - t0
    audio_seconds = wav.shape[0] / SAMPLE_RATE
    out_path = Path(args.out)
    sf.write(out_path, wav, SAMPLE_RATE)
    print(f"[INFO] saved {out_path}")
    print(f"[INFO] raw_reference_codes={raw_reference_codes.shape} delayed={delayed_reference_codes.shape}")
    print(f"[INFO] prompt_tokens={prompt_ids.shape[0]} placeholders={int(placeholder_mask.sum())}")
    print(f"[INFO] audio={audio_seconds:.2f}s total={ar_dt:.2f}s rtf={(ar_dt/audio_seconds if audio_seconds else math.inf):.3f}")


if __name__ == "__main__":
    main()
