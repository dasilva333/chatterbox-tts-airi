from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT
VOICE_DIR = ROOT.parent / "voices"

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
AUDIO_PLACEHOLDER_ID = -100

# Fallback ids for the local tokenizer.json shipped with the current ONNX bundle.
# The runtime prefers tokenizer.get_added_vocab() so this remains checkpoint-safe.
FALLBACK_SPECIAL_IDS = {
    "<|tts|>": 151667,
    "<|audio|>": 151670,
    "<|text|>": 151672,
    "<|ref_audio|>": 151679,
    "<|ref_text|>": 151680,
}

# Names used by known / likely clone-capable exports.  The current Reza2kn ONNX
# bundle only has input_ids + attention_mask, so these are for future-compatible
# exports or local graph surgery outputs.
REFERENCE_CODE_INPUT_CANDIDATES = (
    "reference_codes_delayed",
    "ref_codes_delayed",
    "prompt_audio_codes_delayed",
    "reference_codes",
    "ref_codes",
    "prompt_audio_codes",
)
REFERENCE_LENGTH_INPUT_CANDIDATES = (
    "num_ref_tokens",
    "reference_num_tokens",
    "reference_codes_len",
    "reference_codes_length",
    "reference_lengths",
    "ref_codes_len",
)
INPUT_EMBEDS_CANDIDATES = ("input_embeds", "inputs_embeds")


class CloneUnsupportedError(RuntimeError):
    """Raised when the loaded ONNX graphs cannot consume reference voice data."""


@dataclass(frozen=True)
class HiggsSpecialTokens:
    tts_id: int
    text_id: int
    audio_id: int
    ref_audio_id: int
    ref_text_id: int | None

    @classmethod
    def from_tokenizer(cls, tokenizer: AutoTokenizer) -> "HiggsSpecialTokens":
        vocab = dict(getattr(tokenizer, "get_added_vocab", lambda: {})())

        def get_id(token: str, *, required: bool = True) -> int | None:
            value = vocab.get(token, FALLBACK_SPECIAL_IDS.get(token))
            if value is None and required:
                raise ValueError(f"Tokenizer is missing required Higgs special token {token!r}.")
            return int(value) if value is not None else None

        return cls(
            tts_id=int(get_id("<|tts|>")),
            text_id=int(get_id("<|text|>")),
            audio_id=int(get_id("<|audio|>")),
            ref_audio_id=int(get_id("<|ref_audio|>")),
            ref_text_id=get_id("<|ref_text|>", required=False),
        )


@dataclass(frozen=True)
class PromptBundle:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    delayed_reference_codes: np.ndarray | None
    raw_reference_codes: np.ndarray | None

    @property
    def has_reference(self) -> bool:
        return self.delayed_reference_codes is not None


@dataclass(frozen=True)
class GenerationResult:
    waveform: np.ndarray
    raw_output_codes: list[list[int]]
    metrics: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Higgs ONNX TTS runner with the SGLang voice-cloning prompt contract. "
            "It performs real cloning only when the AR prefill ONNX graph exposes "
            "a reference-code or input-embedding conditioning surface."
        )
    )
    parser.add_argument("--text", default="Have a nice day and enjoy south california sunshine.")
    parser.add_argument("--out", default=str(ROOT / "higgs_voice_clone.wav"))
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
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

    reference = parser.add_argument_group("voice cloning inputs")
    reference.add_argument(
        "--voice",
        default=None,
        help="Optional voice sample stem under ../voices, used as --reference-audio when set.",
    )
    reference.add_argument(
        "--reference-audio",
        default=None,
        help="Reference wav/flac/ogg/mp3/m4a path. Requires --codec-model-path to encode.",
    )
    reference.add_argument(
        "--reference-codes",
        default=None,
        help=(
            "Pre-encoded Higgs reference codes. Accepted formats: .json, .npy, .npz, .csv/.txt. "
            "Default layout is pre-delay [T, 8], matching SGLang's reference_codes API."
        ),
    )
    reference.add_argument(
        "--reference-codes-are-delayed",
        action="store_true",
        help="Treat --reference-codes as already delayed [T+7, 8] rows.",
    )
    reference.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of the reference audio. Strongly recommended for cloning quality.",
    )
    reference.add_argument(
        "--reference-text-file",
        default=None,
        help="Read reference transcript from a UTF-8 text file.",
    )
    reference.add_argument(
        "--i-have-rights-to-this-voice",
        action="store_true",
        help="Required when reference audio/codes are supplied.",
    )

    codec = parser.add_argument_group("optional reference-audio codec encode")
    codec.add_argument(
        "--codec-model-path",
        default=None,
        help=(
            "Path or HF repo id for a full Higgs TTS checkpoint containing bundled codec weights. "
            "Only needed when --reference-audio is used instead of --reference-codes."
        ),
    )
    codec.add_argument("--codec-device", default="cuda:0")
    codec.add_argument("--codec-dtype", default="float32")
    codec.add_argument(
        "--save-reference-codes",
        default=None,
        help="Optional .npy/.json path to save raw [T,8] reference codes after encoding audio.",
    )
    codec.add_argument(
        "--encode-reference-only",
        action="store_true",
        help="Encode and save --reference-audio codes, then exit before AR inference.",
    )

    debug = parser.add_argument_group("diagnostics and fallbacks")
    debug.add_argument(
        "--dry-run-contract",
        action="store_true",
        help="Build and print the Higgs cloning prompt contract without running ONNX inference.",
    )
    debug.add_argument(
        "--allow-zero-shot-fallback",
        action="store_true",
        help=(
            "If references are supplied but this ONNX export lacks a conditioning input, "
            "synthesize zero-shot instead of failing. Disabled by default to avoid fake cloning."
        ),
    )
    debug.add_argument(
        "--print-graph-inputs",
        action="store_true",
        help="Print ONNX graph input names before generation.",
    )
    return parser.parse_args()


def ensure_files(paths: Iterable[Path]) -> None:
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


def encode_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text or "", add_special_tokens=False))


def build_higgs_prompt_ids(
    tokenizer: AutoTokenizer,
    specials: HiggsSpecialTokens,
    text: str,
    *,
    delayed_reference_codes: np.ndarray | None = None,
    reference_text: str | None = None,
) -> np.ndarray:
    """Build the SGLang Higgs TTS prompt.

    Zero-shot:
        <|tts|> <|text|> tok(text) <|audio|>

    Clone with reference transcript:
        <|tts|> <|ref_text|> tok(reference_text)
        <|ref_audio|> [-100] * delayed_ref_len
        <|text|> tok(text) <|audio|>

    Clone without reference transcript:
        <|tts|> <|ref_audio|> [-100] * delayed_ref_len
        <|text|> tok(text) <|audio|>
    """
    ids: list[int] = [specials.tts_id]
    n_ref = int(delayed_reference_codes.shape[0]) if delayed_reference_codes is not None else 0

    if n_ref > 0:
        if reference_text and specials.ref_text_id is not None:
            ids.append(specials.ref_text_id)
            ids.extend(encode_text(tokenizer, reference_text))
        ids.append(specials.ref_audio_id)
        ids.extend([AUDIO_PLACEHOLDER_ID] * n_ref)

    ids.append(specials.text_id)
    ids.extend(encode_text(tokenizer, text.strip()))
    ids.append(specials.audio_id)
    return np.asarray(ids, dtype=np.int64)


def resolve_voice_path(stem: str) -> Path:
    for ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
        path = VOICE_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No voice sample found for {stem!r} under {VOICE_DIR}")


def read_reference_text(args: argparse.Namespace) -> str | None:
    if args.reference_text_file:
        return Path(args.reference_text_file).read_text(encoding="utf-8").strip()
    return args.reference_text


def _extract_codes_from_json(value: Any) -> Any:
    if isinstance(value, dict):
        for key in (
            "reference_codes",
            "codes",
            "audio_codes",
            "codes_TN",
            "reference_codes_TN",
        ):
            if key in value:
                return value[key]
        raise ValueError(
            "JSON reference-code dict must contain one of: "
            "reference_codes, codes, audio_codes, codes_TN, reference_codes_TN"
        )
    return value


def normalize_codes_shape(raw: Any, *, num_codebooks: int = NUM_CODEBOOKS) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("reference_codes is empty")

    # Common codec layout [1, N, T] or [N, T] -> [T, N].
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == num_codebooks:
            arr = arr[0].T
        elif arr.shape[0] == 1 and arr.shape[2] == num_codebooks:
            arr = arr[0]
        else:
            raise ValueError(
                f"Unsupported 3-D reference_codes shape {arr.shape}; expected [1,{num_codebooks},T] or [1,T,{num_codebooks}]."
            )
    elif arr.ndim == 2:
        if arr.shape[1] == num_codebooks:
            pass
        elif arr.shape[0] == num_codebooks:
            arr = arr.T
        else:
            raise ValueError(
                f"reference_codes must be [T,{num_codebooks}] or [{num_codebooks},T], got {arr.shape}."
            )
    else:
        raise ValueError(f"reference_codes must be 2-D or 3-D, got shape {arr.shape}.")

    return np.ascontiguousarray(arr, dtype=np.int64)


def load_reference_codes(path_like: str) -> np.ndarray:
    path = Path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _extract_codes_from_json(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        for key in (
            "reference_codes",
            "codes",
            "audio_codes",
            "codes_TN",
            "arr_0",
        ):
            if key in data:
                raw = data[key]
                break
        else:
            raise ValueError(f"No known reference-code array key found in {path}")
    elif suffix in {".csv", ".txt"}:
        try:
            raw = np.loadtxt(path, delimiter=",", dtype=np.int64)
        except ValueError:
            raw = np.loadtxt(path, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported reference code file type {suffix!r}: {path}")
    return normalize_codes_shape(raw)


def validate_code_range(codes: np.ndarray, *, delayed: bool) -> None:
    max_allowed = EOC_ID if delayed else CODEC_CODEBOOK_SIZE - 1
    lo = int(codes.min())
    hi = int(codes.max())
    if lo < 0 or hi > max_allowed:
        label = "delayed" if delayed else "raw pre-delay"
        raise ValueError(
            f"{label} reference codes must be in [0, {max_allowed}], got min={lo}, max={hi}."
        )


def apply_delay_pattern(codes_TN: np.ndarray) -> np.ndarray:
    codes_TN = normalize_codes_shape(codes_TN)
    validate_code_range(codes_TN, delayed=False)
    T, N = codes_TN.shape
    out = np.full((T + N - 1, N), EOC_ID, dtype=np.int64)
    for c in range(N):
        if c:
            out[:c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: np.ndarray) -> np.ndarray:
    delayed_LN = normalize_codes_shape(delayed_LN)
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(f"Delayed codes need L >= N, got {delayed_LN.shape}")
    out = np.empty((T, N), dtype=np.int64)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def save_codes(path_like: str, codes_TN: np.ndarray) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(codes_TN.tolist()), encoding="utf-8")
    else:
        np.save(path, codes_TN)


def encode_reference_audio_with_sglang_codec(args: argparse.Namespace, audio_path: Path) -> np.ndarray:
    if not args.codec_model_path:
        raise CloneUnsupportedError(
            "Raw reference audio needs the Higgs audio codec. Provide either:\n"
            "  1. --reference-codes with pre-encoded [T,8] Higgs codec codes, or\n"
            "  2. --reference-audio plus --codec-model-path pointing at a full Higgs TTS checkpoint "
            "that contains tied.embedding.modality_embeddings.0.model.* codec weights."
        )
    try:
        import torch
        from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
    except Exception as exc:  # pragma: no cover - optional external stack
        raise CloneUnsupportedError(
            "Could not import sglang_omni HiggsAudioCodec. Install/use sglang-omni, or pass "
            "pre-encoded --reference-codes instead."
        ) from exc

    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        # HiggsAudioCodec accepts channel-first 2-D, while soundfile returns [L, C].
        waveform = waveform.T
    dtype = getattr(torch, args.codec_dtype)
    codec = HiggsAudioCodec.from_pretrained(
        args.codec_model_path,
        device=args.codec_device,
        dtype=dtype,
    )
    codes = codec.encode_reference(waveform, sample_rate=int(sample_rate)).numpy()
    return normalize_codes_shape(codes)


def maybe_load_or_encode_reference_codes(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, Path | None]:
    reference_audio_path: Path | None = None
    if args.reference_audio:
        reference_audio_path = Path(args.reference_audio)
    elif args.voice:
        reference_audio_path = resolve_voice_path(args.voice)

    if args.reference_codes and reference_audio_path:
        raise ValueError("Use either --reference-codes or --reference-audio/--voice, not both.")

    if args.reference_codes:
        codes = load_reference_codes(args.reference_codes)
        if args.reference_codes_are_delayed:
            validate_code_range(codes, delayed=True)
            delayed = codes
            raw = reverse_delay_pattern(codes)
        else:
            validate_code_range(codes, delayed=False)
            raw = codes
            delayed = apply_delay_pattern(codes)
        return raw, delayed, None

    if reference_audio_path:
        raw = encode_reference_audio_with_sglang_codec(args, reference_audio_path)
        validate_code_range(raw, delayed=False)
        if args.save_reference_codes:
            save_codes(args.save_reference_codes, raw)
            print(f"[INFO] Saved raw reference codes [T,8] to {args.save_reference_codes}")
        delayed = apply_delay_pattern(raw)
        return raw, delayed, reference_audio_path

    return None, None, None


def get_session_input_names(session: ort.InferenceSession) -> list[str]:
    return [inp.name for inp in session.get_inputs()]


def get_input_rank(session: ort.InferenceSession, name: str) -> int | None:
    for inp in session.get_inputs():
        if inp.name == name:
            shape = getattr(inp, "shape", None)
            if shape is None:
                return None
            return len(shape)
    return None


def make_reference_feed_value(
    session: ort.InferenceSession,
    input_name: str,
    delayed_reference_codes: np.ndarray,
) -> np.ndarray:
    rank = get_input_rank(session, input_name)
    if rank == 3:
        return delayed_reference_codes.reshape(1, *delayed_reference_codes.shape).astype(np.int64)
    return delayed_reference_codes.astype(np.int64)


def describe_prefill_clone_support(prefill_sess: ort.InferenceSession) -> str:
    names = set(get_session_input_names(prefill_sess))
    code_inputs = [name for name in REFERENCE_CODE_INPUT_CANDIDATES if name in names]
    embed_inputs = [name for name in INPUT_EMBEDS_CANDIDATES if name in names]
    if code_inputs:
        return f"reference-code input(s): {code_inputs}"
    if embed_inputs:
        return f"input-embedding input(s): {embed_inputs}"
    return "none"


def build_prefill_feed(
    prefill_sess: ort.InferenceSession,
    prompt: PromptBundle,
) -> dict[str, np.ndarray]:
    names = set(get_session_input_names(prefill_sess))
    feed: dict[str, np.ndarray] = {
        "input_ids": prompt.input_ids.reshape(1, -1),
        "attention_mask": prompt.attention_mask.reshape(1, -1),
    }

    if not prompt.has_reference:
        return feed

    if "input_ids" not in names or "attention_mask" not in names:
        raise CloneUnsupportedError(
            f"Unexpected prefill graph inputs {sorted(names)}; expected at least input_ids and attention_mask."
        )

    for name in REFERENCE_CODE_INPUT_CANDIDATES:
        if name in names:
            feed[name] = make_reference_feed_value(prefill_sess, name, prompt.delayed_reference_codes)
            for length_name in REFERENCE_LENGTH_INPUT_CANDIDATES:
                if length_name in names:
                    feed[length_name] = np.asarray([prompt.delayed_reference_codes.shape[0]], dtype=np.int64)
            return feed

    if any(name in names for name in INPUT_EMBEDS_CANDIDATES):
        raise CloneUnsupportedError(
            "This ONNX prefill graph exposes input_embeds/inputs_embeds, but this runner cannot yet "
            "materialize Higgs text+audio embeddings from the quantized ONNX weights. Use/export a graph "
            "that accepts reference_codes_delayed, or add an embedding extractor sidecar."
        )

    raise CloneUnsupportedError(
        "The loaded AR prefill ONNX graph has no reference conditioning input.\n"
        f"Prefill inputs: {sorted(names)}\n\n"
        "SGLang voice cloning is not a plain token-only prompt. Its prompt contains -100 "
        "reference-audio placeholders, and the runtime overlays fused multi-codebook audio "
        "embeddings at those positions during prefill. This ONNX export only accepts input_ids "
        "and attention_mask, so feeding the -100 prompt would either fail or become fake cloning.\n\n"
        "Needed to run real local cloning with this runner:\n"
        "  - a clone-capable prefill ONNX export with reference_codes_delayed/ref_codes_delayed input, or\n"
        "  - a prefill export that accepts fully prepared input_embeds plus a local embedding extractor."
    )


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


def apply_generation_delay_mask(codes: list[int], step: int) -> list[int]:
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
    prompt: PromptBundle,
    args: argparse.Namespace,
) -> list[list[int]]:
    rng = np.random.default_rng(args.seed)

    prefill_feed = build_prefill_feed(prefill_sess, prompt)
    outputs = prefill_sess.run(None, prefill_feed)
    outputs_dict = dict(zip((o.name for o in prefill_sess.get_outputs()), outputs))

    delay_count = 0
    eoc_countdown: int | None = None
    done = False

    codes = sample_codebooks(outputs_dict["logits"], args.temperature, args.top_p, args.top_k, rng)
    codes = apply_generation_delay_mask(codes, delay_count)
    delay_count += 1
    rows = [codes[:]]
    past = extract_past(outputs_dict)

    for step in range(1, args.max_steps):
        if done:
            break
        feeds = {
            "codes": to_codes_tensor(codes),
            "position_ids": np.asarray([[prompt.input_ids.shape[0] + step - 1]], dtype=np.int64),
            **past,
        }
        outputs = decode_sess.run(None, feeds)
        outputs_dict = dict(zip((o.name for o in decode_sess.get_outputs()), outputs))
        codes = sample_codebooks(outputs_dict["logits"], args.temperature, args.top_p, args.top_k, rng)
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


def build_prompt_bundle(
    tokenizer: AutoTokenizer,
    specials: HiggsSpecialTokens,
    args: argparse.Namespace,
    *,
    raw_reference_codes: np.ndarray | None,
    delayed_reference_codes: np.ndarray | None,
    force_zero_shot: bool = False,
) -> PromptBundle:
    ref_codes = None if force_zero_shot else delayed_reference_codes
    ref_text = None if force_zero_shot else read_reference_text(args)
    input_ids = build_higgs_prompt_ids(
        tokenizer,
        specials,
        args.text,
        delayed_reference_codes=ref_codes,
        reference_text=ref_text,
    )
    return PromptBundle(
        input_ids=input_ids,
        attention_mask=np.ones_like(input_ids, dtype=np.int64),
        delayed_reference_codes=ref_codes,
        raw_reference_codes=None if force_zero_shot else raw_reference_codes,
    )


def print_contract(prompt: PromptBundle, args: argparse.Namespace) -> None:
    ref_raw = prompt.raw_reference_codes
    ref_delayed = prompt.delayed_reference_codes
    placeholder_count = int((prompt.input_ids == AUDIO_PLACEHOLDER_ID).sum())
    print("[CONTRACT] Higgs prompt/token contract")
    print(f"  prompt_tokens: {prompt.input_ids.shape[0]}")
    print(f"  placeholders(-100): {placeholder_count}")
    if ref_raw is not None and ref_delayed is not None:
        print(f"  raw_reference_codes: {tuple(ref_raw.shape)} [T, 8]")
        print(f"  delayed_reference_codes: {tuple(ref_delayed.shape)} [T+7, 8]")
        print(f"  reference_text_present: {bool(read_reference_text(args))}")
    else:
        print("  reference: none / zero-shot")
    preview = prompt.input_ids[: min(prompt.input_ids.shape[0], 32)].tolist()
    print(f"  input_ids_preview: {preview}{' ...' if prompt.input_ids.shape[0] > 32 else ''}")


def main() -> None:
    args = parse_args()

    has_reference = bool(args.reference_codes or args.reference_audio or args.voice)
    if has_reference and not args.i_have_rights_to_this_voice:
        raise SystemExit(
            "Refusing to clone a voice without --i-have-rights-to-this-voice. "
            "Only use reference voices you own or have permission to synthesize."
        )

    vocoder_onnx = VOCODER_FP32_ONNX if args.vocoder == "fp32" else VOCODER_INT4_ONNX
    ensure_files([PREFILL_ONNX, DECODE_ONNX, vocoder_onnx])

    raw_reference_codes, delayed_reference_codes, reference_audio_path = maybe_load_or_encode_reference_codes(args)
    if args.encode_reference_only:
        if raw_reference_codes is None:
            raise SystemExit("--encode-reference-only requires --reference-audio or --voice.")
        if not args.save_reference_codes:
            raise SystemExit("--encode-reference-only also requires --save-reference-codes.")
        return

    print(f"[INFO] Loading tokenizer from {TOKENIZER_DIR}")
    tokenizer = load_tokenizer()
    specials = HiggsSpecialTokens.from_tokenizer(tokenizer)
    prompt = build_prompt_bundle(
        tokenizer,
        specials,
        args,
        raw_reference_codes=raw_reference_codes,
        delayed_reference_codes=delayed_reference_codes,
    )
    print_contract(prompt, args)

    if args.dry_run_contract:
        return

    print(f"[INFO] Loading prefill session: {PREFILL_ONNX.name}")
    prefill_sess = make_session(PREFILL_ONNX, args.provider)
    print(f"[INFO] Loading decode session: {DECODE_ONNX.name}")
    decode_sess = make_session(DECODE_ONNX, args.provider)
    print(f"[INFO] Loading vocoder session: {vocoder_onnx.name}")
    vocoder_sess = make_session(vocoder_onnx, args.provider)

    if args.print_graph_inputs:
        print(f"[INFO] Prefill inputs: {get_session_input_names(prefill_sess)}")
        print(f"[INFO] Decode inputs: {get_session_input_names(decode_sess)[:6]} ...")
        print(f"[INFO] Vocoder inputs: {get_session_input_names(vocoder_sess)}")
    print(f"[INFO] Prefill clone support: {describe_prefill_clone_support(prefill_sess)}")

    if prompt.has_reference and describe_prefill_clone_support(prefill_sess) == "none":
        if not args.allow_zero_shot_fallback:
            # Trigger the detailed error from build_prefill_feed without starting inference.
            build_prefill_feed(prefill_sess, prompt)
        print("[WARN] Falling back to zero-shot synthesis because --allow-zero-shot-fallback was set.")
        prompt = build_prompt_bundle(
            tokenizer,
            specials,
            args,
            raw_reference_codes=None,
            delayed_reference_codes=None,
            force_zero_shot=True,
        )
        print_contract(prompt, args)

    t0 = time.perf_counter()
    codec_rows = synthesize_codes(prefill_sess, decode_sess, prompt, args)
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
        f"[INFO] mode={'voice-clone' if prompt.has_reference else 'zero-shot'} "
        f"audio={audio_seconds:.2f}s ar={ar_dt:.2f}s vocoder={voc_dt:.2f}s total_rtf={rtf:.3f}"
    )
    if reference_audio_path is not None:
        print(f"[INFO] Reference audio: {reference_audio_path}")


if __name__ == "__main__":
    try:
        main()
    except CloneUnsupportedError as exc:
        print(f"[UNSUPPORTED] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
