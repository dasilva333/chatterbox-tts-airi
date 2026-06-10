#!/usr/bin/env python3
"""Higgs ONNX voice-cloning integration patch (self-extracting).

Writes three files into the higgs working folder:

  encode_reference_audio.py            .wav/.mp3/.flac -> [T,8] codec codes
  higgs_clone_onnx.py                  unified zero-shot / cloning runner
  integration-notes-voice-cloning.md   docs, deps, run order, known risks

Prerequisite: the inputs_embeds surgery artifacts (embedder.onnx + patched
prefill) from higgs_onnx_surgery_patch.zip, or the earlier inputs_embeds
variant — the runner auto-detects either.

Usage, from the folder containing higgs_audio_v3_ar_prefill_matmul4.onnx:

  python higgs_voice_cloning_onnx_patch.py --dry-run
  python higgs_voice_cloning_onnx_patch.py
  python higgs_voice_cloning_onnx_patch.py --force     # overwrite existing
"""
from __future__ import annotations

import argparse
from pathlib import Path

FILES: dict[str, str] = {}

# --------------------------------------------------------------------------
FILES["encode_reference_audio.py"] = r'''
from __future__ import annotations

"""Encode a reference .wav/.mp3/.flac into Higgs [T, 8] codec codes.

Uses the vendored HiggsAudioV2Tokenizer (PyTorch) with weights from the public
codec repo. Output .json/.npy is directly consumable by higgs_clone_onnx.py
(--reference-codes) and run_higgs_inputs_embeds_voice_demo.py.

Deps beyond the ONNX stack: torch, torchaudio, safetensors, huggingface_hub
(CPU torch is fine; a few seconds per clip). For mp3, soundfile>=0.12 or
torchaudio with ffmpeg available.

Usage:
  python encode_reference_audio.py --audio voices/me.wav --out my_codes.json
  python encode_reference_audio.py --audio ref.mp3 --codec bosonai/higgs-audio-v2-tokenizer
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VENDORED = ROOT / "sglang_reference" / "sglang_omni" / "models" / "higgs_tts" / "_vendored"
SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (mono float32 [-1,1] [L], sample_rate). Tries soundfile, torchaudio, librosa."""
    err: list[str] = []
    try:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        return wav[:, 0], int(sr)  # first channel, matching SGLang's _to_mono_3d
    except Exception as e:  # noqa: BLE001
        err.append(f"soundfile: {e}")
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        return wav[0].numpy().astype(np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        err.append(f"torchaudio: {e}")
    try:
        import librosa

        wav, sr = librosa.load(path, sr=None, mono=True)
        return wav.astype(np.float32), int(sr)
    except Exception as e:  # noqa: BLE001
        err.append(f"librosa: {e}")
    raise RuntimeError("Could not decode audio file:\n  " + "\n  ".join(err))


def load_codec(codec: str, device: str = "cpu"):
    """Instantiate the vendored codec model and load weights from a local dir or HF repo."""
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(VENDORED))
    from higgs_audio_v2_tokenizer_hf import (  # type: ignore
        HiggsAudioV2TokenizerConfig,
        HiggsAudioV2TokenizerModel,
    )

    cfg_dict = json.loads((VENDORED / "higgs_audio_v2_tokenizer_config.json").read_text())
    for k in ("architectures", "torch_dtype", "transformers_version"):
        cfg_dict.pop(k, None)
    model = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**cfg_dict)).float().eval()

    ckpt_dir = Path(codec)
    if not ckpt_dir.is_dir():
        from huggingface_hub import snapshot_download

        ckpt_dir = Path(snapshot_download(codec, allow_patterns=["*.safetensors", "*.json", "*.bin"]))
    st_files = sorted(ckpt_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"no .safetensors under {ckpt_dir}")

    raw: dict = {}
    for f in st_files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                raw[k] = h.get_tensor(k)

    # try direct keys, then common prefix-strips (TTS-bundled layout, 'model.' wrapper)
    own = set(model.state_dict().keys())
    candidates = [raw]
    for prefix in ("tied.embedding.modality_embeddings.0.model.", "model."):
        stripped = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        if stripped:
            candidates.append(stripped)
    state = max(candidates, key=lambda s: len(own & set(s.keys())))
    matched = len(own & set(state.keys()))
    if matched < len(own) // 2:
        raise RuntimeError(
            f"checkpoint/model key mismatch: only {matched}/{len(own)} keys align; "
            f"sample ckpt keys: {list(raw)[:5]}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[codec] loaded {matched}/{len(own)} keys (missing={len(missing)}, unexpected={len(unexpected)};"
          " weight-norm keys are regenerated, sparse missing is expected)")
    return model.to(device)


def encode_file(audio_path: str, codec: str = "bosonai/higgs-audio-v2-tokenizer",
                device: str = "cpu") -> np.ndarray:
    """wav/mp3/flac file -> [T, 8] int64 codes (25 fps)."""
    import torch
    import torch.nn.functional as F

    wav_np, sr = load_audio(audio_path)
    wav = torch.from_numpy(wav_np).view(1, 1, -1)
    if sr != SAMPLE_RATE:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[-1] < SAMPLE_RATE:  # encoder errors on clips < 1 s
        wav = F.pad(wav, (0, SAMPLE_RATE - wav.shape[-1]))

    model = load_codec(codec, device)
    with torch.no_grad():
        codes_BNT = model.encode(wav.to(device)).audio_codes  # [1, 8, T]
    codes = codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu().numpy()
    assert codes.ndim == 2 and codes.shape[1] == NUM_CODEBOOKS, codes.shape
    return codes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--audio", required=True, help=".wav/.mp3/.flac reference clip (you must have rights to this voice)")
    p.add_argument("--out", default=None, help="output .json or .npy (default: <audio>.codes.json)")
    p.add_argument("--codec", default="bosonai/higgs-audio-v2-tokenizer",
                   help="codec weights: HF repo id or local dir (also accepts a full Higgs TTS checkpoint dir)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference-text", default=None, help="transcript, stored in the fixture for convenience")
    args = p.parse_args()

    codes = encode_file(args.audio, args.codec, args.device)
    out = Path(args.out) if args.out else Path(args.audio).with_suffix(".codes.json")
    if out.suffix.lower() == ".npy":
        np.save(out, codes)
    else:
        out.write_text(json.dumps({
            "reference_codes": codes.tolist(),
            "reference_text": args.reference_text,
            "source_audio": str(args.audio),
            "frames": int(codes.shape[0]),
            "num_codebooks": NUM_CODEBOOKS,
            "sample_rate": SAMPLE_RATE,
        }, indent=2), encoding="utf-8")
    print(f"wrote {out}  frames={codes.shape[0]}  audio_seconds={codes.shape[0]/25:.2f}")


if __name__ == "__main__":
    main()
'''

# --------------------------------------------------------------------------
FILES["higgs_clone_onnx.py"] = r'''
from __future__ import annotations

"""Higgs ONNX voice cloning — integrated runner.

Merges the working inputs_embeds demo loop with the surgery-patch artifacts:
  - text embeddings via the 624-byte embedder.onnx (bit-exact, no extra full
    prefill pass) when present; falls back to the exposed-embedding trick
  - fused audio table from modality_embedding_fp16.npy when present; else a
    fast byte-offset extraction from the prefill proto (no 2 GB graph load),
    cached to .npy for next time
  - patched prefill: auto-supports both surgery variants by inspecting graph
    inputs ('embedding' = surgery-patch zip; 'inputs_embeds' = earlier variant)
  - real reference audio: --reference-audio ref.wav|.mp3 encodes via
    encode_reference_audio.py (vendored Higgs codec, needs torch+torchaudio)

Examples:
  # clone from a wav, transcript recommended for quality
  python higgs_clone_onnx.py --reference-audio voices/me.wav \
      --reference-text "exact transcript of the clip" \
      --text "Have a nice day and enjoy south california sunshine." \
      --i-have-rights-to-this-voice --out clone.wav

  # reuse pre-encoded codes (skips torch entirely)
  python higgs_clone_onnx.py --reference-codes me.codes.json --text "..." \
      --i-have-rights-to-this-voice --out clone.wav

  # zero-shot (no reference)
  python higgs_clone_onnx.py --text "hello world" --out zs.wav
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 8
VOCAB_SIZE = 1026
BOC_ID = 1024
EOC_ID = 1025
AUDIO_PLACEHOLDER_ID = -100
EMB_DIM = 2560

TEXT_SPECIALS = {
    "tts": 151667,
    "audio": 151670,
    "text": 151672,
    "ref_audio": 151679,
    "ref_text": 151680,
}


# ---------------------------------------------------------------- file layout
def find(name: str) -> Path | None:
    for cand in (ROOT / name, ROOT / "ar" / name):
        if cand.exists():
            return cand
    return None


def require(name: str) -> Path:
    p = find(name)
    if p is None:
        raise FileNotFoundError(f"{name} not found in {ROOT} or {ROOT/'ar'}")
    return p


# ---------------------------------------------------------- model components
def make_session(path: Path, provider: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def get_patched_prefill(provider: str) -> ort.InferenceSession:
    for name in ("higgs_audio_v3_ar_prefill_matmul4_embeds.onnx",
                 "higgs_audio_v3_ar_prefill_inputs_embeds.onnx"):
        p = find(name)
        if p is not None:
            return make_session(p, provider)
    raise FileNotFoundError(
        "no patched prefill found — run surgery_inputs_embeds.py (surgery patch zip) first")


def get_text_embedder(provider: str):
    """Returns fn(input_ids [1,n]) -> fp16 [1,n,2560]."""
    emb = find("higgs_audio_v3_embedder.onnx")
    if emb is not None:
        sess = make_session(emb, "CPUExecutionProvider")  # tiny gather, CPU is fine
        return lambda ids: sess.run(["embedding"], {"input_ids": ids})[0]

    # fallback: expose `embedding` as an output of the original prefill (slow path)
    import onnx
    from onnx import helper

    src = require("higgs_audio_v3_ar_prefill_matmul4.onnx")
    exposed = src.with_name("higgs_audio_v3_ar_prefill_with_embedding_output.onnx")
    if not exposed.exists():
        model = onnx.load(str(src), load_external_data=True)
        vi = next(v for v in model.graph.value_info if v.name == "embedding")
        model.graph.output.append(helper.make_tensor_value_info(
            "embedding", vi.type.tensor_type.elem_type,
            [d.dim_param or d.dim_value or None for d in vi.type.tensor_type.shape.dim]))
        onnx.save(model, str(exposed))
    sess = make_session(exposed, provider)
    names = [o.name for o in sess.get_outputs()]

    def run(ids: np.ndarray) -> np.ndarray:
        outs = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        return dict(zip(names, outs))["embedding"]
    return run


def get_fused_table() -> np.ndarray:
    """modality_embedding [8208, 2560] fp16 — npy cache, else byte-offset extraction."""
    npy = find("modality_embedding_fp16.npy")
    if npy is not None:
        return np.load(npy)

    import onnx

    src = require("higgs_audio_v3_ar_prefill_matmul4.onnx")
    m = onnx.load(str(src), load_external_data=False)
    init = next(i for i in m.graph.initializer if i.name == "model.modality_embedding.weight")
    kv = {e.key: e.value for e in init.external_data}
    if "location" in kv:
        with open(src.parent / kv["location"], "rb") as f:
            f.seek(int(kv["offset"]))
            tbl = np.frombuffer(f.read(int(kv["length"])), dtype=np.float16).reshape(8208, EMB_DIM)
    else:  # weights embedded in the proto
        tbl = onnx.numpy_helper.to_array(init).astype(np.float16)
    out = src.parent / "modality_embedding_fp16.npy"
    np.save(out, tbl)
    print(f"[init] cached fused table -> {out}")
    return tbl


# ------------------------------------------------------------- cloning maths
def apply_delay_pattern(codes_tn: np.ndarray) -> np.ndarray:
    t, n = codes_tn.shape
    out = np.full((t + n - 1, n), EOC_ID, dtype=np.int64)
    for c in range(n):
        out[:c, c] = BOC_ID
        out[c:c + t, c] = codes_tn[:, c]
    return out


def fused_reference_embeddings(delayed: np.ndarray, table: np.ndarray) -> np.ndarray:
    offsets = (np.arange(NUM_CODEBOOKS, dtype=np.int64) * VOCAB_SIZE)[None, :]
    return table[delayed.astype(np.int64) + offsets].sum(axis=-2)


def load_reference_codes(path: str) -> tuple[np.ndarray, str | None]:
    p = Path(path)
    ref_text = None
    if p.suffix.lower() == ".json":
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            ref_text = loaded.get("reference_text")
            loaded = loaded.get("reference_codes", loaded.get("codes", loaded))
        arr = np.asarray(loaded, dtype=np.int64)
    elif p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        raise ValueError(f"unsupported reference-codes file: {path}")
    if arr.ndim != 2 or arr.shape[1] != NUM_CODEBOOKS:
        raise ValueError(f"expected [T, {NUM_CODEBOOKS}], got {arr.shape}")
    return arr, ref_text


def build_prompt_ids(tokenizer, text: str, num_ref_rows: int, reference_text: str | None) -> np.ndarray:
    ids = [TEXT_SPECIALS["tts"]]
    if reference_text and num_ref_rows > 0:
        ids.append(TEXT_SPECIALS["ref_text"])
        ids.extend(tokenizer.encode(reference_text, add_special_tokens=False))
    if num_ref_rows > 0:
        ids.append(TEXT_SPECIALS["ref_audio"])
        ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_rows)
    ids.append(TEXT_SPECIALS["text"])
    ids.extend(tokenizer.encode(text.strip(), add_special_tokens=False))
    ids.append(TEXT_SPECIALS["audio"])
    return np.asarray(ids, dtype=np.int64)


# ------------------------------------------------------- generation (proven)
def sample_codebooks(logits, temperature, top_p, top_k, rng) -> list[int]:
    out: list[int] = []
    for cb in range(NUM_CODEBOOKS):
        scores = logits[0, cb].astype(np.float64)
        if temperature <= 1e-5 or top_k == 1:
            out.append(int(scores.argmax()))
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
            probs = probs[:len(order)]
            probs /= probs.sum()
        out.append(int(rng.choice(order, p=probs)))
    return out


def apply_generation_delay_mask(codes: list[int], delay_count: int) -> list[int]:
    masked = codes[:]
    for i in range(min(delay_count, NUM_CODEBOOKS)):
        masked[i] = BOC_ID
    return masked


def delayed_to_codec_rows(rows: list[list[int]]) -> list[list[int]]:
    codec: list[list[int]] = []
    for frame in range(len(rows) - (NUM_CODEBOOKS - 1)):
        row = []
        for cb in range(NUM_CODEBOOKS):
            code = int(rows[frame + cb][cb])
            if code < 0 or code >= VOCAB_SIZE - 2:
                row = []
                break
            row.append(code)
        if row:
            codec.append(row)
    return codec


def extract_past(out_map: dict) -> dict:
    return {("past_" + k[len("present_"):] if k.startswith("present_") else k): v
            for k, v in out_map.items() if k.startswith(("present_", "past_"))}


# ----------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Higgs ONNX voice cloning (integrated runner)")
    p.add_argument("--text", default="Hello world, this is a local Higgs voice demo.")
    p.add_argument("--reference-audio", default=None, help=".wav/.mp3/.flac clip to clone")
    p.add_argument("--reference-codes", default=None, help="pre-encoded [T,8] .json/.npy")
    p.add_argument("--reference-text", default=None, help="transcript of the reference clip")
    p.add_argument("--codec", default="bosonai/higgs-audio-v2-tokenizer")
    p.add_argument("--i-have-rights-to-this-voice", action="store_true")
    p.add_argument("--out", default=str(ROOT / "higgs_clone.wav"))
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--provider", default="CUDAExecutionProvider")
    args = p.parse_args()

    if (args.reference_audio or args.reference_codes) and not args.i_have_rights_to_this_voice:
        raise SystemExit("cloning a voice requires --i-have-rights-to-this-voice")

    tokenizer = AutoTokenizer.from_pretrained(ROOT, local_files_only=True)

    # ---- reference -> delayed codes
    raw_codes = None
    if args.reference_audio:
        from encode_reference_audio import encode_file

        raw_codes = encode_file(args.reference_audio, args.codec)
        print(f"[ref] encoded {args.reference_audio}: {raw_codes.shape[0]} frames "
              f"({raw_codes.shape[0]/25:.2f}s)")
    elif args.reference_codes:
        raw_codes, fixture_text = load_reference_codes(args.reference_codes)
        args.reference_text = args.reference_text or fixture_text

    delayed = apply_delay_pattern(raw_codes) if raw_codes is not None else None
    num_ref_rows = 0 if delayed is None else len(delayed)

    prompt_ids = build_prompt_ids(tokenizer, args.text, num_ref_rows, args.reference_text)
    placeholder_idx = np.flatnonzero(prompt_ids == AUDIO_PLACEHOLDER_ID)
    safe_ids = np.where(prompt_ids == AUDIO_PLACEHOLDER_ID,
                        TEXT_SPECIALS["audio"], prompt_ids).astype(np.int64).reshape(1, -1)
    mask = np.ones_like(safe_ids)

    # ---- inputs_embeds
    embed = get_text_embedder(args.provider)
    inputs_embeds = embed(safe_ids).astype(np.float16, copy=True)
    if delayed is not None:
        table = get_fused_table()
        ref_embeds = fused_reference_embeddings(delayed, table).astype(np.float16)
        assert len(placeholder_idx) == len(ref_embeds), (len(placeholder_idx), len(ref_embeds))
        inputs_embeds[0, placeholder_idx, :] = ref_embeds

    # ---- prefill (auto-detect surgery variant by input names)
    prefill = get_patched_prefill(args.provider)
    in_names = {i.name for i in prefill.get_inputs()}
    if "embedding" in in_names:          # surgery-patch zip variant
        feed = {"embedding": inputs_embeds, "attention_mask": mask}
    elif "inputs_embeds" in in_names:    # earlier variant (keeps input_ids for Shape)
        feed = {"input_ids": safe_ids, "attention_mask": mask, "inputs_embeds": inputs_embeds}
    else:
        raise RuntimeError(f"unrecognized patched prefill inputs: {sorted(in_names)}")

    decode_sess = make_session(require("higgs_audio_v3_ar_decode_matmul4.onnx"), args.provider)
    vocoder_sess = make_session(require("higgs_audio_v3_vocoder_decode.onnx"), args.provider)

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    out_map = dict(zip((o.name for o in prefill.get_outputs()), prefill.run(None, feed)))

    delay_count, eoc_countdown, done = 0, None, False
    codes = sample_codebooks(out_map["logits"], args.temperature, args.top_p, args.top_k, rng)
    codes = apply_generation_delay_mask(codes, delay_count)
    delay_count += 1
    rows = [codes[:]]
    past = extract_past(out_map)
    prompt_len = prompt_ids.shape[0]

    for step in range(1, args.max_steps):
        if done:
            break
        feeds = {"codes": np.asarray(codes, dtype=np.int64).reshape(1, NUM_CODEBOOKS),
                 "position_ids": np.asarray([[prompt_len + step - 1]], dtype=np.int64), **past}
        out_map = dict(zip((o.name for o in decode_sess.get_outputs()), decode_sess.run(None, feeds)))
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
    if not codec_rows:
        raise RuntimeError("no codec rows generated — try a different seed/temperature")
    flat = np.asarray(codec_rows, dtype=np.int64).T[None]  # [1, 8, T]
    wav = vocoder_sess.run(None, {"audio_codes": flat})[0][0, 0].astype(np.float32)
    dt = time.perf_counter() - t0

    import soundfile as sf

    sf.write(args.out, wav, SAMPLE_RATE)
    secs = wav.shape[0] / SAMPLE_RATE
    print(f"[done] {args.out}  audio={secs:.2f}s  total={dt:.2f}s  "
          f"rtf={(dt/secs if secs else math.inf):.3f}  "
          f"mode={'clone' if num_ref_rows else 'zero-shot'}  prompt_tokens={prompt_len}")


if __name__ == "__main__":
    main()
'''

# --------------------------------------------------------------------------
FILES["integration-notes-voice-cloning.md"] = r'''
# Integration Notes — Real Reference Audio → ONNX Cloning

Date: 2026-06-11. Integrates the working inputs_embeds demo with the surgery-patch
artifacts and adds the missing piece: **.wav/.mp3 → [T,8] codec codes**.

## New files

| File | Role |
|---|---|
| `encode_reference_audio.py` | wav/mp3/flac → `[T,8]` codes (vendored Higgs codec, PyTorch). Standalone CLI + importable `encode_file()`. |
| `higgs_clone_onnx.py` | Unified runner: zero-shot / `--reference-codes` / `--reference-audio`. Supersedes `run_higgs_inputs_embeds_voice_demo.py` (kept untouched). |

## What got cleaner vs the demo script

- Text embeddings come from the 624-byte `higgs_audio_v3_embedder.onnx` (bit-exact,
  CPU, instant) instead of running a full 36-layer prefill pass just to read the
  `embedding` output. Falls back to the exposed-output trick if the embedder isn't there.
- The fused audio table loads from `modality_embedding_fp16.npy`; if absent it's
  extracted by **byte offset** from the prefill proto (no 2 GB `load_external_data`)
  and cached.
- Patched prefill auto-detected by graph inputs: works with both the surgery-patch
  variant (`embedding`, no input_ids) and the earlier variant
  (`inputs_embeds` + input_ids). Drop either file next to the script.
- Generation loop (sampling, delay mask, EOC countdown, de-delay, vocoder) is the
  proven demo code, unchanged.

## Reference-audio encode path

`audio_codec.py` (SGLang) confirms the contract: mono fp32 `[1,1,L]`, resample to
24 kHz, zero-pad clips < 1 s, `model.encode(...).audio_codes` → `[1,8,T]` →
transpose → `[T,8]` at 25 fps. `encode_reference_audio.py` replicates exactly that
with the vendored `HiggsAudioV2TokenizerModel`.

Weights: `--codec` accepts the public `bosonai/higgs-audio-v2-tokenizer` repo id
(default), a local dir of it, or a full Higgs TTS checkpoint dir (codec tensors are
auto-detected under the `tied.embedding.modality_embeddings.0.model.*` prefix).

## Extra deps (encode step only)

```bash
pip install torch torchaudio safetensors          # CPU torch is enough
# transformers needs PreTrainedAudioTokenizerBase — upgrade if the import fails:
pip install -U transformers
```

mp3 input: decoded via soundfile → torchaudio → librosa chain; if all three fail,
`pip install librosa` or convert with ffmpeg.

## Run

```bash
# 1) one-time: encode a voice (cacheable per voice)
python encode_reference_audio.py --audio voices/me.wav --reference-text "transcript..." --out me.codes.json

# 2) clone
python higgs_clone_onnx.py --reference-codes me.codes.json \
    --text "Have a nice day and enjoy south california sunshine." \
    --i-have-rights-to-this-voice --out clone.wav

# or one shot (encodes inline)
python higgs_clone_onnx.py --reference-audio voices/me.wav --reference-text "..." \
    --text "..." --i-have-rights-to-this-voice --out clone.wav
```

## Known risks / not yet verified

1. **Codec checkpoint key alignment**: the vendored model follows the upstream
   transformers PR #40294 naming; the public `bosonai/higgs-audio-v2-tokenizer`
   snapshot may name tensors differently. The loader tries direct keys plus two
   prefix-strips and aborts with sample keys printed if <50% align — if you hit
   that, paste the printed keys and a remap can be added in minutes.
2. Scripts are syntax-reviewed but not executed here (no GPU/torch in this
   sandbox) — first run may need small fixes; errors will be precise.
3. Clone *quality* through the int4 matmul4 AR graphs is unmeasured — compare
   WavLM similarity informally against the SGLang reference outputs if it matters.
4. Long references: prompt grows by T+7 placeholder rows (10 s clip ≈ 257 rows).
   Keep references ≤ ~15 s for prefill speed and the 8k context.
'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the Higgs ONNX voice-cloning integration patch")
    ap.add_argument("--target", default=".", help="higgs working folder (default: current dir)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    tgt = Path(args.target)
    marker = tgt / "higgs_audio_v3_ar_prefill_matmul4.onnx"
    if not marker.exists() and not (tgt / "ar" / marker.name).exists():
        print(f"[warn] {marker.name} not found in {tgt.resolve()} — is this the higgs folder?")

    for name, payload in FILES.items():
        content = payload.lstrip("\n")
        dest = tgt / name
        if dest.exists() and not args.force:
            print(f"[skip] {dest} exists (use --force to overwrite)")
            continue
        print(f"[{'dry' if args.dry_run else 'write'}] {dest}  ({len(content)} bytes)")
        if not args.dry_run:
            dest.write_text(content, encoding="utf-8", newline="\n")
    print("done." if not args.dry_run else "dry-run complete, nothing written.")


if __name__ == "__main__":
    main()
