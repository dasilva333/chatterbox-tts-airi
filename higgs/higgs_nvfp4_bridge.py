#!/usr/bin/env python3
"""Bridge utilities for running Higgs Audio v3 NVFP4 through SGLang-Omni.

This script does three concrete things:

1. Prepares the Reza2kn NVFP4 Hugging Face snapshot for SGLang's ModelOpt FP4
   loader by adding the missing SGLang-style quantization metadata, fixing the
   model index if it points at a nonexistent safetensors file, and replacing a
   Git-LFS tokenizer pointer with a real tokenizer.json when supplied.
2. Installs/patches the Higgs-specific SGLang-Omni reference files into the
   active Python environment, including the Higgs stages.py file that wires
   preprocessing -> audio_encoder -> tts_engine -> vocoder.
3. Provides preflight, serve, and speech-test commands so failures identify the
   exact remaining seam: metadata, safetensors layout, SGLang quantization, or
   runtime serving.

Safety: the speech command requires --i-have-rights-to-this-voice when a
reference voice is used. Do not clone a real person's voice without permission.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_NVFP4_REPO = "Reza2kn/Higgs-Audio-v3-TTS-4bit-NVFP4"
DEFAULT_BASE_TOKENIZER_REPO = "bosonai/higgs-audio-v3-tts-4b"
SGLANG_QUANT_ALGO = "NVFP4"
SGLANG_QUANT_METHOD = "modelopt"
DEFAULT_GROUP_SIZE = 16

# Original Higgs checkpoint prefixes that are known to be non-quantized in the
# Reza2kn artifact. The SGLang quant loader maps/expands these where possible.
DEFAULT_EXCLUDE_MODULES = [
    "tied.embedding.text_embedding",
    "tied.embedding.modality_embeddings",
    "tied.head.text_head",
    "tied.head.modality_heads",
    "body.norm",
    "lm_head",
    "embed_tokens",
    "model.embed_tokens",
    "model.norm",
    "multimodal_embedding",
    "modality_head",
]

PACKED_MODULES_MAPPING = {
    # SGLang's Qwen3 stacks q/k/v and gate/up while loading.
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

SAFETENSORS_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.jinja",
    "LICENSE",
    "README.md",
    "samples/*",
]


class BridgeError(RuntimeError):
    pass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str | None = None

    def render(self) -> str:
        icon = "OK" if self.ok else "FAIL"
        msg = f"[{icon}] {self.name}: {self.detail}"
        if self.fix and not self.ok:
            msg += f"\n      fix: {self.fix}"
        return msg


def log(msg: str) -> None:
    print(f"[higgs-nvfp4] {msg}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        text = path.read_text("utf-8", errors="ignore")
    except OSError:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1")


def resolve_source(source: str, *, revision: str | None = None) -> Path:
    """Resolve a local path or Hugging Face repo id to a local snapshot path."""
    p = Path(source).expanduser()
    if p.exists():
        return p.resolve()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - env-dependent
        raise BridgeError(
            f"{source!r} is not a local path and huggingface_hub is unavailable: {exc}"
        ) from exc

    log(f"Downloading HF snapshot {source} ...")
    snapshot = snapshot_download(
        repo_id=source,
        revision=revision,
        allow_patterns=SAFETENSORS_ALLOW_PATTERNS,
    )
    return Path(snapshot).resolve()


def copy_or_sync_source(source_dir: Path, out_dir: Path, *, in_place: bool) -> Path:
    if in_place:
        return source_dir
    if out_dir.exists():
        log(f"Output already exists: {out_dir}")
    else:
        log(f"Copying snapshot to {out_dir}")
        shutil.copytree(source_dir, out_dir, symlinks=True)
    return out_dir.resolve()


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def build_sglang_quantization_config(group_size: int) -> dict[str, Any]:
    return {
        "quant_method": SGLANG_QUANT_METHOD,
        "quant_algo": SGLANG_QUANT_ALGO,
        "group_size": int(group_size),
        "kv_cache_scheme": "auto",
        "ignore": DEFAULT_EXCLUDE_MODULES,
        "exclude_modules": DEFAULT_EXCLUDE_MODULES,
        "packed_modules_mapping": PACKED_MODULES_MAPPING,
        "_comment": (
            "Generated by higgs_nvfp4_bridge.py. SGLang ModelOptFp4Config "
            "requires quant_algo=NVFP4, group_size, and ignore/exclude_modules."
        ),
    }


def build_hf_quant_config(group_size: int) -> dict[str, Any]:
    return {
        "quantization": {
            "quant_algo": SGLANG_QUANT_ALGO,
            "kv_cache_quant_algo": "auto",
            "exclude_modules": DEFAULT_EXCLUDE_MODULES,
        },
        "group_size": int(group_size),
        "packed_modules_mapping": PACKED_MODULES_MAPPING,
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "block",
                    "group_size": int(group_size),
                }
            }
        },
        "_comment": "Legacy sidecar for SGLang ModelOpt FP4 loader.",
    }


def patch_config_json(model_dir: Path, *, group_size: int) -> None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise BridgeError(f"Missing config.json under {model_dir}")
    cfg = read_json(cfg_path)
    if cfg.get("model_type") != "higgs_multimodal_qwen3":
        log(f"Warning: config model_type is {cfg.get('model_type')!r}, expected higgs_multimodal_qwen3")
    backup_once(cfg_path)
    cfg["quantization_config"] = build_sglang_quantization_config(group_size)
    # SGLang's Qwen3 loader uses this mapping when it needs to reason about
    # fused linears; keeping it top-level also matches some ModelOpt examples.
    cfg["packed_modules_mapping"] = PACKED_MODULES_MAPPING
    write_json(cfg_path, cfg)
    write_json(model_dir / "hf_quant_config.json", build_hf_quant_config(group_size))
    write_json(model_dir / "quantization_config.sglang_modelopt.json", cfg["quantization_config"])
    log("Wrote SGLang ModelOpt FP4 quantization metadata")


def maybe_fix_tokenizer(model_dir: Path, tokenizer_source: str | None) -> None:
    tok = model_dir / "tokenizer.json"
    if tok.exists() and not is_lfs_pointer(tok):
        log("tokenizer.json is already materialized")
        return
    if tokenizer_source:
        src = Path(tokenizer_source).expanduser()
        if src.is_dir():
            src = src / "tokenizer.json"
        if not src.exists() or is_lfs_pointer(src):
            raise BridgeError(f"Tokenizer source is missing or still an LFS pointer: {src}")
        backup_once(tok)
        shutil.copy2(src, tok)
        log(f"Copied real tokenizer.json from {src}")
        return
    log("tokenizer.json is missing or an LFS pointer; attempting HF download from base model")
    try:
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(DEFAULT_BASE_TOKENIZER_REPO, "tokenizer.json")
        backup_once(tok)
        shutil.copy2(src, tok)
        log(f"Downloaded tokenizer.json from {DEFAULT_BASE_TOKENIZER_REPO}")
    except Exception as exc:
        raise BridgeError(
            "tokenizer.json is a Git-LFS pointer and no usable --tokenizer-source was provided. "
            "Pass --tokenizer-source /path/to/real/tokenizer.json from your working Higgs/ONNX folder."
        ) from exc


def safetensors_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def fix_index_weight_map(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file safetensors models don't strictly need an index.
        return
    idx = read_json(index_path)
    weight_map = idx.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        raise BridgeError("model.safetensors.index.json has no weight_map object")
    referenced = set(str(v) for v in weight_map.values())
    existing = {p.name for p in safetensors_files(model_dir)}
    missing = sorted(referenced - existing)
    if not missing:
        log("model.safetensors.index.json references existing safetensors files")
        return
    if len(existing) == 1:
        replacement = next(iter(existing))
        backup_once(index_path)
        idx["weight_map"] = {k: (replacement if v in missing else v) for k, v in weight_map.items()}
        write_json(index_path, idx)
        log(f"Patched index: replaced missing {missing} with {replacement}")
        return
    raise BridgeError(
        f"Index references missing safetensors files {missing}; existing files are {sorted(existing)}. "
        "Manually fix model.safetensors.index.json or provide the missing file(s)."
    )


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise BridgeError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = f.read(header_len)
    try:
        return json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        raise BridgeError(f"Could not parse safetensors header for {path}: {exc}") from exc


def first_safetensors_header(model_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    files = safetensors_files(model_dir)
    if not files:
        return None, None
    return files[0], read_safetensors_header(files[0])


def infer_weight_layout(model_dir: Path) -> dict[str, Any]:
    path, header = first_safetensors_header(model_dir)
    if path is None or header is None:
        return {"has_safetensors": False, "detail": "no local .safetensors file"}
    tensor_keys = [k for k in header.keys() if k != "__metadata__"]
    lower = [k.lower() for k in tensor_keys]
    scale_keys = [k for k in tensor_keys if any(s in k.lower() for s in ("weight_scale", "input_scale", "scale_2", "alpha"))]
    q_body = [k for k in tensor_keys if k.startswith("body.layers.") and k.endswith(".weight")]
    dtypes = {}
    shapes = {}
    for k in tensor_keys[:5000]:
        meta = header.get(k) or {}
        if isinstance(meta, dict):
            dtypes[meta.get("dtype", "?")] = dtypes.get(meta.get("dtype", "?"), 0) + 1
            shape = meta.get("shape")
            if isinstance(shape, list):
                shapes[tuple(shape)] = shapes.get(tuple(shape), 0) + 1
    return {
        "has_safetensors": True,
        "file": str(path),
        "num_tensors": len(tensor_keys),
        "num_body_weight_keys": len(q_body),
        "num_scale_like_keys": len(scale_keys),
        "scale_examples": scale_keys[:12],
        "dtype_counts": dtypes,
        "shape_examples": list(shapes.items())[:12],
    }


def check_model_dir(model_dir: Path) -> list[CheckResult]:
    out: list[CheckResult] = []
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = read_json(cfg_path)
        out.append(CheckResult("config.json", True, f"model_type={cfg.get('model_type')!r}"))
        q = cfg.get("quantization_config") or {}
        ok_q = q.get("quant_algo") == SGLANG_QUANT_ALGO and q.get("group_size") is not None
        out.append(
            CheckResult(
                "SGLang quantization_config",
                bool(ok_q),
                f"quant_algo={q.get('quant_algo')!r}, group_size={q.get('group_size')!r}",
                "run: python higgs_nvfp4_bridge.py prepare --model-path <dir>",
            )
        )
    else:
        out.append(CheckResult("config.json", False, "missing", "download or pass the real NVFP4 snapshot"))

    tok = model_dir / "tokenizer.json"
    out.append(
        CheckResult(
            "tokenizer.json",
            tok.exists() and not is_lfs_pointer(tok),
            "materialized" if tok.exists() and not is_lfs_pointer(tok) else "missing or Git-LFS pointer",
            "pass --tokenizer-source from your working Higgs/ONNX bundle or let prepare download it",
        )
    )

    st_files = safetensors_files(model_dir)
    out.append(
        CheckResult(
            "safetensors file",
            bool(st_files),
            ", ".join(p.name for p in st_files) if st_files else "none found",
            "download quantized.safetensors/model.safetensors from the NVFP4 repo",
        )
    )

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = read_json(index_path)
        refs = set((idx.get("weight_map") or {}).values())
        existing = {p.name for p in st_files}
        missing = sorted(refs - existing)
        out.append(
            CheckResult(
                "safetensors index",
                not missing,
                "all referenced files exist" if not missing else f"missing references: {missing}",
                "run prepare; it patches a single-file snapshot index automatically",
            )
        )

    layout = infer_weight_layout(model_dir)
    if layout.get("has_safetensors"):
        scale_n = int(layout.get("num_scale_like_keys") or 0)
        out.append(
            CheckResult(
                "safetensors header",
                True,
                f"{layout['num_tensors']} tensors; dtypes={layout['dtype_counts']}",
            )
        )
        out.append(
            CheckResult(
                "ModelOpt FP4 scale tensors",
                scale_n > 0,
                f"found {scale_n} scale-like keys; examples={layout.get('scale_examples')}",
                (
                    "If this stays 0, the artifact is not serialized in SGLang's native ModelOpt FP4 layout. "
                    "You then need a Reza-specific dequant/packing loader or a re-export that includes input_scale, weight_scale, and weight_scale_2 tensors."
                ),
            )
        )
    return out


def preflight(model_dir: Path, *, strict: bool) -> None:
    checks = check_model_dir(model_dir)
    for c in checks:
        print(c.render())
    failed = [c for c in checks if not c.ok]
    if failed and strict:
        raise SystemExit(2)


def install_overlay(overlay_dir: Path, *, dry_run: bool = False) -> None:
    try:
        import sglang_omni  # type: ignore
    except Exception as exc:
        raise BridgeError(
            "Cannot import sglang_omni. Install sglang-omni in the active venv first."
        ) from exc

    pkg_root = Path(sglang_omni.__file__).resolve().parent
    src_root = overlay_dir / "sglang_omni"
    if not src_root.exists():
        raise BridgeError(f"Overlay source missing: {src_root}")
    log(f"Installing Higgs overlay into {pkg_root}")
    for src in sorted(src_root.rglob("*.py")):
        rel = src.relative_to(src_root)
        dst = pkg_root / rel
        if dry_run:
            print(f"would copy {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup_once(dst)
        shutil.copy2(src, dst)
    # Copy vendored JSON config too.
    for src in sorted(src_root.rglob("*.json")):
        rel = src.relative_to(src_root)
        dst = pkg_root / rel
        if dry_run:
            print(f"would copy {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup_once(dst)
        shutil.copy2(src, dst)
    if not dry_run:
        log("Overlay installed. Backups use *.bak next to original files.")


def prepare(args: argparse.Namespace) -> Path:
    if args.model_path:
        source_dir = Path(args.model_path).expanduser().resolve()
        if not source_dir.exists():
            raise BridgeError(f"--model-path does not exist: {source_dir}")
        work_dir = source_dir
    else:
        source_dir = resolve_source(args.source, revision=args.revision)
        out_dir = Path(args.output).expanduser().resolve() if args.output else source_dir
        work_dir = copy_or_sync_source(source_dir, out_dir, in_place=args.in_place or args.output is None)
    patch_config_json(work_dir, group_size=args.group_size)
    maybe_fix_tokenizer(work_dir, args.tokenizer_source)
    fix_index_weight_map(work_dir)
    if args.preflight:
        preflight(work_dir, strict=False)
    log(f"Prepared NVFP4 model directory: {work_dir}")
    return work_dir


def run_serve(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    env = os.environ.copy()
    if args.fp4_gemm_backend:
        env["HIGGS_NVFP4_FP4_GEMM_BACKEND"] = args.fp4_gemm_backend
    if args.extra_pythonpath:
        env["PYTHONPATH"] = args.extra_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [args.sgl_omni_bin, "serve", "--model-path", str(model_path), "--port", str(args.port)]
    if args.host:
        cmd.extend(["--host", args.host])
    if args.extra_args:
        cmd.extend(args.extra_args)
    log("Launching: " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, env=env))


def encode_file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_speech(args: argparse.Namespace) -> None:
    if args.reference_audio and not args.i_have_rights_to_this_voice:
        raise BridgeError(
            "Voice reference supplied. Re-run with --i-have-rights-to-this-voice only for a voice you own or have permission to clone."
        )
    payload: dict[str, Any] = {
        "input": args.text,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.reference_audio:
        ref: dict[str, Any] = {}
        ref_path = Path(args.reference_audio).expanduser()
        if args.embed_reference_bytes:
            ref["base64"] = encode_file_b64(ref_path)
        else:
            ref["audio_path"] = str(ref_path.resolve())
        if args.reference_text:
            ref["text"] = args.reference_text
        payload["references"] = [ref]
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        args.url.rstrip("/") + "/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    log(f"POST {req.full_url}")
    with urlopen(req, timeout=args.timeout) as resp:  # noqa: S310 - user-provided local server URL
        body = resp.read()
    out = Path(args.out).expanduser()
    out.write_bytes(body)
    log(f"Wrote {out} ({len(body)} bytes)")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Prepare an NVFP4 snapshot for SGLang ModelOpt FP4 loading")
    prep.add_argument("--model-path", help="Patch an existing local model directory in place")
    prep.add_argument("--source", default=DEFAULT_NVFP4_REPO, help="HF repo id or local source path")
    prep.add_argument("--revision")
    prep.add_argument("--output", help="Output directory when --source is used")
    prep.add_argument("--in-place", action="store_true", help="Patch source snapshot in place")
    prep.add_argument("--tokenizer-source", help="Real tokenizer.json or directory containing it")
    prep.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    prep.add_argument("--preflight", action="store_true", help="Run checks after preparation")
    prep.set_defaults(func=prepare)

    pf = sub.add_parser("preflight", help="Check whether a prepared NVFP4 snapshot is loadable")
    pf.add_argument("--model-path", required=True)
    pf.add_argument("--strict", action="store_true")
    pf.set_defaults(func=lambda a: preflight(Path(a.model_path).expanduser().resolve(), strict=a.strict))

    inst = sub.add_parser("install-overlay", help="Install patched Higgs SGLang-Omni files into the active venv")
    inst.add_argument("--overlay-dir", default=str(Path(__file__).resolve().parent / "sglang_overlay"))
    inst.add_argument("--dry-run", action="store_true")
    inst.set_defaults(func=lambda a: install_overlay(Path(a.overlay_dir).expanduser().resolve(), dry_run=a.dry_run))

    serve = sub.add_parser("serve", help="Launch sgl-omni serve against a prepared NVFP4 model directory")
    serve.add_argument("--model-path", required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host")
    serve.add_argument("--sgl-omni-bin", default="sgl-omni")
    serve.add_argument("--fp4-gemm-backend", choices=("marlin", "cutlass", "flashinfer"))
    serve.add_argument("--extra-pythonpath")
    serve.add_argument("extra_args", nargs=argparse.REMAINDER)
    serve.set_defaults(func=run_serve)

    sp = sub.add_parser("speech", help="Send a cloning or zero-shot speech request to a running server")
    sp.add_argument("--url", default="http://127.0.0.1:8000")
    sp.add_argument("--text", required=True)
    sp.add_argument("--out", default="nvfp4_speech.wav")
    sp.add_argument("--reference-audio")
    sp.add_argument("--reference-text")
    sp.add_argument("--embed-reference-bytes", action="store_true")
    sp.add_argument("--i-have-rights-to-this-voice", action="store_true")
    sp.add_argument("--temperature", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=50)
    sp.add_argument("--max-new-tokens", type=int, default=1024)
    sp.add_argument("--timeout", type=float, default=300.0)
    sp.set_defaults(func=run_speech)
    return p


def main(argv: list[str] | None = None) -> None:
    args = make_argparser().parse_args(argv)
    try:
        args.func(args)
    except BridgeError as exc:
        print(f"[higgs-nvfp4][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
