from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference
from transformers import AutoTokenizer


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
TOKENIZER_DIR = PROJECT_ROOT
ORIG_PREFILL = ROOT / "higgs_audio_v3_ar_prefill_matmul4.onnx"
PATCHED_PREFILL = ROOT / "higgs_audio_v3_ar_prefill_inputs_embeds.onnx"

TTS_ID = 151667
TEXT_ID = 151672
AUDIO_ID = 151670


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Higgs inputs_embeds from the original prefill graph")
    p.add_argument("--text", default="hello world")
    p.add_argument("--out", default=str(ROOT / "higgs_inputs_embeds.npy"))
    p.add_argument("--provider", default="CPUExecutionProvider")
    return p.parse_args()


def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)


def build_input_ids(tokenizer: AutoTokenizer, text: str) -> np.ndarray:
    encoded = tokenizer(text.strip(), add_special_tokens=False)
    ids = [TTS_ID, TEXT_ID, *encoded["input_ids"], AUDIO_ID]
    return np.asarray(ids, dtype=np.int64).reshape(1, -1)


def patch_original_to_expose_embedding(src: Path, dst: Path) -> Path:
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph

    if any(o.name == "embedding" for o in graph.output):
        onnx.save(model, str(dst))
        return dst

    embedding_vi = None
    for vi in list(graph.value_info):
        if vi.name == "embedding":
            embedding_vi = vi
            break

    if embedding_vi is None:
        raise RuntimeError("Could not find value_info for embedding in the original graph.")

    graph.output.append(
        helper.make_tensor_value_info(
            "embedding",
            embedding_vi.type.tensor_type.elem_type,
            [
                d.dim_param if d.dim_param else (d.dim_value if d.dim_value != 0 else None)
                for d in embedding_vi.type.tensor_type.shape.dim
            ],
        )
    )
    model = shape_inference.infer_shapes(model)
    onnx.save(model, str(dst))
    return dst


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()
    input_ids = build_input_ids(tokenizer, args.text)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    exposed = ROOT / "higgs_audio_v3_ar_prefill_with_embedding_output.onnx"
    patch_original_to_expose_embedding(ORIG_PREFILL, exposed)

    sess = make_session(exposed, args.provider)
    outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    out_names = [o.name for o in sess.get_outputs()]
    out_map = dict(zip(out_names, outputs))
    embeds = out_map["embedding"]
    np.save(Path(args.out), embeds)
    print(f"wrote {args.out}")
    print("embedding shape:", embeds.shape, "dtype:", embeds.dtype)

    patched = make_session(PATCHED_PREFILL, args.provider)
    patched_outs = patched.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": embeds.astype(np.float16),
        },
    )
    print("patched outputs:", len(patched_outs))
    print("logits shape:", patched_outs[0].shape)


if __name__ == "__main__":
    main()
