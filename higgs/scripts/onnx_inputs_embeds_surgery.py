from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, checker, shape_inference


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


def find_node_by_output(model: onnx.ModelProto, output_name: str):
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def collect_consumers(model: onnx.ModelProto):
    consumers: dict[str, list[int]] = {}
    for idx, node in enumerate(model.graph.node):
        for inp in node.input:
            consumers.setdefault(inp, []).append(idx)
    return consumers


def make_value_info_from_tensor(name: str, tensor_type: onnx.TypeProto) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name,
        tensor_type.tensor_type.elem_type,
        [
            d.dim_param if d.dim_param else (d.dim_value if d.dim_value != 0 else None)
            for d in tensor_type.tensor_type.shape.dim
        ],
    )


def surgery(src: Path, dst: Path) -> None:
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph
    consumers = collect_consumers(model)

    embedding_node = None
    embedding_out = None
    for node in graph.node:
        if node.op_type == "GatherBlockQuantized" and "input_ids" in node.input:
            embedding_node = node
            embedding_out = node.output[0]
            break
    if embedding_node is None or embedding_out is None:
        raise RuntimeError("Could not find a GatherBlockQuantized node fed by input_ids.")

    # Derive the external input type from the existing embedding output if available.
    inferred = None
    for vi in list(graph.value_info) + list(graph.output):
        if vi.name == embedding_out:
            inferred = vi
            break

    if inferred is None:
        raise RuntimeError(
            f"Could not find value_info for {embedding_out!r}; need shape metadata to add inputs_embeds."
        )

    inputs_embeds = make_value_info_from_tensor("inputs_embeds", inferred.type)
    graph.input.append(inputs_embeds)

    # Rewire all consumers of the embedding output to consume inputs_embeds instead.
    for node in graph.node:
        rewired = ["inputs_embeds" if inp == embedding_out else inp for inp in node.input]
        if list(node.input) != rewired:
            node.input[:] = rewired

    # Remove the embedding node if it no longer feeds any node.
    new_nodes = []
    for node in graph.node:
        if node is embedding_node:
            continue
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Remove the explicit input_ids reference from the embedding node's removed path only.
    # Keep input_ids in the graph because the remaining shape/position logic still uses it.

    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    onnx.save(model, str(dst))


def parse_args():
    p = argparse.ArgumentParser(description="Patch Higgs prefill ONNX to expose inputs_embeds")
    p.add_argument("--src", default=str(MODELS_DIR / "higgs_audio_v3_ar_prefill_matmul4.onnx"))
    p.add_argument("--dst", default=str(MODELS_DIR / "higgs_audio_v3_ar_prefill_inputs_embeds.onnx"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    surgery(Path(args.src), Path(args.dst))
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
