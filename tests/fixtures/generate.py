#!/usr/bin/env python3
"""Regenerate the deterministic Phase 3 ONNX fixtures.

PHPT runs use the committed .onnx files and never download models.  This
script is only for maintainers; it requires `onnx`:

    pip install onnx
    python3 tests/fixtures/generate.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

BASE = Path(__file__).resolve().parent
OPSET = 13
IR_VERSION = 10


def save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, BASE / name)


def make_model(nodes, inputs, outputs, name: str):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION,
    )


def main() -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    save(make_model([helper.make_node("Identity", ["X"], ["Y"])],
                    [x], [y], "identity"), "identity.onnx")

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
    total = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 3])
    diff = helper.make_tensor_value_info("diff", TensorProto.FLOAT, [2, 3])
    save(make_model(
        [
            helper.make_node("Add", ["A", "B"], ["sum"]),
            helper.make_node("Sub", ["A", "B"], ["diff"]),
        ],
        [a, b], [total, diff], "multi_io"), "multi_io.onnx")

    seq_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    seq_y = helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, [2])
    save(make_model([helper.make_node("SequenceConstruct", ["X"], ["seq"])],
                    [seq_x], [seq_y], "sequence_io"), "sequence_io.onnx")

    f16_x = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2])
    f16_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2])
    save(make_model([helper.make_node("Identity", ["X"], ["Y"])],
                    [f16_x], [f16_y], "float16_io"), "float16_io.onnx")

    dynamic_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])
    dynamic_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])
    save(make_model([helper.make_node("Identity", ["X"], ["Y"])],
                    [dynamic_x], [dynamic_y], "dynamic"), "dynamic.onnx")

    for name, shape in (("zero_1d", [0]), ("zero_2d", [2, 0])):
        zx = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
        zy = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
        save(make_model([helper.make_node("Identity", ["X"], ["Y"])],
                        [zx], [zy], name), name + ".onnx")

    # Deliberately not a valid protobuf/model.
    (BASE / "corrupt.onnx").write_bytes(
        b"not an ONNX model\n" + bytes(range(16))
    )


if __name__ == "__main__":
    main()
