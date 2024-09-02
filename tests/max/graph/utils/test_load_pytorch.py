# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external PyTorch weights into Max Graph."""

from max.dtype import DType
from max.graph import Graph, TensorType, Value
from max.graph.utils import load_pytorch


def test_load_pytorch(testdata_directory):
    # Loads the values saved in gen_external_checkpoints.py.
    weights = load_pytorch(testdata_directory / "example_data.pt")
    with Graph("test_load_pytorch") as graph:
        values = {key: Value(weight) for key, weight in weights.items()}
        assert len(values) == 5
        assert values["a"].type == TensorType(DType.int32, [5, 2])
        assert values["b"].type == TensorType(DType.float64, [1, 2, 3])
        assert values["c"].type == TensorType(DType.float32, [])
        assert values["fancy/name"].type == TensorType(DType.int64, [3])
        assert values["bf16"].type == TensorType(DType.bfloat16, [2])
