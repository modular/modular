# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external PyTorch weights into Max Graph."""

from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.utils import load_pytorch


def test_load_pytorch(testdata_directory):
    with Graph("test_load_pytorch") as graph:
        # Loads the data saved in gen_external_checkpoints.py.
        data = load_pytorch(testdata_directory / "example_data.pt")
        assert len(data) == 5
        assert data["a"].value.tensor_type == TensorType(DType.int32, [5, 2])
        assert data["b"].value.tensor_type == TensorType(
            DType.float64, [1, 2, 3]
        )
        assert data["c"].value.tensor_type == TensorType(DType.float32, [])
        assert data["fancy/name"].value.tensor_type == TensorType(
            DType.int64, [3]
        )
        assert data["bf16"].value.tensor_type == TensorType(DType.bfloat16, [2])
