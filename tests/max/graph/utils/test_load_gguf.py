# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external GGUF weights into Max Graph."""
import pytest
from max.graph import DType, Graph, TensorType
from max.graph.utils import load_gguf


def test_load_gguf(testdata_directory) -> None:
    with Graph("test_load_gguf") as graph:
        data = load_gguf(testdata_directory / "example_data.gguf")
        assert len(data) == 6
        assert data["a"].value.tensor_type == TensorType(DType.int32, [5, 2])
        assert data["b"].value.tensor_type == TensorType(
            DType.float64, [1, 2, 3]
        )
        assert data["c"].value.tensor_type == TensorType(DType.float32, [])
        assert data["fancy/name"].value.tensor_type == TensorType(
            DType.int64, [3]
        )
        assert data["bf16"].value.tensor_type == TensorType(DType.bfloat16, [2])
        assert data["quantized"].value.tensor_type == TensorType(
            DType.uint8, [2, 144]
        )
