# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for max.graph linear algebra operations."""

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.type import Shape, ShapeLike


def matmul_graph(
    name: str,
    shapes: tuple[ShapeLike, ShapeLike],
    dtype: DType = DType.float32,
) -> Graph:
    """Creates a graph op containing a matmul."""

    def tensor_type(dims: ShapeLike) -> TensorType:
        return TensorType(dtype, Shape(dims), device=DeviceRef.CPU())

    return Graph(
        name,
        lambda x, y: x @ y,
        (tensor_type(shapes[0]), tensor_type(shapes[1])),
    )


def assert_matmul_properties(
    graph: Graph,
    expected_output_shape: ShapeLike,
    dtype: DType | None = None,
) -> None:
    """Asserts that the graph contains a matmul, has the expected shape and
    dtype.
    """
    assert "rmo.matmul" in str(graph._mlir_op)
    (result_type,) = graph.output_types
    assert Shape(expected_output_shape) == result_type.shape  # type: ignore
    assert dtype in (None, result_type.dtype)  # type: ignore


# TODO(MSDK-1234): add f8e5m2 and f8e4m3fn to test date types
@pytest.mark.parametrize(
    "dtype",
    [
        d
        for d in DType
        if d
        not in [
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
        ]
    ],
)
def test_matmul_static(dtype: DType) -> None:
    """Tests for static matmul."""

    graph = matmul_graph("matmul_static", ([2, 3], [3, 2]), dtype)
    assert_matmul_properties(graph, (2, 2), dtype)

    # Test matrix-vector multiplication.
    graph = matmul_graph("matmul_static_matrix_vector", ([2, 3], [3]), dtype)
    assert_matmul_properties(graph, (2,), dtype)

    # Test vector-matrix multiplication.
    graph = matmul_graph("matmul_static_vector_matrix", ([3], [3, 2]), dtype)
    assert_matmul_properties(graph, (2,), dtype)

    with pytest.raises(ValueError):
        # Test that a shape error is raised.
        matmul_graph("matmul_static_invalid_axes", ([2, 3], [2, 3]), dtype)


def test_matmul_symbolic() -> None:
    """Tests for symbolic matmul."""

    graph = matmul_graph("matmul_symbolic", (["M", "K"], ["K", "N"]))
    assert_matmul_properties(graph, ["M", "N"])

    # Test that a shape error is raised for incompatible symbolic dims.
    with pytest.raises(ValueError):
        matmul_graph("matmul_symbolic_invalid_axes", (["M", "K"], ["J", "N"]))

    # Test symbolic matrix-vector multiplication.
    graph = matmul_graph("matmul_symbolic_matrix_vector", (["M", "K"], ["K"]))
    assert_matmul_properties(graph, ["M"])

    # Test symbolic vector-matrix multiplication
    graph = matmul_graph("matmul_symbolic_vector_matrix", (["K"], ["K", "N"]))
    assert_matmul_properties(graph, ["N"])


def test_batch_matmul() -> None:
    """Test broadcasting behaviour with batch matmul."""

    # Test basic batch matmul.
    graph = matmul_graph("batch_matmul", ([2, 3, 4], [2, 4, 5]))
    assert_matmul_properties(graph, [2, 3, 5])

    # Test broadcasting in the batch dimension.
    graph = matmul_graph("batch_matmul_broadcast", ([1, 3, 4], [2, 4, 5]))
    assert_matmul_properties(graph, [2, 3, 5])

    # Test multiple batch dimensions.
    graph = matmul_graph("multi_batch_matmul", ([2, 3, 4, 5], [2, 3, 5, 6]))
    assert_matmul_properties(graph, [2, 3, 4, 6])

    # Test non-broadcastable inputs raise an error.
    with pytest.raises(ValueError):
        matmul_graph("batch_matmul_invalid_broadcast", ([11, 6, 1], [2, 6]))


def test_matmul_edge_cases() -> None:
    """Test edge cases for matmul."""

    # Test 1x1 matrix multiplication
    graph = matmul_graph("matmul_1x1", ([1, 1], [1, 1]))
    assert_matmul_properties(graph, [1, 1])

    # Test vector-vector matmul.
    graph = matmul_graph("matmul_1d_tensors", ([3], [3]))
    # Vector-vector matmul should produce a scalar.
    assert_matmul_properties(graph, [])

    # Test that an error is raised for matmul with a scalar.
    with pytest.raises(ValueError):
        matmul_graph("matmul_scalar", ([3], []))


def test_matmul_symbolic_edge_cases() -> None:
    """Test edge cases for symbolic matmul."""

    # Test symbolic matmul with 1 in inner dimension.
    graph = matmul_graph("symbolic_matmul_1_inner", (["M", 1], [1, "N"]))
    assert_matmul_properties(graph, ["M", "N"])

    # Test symbolic batch matmul.
    graph = matmul_graph(
        "symbolic_batch_matmul", (["B", "M", "K"], ["B", "K", "N"])
    )
    assert_matmul_properties(graph, ["B", "M", "N"])

    # Test symbolic batch matmul with broadcasting.
    graph = matmul_graph(
        "symbolic_batch_matmul_broadcast", ([1, "M", "K"], ["B", "K", "N"])
    )
    assert_matmul_properties(graph, ["B", "M", "N"])

    # Test symbolic matmul with mixed static and dynamic dimensions
    graph = matmul_graph(
        "symbolic_matmul_mixed_static_dynamic", (["M", 3], [3, "N"])
    )
    assert_matmul_properties(graph, ["M", "N"])
    # Check static dimension exists in op signature.
    assert "3" in str(graph._mlir_op)

    # Test that an error is raised for incompatible symbolic dimensions.
    with pytest.raises(ValueError):
        matmul_graph("symbolic_matmul_incompatible", (["M", "K"], ["J", "N"]))

    # Test that an error is raised for 1D symbolic matmul with invalid dims.
    with pytest.raises(ValueError):
        matmul_graph("symbolic_matmul_1d_tensors", (["M"], ["N"]))


def test_matmul_higher_rank_symbolic() -> None:
    """Test matmul with higher rank symbolic tensors."""

    # Test 3D tensor multiplication.
    graph = matmul_graph("symbolic_3d_matmul", (["A", "B", "C"], ["C", "D"]))
    assert_matmul_properties(graph, ["A", "B", "D"])

    # Test 4D tensor multiplication.
    graph = matmul_graph(
        "symbolic_4d_matmul", (["A", "B", "C", "D"], ["D", "E"])
    )
    assert_matmul_properties(graph, ["A", "B", "C", "E"])

    # Test higher rank tensor multiplication with broadcasting.
    graph = matmul_graph(
        "symbolic_higher_rank_broadcast_matmul",
        (["A", 1, "B", "C"], ["D", "C", "E"]),
    )
    assert_matmul_properties(graph, ["A", "D", "B", "E"])

    # Test that an error is raised for incompatible higher rank tensors.
    with pytest.raises(ValueError):
        matmul_graph(
            "symbolic_higher_rank_incompatible",
            (["A", "B", "C", "D"], ["E", "F", "G"]),
        )


def test_builder_failure_message() -> None:
    """Test that we get a reasonable error message for a matmul op builder
    failure.

    This is really a test that we are able to bubble up MLIR diagnostics into
    Python exceptions from RMO builders.
    """
    with pytest.raises(ValueError):
        matmul_graph(
            "symbolic_higher_rank_incompatible",
            (["A", "B", "C", "D"], ["E", "F", "G"]),
        )


def test_matmul_chaining() -> None:
    """Test chaining multiple matmul operations."""

    # Test chaining two matmul operations.
    graph = Graph(
        "symbolic_chained_matmul",
        lambda x, y, z: (x @ y) @ z,
        (
            TensorType(DType.float32, ["A", "B"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["B", "C"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["C", "D"], device=DeviceRef.CPU()),
        ),
    )
    assert str(graph._mlir_op).count("rmo.matmul") == 2
    assert "A, D" in str(graph._mlir_op)

    # Test chaining three matmul operations.
    graph = Graph(
        "symbolic_triple_chained_matmul",
        lambda w, x, y, z: ((w @ x) @ y) @ z,
        (
            TensorType(DType.float32, ["A", "B"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["B", "C"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["C", "D"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["D", "E"], device=DeviceRef.CPU()),
        ),
    )
    assert str(graph._mlir_op).count("rmo.matmul") == 3
    assert "A, E" in str(graph._mlir_op)

    # Test chaining matmul with different ranks.
    graph = Graph(
        "symbolic_chained_matmul_different_ranks",
        lambda x, y, z: (x @ y) @ z,
        (
            TensorType(DType.float32, ["A", "B", "C"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["C", "D"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["D", "E"], device=DeviceRef.CPU()),
        ),
    )
    assert str(graph._mlir_op).count("rmo.matmul") == 2
    assert "A, B, E" in str(graph._mlir_op)


def test_matmul_invalid_param_name() -> None:
    """Test edge cases for symbolic matmul."""

    with pytest.raises(ValueError, match="Invalid name"):
        matmul_graph("matmul_invalid_param_name", (["M", "1"], ["1", "N"]))


def test_layer_norm() -> None:
    """Basic test for layer norm."""

    def _layer_norm(input: TensorValue) -> TensorValue:
        return ops.layer_norm(
            input,
            gamma=np.array((0.5, 0.25), np.float32),
            beta=np.array((3.14, 2.72), np.float32),
            epsilon=1e-3,
        )

    graph = Graph(
        "layer_norm",
        _layer_norm,
        input_types=(
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
        ),
    )

    assert "mo.layer_norm" in str(graph._mlir_op)


def test_matmul_dtype_promotion() -> None:
    """Tests for dtype promotion in matmul."""
    graph = Graph(
        "matmul_dtype_promotion",
        lambda x, y: x @ y,
        (
            TensorType(DType.float32, (4, 2, 3), device=DeviceRef.CPU()),
            TensorType(DType.float64, (3, 2), device=DeviceRef.CPU()),
        ),
    )
    assert_matmul_properties(graph, (4, 2, 2), DType.float64)


def test_band_part() -> None:
    def _band_part(input: TensorValue) -> TensorValue:
        return ops.band_part(input, -1, 0)

    graph = Graph(
        "band_part",
        _band_part,
        input_types=(
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
        ),
    )

    assert "rmo.mo.linalg.band_part" in str(graph._mlir_op)
