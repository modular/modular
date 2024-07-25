# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests type factories and accessors."""
import pytest

from max import _graph
from max.graph import DType


@pytest.mark.parametrize(
    "dtype_mlir", [dtype._mlir for dtype in DType if dtype._mlir != "bool"]
)
def test_dtype_type(mlir_context, dtype_mlir) -> None:
    """Tests dtype to MLIR type conversion."""
    dtype = _graph.dtype_type(mlir_context, dtype_mlir)
    assert str(dtype) == dtype_mlir


def test_bool_type(mlir_context) -> None:
    """Tests bool to MLIR type conversion."""
    dtype = _graph.dtype_type(mlir_context, "bool")
    # MO::toMLIRType converts bool to signless integer i1.
    assert str(dtype) == "i1"


def test_tensor_type(mlir_context) -> None:
    """Tests tensor type creation."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    tensor_type = _graph.tensor_type(mlir_context, dtype, [dim1, dim2])
    assert str(tensor_type) == "!mo.tensor<[3, x], f32>"


def test_tensor_type_accessors(mlir_context) -> None:
    """Tests tensor type property accessors."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    tensor_type = _graph.tensor_type(mlir_context, dtype, [dim1, dim2])

    assert _graph.tensor_type_get_dtype(tensor_type) == DType.float32._mlir
    assert _graph.tensor_type_get_rank(tensor_type) == 2


def test_opaque_type(mlir_context) -> None:
    """Tests opaque type creation and properties."""
    opaque = _graph.opaque_type(mlir_context, "custom_type")
    assert _graph.type_is_opaque(opaque)
    assert not _graph.type_is_tensor(opaque)
    assert _graph.opaque_type_name(opaque) == "custom_type"


def test_type_checking(mlir_context) -> None:
    """Tests type checking functions."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim = _graph.static_dim(mlir_context, 3)
    tensor_type = _graph.tensor_type(mlir_context, dtype, [dim])
    opaque = _graph.opaque_type(mlir_context, "custom_type")

    assert _graph.type_is_tensor(tensor_type)
    assert not _graph.type_is_opaque(tensor_type)
    assert _graph.type_is_opaque(opaque)
    assert not _graph.type_is_tensor(opaque)
