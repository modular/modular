# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests type factories and accessors."""

import re

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max import _graph
from max.dtype import DType
from max.graph.type import Dim, StaticDim, SymbolicDim, TensorType, _OpaqueType


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(dtype=...)
def test_dtype_type(mlir_context, dtype: DType) -> None:
    """Tests dtype to MLIR type conversion."""
    # bool dtype prints as i1 if printing the raw mlir type.
    # We would need `DType.from_mlir`, like `tensor_type_get_dtype` to get "bool".
    assume(dtype != DType.bool)
    new_dtype = _graph.dtype_type(mlir_context, dtype._mlir)
    assert dtype._mlir == str(new_dtype)


@given(dim=...)
def test_static_dim(dim: int):
    assume(-1 <= dim < 2**63)
    assert StaticDim(dim).dim == dim


@given(i=...)
def test_static_dim__equals_dim_value(i: int):
    assume(-1 <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert dim == i
    assert dim == dim


@given(i=...)
def test_static_dim__compares_to_dim_value(i: int):
    assume(-1 <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert i <= dim < i + 1


@given(dim=...)
def test_static_dim_negative(dim: int):
    assume(dim < -1)
    with pytest.raises(ValueError):
        StaticDim(dim)


@given(dim=st.integers(min_value=2**63))
def test_static_dim_too_big(dim: int):
    with pytest.raises(ValueError):
        StaticDim(dim)


# TODO(MSDK-695): less restrictive dim names
@given(
    name=st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))
    )
)
def test_symbolic_dim(name: str):
    assume(name != "")
    SymbolicDim(name)


# TODO(MSDK-695): less restrictive dim names
@given(
    name=st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))
    )
)
def test_symbolic_dim__equals_name(name: str):
    assume(name != "")
    dim = SymbolicDim(name)
    assert isinstance(dim, Dim)
    assert dim == name
    assert dim == dim


# TODO(MSDK-695): less restrictive dim names
@given(name=st.text())
def test_symbolic_dim_invalid(name: str):
    assume(not re.match(r"^[a-zA-Z_]\w*$", name))
    with pytest.raises(ValueError):
        SymbolicDim(name)


@given(dim=...)
def test_dim_to_mlir_no_context(dim: Dim):
    with pytest.raises(RuntimeError):
        print(dim.to_mlir())


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tensor_type=...)
def test_tensor_type_to_mlir(mlir_context, tensor_type: TensorType):
    assert tensor_type == TensorType.from_mlir(tensor_type.to_mlir())


@given(tensor_type=...)
def test_tensor_type_to_mlir_no_context(tensor_type: TensorType):
    with pytest.raises(RuntimeError):
        tensor_type.to_mlir()


def test_opaque_type_to_mlir_no_context():
    with pytest.raises(RuntimeError):
        _OpaqueType("something").to_mlir()


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
