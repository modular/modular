# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests type factories and accessors."""

import re

import pytest
from conftest import static_dims
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max import mlir
from max._core import graph as _graph
from max._core.dialects import mo  # type: ignore
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    StaticDim,
    SymbolicDim,
    TensorType,
    _ChainType,
)
from max.graph.type import FilterLayout, Type


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(dtype=...)
def test_dtype_type(mlir_context, dtype: DType) -> None:
    """Tests dtype to MLIR type conversion."""
    # bool dtype prints as i1 if printing the raw mlir type.
    # We would need `DType.from_mlir`, like `tensor_type_get_dtype` to get "bool".
    assume(dtype != DType.bool)
    new_dtype = _graph.dtype_type(mlir_context, dtype._mlir)
    assert dtype._mlir == str(new_dtype).lower()


@given(dim=...)
def test_static_dim(dim: int):
    assume(-(2**63) <= dim < 2**63)
    assert StaticDim(dim).dim == dim


@given(i=...)
def test_static_dim__equals_dim_value(i: int):
    assume(-(2**63) <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert dim == i
    assert dim == dim


@given(i=...)
def test_static_dim__compares_to_dim_value(i: int):
    assume(-(2**63) <= i < 2**63)
    dim = StaticDim(i)
    assert isinstance(dim, Dim)
    assert i <= dim < i + 1


@given(dim=st.integers(min_value=2**63))
def test_static_dim_too_big(dim: int):
    with pytest.raises(ValueError):
        StaticDim(dim)


@given(numerator=static_dims())
def test_static_dim__division_by_zero(numerator: StaticDim):
    with pytest.raises(ZeroDivisionError):
        _ = numerator // 0


def test_algebraic_dim_simplify_and_comparison(mlir_context):
    assert 4 * Dim("x") + 4 == (Dim("x") + 1) * 4
    assert 4 * Dim("x") // 5 != Dim(4) // 5 * "x"
    assert 0 == Dim(4) // 5 * "x"
    assert -Dim("x") - 4 == -(Dim("x") + 4)


def test_dims_print_reasonably(mlir_context):
    assert str(Dim(23)) == "23"
    assert str(Dim("test")) == "test"
    assert str((Dim("x") + "y" - 4) // 5) == "(x + y + -4) // 5"

    assert repr(Dim(23)) == "Dim(23)"
    assert repr(Dim("test")) == "Dim('test')"
    assert (
        repr((Dim("x") + "y" - 4) // 5)
        == "(Dim('x') + Dim('y') + Dim(-4)) // Dim(5)"
    )


# TODO(MSDK-695): less restrictive dim names
@given(
    name=st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z"))
    )
)
def test_symbolic_dim(name: str):
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


def test_symbolic_dim_to_int_error() -> None:
    """Checks the error message when creating an int from a SymbolicDim."""
    with pytest.raises(
        TypeError, match="conversions only supported for static dims"
    ):
        int(SymbolicDim("x"))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(tensor_type=...)
def test_tensor_type_to_mlir(mlir_context, tensor_type: TensorType):
    assert tensor_type == TensorType.from_mlir(tensor_type.to_mlir())


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


def test_tensor_type__negative_dim(mlir_context) -> None:
    with pytest.raises(TypeError, match="dimensions must be non-negative"):
        tensor_type = TensorType(DType.float32, [-1], device=DeviceRef.CPU())


def test_tensor_type_with_device(mlir_context: mlir.Context) -> None:
    """Tests tensor type creation."""
    device_type = DeviceRef.GPU(id=2)
    tensor_type = TensorType(DType.float32, shape=[3], device=device_type)
    assert TensorType.from_mlir(tensor_type.to_mlir()) == tensor_type
    assert tensor_type.to_mlir().device_ref == device_type.to_mlir()


def test_tensor_type_accessors(mlir_context) -> None:
    """Tests tensor type property accessors."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    tensor_type = _graph.tensor_type(mlir_context, dtype, [dim1, dim2])

    assert _graph.tensor_type_get_dtype(tensor_type) == DType.float32._mlir
    assert _graph.tensor_type_get_rank(tensor_type) == 2


def test_tensor_type_layout(mlir_context):
    t = TensorType(DType.float32, ["r", "s", "f", "c"], DeviceRef.CPU())
    t_copy = Type.from_mlir(t.to_mlir())
    t._layout = FilterLayout.RSCF
    t2 = Type.from_mlir(t.to_mlir())
    assert t._layout == FilterLayout.RSCF
    assert t2._layout == FilterLayout.RSCF
    assert t == t2
    # layout is not considered for equality checks
    assert t == t_copy


def test_tensor_type_with_device_accessors(mlir_context: mlir.Context) -> None:
    """Tests tensor type with device property accessors."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    cpu_device = _graph.device_attr(mlir_context, "cpu", 0)
    cpu_tensor_type = _graph.tensor_type_with_device(
        mlir_context, dtype, [dim1, dim2], cpu_device
    )
    cuda_device0 = _graph.device_attr(mlir_context, "cuda", 0)
    cuda_tensor_type = _graph.tensor_type_with_device(
        mlir_context, dtype, [dim1, dim2], cuda_device0
    )
    default_tensor_type = _graph.tensor_type(mlir_context, dtype, [dim1, dim2])

    assert _graph.tensor_type_get_device(cpu_tensor_type) == cpu_device
    assert _graph.tensor_type_get_device(cuda_tensor_type) == cuda_device0
    assert _graph.tensor_type_get_device(default_tensor_type) == cpu_device


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
    opaque_type = _graph.opaque_type(mlir_context, "custom_type")
    buffer_type = _graph.buffer_type(mlir_context, dtype, [dim])
    chain_type = mo.ChainType()

    assert _graph.type_is_tensor(tensor_type)
    assert not _graph.type_is_opaque(tensor_type)
    assert not _graph.type_is_buffer(tensor_type)
    assert not isinstance(tensor_type, mo.ChainType)

    assert _graph.type_is_opaque(opaque_type)
    assert not _graph.type_is_tensor(opaque_type)
    assert not _graph.type_is_buffer(opaque_type)
    assert not isinstance(opaque_type, mo.ChainType)

    assert _graph.type_is_buffer(buffer_type)
    assert not _graph.type_is_opaque(buffer_type)
    assert not _graph.type_is_tensor(buffer_type)
    assert not isinstance(buffer_type, mo.ChainType)

    assert isinstance(chain_type, mo.ChainType)
    assert not _graph.type_is_opaque(chain_type)
    assert not _graph.type_is_tensor(chain_type)
    assert not _graph.type_is_buffer(chain_type)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(buffer_type=...)
def test_buffer_mlir_roundtrip(mlir_context, buffer_type: BufferType):
    assert buffer_type == BufferType.from_mlir(buffer_type.to_mlir())


def test_buffer_type(mlir_context) -> None:
    """Tests buffer type creation."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    buffer_type = _graph.buffer_type(mlir_context, dtype, [dim1, dim2])
    assert str(buffer_type) == "!mo.buffer<[3, x], f32>"


def test_buffer_type_accessors(mlir_context) -> None:
    """Tests buffer type property accessors."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    buffer_type = _graph.buffer_type(mlir_context, dtype, [dim1, dim2])

    assert _graph.buffer_type_get_dtype(buffer_type) == DType.float32._mlir
    assert _graph.buffer_type_get_rank(buffer_type) == 2


def test_buffer_type_with_device_accessors(mlir_context) -> None:
    """Tests buffer type with device property accessors."""
    dtype = _graph.dtype_type(mlir_context, "f32")
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    cpu_device = _graph.device_attr(mlir_context, "cpu", 0)
    cpu_buffer_type = _graph.buffer_type_with_device(
        mlir_context, dtype, [dim1, dim2], cpu_device
    )

    cuda_device0 = _graph.device_attr(mlir_context, "cuda", 0)
    cuda_buffer_type = _graph.buffer_type_with_device(
        mlir_context, dtype, [dim1, dim2], cuda_device0
    )

    default_buffer_type = _graph.buffer_type(mlir_context, dtype, [dim1, dim2])

    assert _graph.buffer_type_get_device(cpu_buffer_type) == cpu_device
    assert _graph.buffer_type_get_device(cuda_buffer_type) == cuda_device0
    assert _graph.buffer_type_get_device(default_buffer_type) == cpu_device


def test_chain_type(mlir_context):
    assert _ChainType() == _ChainType.from_mlir(_ChainType().to_mlir())


def test_invalid_dimension(mlir_context):
    with pytest.raises(TypeError):
        _ = TensorType(
            DType.bfloat16, [-7095393036038990704], device=DeviceRef.CPU()
        )


@pytest.mark.skip("GEX-1918")
def test_GEX_1918(mlir_context) -> None:
    with pytest.raises(ValueError):
        _ = Dim(2**63 - 1) * 2
    with pytest.raises(ValueError):
        _ = Dim(2**63 - 1) + 1


def test_MAXPLAT_148(mlir_context):
    with pytest.raises(TypeError):
        graph = Graph(
            "MAXPLAT-148",
            input_types=[
                TensorType(DType.float32, [-1, 2], device=DeviceRef.CPU())
            ],
        )


def test_device_type(mlir_context) -> None:
    """Tests Device type."""
    host = DeviceRef.CPU(0)
    cuda0 = DeviceRef.GPU(0)
    cuda1 = DeviceRef.GPU(1)
    cuda1_2 = DeviceRef.GPU(1)
    assert cuda0 != cuda1 != host
    assert cuda0 != cuda1_2 != host
    assert cuda0 != DeviceRef.CPU()
    assert cuda1 == cuda1_2
