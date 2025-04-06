# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

import functools

# TODO(MAXPLAT-75): Generate mypy stubs
from max import mlir
from max._core import OpBuilder, Type

# TODO(MAXPLAT-75): typing
from max._core.dialects import builtin, m, mo, mosh  # type: ignore
from max._core.dtype import DType


def test_mo_attr(mlir_context):
    attr = mo.DTypeAttr(mlir_context, DType.bool)
    assert attr.dtype == DType.bool
    assert attr == mo.DTypeAttr(mlir_context, DType.bool)
    assert attr != mo.DTypeAttr(mlir_context, DType.int8)


def test_mosh(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    assert isinstance(shape_type, mosh.ShapeType)
    assert isinstance(shape_type, Type)
    assert shape_type == mosh.ShapeType(mlir_context)


def test_mosh_shapeattr(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    attr = mosh.ShapeAttr([1, 2, 3], shape_type)
    dims = attr.values
    index_type = builtin.IntegerType(
        mlir_context, 64, builtin.SignednessSemantics.signed
    )
    uint8_type = builtin.IntegerType(
        mlir_context, 8, builtin.SignednessSemantics.unsigned
    )
    Index = functools.partial(builtin.IntegerAttr, index_type)
    UInt8 = functools.partial(builtin.IntegerAttr, uint8_type)
    expected = [Index(1), Index(2), Index(3)]
    assert attr.values == expected
    assert attr.values != []
    assert attr.values != [Index(1), Index(1), Index(1)]
    assert attr.values != [UInt8(1), UInt8(2), UInt8(3)]


def test_mosh_shapeattr_empty(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    attr = mosh.ShapeAttr([], shape_type)
    assert list(attr.values) == []


def test_mosh_shapeattr__no_active_context():
    with mlir.Context() as ctx:
        shape_type = mosh.ShapeType(ctx)
    attr = mosh.ShapeAttr([], shape_type)
    assert attr.values == []
    assert attr.type == shape_type


def test_builtin_integerattr(mlir_context):
    int_type = builtin.IntegerType(
        mlir_context, 1, builtin.SignednessSemantics.unsigned
    )
    int_attr = builtin.IntegerAttr(int_type, 1)
    assert int_attr.type == int_type


def test_builtin_moduleop(mlir_context):
    op = builtin.ModuleOp(mlir.Location.current)


def test_mo_graph_op(mlir_context):
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc, "hello", [], [], isSubgraph=False)

    # TODO(MAXPLAT-75): typing
    assert graph.name == "hello"  # type: ignore
    assert graph.input_params == []  # type: ignore
    assert graph.function_type == builtin.FunctionType(mlir_context, [], [])  # type: ignore


def test_device_ref_attr(mlir_context):
    attr = m.DeviceRefAttr(mlir_context, "cpu", 0)
    assert attr.label == "cpu"
    assert attr.id == 0
