# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

import functools

import pytest
from max import mlir
from max._core import OpBuilder, Type
from max._core.dialects import builtin, m, mo, mosh
from max._core.dtype import DType


def test_mo_attr(mlir_context):
    attr = mo.DTypeAttr(DType.bool)
    assert attr.dtype == DType.bool
    assert attr == mo.DTypeAttr(DType.bool)
    assert attr != mo.DTypeAttr(DType.int8)


def test_mosh(mlir_context):
    shape_type = mosh.ShapeType()
    assert isinstance(shape_type, mosh.ShapeType)
    assert isinstance(shape_type, Type)
    assert shape_type == mosh.ShapeType()


def test_mosh_shapeattr(mlir_context):
    shape_type = mosh.ShapeType()
    attr = mosh.ShapeAttr([1, 2, 3], shape_type)
    dims = attr.values
    index_type = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
    uint8_type = builtin.IntegerType(8, builtin.SignednessSemantics.unsigned)
    Index = functools.partial(builtin.IntegerAttr, index_type)
    UInt8 = functools.partial(builtin.IntegerAttr, uint8_type)
    expected = [Index(1), Index(2), Index(3)]
    assert attr.values == expected
    assert attr.values != []
    assert attr.values != [Index(1), Index(1), Index(1)]
    assert attr.values != [UInt8(1), UInt8(2), UInt8(3)]


def test_mosh_shapeattr_empty(mlir_context):
    shape_type = mosh.ShapeType()
    attr = mosh.ShapeAttr([], shape_type)
    assert list(attr.values) == []


def test_no_active_context():
    with pytest.raises(RuntimeError):
        shape_type = mosh.ShapeType()


def test_builtin_integerattr(mlir_context):
    int_type = builtin.IntegerType(1, builtin.SignednessSemantics.unsigned)
    int_attr = builtin.IntegerAttr(int_type, 1)
    assert int_attr.type == int_type


def test_builtin_moduleop(mlir_context):
    op = builtin.ModuleOp(mlir.Location.current)


def test_mo_graph_op(mlir_context):
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [], is_subgraph=False)

    assert graph.name == "hello"
    assert graph.input_parameters == []
    assert graph.function_type == builtin.FunctionType([], [])


def test_device_ref_attr(mlir_context):
    attr = m.DeviceRefAttr("cpu", 0)
    assert attr.label == "cpu"
    assert attr.id == 0
