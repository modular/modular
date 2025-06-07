# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

import functools

import pytest
from max import mlir
from max._core import NamedAttribute, OpBuilder, Type
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
    dims = list(attr.values)
    index_type = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
    uint8_type = builtin.IntegerType(8, builtin.SignednessSemantics.unsigned)
    Index = functools.partial(builtin.IntegerAttr, index_type)
    UInt8 = functools.partial(builtin.IntegerAttr, uint8_type)
    expected = [Index(1), Index(2), Index(3)]
    assert dims == expected
    assert dims != []
    assert dims != [Index(1), Index(1), Index(1)]
    assert dims != [UInt8(1), UInt8(2), UInt8(3)]


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
    assert list(graph.input_parameters) == []
    assert graph.function_type == builtin.FunctionType([], [])


def test_device_ref_attr(mlir_context):
    attr = m.DeviceRefAttr("cpu", 0)
    assert attr.label == "cpu"
    assert attr.id == 0


def test_dictattr_arrayview(mlir_context):
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    print(na)
    print(na.name)
    print(na.value)
    attr = builtin.DictionaryAttr([na])
    assert list(attr.value) == [na]


def test_arrayview_dead_attr_reference(mlir_context):
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    array_view = attr.value
    del attr
    assert array_view[0] == na


def test_arrayview_dead_array_reference(mlir_context):
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    out = attr.value[0]
    del attr
    assert out == na


def test_discardable_attributes(mlir_context):
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [])

    attrs = graph.discardable_attributes

    # empty, even though graph has inherent attributes
    assert not attrs
    assert len(attrs) == 0
    assert dict(attrs.items()) == {}

    # __setitem__, __getitem__
    attrs["foo"] = builtin.StringAttr("foo")
    assert attrs
    assert len(attrs) == 1
    assert dict(attrs.items()) == {"foo": builtin.StringAttr("foo")}
    assert attrs["foo"] == builtin.StringAttr("foo")
    with pytest.raises(KeyError):
        attrs["bar"]

    signature = graph.signature
    # Set an inherent attribute
    # "signature" is inherent on `mo.GraphOp`
    attrs["signature"] = builtin.StringAttr("foo")
    assert "signature" in attrs
    assert signature == graph.signature  # inherent attribute is still fine
    assert attrs["signature"] == builtin.StringAttr("foo")
    del attrs["signature"]

    # __contains__
    assert "foo" in attrs
    assert "bar" not in attrs

    # __iter__
    assert list(attrs) == list(attrs.keys()) == ["foo"]

    # __del__
    del attrs["foo"]
    assert not attrs
    assert len(attrs) == 0
    assert dict(attrs.items()) == {}


def test_discardable_attrs__op_deleted(mlir_context):
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    del graph
    del builder
    del module
    assert list(attrs) == ["foo"]


def test_discardable_attrs__dict_deleted(mlir_context):
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs
    del graph
    del builder
    del module
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__attr_deleted(mlir_context):
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs["foo"]
    assert "foo" not in attrs
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__concurrent_modification(mlir_context):
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = builder.create(mo.GraphOp, loc)("hello", [], [])
    attrs = graph.discardable_attributes

    attrs["foo"] = builtin.StringAttr("foo")
    attrs["bar"] = builtin.StringAttr("bar")

    keys = iter(attrs)
    items = iter(attrs.items())
    values = iter(attrs.values())

    keys2 = iter(attrs)
    next(keys2)
    items2 = iter(attrs.items())
    next(items2)
    values2 = iter(attrs.values())
    next(values2)

    del attrs["foo"]
    del attrs["bar"]
    assert not attrs

    # just make sure these don't error, there's not a defined behavior
    _ = list(keys)
    _ = list(items)
    _ = list(values)
    _ = list(keys2)
    _ = list(items2)
    _ = list(values2)
