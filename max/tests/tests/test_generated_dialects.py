# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

import functools

import pytest
from max import mlir
from max._core import (
    Block,
    InsertPoint,
    NamedAttribute,
    OpBuilder,
    Operation,
    Pass,
    Type,
    lower,
)
from max._core.dialects import builtin, m, mo, mosh
from max._core.dtype import DType
from max.graph import Graph


def test_mo_attr(mlir_context) -> None:  # noqa: ANN001
    attr = mo.DTypeAttr(DType.bool)
    assert attr.dtype == DType.bool
    assert attr == mo.DTypeAttr(DType.bool)
    assert attr != mo.DTypeAttr(DType.int8)


def test_mosh(mlir_context) -> None:  # noqa: ANN001
    shape_type = mosh.ShapeType()
    assert isinstance(shape_type, mosh.ShapeType)
    assert isinstance(shape_type, Type)
    assert shape_type == mosh.ShapeType()


def test_mosh_shapeattr(mlir_context) -> None:  # noqa: ANN001
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


def test_mosh_shapeattr_empty(mlir_context) -> None:  # noqa: ANN001
    shape_type = mosh.ShapeType()
    attr = mosh.ShapeAttr([], shape_type)
    assert list(attr.values) == []


def test_no_active_context() -> None:
    with pytest.raises(RuntimeError):
        shape_type = mosh.ShapeType()


def test_builtin_integerattr(mlir_context) -> None:  # noqa: ANN001
    int_type = builtin.IntegerType(1, builtin.SignednessSemantics.unsigned)
    int_attr = builtin.IntegerAttr(int_type, 1)
    assert int_attr.type == int_type


def test_builtin_moduleop(mlir_context) -> None:  # noqa: ANN001
    op = builtin.ModuleOp(mlir.Location.current)


def test_mo_graph_op(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [], is_subgraph=False)

    assert graph.name == "hello"
    assert list(graph.input_parameters) == []
    assert graph.function_type == builtin.FunctionType([], [])


def test_regions_and_blocks(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [], is_subgraph=False)

    block = graph.regions[0].front
    assert isinstance(block, Block)

    del builder
    del graph
    del module

    # check that we can still safely access the block
    ip = block.end
    assert isinstance(ip, InsertPoint)


def test_device_ref_attr(mlir_context) -> None:  # noqa: ANN001
    attr = m.DeviceRefAttr("cpu", 0)
    assert attr.label == "cpu"
    assert attr.id == 0


def test_dictattr_arrayview(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    assert list(attr.value) == [na]


def test_arrayview_dead_attr_reference(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    array_view = attr.value
    del attr
    assert array_view[0] == na


def test_arrayview_dead_array_reference(mlir_context) -> None:  # noqa: ANN001
    na = NamedAttribute("foo", builtin.StringAttr("bar"))
    attr = builtin.DictionaryAttr([na])
    out = attr.value[0]
    del attr
    assert out == na


def test_discardable_attributes(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current

    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])

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


def test_discardable_attrs__op_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    del graph
    del builder
    del module
    assert list(attrs) == ["foo"]


def test_discardable_attrs__dict_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs
    del graph
    del builder
    del module
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__attr_deleted(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
    attrs = graph.discardable_attributes
    attrs["foo"] = builtin.StringAttr("foo")
    foo = attrs["foo"]

    del attrs["foo"]
    assert "foo" not in attrs
    assert isinstance(foo, builtin.StringAttr)
    assert foo.value == "foo"


def test_discardable_attrs__concurrent_modification(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    builder = OpBuilder(module.body.end)
    graph = mo.GraphOp(builder, loc, "hello", [], [])
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


def test_lower_remove_dead_values(mlir_context) -> None:  # noqa: ANN001
    with Graph("empty", input_types=[]) as graph:
        graph.output()
    module = Operation._from_cmlir(graph._module.operation)
    assert isinstance(module, builtin.ModuleOp)
    assert "mo.chain.create()" in str(module)
    lower(module, [builtin.passes.RemoveDeadValues()])
    assert isinstance(module, builtin.ModuleOp)
    assert "mo.chain.create()" not in str(module)


def test_lowering_failure_diagnostic(mlir_context) -> None:  # noqa: ANN001
    # graph with no output!
    graph = Graph("empty", input_types=[])
    module = Operation._from_cmlir(graph._module.operation)
    assert isinstance(module, builtin.ModuleOp)
    with pytest.raises(Exception):
        module.verify()
    with pytest.raises(Exception):
        lower(module, [builtin.passes.RemoveDeadValues()])


def test_construct_pass_with_options(mlir_context) -> None:  # noqa: ANN001
    # Tablegen doesn't generate a public-visibility way to inspect
    # pass options, so don't try to test the actual pass option values.

    no_options = mo.passes.MOToMOGG()
    assert isinstance(no_options, Pass)
    assert no_options.name == "MOToMOGG"

    with_options = mo.passes.MOToMOGG(
        kernel_library_paths=["foo", "bar"],
        force_sync=True,
    )
    assert isinstance(with_options, Pass)
    assert with_options.name == "MOToMOGG"


def test_get_context_from_cpp(mlir_context) -> None:  # noqa: ANN001
    loc = mlir.Location.current
    module = builtin.ModuleOp(loc)
    assert module.context is mlir_context
