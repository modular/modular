# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

# TODO(MAXPLAT-75): Generate mypy stubs
from max._core.dialects import mo, mosh  # type: ignore
from max._core.dtype import DType


def test_mo_attr(mlir_context):
    # TODO(GEX-1848): use max.dtype.DType
    attr = mo.DTypeAttr(mlir_context, 1)
    assert attr.dtype == DType.bool


def test_mosh(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    assert isinstance(shape_type, mosh.ShapeType)
    # TODO(MAXPLAT-67)
    # assert isinstance(shape_type, mlir.Type)


def test_mosh_shapeattr(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    attr = mosh.MKGENShapeAttr([1, 2, 3], shape_type)
    # Next: validate `attr.values`!


def test_mosh_shapeattr_empty(mlir_context):
    shape_type = mosh.ShapeType(mlir_context)
    attr = mosh.MKGENShapeAttr([], shape_type)
    # assert list(attr.values) == []
