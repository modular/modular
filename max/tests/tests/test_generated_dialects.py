# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

# TODO(MAXPLAT-75): Generate mypy stubs
from max._core.dialects import mo  # type: ignore
from max._core.dtype import DType


def test_mo_attr(mlir_context):
    # TODO(GEX-1848): use max.dtype.DType
    attr = mo.DTypeAttr(mlir_context, 1)
    assert attr.dtype == DType.bool
