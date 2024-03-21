# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the closed source algorithm package."""

from .functional import (
    map,
    vectorize,
    async_parallelize,
    sync_parallelize,
    parallelize,
    Static1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    BinaryTile1DTileUnitFunc,
    tile,
    Static2DTileUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    unswitch,
    Static1DTileUnswitchUnitFunc,
    tile_and_unswitch,
    Dynamic1DTileUnswitchUnitFunc,
    tile_middle_unswitch_boundaries,
    Static1DTileUnitFuncWithFlags,
    tile_middle_unswitch_boundaries,
    elementwise,
    parallelize_over_rows,
    stencil,
)
from .reduction import (
    map_reduce,
    reduce,
    reduce_boolean,
    max,
    min,
    sum,
    product,
    mean,
    variance,
    all_true,
    any_true,
    none_true,
    argmax,
    argmin,
    reduce_shape,
    cumsum,
)
from .sort import partition, sort
from .swap import swap
