# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the algorithm package."""

from .functional import (
    BinaryTile1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    Dynamic1DTileUnswitchUnitFunc,
    Static1DTileUnitFunc,
    Static1DTileUnitFuncWithFlags,
    Static1DTileUnswitchUnitFunc,
    Static2DTileUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    elementwise,
    map,
    parallelize,
    parallelize_over_rows,
    stencil,
    sync_parallelize,
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)
from .memory import parallel_memcpy
from .reduction import (
    all_true,
    any_true,
    cumsum,
    map_reduce,
    max,
    mean,
    min,
    none_true,
    product,
    reduce,
    reduce_boolean,
    sum,
    variance,
)
