# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Exhaustive checks over every swizzle a committed copy uses.

Two sources of swizzle cases:

1. The swizzled box `Term`s of the committed Moka dumps under
   `Kernels/test/gpu/Moka/tools/`. Each is a `rows x cols` tile of `eb`-byte
   elements whose columns are cut into `swz`-byte chunks, each chunk swizzled
   on its own. The dump tuple is `(rows, cols, eb, swz)`.
2. The TMA swizzle modes of the SM100 matmuls in `max/kernels`, which load a
   flat row-major `BM x BK` tile and scatter it to `swizzle(m * BK + k)`.

For each case the `TwoLevelLayout` the swizzle constructor builds must give
the same address as the integer layout with the swizzle applied, at every
coordinate of the tile. Checks are exhaustive.
"""

from layout import (
    ComptimeInt,
    Coord,
    row_major,
    two_level_layout_from_swizzle,
)
from layout.swizzle import Swizzle, make_swizzle
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.bit import log2_floor
from std.sys import size_of
from std.testing import TestSuite, assert_equal


# The distinct swizzled box terms of the committed Moka dumps
# (`Kernels/test/gpu/Moka/tools/**`), with `swz > 0`. Duplicates across dumps
# are listed once.
def test_moka_dump_buf_terms() raises:
    _check_moka_case[1, 64, 2, 128]()
    _check_moka_case[8, 64, 2, 128]()
    _check_moka_case[8, 128, 1, 128]()
    _check_moka_case[32, 32, 2, 32]()
    _check_moka_case[32, 64, 2, 128]()
    _check_moka_case[32, 128, 2, 128]()
    _check_moka_case[32, 192, 2, 128]()
    _check_moka_case[64, 64, 1, 64]()
    _check_moka_case[64, 64, 2, 128]()
    _check_moka_case[64, 256, 2, 128]()
    _check_moka_case[64, 576, 1, 64]()
    _check_moka_case[128, 32, 2, 64]()
    _check_moka_case[128, 64, 1, 64]()
    _check_moka_case[128, 64, 2, 128]()
    _check_moka_case[128, 128, 1, 128]()
    _check_moka_case[128, 128, 2, 128]()


def _check_moka_case[rows: Int, cols: Int, eb: Int, swz: Int]() raises:
    """Checks one swizzled Moka box term at every `(r, c)`.

    A term wider than the swizzle span is consecutive `rows x swz` byte
    chunks. Within one chunk the map is the unswizzled `rows x (swz / eb)`
    element tile, so the swizzle constructor covers each chunk exactly and the
    chunk's base is plain integer arithmetic.

    Parameters:
        rows: Rows of the term.
        cols: Columns of the term.
        eb: Bytes per element.
        swz: Swizzle width in bytes.
    """
    comptime lb = log2_floor(eb)
    comptime bits = log2_floor(swz // 16)
    comptime per = swz // eb
    comptime chunk_swizzle = Swizzle(bits, 4 - lb, 3)
    comptime plain = row_major[rows, per]()
    comptime layout = two_level_layout_from_swizzle[
        type_of(plain), chunk_swizzle
    ](plain)
    var value = layout
    comptime mask = swz // 16 - 1
    for r in range(rows):
        for c in range(cols):
            var chunk = c // per
            var within = c % per
            # The committed copy's address: the chunk base plus the swizzle
            # applied to the byte offset inside the chunk.
            var a = r * swz + within * eb
            var expected = chunk * rows * swz + (a ^ (((a >> 7) & mask) << 4))
            var got = Int(value(Coord(r, within))) * eb + chunk * rows * swz
            assert_equal(got, expected)


# The TMA swizzle modes of the SM100 matmuls. The tile shapes and dtypes
# mirror `test_tile_layout_typed_equivalence.mojo`; a shape is legal for a
# mode when a K row spans a whole number of swizzle atoms.
comptime _fits_swizzle[dtype: DType, BK: Int, mode: TensorMapSwizzle]: Bool = (
    BK * size_of[dtype]()
) % mode.bytes() == 0


def test_tma_swizzle_modes() raises:
    _check_all_tma_shapes[.bfloat16]()
    _check_all_tma_shapes[.float16]()
    _check_all_tma_shapes[.float32]()
    _check_all_tma_shapes[.float8_e4m3fn]()


def _check_all_tma_shapes[dtype: DType]() raises:
    """Runs every tile shape for one element type.

    Parameters:
        dtype: Element data type of the tile.
    """
    _check_all_tma_modes[dtype, TensorMapSwizzle.SWIZZLE_32B]()
    _check_all_tma_modes[dtype, TensorMapSwizzle.SWIZZLE_64B]()
    _check_all_tma_modes[dtype, TensorMapSwizzle.SWIZZLE_128B]()


def _check_all_tma_modes[dtype: DType, mode: TensorMapSwizzle]() raises:
    """Runs every tile shape for one element type and swizzle mode.

    Parameters:
        dtype: Element data type of the tile.
        mode: Memory access pattern swizzling mode.
    """
    _check_tma_case[dtype, 8, 8, mode]()
    _check_tma_case[dtype, 8, 64, mode]()
    _check_tma_case[dtype, 8, 128, mode]()
    _check_tma_case[dtype, 64, 64, mode]()
    _check_tma_case[dtype, 64, 128, mode]()
    _check_tma_case[dtype, 128, 64, mode]()
    _check_tma_case[dtype, 128, 256, mode]()
    _check_tma_case[dtype, 256, 128, mode]()


def _check_tma_case[
    dtype: DType, BM: Int, BK: Int, mode: TensorMapSwizzle
]() raises:
    """Checks one flat `BM x BK` tile at every `m * BK + k`.

    Parameters:
        dtype: Element data type of the tile.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        mode: Memory access pattern swizzling mode.
    """
    comptime if _fits_swizzle[dtype, BK, mode]:
        comptime swizzle = make_swizzle[dtype, mode]()
        comptime plain = row_major[BM * BK]()
        comptime layout = two_level_layout_from_swizzle[
            type_of(plain), swizzle
        ](plain)
        var value = layout
        for i in range(BM * BK):
            assert_equal(Int(value(Coord(i))), swizzle(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
