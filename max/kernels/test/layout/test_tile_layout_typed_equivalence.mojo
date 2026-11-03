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
"""Equivalence of the typed tile layouts with their legacy counterparts.

Every `tile_layout_*_typed` must lower through `to_layout()` to exactly the
legacy `Layout` that the corresponding `tile_layout_*()` builds, so callers can
switch to the typed form without moving a byte in shared memory.

Layouts are compared through their printed form rather than `Layout.__eq__`:
`IntTuple` equality treats a single-element tuple as its scalar, which would
hide a change in the mode nesting.
"""

from std.sys import size_of
from std.testing import TestSuite, assert_equal

from layout import IntTuple, Layout, coord
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_k_major_typed,
    tile_layout_mn_major,
    tile_layout_mn_major_typed,
    tile_sf_layout_k_major,
    tile_sf_layout_k_major_typed,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle

# A K extent is tileable only when it spans a whole number of swizzle atoms;
# other combinations are rejected by the legacy layouts themselves.
comptime _k_fits_swizzle[
    dtype: DType, BK: Int, swizzle_mode: TensorMapSwizzle
] = (BK * size_of[dtype]()) % swizzle_mode.bytes() == 0


def _check_k_and_mn_major[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
]() raises:
    """Compares both major orders for one tile configuration.

    The MN-major layouts are the transpose of the K-major ones with the extents
    swapped, so a single legality check covers both.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
    """
    comptime if _k_fits_swizzle[dtype, BK, swizzle_mode]:
        assert_equal(
            String(
                tile_layout_k_major_typed[
                    dtype, BM, BK, swizzle_mode
                ].to_layout()
            ),
            String(tile_layout_k_major[dtype, BM, BK, swizzle_mode]()),
        )
        assert_equal(
            String(
                tile_layout_mn_major_typed[
                    dtype, BK, BM, swizzle_mode
                ].to_layout()
            ),
            String(tile_layout_mn_major[dtype, BK, BM, swizzle_mode]()),
        )


def _check_all_swizzles[dtype: DType, BM: Int, BK: Int]() raises:
    """Runs `_check_k_and_mn_major` over every swizzle mode.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
    """
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_NONE]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_32B]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_64B]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_128B]()


def _check_all_shapes[dtype: DType]() raises:
    """Runs `_check_all_swizzles` over a range of tile shapes.

    `BM == 8` and `BK == 8` collapse an outer mode to extent 1, the case where
    the legacy layouts drop the stride to 0.

    Parameters:
        dtype: Element data type of the tensor.
    """
    _check_all_swizzles[dtype, 8, 8]()
    _check_all_swizzles[dtype, 8, 64]()
    _check_all_swizzles[dtype, 8, 128]()
    _check_all_swizzles[dtype, 64, 64]()
    _check_all_swizzles[dtype, 64, 128]()
    _check_all_swizzles[dtype, 128, 64]()
    _check_all_swizzles[dtype, 128, 256]()
    _check_all_swizzles[dtype, 256, 128]()


# Scale-factor atom extents: 32 MN lanes by 16 K elements. `_copy_sf_to_tmem_tt`
# indexes the SF SMEM tile at (mn_atom * 32, k_atom * 16).
comptime _SF_ATOM_MN = 32
comptime _SF_ATOM_K = 16


def _assert_sf_offsets[
    BM: Int,
    BK: Int,
    SF_SCALE_SIZE: Int,
    legacy: Layout,
]() raises:
    """Compares the typed and legacy SF offset at every atom coordinate.

    Parameters:
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.
        legacy: The legacy scale-factor `Layout` for this configuration.
    """
    comptime for i in range(BM // 128):
        comptime for k in range(BK // (4 * SF_SCALE_SIZE)):
            comptime expected = legacy(
                IntTuple(i * _SF_ATOM_MN, k * _SF_ATOM_K)
            )
            comptime actual = Int(
                tile_sf_layout_k_major_typed[BM, BK, SF_SCALE_SIZE](
                    coord[i * _SF_ATOM_MN, k * _SF_ATOM_K]
                )
            )
            assert_equal(
                actual,
                expected,
                String(
                    "SF offset divergence at atom (",
                    i,
                    ", ",
                    k,
                    ") for BM=",
                    BM,
                    " BK=",
                    BK,
                ),
            )


def _check_sf[BM: Int, BK: Int, SF_SCALE_SIZE: Int]() raises:
    """Compares the scale-factor layout for one tile configuration.

    Parameters:
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.
    """
    assert_equal(
        String(tile_sf_layout_k_major_typed[BM, BK, SF_SCALE_SIZE].to_layout()),
        String(tile_sf_layout_k_major[BM, BK, SF_SCALE_SIZE]()),
    )
    _assert_sf_offsets[
        BM,
        BK,
        SF_SCALE_SIZE,
        tile_sf_layout_k_major[BM, BK, SF_SCALE_SIZE](),
    ]()


def _assert_k_major_offsets[
    dtype: DType,
    BM: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    legacy: Layout,
]() raises:
    """Compares the typed and legacy K-major offsets at query coordinates.

    The printed-form check above pins the layout structure; this probes the
    coordinates kernels actually index. The SM90 attention and MSA Q tiles
    read the K offsets as `(0, d)` for every swizzle atom `d`, so those are
    checked exhaustively, plus the corners and the middle of the tile.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
        legacy: The legacy K-major `Layout` for this configuration.
    """
    comptime for j in range(BK):
        comptime expected = legacy(IntTuple(0, j))
        comptime actual = Int(
            tile_layout_k_major_typed[dtype, BM, BK, swizzle_mode](coord[0, j])
        )
        assert_equal(
            actual,
            expected,
            String(
                "K-major K-offset divergence at (0, ",
                j,
                ") for BM=",
                BM,
                " BK=",
                BK,
            ),
        )
    # Interior and corner probes beyond the K-offset column.
    comptime expected_10 = legacy(IntTuple(1, 0))
    comptime actual_10 = Int(
        tile_layout_k_major_typed[dtype, BM, BK, swizzle_mode](coord[1, 0])
    )
    assert_equal(actual_10, expected_10, "K-major offset divergence at (1, 0)")

    comptime expected_mid = legacy(IntTuple(BM // 2, BK // 2))
    comptime actual_mid = Int(
        tile_layout_k_major_typed[dtype, BM, BK, swizzle_mode](
            coord[BM // 2, BK // 2]
        )
    )
    assert_equal(
        actual_mid, expected_mid, "K-major offset divergence at middle"
    )

    comptime expected_last = legacy(IntTuple(BM - 1, BK - 1))
    comptime actual_last = Int(
        tile_layout_k_major_typed[dtype, BM, BK, swizzle_mode](
            coord[BM - 1, BK - 1]
        )
    )
    assert_equal(
        actual_last, expected_last, "K-major offset divergence at last"
    )


def _check_k_major_offsets[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
]() raises:
    """Compares the typed and legacy K-major offsets at query coordinates.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
    """
    comptime if _k_fits_swizzle[dtype, BK, swizzle_mode]:
        _assert_k_major_offsets[
            dtype,
            BM,
            BK,
            swizzle_mode,
            tile_layout_k_major[dtype, BM, BK, swizzle_mode](),
        ]()


def test_k_major_offsets_match_legacy() raises:
    _check_k_major_offsets[.bfloat16, 64, 64, TensorMapSwizzle.SWIZZLE_128B]()
    _check_k_major_offsets[.bfloat16, 64, 128, TensorMapSwizzle.SWIZZLE_128B]()
    _check_k_major_offsets[.float16, 128, 64, TensorMapSwizzle.SWIZZLE_64B]()
    _check_k_major_offsets[.bfloat16, 8, 32, TensorMapSwizzle.SWIZZLE_32B]()


def test_k_major_and_mn_major_match_legacy() raises:
    _check_all_shapes[.bfloat16]()
    _check_all_shapes[.float16]()
    _check_all_shapes[.float32]()
    _check_all_shapes[.float8_e4m3fn]()


def test_sf_layout_matches_legacy() raises:
    _check_sf[128, 64, 16]()
    _check_sf[128, 128, 16]()
    _check_sf[128, 128, 32]()
    _check_sf[128, 256, 32]()
    _check_sf[128, 512, 32]()
    _check_sf[256, 128, 16]()
    _check_sf[256, 256, 32]()
    _check_sf[256, 512, 32]()
    _check_sf[384, 192, 16]()
    _check_sf[512, 128, 32]()
    _check_sf[512, 256, 32]()
    _check_sf[640, 128, 32]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
