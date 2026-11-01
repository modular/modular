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
"""Parity of typed layout algebra with the legacy `Layout` algebra.

The kernels that moved off `RuntimeLayout` and `RuntimeTuple` now build their
coordinate maps with the typed algebra in `layout.tile_layout`. This test
rebuilds each map with the legacy algebra and checks that both map every
linear index of the domain to the same offset, so the strides of every leaf
agree. A linear index decomposes colexicographically over the flattened shape
in both implementations, so walking the whole domain covers every leaf.

Covered maps:

- `st_matrix_n_layout` / `st_matrix_m_layout`, formerly
  `downcast(logical_product(atom, tiler), f)`, now closed-form strides.
- `zipped_divide(upcast(layout, simd), thread_layout)` as used by the SM100
  structured epilogues to distribute SIMD vectors over threads.
- `blocked_product(..., coalesce_output=True)` as used by the SM100
  non-transposed shared-memory epilogue.
"""

from std.sys import size_of
from std.testing import TestSuite, assert_equal

from layout import Coord, Idx, IntTuple, Layout as LegacyLayout, coord
from layout.layout import blocked_product as legacy_blocked_product
from layout.layout import downcast, logical_product
from layout.layout import upcast as legacy_upcast
from layout.layout import zipped_divide as legacy_zipped_divide
from layout.tensor_core_async import st_matrix_m_layout, st_matrix_n_layout
from layout.tile_layout import (
    Layout as TileLayout,
    TensorLayout,
    blocked_product,
    col_major,
    row_major,
    upcast,
    zipped_divide,
)


def _legacy_n_layout[
    c_type: DType, WG_BN: Int, num_m_mmas: Int, num_consumer: Int
]() -> LegacyLayout:
    comptime n_stmatrix = WG_BN // 16
    comptime C = LegacyLayout.row_major(64, 2 * n_stmatrix)
    comptime atom = LegacyLayout([16, 2, 4], [C([1, 0]), C([0, 1]), C([16, 0])])
    comptime b128_layout = logical_product(
        atom, LegacyLayout.col_major(n_stmatrix, num_m_mmas, num_consumer)
    )
    return downcast(materialize[b128_layout](), 128 // (8 * size_of[c_type]()))


def _legacy_m_layout[
    c_type: DType, WG_BM: Int, num_m_mmas: Int, num_consumer: Int
]() -> LegacyLayout:
    comptime n_stmatrix = WG_BM // 16
    comptime C = LegacyLayout.row_major(2 * n_stmatrix, 8 * num_consumer)
    comptime atom = LegacyLayout(
        [8, 2, 2, 4], [C([1, 0]), C([0, 1]), C([8, 0]), C([0, 2])]
    )
    comptime b128_layout = logical_product(
        atom, LegacyLayout.row_major(n_stmatrix, num_m_mmas, num_consumer)
    )
    return downcast(materialize[b128_layout](), 128 // (8 * size_of[c_type]()))


def _assert_same_map(legacy: LegacyLayout, typed: Some[TensorLayout]) raises:
    """Checks that both layouts agree on every linear index of the domain."""
    assert_equal(typed.size(), legacy.size())
    for idx in range(legacy.size()):
        assert_equal(
            Int(typed(Coord(idx))),
            legacy(IntTuple(idx)),
            String(t"mismatch at linear index {idx} for {legacy}"),
        )


def _check_st_matrix[
    c_type: DType, WG_BN: Int, num_m_mmas: Int, num_consumer: Int
]() raises:
    _assert_same_map(
        _legacy_n_layout[c_type, WG_BN, num_m_mmas, num_consumer](),
        st_matrix_n_layout[c_type, WG_BN, num_m_mmas, num_consumer],
    )
    _assert_same_map(
        _legacy_m_layout[c_type, WG_BN, num_m_mmas, num_consumer](),
        st_matrix_m_layout[c_type, WG_BN, num_m_mmas, num_consumer],
    )


def _check_st_matrix_dtype[c_type: DType]() raises:
    _check_st_matrix[c_type, 16, 1, 1]()
    _check_st_matrix[c_type, 32, 1, 1]()
    _check_st_matrix[c_type, 64, 1, 2]()
    _check_st_matrix[c_type, 64, 2, 1]()
    _check_st_matrix[c_type, 128, 1, 2]()
    _check_st_matrix[c_type, 128, 2, 2]()
    _check_st_matrix[c_type, 256, 4, 2]()


def test_st_matrix_bfloat16() raises:
    _check_st_matrix_dtype[.bfloat16]()


def test_st_matrix_float16() raises:
    _check_st_matrix_dtype[.float16]()


def test_st_matrix_float32() raises:
    _check_st_matrix_dtype[.float32]()


def _check_zipped_upcast[
    simd_size: Int, thread_shape: Coord
](
    legacy: LegacyLayout,
    typed: Some[TensorLayout],
    legacy_thread_shape: IntTuple,
) raises:
    """Checks the SIMD-vector-over-threads distribution of one SMEM layout."""
    var legacy_zipped = legacy_zipped_divide(
        legacy_upcast(legacy.copy(), simd_size),
        LegacyLayout.row_major(legacy_thread_shape),
    )
    var typed_zipped = zipped_divide[thread_shape](
        upcast[factor=simd_size](typed)
    )
    _assert_same_map(legacy_zipped, typed_zipped)


def _check_transpose_epilogue_cta2[
    stageN: Int, swizzle_width: Int, simd_size: Int
]() raises:
    """The 4D `(num_swblocks, stageN, 2, 32)` layout of the cta_group=2 path."""
    comptime num_swblocks = 2
    comptime tile_width = 32
    _check_zipped_upcast[simd_size, coord[1, 8, 1, 4]](
        LegacyLayout(
            IntTuple(num_swblocks, stageN, 2, tile_width),
            IntTuple(stageN * swizzle_width, 2 * tile_width, tile_width, 1),
        ),
        TileLayout(
            Coord(Idx[num_swblocks], Idx[stageN], Idx[2], Idx[tile_width]),
            Coord(
                Idx[stageN * swizzle_width],
                Idx[2 * tile_width],
                Idx[tile_width],
                Idx[1],
            ),
        ),
        IntTuple(1, 8, 1, 4),
    )


def _check_transpose_epilogue_cta1[stageN: Int, simd_size: Int]() raises:
    """The 3D `(stageN, 4, 16)` row-major layout of the cta_group=1 path."""
    comptime tile_width = 16
    _check_zipped_upcast[simd_size, coord[min(16, stageN), 1, 2]](
        LegacyLayout.row_major(stageN, 4, tile_width),
        row_major[stageN, 4, tile_width](),
        IntTuple(min(16, stageN), 1, 2),
    )


def _check_bounds_checked_store[
    TMA_BM: Int, stageN: Int, c_smem_dim1: Int, threads: Int, simd_size: Int
]() raises:
    """The `(TMA_BM, stageN)` split of the C staging tile over all threads."""
    comptime thread_n = stageN // simd_size
    _check_zipped_upcast[simd_size, coord[threads // thread_n, thread_n]](
        LegacyLayout(IntTuple(TMA_BM, stageN), IntTuple(c_smem_dim1, 1)),
        TileLayout(
            Coord(Idx[TMA_BM], Idx[stageN]), Coord(Idx[c_smem_dim1], Idx[1])
        ),
        IntTuple(threads // thread_n, thread_n),
    )


def test_zipped_upcast_transpose_epilogue() raises:
    _check_transpose_epilogue_cta2[32, 64, 8]()
    _check_transpose_epilogue_cta2[64, 64, 8]()
    _check_transpose_epilogue_cta2[32, 32, 4]()
    _check_transpose_epilogue_cta1[8, 8]()
    _check_transpose_epilogue_cta1[16, 8]()
    _check_transpose_epilogue_cta1[32, 8]()
    _check_transpose_epilogue_cta1[64, 4]()


def test_zipped_upcast_bounds_checked_store() raises:
    _check_bounds_checked_store[64, 32, 32, 128, 8]()
    _check_bounds_checked_store[64, 64, 64, 128, 8]()
    _check_bounds_checked_store[128, 64, 64, 128, 8]()
    _check_bounds_checked_store[128, 128, 256, 128, 8]()
    _check_bounds_checked_store[64, 64, 64, 128, 4]()


def _check_blocked_m128[stageN: Int]() raises:
    """The coalesced `(32, stageN) x col_major(2, 2)` map of the MMA_M=128 path.

    The epilogue reads the result as `(row, (col, section))`, so the nesting
    must match exactly, not only the linear map.
    """
    comptime data_paths = 16
    var legacy = legacy_blocked_product(
        LegacyLayout.row_major(data_paths * 2, stageN),
        LegacyLayout.col_major(2, 2),
        coalesce_output=True,
    )
    var typed = blocked_product[coalesce_output=True](
        row_major[data_paths * 2, stageN](), col_major[2, 2]()
    )
    _assert_same_map(legacy, typed)
    assert_equal(String(typed.to_layout()), String(legacy))


def test_blocked_product_coalesced() raises:
    _check_blocked_m128[8]()
    _check_blocked_m128[16]()
    _check_blocked_m128[32]()
    _check_blocked_m128[64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
