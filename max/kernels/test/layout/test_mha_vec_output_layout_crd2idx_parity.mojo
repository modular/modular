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
"""Crd2idx parity gate for the SM90 MHA online-softmax fragment layouts.

This test is a *de-risk gate* for the TileTensor migration of the
online-softmax helpers (`_rowmax_online_softmax` / `_rowsum` in
`nn/softmax.mojo`) and their SM90 caller (`sm90/mha.mojo`
`vectorize_p_reg_tile` / `vectorize_o_reg_tile`). Those helpers index a per-warp
MMA register fragment as `score_reg_tile[col_tile, row_tile]`, where the
fragment's view layout is the *nested, non-row-major* `_PVecTTLayout` /
`_OVecTTLayout` built in `_mha_sm90` in
`nn/attention/gpu/nvidia/sm90/mha.mojo`.

The mechanical converter `LTToTTLayout` (`layout/tile_tensor.mojo`) is LOSSY
for these layouts: it `product_each`-collapses both shape AND stride,
destroying the nested non-contiguous stride structure. So the layout must be
HAND-BUILT. This test proves the hand-built nested TileTensor layout
reproduces the fragment geometry EXACTLY for every `[col_tile, row_tile]`
coordinate the helpers visit (the rank-2 "rank-matching" coord form against a
2-mode nested layout), at the default shape (num_m_mmas = num_n_mmas = 1) and
at larger 2x2 shapes.

The geometry is inlined (rather than importing the kernel) so the test stays
in the `//max:layout` package and does not pull a `//max:nn` dep into the
layout test target. The reference offset is the closed form of the nested
shape/stride pair documented at `sm90/mha.mojo` (`_mha_sm90` fragment layout
block). If that geometry changes, update `_fragment_offset` and the `_tt_*`
builder below in lockstep.

If this test ever FAILS, the hand-built layout diverges from the production
fragment geometry and the softmax helpers are NOT safe as written — stop and
re-derive the nested Coord shape/stride before touching any helper.
"""

from layout import ComptimeInt, Idx
from layout.coord import Coord
from layout.tile_layout import Layout as TTLayout
from std.testing import assert_equal, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ===----------------------------------------------------------------------=== #
# Closed-form fragment geometry.
#
# The kernel builds the fragment as the nested shape/stride pair
#
#   shape  = ((num_row_blocks_per_mma, num_m_mmas),
#             (frag_size // (num_row_blocks_per_mma * frag_simdwidth),
#              num_n_mmas))
#   stride = ((frag_simdwidth, frag_size),
#             (num_row_blocks_per_mma * frag_simdwidth,
#              num_m_mmas * frag_size))
#
# A scalar coordinate splits across a mode's sub-modes with the first
# sub-mode fastest, so `[col_tile, row_tile]` maps to
#
#   (col_tile % R) * frag_simdwidth + (col_tile // R) * frag_size
# + (row_tile % C) * (R * frag_simdwidth) + (row_tile // C) * (M * frag_size)
#
# where R = num_row_blocks_per_mma, M = num_m_mmas and
# C = frag_size // (R * frag_simdwidth).
# ===----------------------------------------------------------------------=== #


def _fragment_offset[
    num_row_blocks_per_mma: Int,
    num_m_mmas: Int,
    frag_simdwidth: Int,
    frag_size: Int,
    frag_cols: Int,
](col_tile: Int, row_tile: Int) -> Int:
    """Returns the reference linear offset of `[col_tile, row_tile]`.

    Parameters:
        num_row_blocks_per_mma: Row blocks per MMA fragment.
        num_m_mmas: MMA tiles along M.
        frag_simdwidth: Elements per fragment vector.
        frag_size: Elements per MMA fragment.
        frag_cols: Columns per fragment mode.

    Args:
        col_tile: Row-block coordinate within the fragment.
        row_tile: Fragment-column coordinate within the fragment.

    Returns:
        The linear offset the nested fragment geometry maps the coordinate to.
    """
    return (
        (col_tile % num_row_blocks_per_mma) * frag_simdwidth
        + (col_tile // num_row_blocks_per_mma) * frag_size
        + (row_tile % frag_cols) * (num_row_blocks_per_mma * frag_simdwidth)
        + (row_tile // frag_cols) * (num_m_mmas * frag_size)
    )


# ===----------------------------------------------------------------------=== #
# Hand-built nested TileTensor `Layout`. THIS IS THE MIGRATION BUILDING BLOCK.
# Shape and stride mirror the fragment geometry above leaf for leaf as nested
# `Coord(Coord(...), Coord(...))` element-type lists. The strides are explicit
# (NOT row_major_nested / col_major_nested, which would impose a contiguous
# convention the production layout does not use).
#
# `frag_cols` (= frag_size // (num_row_blocks_per_mma * frag_simdwidth)) is a
# separate explicit parameter: a parameterized comptime alias whose type
# arguments perform division trips a KGEN interpreter crash when the alias is
# instantiated through a function return type (KGEN FuncType::readFrom), so
# the division is hoisted to the caller exactly as the kernel does.
# ===----------------------------------------------------------------------=== #


comptime _TTVecOutputLayout[
    num_row_blocks_per_mma: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    frag_simdwidth: Int,
    frag_size: Int,
    frag_cols: Int,
] = TTLayout[
    Coord[
        Coord[ComptimeInt[num_row_blocks_per_mma], ComptimeInt[num_m_mmas]],
        Coord[ComptimeInt[frag_cols], ComptimeInt[num_n_mmas]],
    ].element_types,
    Coord[
        Coord[ComptimeInt[frag_simdwidth], ComptimeInt[frag_size]],
        Coord[
            ComptimeInt[num_row_blocks_per_mma * frag_simdwidth],
            ComptimeInt[num_m_mmas * frag_size],
        ],
    ].element_types,
]


def _tt_vec_output_layout[
    num_row_blocks_per_mma: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    frag_simdwidth: Int,
    frag_size: Int,
    frag_cols: Int,
]() -> _TTVecOutputLayout[
    num_row_blocks_per_mma,
    num_m_mmas,
    num_n_mmas,
    frag_simdwidth,
    frag_size,
    frag_cols,
]:
    # Fully static layout: the default constructor materializes shape/stride
    # from the `ComptimeInt` type parameters.
    return _TTVecOutputLayout[
        num_row_blocks_per_mma,
        num_m_mmas,
        num_n_mmas,
        frag_simdwidth,
        frag_size,
        frag_cols,
    ]()


# ===----------------------------------------------------------------------=== #
# Parity driver: walk every `[col_tile, row_tile]` coord the helpers visit and
# assert the hand-built TileTensor crd2idx matches the closed-form geometry.
#
# `col_tile in [0, mode0_size)` where mode0_size = product of mode-0 sub-shape
#   = num_row_blocks_per_mma * num_m_mmas         (== num_rows_per_warp)
# `row_tile in [0, mode1_size)` where mode1_size = product of mode-1 sub-shape
#   = frag_cols * num_n_mmas
#
# These are the SINGLE-INT-into-a-NESTED-MODE accesses (rank-2 coord, 2-mode
# nested layout). TileTensor uses
# `layout(Coord(Idx[col_tile], Idx[row_tile]))`, which must route through CuTe
# crd2idx to the reference offset `_fragment_offset` computes.
# ===----------------------------------------------------------------------=== #


def _assert_crd2idx_parity[
    num_row_blocks_per_mma: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    frag_simdwidth: Int,
    frag_size: Int,
]() raises:
    comptime frag_cols = (
        frag_size // (num_row_blocks_per_mma * frag_simdwidth)
    )
    comptime mode0_size = num_row_blocks_per_mma * num_m_mmas
    comptime mode1_size = frag_cols * num_n_mmas

    comptime tt = _tt_vec_output_layout[
        num_row_blocks_per_mma,
        num_m_mmas,
        num_n_mmas,
        frag_simdwidth,
        frag_size,
        frag_cols,
    ]()

    # Exhaustive coord parity over the helper iteration domain.
    comptime for col_tile in range(mode0_size):
        comptime for row_tile in range(mode1_size):
            comptime expected_off = _fragment_offset[
                num_row_blocks_per_mma,
                num_m_mmas,
                frag_simdwidth,
                frag_size,
                frag_cols,
            ](col_tile, row_tile)
            comptime tt_off = Int(tt(Coord(Idx[col_tile], Idx[row_tile])))
            assert_equal(
                tt_off,
                expected_off,
                String(
                    "crd2idx divergence at (col_tile=",
                    col_tile,
                    ", row_tile=",
                    row_tile,
                    "): expected=",
                    expected_off,
                    " tt=",
                    tt_off,
                ),
            )


# ===----------------------------------------------------------------------=== #
# Default shape: num_m_mmas = num_n_mmas = 1, num_row_blocks_per_mma = 2,
# frag_simdwidth = 2 (the geometry `sm90/mha.mojo::_mha_sm90` instantiates:
# num_row_blocks_per_mma=2, frag_simdwidth=2, num_m_mmas/num_n_mmas from
# WM/WN vs the WGMMA atom).
#
# p_frag_size = MMA_M * MMA_N0 // WARP_SIZE; o_frag_size = MMA_M * MMA_N1 //
# WARP_SIZE. We test a representative spread of frag_size values so the gate
# is not pinned to one (MMA_M, MMA_N) combination.
# ===----------------------------------------------------------------------=== #


def test_mha_default_p_frag_size_16() raises:
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=1,
        num_n_mmas=1,
        frag_simdwidth=2,
        frag_size=16,
    ]()


def test_mha_default_p_frag_size_8() raises:
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=1,
        num_n_mmas=1,
        frag_simdwidth=2,
        frag_size=8,
    ]()


def test_mha_default_p_frag_size_32() raises:
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=1,
        num_n_mmas=1,
        frag_simdwidth=2,
        frag_size=32,
    ]()


def test_mha_default_o_frag_size_64() raises:
    # o_frag_size differs from p_frag_size in general (MMA_N1 != MMA_N0); the
    # layout formula is identical, only frag_size changes.
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=1,
        num_n_mmas=1,
        frag_simdwidth=2,
        frag_size=64,
    ]()


# ===----------------------------------------------------------------------=== #
# Larger shapes: num_m_mmas / num_n_mmas > 1 (exercises BOTH nested sub-modes
# having a non-trivial outer index, i.e. the full nested-stride decomposition).
# ===----------------------------------------------------------------------=== #


def test_mha_multi_mma_2x2_frag_size_16() raises:
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=2,
        num_n_mmas=2,
        frag_simdwidth=2,
        frag_size=16,
    ]()


def test_mha_multi_mma_2x2_frag_size_32() raises:
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=2,
        num_n_mmas=2,
        frag_simdwidth=2,
        frag_size=32,
    ]()


def test_mha_multi_mma_2x1_frag_size_16() raises:
    # Asymmetric: num_m_mmas=2, num_n_mmas=1.
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=2,
        num_n_mmas=1,
        frag_simdwidth=2,
        frag_size=16,
    ]()


def test_mha_multi_mma_1x2_frag_size_16() raises:
    # Asymmetric: num_m_mmas=1, num_n_mmas=2.
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=1,
        num_n_mmas=2,
        frag_simdwidth=2,
        frag_size=16,
    ]()


def test_mha_multi_mma_2x2_o_frag_size_64() raises:
    # The o-fragment at a multi-MMA shape (deeper split-K configs).
    _assert_crd2idx_parity[
        num_row_blocks_per_mma=2,
        num_m_mmas=2,
        num_n_mmas=2,
        frag_simdwidth=2,
        frag_size=64,
    ]()
