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
"""Softmax primitives for `HKMhaPrefill`: cross-lane reductions and
column-broadcast ops over col_l rt_32x32 register tiles.

Each column of an rt_32x32 is held redundantly across two half-warps
(lanes `[0, 32)` and `[32, 64)`). Per-column reductions combine
in-lane via `SIMD.reduce_*` and then across half-warps via
`permlane_swap[32]` — a single-cycle DPP-style swap. Using stdlib's
`lane_group_reduce` here would lower to `ds_bpermute_b32` (LDS-routed),
so we go through `permlane_swap` directly.
"""

from std.gpu.intrinsics import permlane_swap
from std.gpu.primitives.warp import vote as warp_vote
from std.math import recip

from layout import TensorLayout

from structured_kernels.amd_tile_io import RegTile


@always_inline
def _col_reduce_at_j[
    layout: TensorLayout,
    //,
    op: StaticString,
    j: Int,
](src: RegTile[DType.float32, layout, _],) -> Float32:
    """Reduces base-tile column `j` of `src` to a single `Float32`.
    `op` is `"max"` or `"add"`; selected at comptime so the unrolled
    body emits the chosen intrinsic directly."""
    comptime assert (
        op == "max" or op == "add"
    ), "_col_reduce_at_j: op must be 'max' or 'add'"

    comptime src_height = layout.static_shape[0]
    comptime base_tile_elts = layout.static_shape[2]
    var src_v = src.vectorize[1, 1, base_tile_elts]()
    comptime assert src_v.flat_rank == 3

    var simd_accum = src_v[0, j, 0]
    comptime for i in range(1, src_height):
        comptime if op == "max":
            simd_accum = max(simd_accum, src_v[i, j, 0])
        else:
            simd_accum = simd_accum + src_v[i, j, 0]

    var lane_val: Float32
    comptime if op == "max":
        lane_val = simd_accum.reduce_max()
    else:
        lane_val = simd_accum.reduce_add()

    # Combine the two half-warps sharing each column.
    var swapped = permlane_swap[32](lane_val, lane_val)
    comptime if op == "max":
        return max(swapped[0], swapped[1])
    else:
        return swapped[0] + swapped[1]


@always_inline
def col_max(
    mut col_accum: RegTile[DType.float32, ...],
    src: RegTile[DType.float32, ...],
):
    """Computes `col_accum[j] = max(src[*, j, *])`."""
    comptime src_width = src.static_shape[1]
    comptime assert (
        col_accum.static_shape[0] == src_width
    ), "col_max: rv outer_dim must equal src.width"
    comptime assert (
        col_accum.static_shape[1] == 1
    ), "col_max: rv inner_dim must be 1 for col_l rt_32x32"
    comptime assert col_accum.flat_rank == 2 and col_accum.origin.mut

    comptime for j in range(src_width):
        col_accum[j, 0] = _col_reduce_at_j["max", j=j](src)


@always_inline
def col_max_acc(
    mut col_accum: RegTile[DType.float32, ...],
    src: RegTile[DType.float32, ...],
    src_accum: RegTile[DType.float32, ...],
):
    """Running-max for online softmax:
    `col_accum[j] = max(src_accum[j], max(src[*, j, *]))`."""
    comptime src_width = src.static_shape[1]
    comptime assert (
        col_accum.static_shape[0] == src_width
    ), "col_max_acc: rv outer_dim must equal src.width"
    comptime assert (
        col_accum.static_shape[1] == 1
    ), "col_max_acc: rv inner_dim must be 1 for col_l rt_32x32"
    comptime assert col_accum.flat_rank == 2 and col_accum.origin.mut
    comptime assert src_accum.flat_rank == 2

    comptime for j in range(src_width):
        col_accum[j, 0] = max(
            src_accum[j, 0][0], _col_reduce_at_j["max", j=j](src)
        )


@always_inline
def col_sum_acc(
    mut col_accum: RegTile[DType.float32, ...],
    src: RegTile[DType.float32, ...],
    src_accum: RegTile[DType.float32, ...],
):
    """Running-norm for online softmax:
    `col_accum[j] = src_accum[j] + sum(src[*, j, *])`."""
    comptime src_width = src.static_shape[1]
    comptime assert (
        col_accum.static_shape[0] == src_width
    ), "col_sum_acc: rv outer_dim must equal src.width"
    comptime assert (
        col_accum.static_shape[1] == 1
    ), "col_sum_acc: rv inner_dim must be 1"
    comptime assert col_accum.flat_rank == 2 and col_accum.origin.mut
    comptime assert src_accum.flat_rank == 2

    comptime for j in range(src_width):
        col_accum[j, 0] = src_accum[j, 0][0] + _col_reduce_at_j["add", j=j](src)


@always_inline
def rv_all_below[
    layout_rv: TensorLayout,
    //,
](
    max_prev: RegTile[DType.float32, layout_rv, MutExternalOrigin],
    max_new: RegTile[DType.float32, layout_rv, MutExternalOrigin],
    threshold: Float32,
) -> Bool:
    """Returns wave-uniform True iff every lane satisfies
    `max_new - max_prev <= threshold`. Wave AND-reduce via a 64-bit
    ballot compared against the full-exec mask (`attend_ker` always
    runs all 64 lanes active). Used by the lazy-rescale skip path."""
    comptime PER_LANE = layout_rv.static_shape[0] * layout_rv.static_shape[1]
    var diff = max_new.raw_load[width=PER_LANE](0) - max_prev.raw_load[
        width=PER_LANE
    ](0)
    var lane_ok = True
    comptime for k in range(PER_LANE):
        if diff[k] > threshold:
            lane_ok = False
    var ballot = warp_vote[DType.uint64](lane_ok)
    return ballot == UInt64(0xFFFFFFFFFFFFFFFF)


@always_inline
def div_col(
    mut dst: RegTile[DType.float32, ...],
    src: RegTile[DType.float32, ...],
    vec: RegTile[DType.float32, ...],
):
    """`dst[gr, gc] = src[gr, gc] / vec[gc]` (final `o_reg / norm_vec`).

    Hand-lowered to `recip(vec) * src` per column. Using `/` directly
    would expand to the IEEE-correct fdiv sequence (`v_div_scale_f32` +
    `v_rcp_f32` + `v_div_fmas_f32` + `v_div_fixup_f32`)."""
    comptime src_height = src.static_shape[0]
    comptime src_width = src.static_shape[1]
    comptime base_tile_elts = src.static_shape[2]
    comptime assert (
        vec.static_shape[0] == src_width
    ), "div_col: rv outer_dim must equal src.width"
    comptime assert vec.static_shape[1] == 1, "div_col: rv inner_dim must be 1"

    var src_v = src.vectorize[1, 1, base_tile_elts]()
    var dst_v = dst.vectorize[1, 1, base_tile_elts]()
    comptime assert src_v.flat_rank == 3
    comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
    comptime assert vec.flat_rank == 2

    comptime for j in range(src_width):
        var v_inv = src_v.ElementType(recip(vec[j, 0][0]))
        comptime for i in range(src_height):
            dst_v[i, j, 0] = src_v[i, j, 0] * v_inv


# `sub_col_inplace` and `mul_col_inplace` share an iteration shape; a
# shared op-parameterized helper would carry layout-typed parameters
# that don't compose with the public generic signatures, so the bodies
# are inlined.


@always_inline
def sub_col_inplace(
    mut dst: RegTile[DType.float32, ...],
    vec: RegTile[DType.float32, ...],
):
    """`dst[gr, gc] -= vec[gc]` in place — `att_block - max_vec` before
    `exp2` in online softmax."""
    comptime src_height = dst.static_shape[0]
    comptime src_width = dst.static_shape[1]
    comptime base_tile_elts = dst.static_shape[2]
    comptime assert (
        vec.static_shape[0] == src_width
    ), "sub_col_inplace: rv outer_dim must equal rt.width"
    comptime assert (
        vec.static_shape[1] == 1
    ), "sub_col_inplace: rv inner_dim must be 1"

    var dst_v = dst.vectorize[1, 1, base_tile_elts]()
    comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
    comptime assert vec.flat_rank == 2

    comptime for j in range(src_width):
        var v_simd = dst_v.ElementType(vec[j, 0][0])
        comptime for i in range(src_height):
            dst_v[i, j, 0] = dst_v[i, j, 0] - v_simd


@always_inline
def mul_col_inplace[
    dtype: DType
](mut dst: RegTile[dtype, ...], vec: RegTile[DType.float32, ...],):
    """`dst[gr, gc] *= vec[gc]` in place — rescale step of online
    softmax (`o_reg *= exp2(max_prev - max_new)` when the running max
    grows). `dst` may be FP32 (the `o_reg` accumulator) or BF16 (the
    pre-cast `att_bf16_full` consumed by the PV MFMA strips that need
    the same scale flip as `o_reg`)."""
    comptime src_height = dst.static_shape[0]
    comptime src_width = dst.static_shape[1]
    comptime base_tile_elts = dst.static_shape[2]
    comptime assert (
        vec.static_shape[0] == src_width
    ), "mul_col_inplace: rv outer_dim must equal rt.width"
    comptime assert (
        vec.static_shape[1] == 1
    ), "mul_col_inplace: rv inner_dim must be 1"

    var dst_v = dst.vectorize[1, 1, base_tile_elts]()
    comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
    comptime assert vec.flat_rank == 2

    comptime for j in range(src_width):
        var v_simd = dst_v.ElementType(vec[j, 0][0])
        comptime for i in range(src_height):
            dst_v[i, j, 0] = dst_v[i, j, 0] * v_simd
