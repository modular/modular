# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for doing matrix-multiply-accumulate (mma)
using tensor cores.
PTX documentation => https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
"""


@adaptive
@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    inout a: SIMD[DType.float32, 4],
    a_ptr: DTypePointer[DType.float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    constrained[m == 16 and n == 8 and k == 8]()
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    let a02_row = group_id
    let a01_col = group_lane_id
    let a13_row = group_id + 8
    let a23_col = group_lane_id + 4

    a[0] = a_ptr.load((tile_row + a02_row) * ldm + (tile_col + a01_col))
    a[1] = a_ptr.load((tile_row + a13_row) * ldm + (tile_col + a01_col))
    a[2] = a_ptr.load((tile_row + a02_row) * ldm + (tile_col + a23_col))
    a[3] = a_ptr.load((tile_row + a13_row) * ldm + (tile_col + a23_col))


@adaptive
@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    inout a: SIMD[DType.float16, 4],
    a_ptr: DTypePointer[DType.float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    constrained[m == 16 and n == 8 and k == 8]()
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    let a01_row = group_id
    let a0_col = (group_lane_id * 2) + (0 & 0x1)
    let a1_col = (group_lane_id * 2) + (1 & 0x1)
    let a23_row = group_id + 8
    let a2_col = (group_lane_id * 2) + (2 & 0x1)
    let a3_col = (group_lane_id * 2) + (3 & 0x1)

    a[0] = a_ptr.load((tile_row + a01_row) * ldm + (tile_col + a0_col))
    a[1] = a_ptr.load((tile_row + a01_row) * ldm + (tile_col + a1_col))
    a[2] = a_ptr.load((tile_row + a23_row) * ldm + (tile_col + a2_col))
    a[3] = a_ptr.load((tile_row + a23_row) * ldm + (tile_col + a3_col))


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    inout b: SIMD[DType.float32, 2],
    b_ptr: DTypePointer[DType.float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    constrained[m == 16 and n == 8 and k == 8]()
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    let b0_row = group_lane_id
    let b01_col = group_id
    let b1_row = group_lane_id + 4

    b[0] = b_ptr.load((tile_row + b0_row) * ldm + (tile_col + b01_col))
    b[1] = b_ptr.load((tile_row + b1_row) * ldm + (tile_col + b01_col))


@adaptive
@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    inout b: SIMD[DType.float16, 2],
    b_ptr: DTypePointer[DType.float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    constrained[m == 16 and n == 8 and k == 8]()
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    let b0_row = (group_lane_id * 2) + (0 & 0x1)
    let b01_col = group_id
    let b1_row = (group_lane_id * 2) + (1 & 0x1)

    b[0] = b_ptr.load((tile_row + b0_row) * ldm + (tile_col + b01_col))
    b[1] = b_ptr.load((tile_row + b1_row) * ldm + (tile_col + b01_col))


@adaptive
@always_inline
fn store_matrix_d[
    dtype: DType, m: Int, n: Int, k: Int
](
    d_ptr: DTypePointer[dtype],
    d: SIMD[dtype, 4],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    let d01_row = group_id
    let d0_col = (group_lane_id * 2) + (0 & 0x1)
    let d1_col = (group_lane_id * 2) + (1 & 0x1)
    let d23_row = group_id + 8
    let d2_col = (group_lane_id * 2) + (2 & 0x1)
    let d3_col = (group_lane_id * 2) + (3 & 0x1)

    d_ptr.store((tile_row + d01_row) * ldm + (tile_col + d0_col), d[0])
    d_ptr.store((tile_row + d01_row) * ldm + (tile_col + d1_col), d[1])
    d_ptr.store((tile_row + d23_row) * ldm + (tile_col + d2_col), d[2])
    d_ptr.store((tile_row + d23_row) * ldm + (tile_col + d3_col), d[3])
