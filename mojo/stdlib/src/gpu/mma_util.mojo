# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides abstractions for doing matrix-multiply-accumulate (mma)
using tensor cores.
PTX documentation => https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
"""


@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    a_ptr: DTypePointer[DType.float32], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.float32, 4]:
    """
    For shape m16n8k8 type tf32 loads matrix A tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var a02_row = group_id
    var a01_col = group_lane_id
    var a13_row = group_id + 8
    var a23_col = group_lane_id + 4

    var a = SIMD[DType.float32, 4]()
    a[0] = a_ptr[(tile_row + a02_row) * ldm + (tile_col + a01_col)]
    a[1] = a_ptr[(tile_row + a13_row) * ldm + (tile_col + a01_col)]
    a[2] = a_ptr[(tile_row + a02_row) * ldm + (tile_col + a23_col)]
    a[3] = a_ptr[(tile_row + a13_row) * ldm + (tile_col + a23_col)]
    return a


@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    a_ptr: DTypePointer[DType.float16], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.float16, 4]:
    """
    For shape m16n8k8 & type fp16 loads matrix A tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var a01_row = group_id
    var a0_col = (group_lane_id * 2) + (0 & 0x1)
    var a1_col = (group_lane_id * 2) + (1 & 0x1)
    var a23_row = group_id + 8
    var a2_col = (group_lane_id * 2) + (2 & 0x1)
    var a3_col = (group_lane_id * 2) + (3 & 0x1)

    var a = SIMD[DType.float16, 4]()
    a[0] = a_ptr[(tile_row + a01_row) * ldm + (tile_col + a0_col)]
    a[1] = a_ptr[(tile_row + a01_row) * ldm + (tile_col + a1_col)]
    a[2] = a_ptr[(tile_row + a23_row) * ldm + (tile_col + a2_col)]
    a[3] = a_ptr[(tile_row + a23_row) * ldm + (tile_col + a3_col)]
    return a


@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    a_ptr: DTypePointer[DType.bfloat16], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.bfloat16, 4]:
    """
    For shape m16n8k8 & type bfp16 loads matrix A tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var a01_row = group_id
    var a0_col = (group_lane_id * 2) + (0 & 0x1)
    var a1_col = (group_lane_id * 2) + (1 & 0x1)
    var a23_row = group_id + 8
    var a2_col = (group_lane_id * 2) + (2 & 0x1)
    var a3_col = (group_lane_id * 2) + (3 & 0x1)

    var a = SIMD[DType.bfloat16, 4]()
    a[0] = a_ptr[(tile_row + a01_row) * ldm + (tile_col + a0_col)]
    a[1] = a_ptr[(tile_row + a01_row) * ldm + (tile_col + a1_col)]
    a[2] = a_ptr[(tile_row + a23_row) * ldm + (tile_col + a2_col)]
    a[3] = a_ptr[(tile_row + a23_row) * ldm + (tile_col + a3_col)]
    return a


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: DTypePointer[DType.float32], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.float32, 2]:
    """
    For shape m16n8k8 & type tf32 loads matrix B tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var b0_row = group_lane_id
    var b01_col = group_id
    var b1_row = group_lane_id + 4

    var b = SIMD[DType.float32, 2]()
    b[0] = b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)]
    b[1] = b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)]
    return b


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: DTypePointer[DType.float16], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.float16, 2]:
    """
    For shape m16n8k8 & type fp16 loads matrix B tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var b0_row = (group_lane_id * 2) + (0 & 0x1)
    var b01_col = group_id
    var b1_row = (group_lane_id * 2) + (1 & 0x1)

    var b = SIMD[DType.float16, 2]()
    b[0] = b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)]
    b[1] = b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)]
    return b


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: DTypePointer[DType.bfloat16], tile_row: Int, tile_col: Int, ldm: Int
) -> SIMD[DType.bfloat16, 2]:
    """
    For shape m16n8k8 & type bfp16 loads matrix B tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    constrained[m == 16 and n == 8 and k == 8]()
    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var b0_row = (group_lane_id * 2) + (0 & 0x1)
    var b01_col = group_id
    var b1_row = (group_lane_id * 2) + (1 & 0x1)

    var b = SIMD[DType.bfloat16, 2]()
    b[0] = b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)]
    b[1] = b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)]
    return b


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
    """
    For shape m16n8k8 stores matrix D tile from registers to memory in specific order after performing
    tensor core based warp sync mma op.
    """

    var group_id = lane_id() >> 2
    var group_lane_id = lane_id() % 4

    var d01_row = group_id
    var d0_col = (group_lane_id * 2) + (0 & 0x1)
    var d1_col = (group_lane_id * 2) + (1 & 0x1)
    var d23_row = group_id + 8
    var d2_col = (group_lane_id * 2) + (2 & 0x1)
    var d3_col = (group_lane_id * 2) + (3 & 0x1)

    d_ptr[(tile_row + d01_row) * ldm + (tile_col + d0_col)] = d[0]
    d_ptr[(tile_row + d01_row) * ldm + (tile_col + d1_col)] = d[1]
    d_ptr[(tile_row + d23_row) * ldm + (tile_col + d2_col)] = d[2]
    d_ptr[(tile_row + d23_row) * ldm + (tile_col + d3_col)] = d[3]
