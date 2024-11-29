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
    a_ptr: UnsafePointer[Float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
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

    return SIMD[DType.float32, 4](
        a_ptr[(tile_row + a02_row) * ldm + (tile_col + a01_col)],
        a_ptr[(tile_row + a13_row) * ldm + (tile_col + a01_col)],
        a_ptr[(tile_row + a02_row) * ldm + (tile_col + a23_col)],
        a_ptr[(tile_row + a13_row) * ldm + (tile_col + a23_col)],
    )


@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
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

    return SIMD[DType.float16, 4](
        a_ptr[(tile_row + a01_row) * ldm + (tile_col + a0_col)],
        a_ptr[(tile_row + a01_row) * ldm + (tile_col + a1_col)],
        a_ptr[(tile_row + a23_row) * ldm + (tile_col + a2_col)],
        a_ptr[(tile_row + a23_row) * ldm + (tile_col + a3_col)],
    )


@always_inline
fn load_matrix_a[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, k // 2]:
    """
    For type bfp16 loads matrix A tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    @parameter
    if m == 16 and n == 8 and k == 8:
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        var a01_row = group_id
        var a0_col = (group_lane_id * 2) + (0 & 0x1)
        var a1_col = (group_lane_id * 2) + (1 & 0x1)
        var a23_row = group_id + 8
        var a2_col = (group_lane_id * 2) + (2 & 0x1)
        var a3_col = (group_lane_id * 2) + (3 & 0x1)

        return SIMD[DType.bfloat16, k // 2](
            a_ptr[(tile_row + a01_row) * ldm + (tile_col + a0_col)],
            a_ptr[(tile_row + a01_row) * ldm + (tile_col + a1_col)],
            a_ptr[(tile_row + a23_row) * ldm + (tile_col + a2_col)],
            a_ptr[(tile_row + a23_row) * ldm + (tile_col + a3_col)],
        )
    else:
        constrained[m == 16 and n == 8 and k == 16]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        var a_row_0 = group_id
        var a_row_1 = group_id + 8

        var a_col_0 = (group_lane_id * 2) + (0 & 0x1)
        var a_col_1 = (group_lane_id * 2) + (1 & 0x1)
        var a_col_2 = (group_lane_id * 2) + (2 & 0x1)
        var a_col_3 = (group_lane_id * 2) + (3 & 0x1)
        var a_col_4 = (group_lane_id * 2) + (4 & 0x1) + 8
        var a_col_5 = (group_lane_id * 2) + (5 & 0x1) + 8
        var a_col_6 = (group_lane_id * 2) + (6 & 0x1) + 8
        var a_col_7 = (group_lane_id * 2) + (7 & 0x1) + 8

        var a = SIMD[DType.bfloat16, k // 2]()
        a[0] = a_ptr[(tile_row + a_row_0) * ldm + (tile_col + a_col_0)]
        a[1] = a_ptr[(tile_row + a_row_0) * ldm + (tile_col + a_col_1)]
        a[2] = a_ptr[(tile_row + a_row_1) * ldm + (tile_col + a_col_2)]
        a[3] = a_ptr[(tile_row + a_row_1) * ldm + (tile_col + a_col_3)]

        a[4] = a_ptr[(tile_row + a_row_0) * ldm + (tile_col + a_col_4)]
        a[5] = a_ptr[(tile_row + a_row_0) * ldm + (tile_col + a_col_5)]
        a[6] = a_ptr[(tile_row + a_row_1) * ldm + (tile_col + a_col_6)]
        a[7] = a_ptr[(tile_row + a_row_1) * ldm + (tile_col + a_col_7)]
        return a


@always_inline
fn load_matrix_a_amd[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[Float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float32, 1]:
    constrained[m == 16 and n == 16 and k == 4]()
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4
    return a_ptr[ldm * (tile_row + thread_x) + tile_col + thread_y]


@always_inline
fn load_matrix_a_amd[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 4]:
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4
    var a = SIMD[DType.float16, 4]()

    @parameter
    for i in range(4):
        var a_idx = ldm * (tile_row + thread_x) + tile_col + i + 4 * thread_y
        a[i] = a_ptr[a_idx]

    return a


@always_inline
fn load_matrix_a_amd[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 4]:
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4
    var a = SIMD[DType.bfloat16, 4]()

    @parameter
    for i in range(4):
        var a_idx = ldm * (tile_row + thread_x) + tile_col + i + 4 * thread_y
        a[i] = a_ptr[a_idx]

    return a


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[Float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
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

    return SIMD[DType.float32, 2](
        b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)],
        b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)],
    )


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
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

    return SIMD[DType.float16, 2](
        b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)],
        b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)],
    )


@always_inline
fn load_matrix_b[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, k // 4]:
    """
    For type bfp16 loads matrix B tile from memory to registers in specific order to be used
    by tensor cores to perform a warp sync mma op.
    """

    @parameter
    if m == 16 and n == 8 and k == 8:
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        var b0_row = (group_lane_id * 2) + (0 & 0x1)
        var b01_col = group_id
        var b1_row = (group_lane_id * 2) + (1 & 0x1)

        return SIMD[DType.bfloat16, k // 4](
            b_ptr[(tile_row + b0_row) * ldm + (tile_col + b01_col)],
            b_ptr[(tile_row + b1_row) * ldm + (tile_col + b01_col)],
        )
    else:
        constrained[m == 16 and n == 8 and k == 16]()
        var group_id = lane_id() >> 2
        var group_lane_id = lane_id() % 4

        var b_row_0 = (group_lane_id * 2) + (0 & 0x1)
        var b_row_1 = (group_lane_id * 2) + (1 & 0x1)
        var b_row_2 = (group_lane_id * 2) + (2 & 0x1) + 8
        var b_row_3 = (group_lane_id * 2) + (3 & 0x1) + 8
        var b_col = group_id

        return SIMD[DType.bfloat16, k // 4](
            b_ptr[(tile_row + b_row_0) * ldm + (tile_col + b_col)],
            b_ptr[(tile_row + b_row_1) * ldm + (tile_col + b_col)],
            b_ptr[(tile_row + b_row_2) * ldm + (tile_col + b_col)],
            b_ptr[(tile_row + b_row_3) * ldm + (tile_col + b_col)],
        )


@always_inline
fn load_matrix_b_amd[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[Float32],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float32, 1]:
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4
    return b_ptr[ldm * (tile_row + thread_y) + tile_col + thread_x]


@always_inline
fn load_matrix_b_amd[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 4]:
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4

    var b = SIMD[DType.float16, 4]()

    @parameter
    for i in range(4):
        var b_idx = ldm * (tile_row + 4 * thread_y + i) + tile_col + thread_x
        b[i] = b_ptr[b_idx]

    return b


@always_inline
fn load_matrix_b_amd[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 4]:
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4

    var b = SIMD[DType.bfloat16, 4]()

    @parameter
    for i in range(4):
        var b_idx = ldm * (tile_row + 4 * thread_y + i) + tile_col + thread_x
        b[i] = b_ptr[b_idx]

    return b


@always_inline
fn _store_matrix_d_nvidia[
    dtype: DType, //, m: Int, n: Int, k: Int
](
    d_ptr: UnsafePointer[Scalar[dtype]],
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


@always_inline
fn _store_matrix_d_amd[
    dtype: DType, //, m: Int, n: Int, k: Int
](
    d_ptr: UnsafePointer[Scalar[dtype]],
    d: SIMD[dtype, 4],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    var lane = lane_id()
    var thread_x = lane & 15
    var thread_y = lane >> 4

    @parameter
    for i in range(4):
        d_ptr[ldm * (tile_row + 4 * thread_y + i) + tile_col + thread_x] = d[i]


@always_inline
fn store_matrix_d[
    dtype: DType, //, m: Int, n: Int, k: Int
](
    d_ptr: UnsafePointer[Scalar[dtype]],
    d: SIMD[dtype, 4],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
):
    """
    Stores matrix D tile from registers to memory in specific order after performing
    tensor core based warp sync mma op.
    """

    @parameter
    if is_nvidia_gpu():
        _store_matrix_d_nvidia[m, n, k](d_ptr, d, tile_row, tile_col, ldm)
    else:
        _store_matrix_d_amd[m, n, k](d_ptr, d, tile_row, tile_col, ldm)
