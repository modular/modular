# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Matrix multiply accumulate (MMA) utilities for GPU tensor cores.

This module provides functions for loading matrix tiles from memory into registers and storing
results back to memory when using tensor cores for matrix multiplication. It supports both
NVIDIA and AMD GPUs with functions specialized for different data types (FP32, FP16, BF16).

The key functions are:

- load_matrix_a: Loads tiles from the first input matrix A
- load_matrix_b: Loads tiles from the second input matrix B
- store_matrix_d: Stores result tiles to the output matrix D

Each function handles the specific memory access patterns required by the tensor core
instructions on each GPU architecture. The tile sizes and data layouts match the hardware
requirements documented in:

NVIDIA PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
AMD Matrix Cores: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
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
    """Loads a tile of matrix A from memory to registers for TF32 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 4 TF32 values loaded from matrix A in the required order.

    Constraints:
        The tile demensions must be m=16, n=8, k=8.
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
    """Loads a tile of matrix A from memory to registers for FP16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 4 FP16 values loaded from matrix A in the required order.

    Constraints:
        The tile demensions must be m=16, n=8, k=8.
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
    """Loads a tile of matrix A from memory to registers for BF16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing k//2 BF16 values loaded from matrix A in the required order.

    Constraints:
        The tile dimensions must be m=16, n=8, k=8 or m=16, n=8, k=16.
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
    """Loads a tile of matrix A from memory to registers for AMD FP32 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 1 FP32 value loaded from matrix A.

    Constraints:
        The tile dimensions must be m=16, n=16, k=4.
    """

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
    """Loads a tile of matrix A from memory to registers for AMD FP16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 4 FP16 values loaded from matrix A.

    Constraints:
        The tile dimensions must be m=16, n=16, k=16.
    """

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
    """Loads a tile of matrix A from memory to registers for AMD BF16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 4 BF16 values loaded from matrix A.

    Constraints:
        The tile dimensions must be m=16, n=16, k=16.
    """

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
    """Loads a tile of matrix B from memory to registers for TF32 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 2 TF32 values loaded from matrix B in the required order.

    Constraints:
        The tile dimensions must be m=16, n=8, k=8.
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
    """Loads a tile of matrix B from memory to registers for FP16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 2 FP16 values loaded from matrix B in the required order.

    Constraints:
        The tile dimensions must be m=16, n=8, k=8.
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
    """Loads a tile of matrix B from memory to registers for BF16 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing k//4 BF16 values loaded from matrix B in the required order.

    Constraints:
        The tile dimensions must be m=16, n=8, k=8 or m=16, n=8, k=16.
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
    """Loads a tile of matrix B from memory to registers for AMD FP32 tensor core operations.

    Parameters:
        m: Number of rows in the output matrix tile.
        n: Number of columns in the output matrix tile.
        k: Inner dimension for matrix multiplication.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 1 FP32 value loaded from matrix B.
    """

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
    """NVIDIA-specific implementation for storing matrix D tile from registers to memory.

    This function handles the specific memory layout required by NVIDIA tensor cores after
    performing a warp-synchronized matrix multiply-accumulate operation. For shape m16n8k8,
    it stores the 4 elements per thread in a specific order based on the thread's position
    in the warp.

    Parameters:
        dtype: Data type of the matrix elements.
        m: Number of rows in matrix D.
        n: Number of columns in matrix D.
        k: Inner dimension for matrix multiply.

    Args:
        d_ptr: Pointer to destination memory for matrix D.
        d: SIMD vector containing 4 elements to store.
        tile_row: Starting row index of the tile in matrix D.
        tile_col: Starting column index of the tile in matrix D.
        ldm: Leading dimension (stride) of matrix D.

    Note:
        - Thread mapping follows NVIDIA's tensor core layout.
        - Each thread stores 4 elements in specific positions based on warp lane ID.
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
    """AMD-specific implementation for storing matrix D tile from registers to memory.

    This function handles the memory layout required by AMD tensor cores after performing
    a warp-synchronized matrix multiply-accumulate operation. It stores 4 elements per
    thread in a linear layout based on the thread's position in the warp.

    Parameters:
        dtype: Data type of the matrix elements.
        m: Number of rows in matrix D.
        n: Number of columns in matrix D.
        k: Inner dimension for matrix multiply.

    Args:
        d_ptr: Pointer to destination memory for matrix D.
        d: SIMD vector containing 4 elements to store.
        tile_row: Starting row index of the tile in matrix D.
        tile_col: Starting column index of the tile in matrix D.
        ldm: Leading dimension (stride) of matrix D.

    Note:
        - Thread mapping follows AMD's tensor core layout.
        - Each thread stores 4 elements in consecutive positions.
    """

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
    """Stores matrix D tile from registers to memory after tensor core operation.

    This function dispatches to architecture-specific implementations for storing the
    results of a tensor core matrix multiply-accumulate operation. It handles the
    different memory layouts required by NVIDIA and AMD tensor cores.

    Parameters:
        dtype: Data type of the matrix elements.
        m: Number of rows in matrix D.
        n: Number of columns in matrix D.
        k: Inner dimension for matrix multiply.

    Args:
        d_ptr: Pointer to destination memory for matrix D.
        d: SIMD vector containing 4 elements to store.
        tile_row: Starting row index of the tile in matrix D.
        tile_col: Starting column index of the tile in matrix D.
        ldm: Leading dimension (stride) of matrix D.

    Note:
        - Automatically selects appropriate implementation based on GPU architecture.
        - Each thread stores 4 elements in architecture-specific positions.
        - Must be called by all threads in a warp.
    """

    @parameter
    if is_nvidia_gpu():
        _store_matrix_d_nvidia[m, n, k](d_ptr, d, tile_row, tile_col, ldm)
    else:
        _store_matrix_d_amd[m, n, k](d_ptr, d, tile_row, tile_col, ldm)
