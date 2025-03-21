# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import align_down, ceildiv

from algorithm.functional import tile_and_unswitch
from buffer import DimList, NDBuffer
from gpu import barrier, block_dim, block_idx, global_idx, thread_idx
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import UnsafePointer, stack_allocation

from utils.index import Index

# Tile size for tiling in shared memory.
# Thread block would have shape (tile_size, tile_size, 1)
alias tile_size = 32


fn matmul_sram(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    c_ptr: UnsafePointer[Float32],
    M: Int,
    N: Int,
    K: Int,
):
    """Matrix Multiplication using shared memory.
    This version loads blocks of size tile_size x tile_size from A and B
    and updates a tile_size x tile_size in C.

    The thread block should have shape (tile_size, tile_size, 1). Each
    thread is mapped one element in C. The grid should have shape
    (N/tile_size, M/tile_size, 1). N is the first dimension for coalesced
    access.
    """

    var a = NDBuffer[DType.float32, 2](a_ptr, Index(M, K))
    var b = NDBuffer[DType.float32, 2](b_ptr, Index(K, N))
    var c = NDBuffer[DType.float32, 2](c_ptr, Index(M, N))

    # Allocate A, B tile in shared memory.
    var a_shared = stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    var col = global_idx.x
    var row = global_idx.y

    # Local index in the c sub-matrix updated by current block.
    var localCol = thread_idx.x
    var localRow = thread_idx.y

    # Result of current thread in C.
    var result = Float32(0.0)

    var K_roundbytile = align_down(K, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    var K_remainder = K - K_roundbytile if K - K_roundbytile > 0 else 1

    @parameter
    @__copy_capture(localCol, a, row, a_shared, localRow, col, b, b_shared)
    @always_inline
    fn update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: Float32

        @parameter
        if not full_tile:
            a_val = a[row, offset + localCol] if (
                row < M and offset + localCol < K
            ) else 0.0
        else:
            a_val = a[row, offset + localCol] if row < M else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Float32

        @parameter
        if not full_tile:
            b_val = b[offset + localRow, col] if (
                col < N and offset + localRow < K
            ) else 0.0
        else:
            b_val = b[offset + localRow, col] if col < N else 0.0
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for k in range(tile_size):
            result += a_shared.load(localRow * tile_size + k) * b_shared.load(
                k * tile_size + localCol
            )

        barrier()

    tile_and_unswitch[update_tile](
        0, K, VariadicList[Int](tile_size, K_remainder)
    )

    if row < M and col < N:
        c[Index(row, col)] = result


# CHECK-LABEL: run_matmul_sram
fn run_matmul(ctx: DeviceContext) raises:
    print("== run_matmul_sram")

    # Should be able to handle non-divisible values.
    alias M = 513
    alias N = 502
    alias K = 511

    var a_host_ptr = UnsafePointer[Float32].alloc(M * K)
    var a_host = NDBuffer[DType.float32, 2, _, DimList(M, K)](a_host_ptr)
    var b_host_ptr = UnsafePointer[Float32].alloc(K * N)
    var b_host = NDBuffer[DType.float32, 2, _, DimList(K, N)](b_host_ptr)
    var c_host_ptr = UnsafePointer[Float32].alloc(M * N)
    var c_host = NDBuffer[DType.float32, 2, _, DimList(M, N)](c_host_ptr)

    for i in range(M):
        for j in range(K):
            a_host[Index(i, j)] = 1

    for i in range(K):
        for j in range(N):
            b_host[Index(i, j)] = 1

    for i in range(M):
        for j in range(N):
            c_host[Index(i, j)] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_function[matmul_sram](
        a_device,
        b_device,
        c_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(N, tile_size), ceildiv(M, tile_size)),
        block_dim=(tile_size, tile_size),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    var failed = False
    for i in range(M - 10, M):
        for j in range(N - 10, N):
            if c_host[i, j] != Float32(K):
                print(
                    "Fail at index = [",
                    i,
                    ",",
                    j,
                    "] the value is",
                    c_host[i, j],
                    "the golden value is",
                    K,
                )
                failed = True

    # CHECK: succeed
    if not failed:
        print("succeed")

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host


def main():
    with DeviceContext() as ctx:
        run_matmul(ctx)
