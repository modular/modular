# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from algorithm.functional import tile_and_unswitch

from math import div_ceil
from pathlib import Path
from sys.info import triple_is_nvidia_cuda
from sys.param_env import env_get_string

from gpu import ThreadIdx, BlockIdx, BlockDim, barrier, AddressSpace
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from math import align_down
from memory import memset_zero, stack_allocation
from memory.buffer import NDBuffer
from tensor import Tensor

from utils.index import Index
from utils.list import DimList


# Tile size for tiling in shared memory.
# Thread block would have shape (tile_size, tile_size, 1)
alias tile_size = 32


fn matmul_sram(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
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

    let a = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_ptr, Index(M, K)
    )
    let b = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        b_ptr, Index(K, N)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        c_ptr, Index(M, N)
    )

    # Allocate A, B tile in shared memory.
    let a_shared = stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()
    let b_shared = stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    let col = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let row = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    # Local index in the c sub-matrix updated by current block.
    let localCol = ThreadIdx.x()
    let localRow = ThreadIdx.y()

    # Result of current thread in C.
    var result = Float32(0.0)

    let K_roundbytile = align_down(K, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    let K_remainder = K - K_roundbytile if K - K_roundbytile > 0 else 1

    @parameter
    @always_inline
    fn update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        let a_val: Float32

        @parameter
        if not full_tile:
            a_val = a[row, offset + localCol] if (
                row < M and offset + localCol < K
            ) else 0.0
        else:
            a_val = a[row, offset + localCol] if row < M else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        let b_val: Float32

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
fn run_matmul() raises:
    print("== run_matmul_sram")

    # Should be able to handle non-divisible values.
    alias M = 513
    alias N = 502
    alias K = 511

    let stream = Stream()

    var a_host = Tensor[DType.float32](M, K)
    var b_host = Tensor[DType.float32](K, N)
    var c_host = Tensor[DType.float32](M, N)

    for i in range(M):
        for j in range(K):
            a_host[Index(i, j)] = 1

    for i in range(K):
        for j in range(N):
            b_host[Index(i, j)] = 1

    for i in range(M):
        for j in range(N):
            c_host[Index(i, j)] = 0

    let a_device = _malloc[Float32](M * K)
    let b_device = _malloc[Float32](K * N)
    let c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host.data(), M * K)
    _copy_host_to_device(b_device, b_host.data(), K * N)

    let func = Function[
        # fmt: off
      fn (DTypePointer[DType.float32],
          DTypePointer[DType.float32],
          DTypePointer[DType.float32],
          Int, Int, Int) -> None,
        # fmt: on
        matmul_sram
    ]()

    func(
        stream,
        (div_ceil(N, tile_size), div_ceil(M, tile_size)),
        (tile_size, tile_size),
        a_device,
        b_device,
        c_device,
        M,
        N,
        K,
    )
    synchronize()

    _copy_device_to_host(c_host.data(), c_device, M * N)

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

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
    except e:
        print("CUDA_ERROR:", e)
