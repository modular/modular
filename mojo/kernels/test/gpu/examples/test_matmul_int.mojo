# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv

from buffer import NDBuffer, DimList
from gpu import AddressSpace, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from memory import memset_zero, stack_allocation

from utils.index import Index

alias TILE_SZ_A = 128
alias TILE_SZ_B = 16
alias TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B


fn matmul(
    a_ptr: DTypePointer[DType.index],
    b_ptr: DTypePointer[DType.index],
    c_ptr: DTypePointer[DType.index],
    m: Int,
    n: Int,
    k: Int,
):
    var a = NDBuffer[DType.index, 2](a_ptr, Index(m, k))
    var b = NDBuffer[DType.index, 2](b_ptr, Index(k, n))
    var c = NDBuffer[DType.index, 2](c_ptr, Index(m, n))

    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    # Allocate B array into shared memory for tiling.
    var b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.index,
        address_space = AddressSpace.SHARED,
    ]()

    # Thread indexing offsets.
    var row = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var col = BlockIdx.y() * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = stack_allocation[TILE_SZ_B, DType.index]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i = ThreadIdx.x() // TILE_SZ_B
        var j = ThreadIdx.x() % TILE_SZ_B

        # Load the B matrix into shared memory.
        var b_val = int(b[tile_idx * TILE_SZ_RATIO + i, col + j])
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Int
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = int(a[row, tile_idx * TILE_SZ_RATIO + idx])
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg[out_idx] += (
                    a_reg * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            c[Index(row, col + out_idx)] = c_reg.load(out_idx)


# CHECK-LABEL: run_matmul
fn run_matmul() raises:
    print("== run_matmul")

    alias m = 512
    alias n = 512
    alias k = 512

    var stream = Stream()

    var a_host = NDBuffer[DType.index, 2, DimList(m, k)].stack_allocation()
    var b_host = NDBuffer[DType.index, 2, DimList(k, n)].stack_allocation()
    var c_host = NDBuffer[DType.index, 2, DimList(m, n)].stack_allocation()

    for i in range(m):
        for j in range(k):
            a_host[Index(i, j)] = 1

    for i in range(k):
        for j in range(n):
            b_host[Index(i, j)] = 1

    for i in range(m):
        for j in range(n):
            c_host[Index(i, j)] = 0

    var a_device = _malloc[DType.index](m * k)
    var b_device = _malloc[DType.index](k * n)
    var c_device = _malloc[DType.index](m * n)

    _copy_host_to_device(a_device, a_host.data, m * k)
    _copy_host_to_device(b_device, b_host.data, k * n)

    var func = Function[matmul]()

    func(
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
        stream=stream,
    )
    synchronize()

    _copy_device_to_host(c_host.data, c_device, m * n)

    for i in range(10):
        for j in range(10):
            print("at index = [", i, ",", j, "]the value is", c_host[i, j])

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
    except e:
        print("CUDA_ERROR:", e)
