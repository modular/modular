# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from math import ceildiv

from buffer import DimList, NDBuffer
from gpu import (
    AddressSpace,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
)
from gpu.host import DeviceContext
from memory import UnsafePointer, memset_zero, stack_allocation

from utils.index import Index

alias TILE_SZ_A = 128
alias TILE_SZ_B = 16
alias TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B


fn matmul(
    a_ptr: UnsafePointer[Scalar[DType.index]],
    b_ptr: UnsafePointer[Scalar[DType.index]],
    c_ptr: UnsafePointer[Scalar[DType.index]],
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
    var row: UInt = global_idx.x
    var col: UInt = block_idx.y * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = stack_allocation[TILE_SZ_B, DType.index]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i: UInt = thread_idx.x // TILE_SZ_B
        var j: UInt = thread_idx.x % TILE_SZ_B

        # Load the B matrix into shared memory.
        var b_val = Int(b[tile_idx * TILE_SZ_RATIO + Int(i), Int(col) + Int(j)])
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Int
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = Int(a[row, tile_idx * TILE_SZ_RATIO + idx])
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
fn run_matmul(ctx: DeviceContext) raises:
    print("== run_matmul")

    alias m = 512
    alias n = 512
    alias k = 512

    var a_host_ptr = UnsafePointer[Scalar[DType.index]].alloc(m * k)
    var b_host_ptr = UnsafePointer[Scalar[DType.index]].alloc(k * n)
    var c_host_ptr = UnsafePointer[Scalar[DType.index]].alloc(m * n)

    var a_host = NDBuffer[DType.index, 2, _, DimList(m, k)](a_host_ptr)
    var b_host = NDBuffer[DType.index, 2, _, DimList(k, n)](b_host_ptr)
    var c_host = NDBuffer[DType.index, 2, _, DimList(m, n)](c_host_ptr)

    for i in range(m):
        for j in range(k):
            a_host[i, j] = 1

    for i in range(k):
        for j in range(n):
            b_host[i, j] = 1

    for i in range(m):
        for j in range(n):
            c_host[i, j] = 0

    var a_device = ctx.enqueue_create_buffer[DType.index](m * k)
    var b_device = ctx.enqueue_create_buffer[DType.index](k * n)
    var c_device = ctx.enqueue_create_buffer[DType.index](m * n)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_function[matmul](
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    for i in range(10):
        for j in range(10):
            print("at index = [", i, ",", j, "]the value is", c_host[i, j])

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host


def main():
    with DeviceContext() as ctx:
        run_matmul(ctx)
