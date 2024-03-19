# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil

from buffer import NDBuffer
from buffer.list import DimList
from gpu import BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import AddressSpace
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer
from tensor import Tensor

from utils.index import Index

alias BLOCK_DIM = 4


fn stencil2d(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    arr_size: Int,
    num_rows: Int,
    num_cols: Int,
    coeff0: Int,
    coeff1: Int,
    coeff2: Int,
    coeff3: Int,
    coeff4: Int,
):
    var tidx = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var tidy = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    var a = NDBuffer[DType.float32, 1](a_ptr, Index(arr_size))
    var b = NDBuffer[DType.float32, 1](b_ptr, Index(arr_size))

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        b[tidy * num_cols + tidx] = (
            coeff0 * a[tidy * num_cols + tidx - 1]
            + coeff1 * a[tidy * num_cols + tidx]
            + coeff2 * a[tidy * num_cols + tidx + 1]
            + coeff3 * a[(tidy - 1) * num_cols + tidx]
            + coeff4 * a[(tidy + 1) * num_cols + tidx]
        )


fn stencil2d_smem(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    arr_size: Int,
    num_rows: Int,
    num_cols: Int,
    coeff0: Int,
    coeff1: Int,
    coeff2: Int,
    coeff3: Int,
    coeff4: Int,
):
    var tidx = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var tidy = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()
    var lindex_x = ThreadIdx.x() + 1
    var lindex_y = ThreadIdx.y() + 1

    var a = NDBuffer[DType.float32, 1](a_ptr, Index(arr_size))
    var b = NDBuffer[DType.float32, 1](b_ptr, Index(arr_size))

    var a_shared = NDBuffer[
        DType.float32,
        2,
        DimList(BLOCK_DIM + 2, BLOCK_DIM + 2),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Each element is loaded in shared memory.
    a_shared[Index(lindex_y, lindex_x)] = a[tidy * num_cols + tidx]

    # First column also loads elements left and right to the block.
    if ThreadIdx.x() == 0:
        a_shared[Index(lindex_y, 0)] = a[tidy * num_cols + (tidx - 1)]
        a_shared[Index(lindex_y, BLOCK_DIM + 1)] = a[
            tidy * num_cols + tidx + BLOCK_DIM
        ]

    # First row also loads elements above and below the block.
    if ThreadIdx.y() == 0:
        a_shared[Index(0, lindex_x)] = a[(tidy - 1) * num_cols + tidx]
        a_shared[Index(BLOCK_DIM + 1, lindex_x)] = a[
            (tidy + BLOCK_DIM) * num_cols + tidx
        ]

    barrier()

    if tidy > 0 and tidx > 0 and tidy < num_rows - 1 and tidx < num_cols - 1:
        b[tidy * num_cols + tidx] = (
            coeff0 * a_shared[Index(lindex_y, lindex_x - 1)]
            + coeff1 * a_shared[Index(lindex_y, lindex_x)]
            + coeff2 * a_shared[Index(lindex_y, lindex_x + 1)]
            + coeff3 * a_shared[Index(lindex_y - 1, lindex_x)]
            + coeff4 * a_shared[Index(lindex_y + 1, lindex_x)]
        )


# CHECK-LABEL: run_stencil2d
fn run_stencil2d[smem: Bool]() raises:
    print("== run_stencil2d")

    alias m = 64
    alias coeff0 = 3
    alias coeff1 = 2
    alias coeff2 = 4
    alias coeff3 = 1
    alias coeff4 = 5
    alias iterations = 4

    alias num_rows = 8
    alias num_cols = 8

    var a_host = Tensor[DType.float32](m)
    var b_host = Tensor[DType.float32](m)

    var stream = Stream()

    for i in range(m):
        a_host[Index(i)] = i
        b_host[Index(i)] = 0

    var a_device = _malloc[Float32](m)
    var b_device = _malloc[Float32](m)

    _copy_host_to_device(a_device, a_host.data(), m)

    alias func_select = stencil2d_smem if smem == True else stencil2d

    var func = Function[__type_of(func_select), func_select]()

    for i in range(iterations):
        func(
            a_device,
            b_device,
            m,
            num_rows,
            num_cols,
            coeff0,
            coeff1,
            coeff2,
            coeff3,
            coeff4,
            grid_dim=(
                div_ceil(num_rows, BLOCK_DIM),
                div_ceil(num_cols, BLOCK_DIM),
            ),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )
        synchronize()

        var tmp_ptr = b_device
        b_device = a_device
        a_device = tmp_ptr

    _copy_device_to_host(b_host.data(), b_device, m)

    # CHECK: == run_stencil2d
    # CHECK: 37729.0 ,52628.0 ,57021.0 ,60037.0 ,58925.0 ,39597.0 ,
    # CHECK: 57888.0 ,80505.0 ,86322.0 ,89682.0 ,86994.0 ,57818.0 ,
    # CHECK: 76680.0 ,106488.0 ,113400.0 ,116775.0 ,112182.0 ,73933.0 ,
    # CHECK: 95424.0 ,132408.0 ,140400.0 ,143775.0 ,137262.0 ,89925.0 ,
    # CHECK: 91968.0 ,135753.0 ,144450.0 ,147450.0 ,138642.0 ,81842.0 ,
    # CHECK: 50277.0 ,73628.0 ,81985.0 ,83565.0 ,71417.0 ,43229.0 ,
    for i in range(1, num_rows - 1):
        for j in range(1, num_cols - 1):
            print(b_host[i * num_cols + j], ",", end="")
        print()

    _free(a_device)
    _free(b_device)

    _ = a_host
    _ = b_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_stencil2d[False]()
            run_stencil2d[True]()
    except e:
        print("CUDA_ERROR:", e)
