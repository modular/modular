# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu import AddressSpace
from gpu.host import Context, Function, synchronize
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from gpu.memory import async_copy_wait_all
from layout import *
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.int_tuple import int


fn async_copy_kernel[
    input_layout: Layout,
    BM: Int,
    BN: Int,
](input: LayoutTensor[input_layout, DType.float32]):
    var input_tile = input.tile[BM, BN](BlockIdx.y(), BlockIdx.x())

    var smem_tile = LayoutTensor[
        Layout(IntTuple(BM, BN)),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async(input_tile)
    async_copy_wait_all()

    var tx = ThreadIdx.x()
    var ty = ThreadIdx.y()
    smem_tile[tx, ty] += ty

    input_tile.copy_from_numa(smem_tile)


fn test_async_copy() raises:
    print("=== test_async_copy")
    # Matrix dimension
    alias M = 6
    alias N = 6
    # Block dimension
    alias BM = 2
    alias BN = 3

    alias input_layout = Layout(IntTuple(M, N), IntTuple(N, 1))
    var input = ManagedLayoutTensor[
        input_layout, DType.float32, gpu_managed_alloc, gpu_free
    ]()

    input.tensor.linspace()

    alias kernel_type = async_copy_kernel[input_layout, BM, BN]

    var kernel = Function[__type_of(kernel_type), kernel_type]()

    kernel(
        input,
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    synchronize()
    input.tensor.print()

    _ = input^


fn main() raises:
    with Context() as ctx:
        # CHECK: === test_async_copy
        # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
        # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
        # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
        # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
        # CHECK: 24.0   26.0   28.0   27.0   28.0   29.0
        # CHECK: 30.0   31.0   32.0   33.0   34.0   35.0
        test_async_copy()
