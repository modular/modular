# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: cuda
# RUN: %mojo %s | FileCheck %s

from gpu.host import Function, Context, synchronize
from gpu.id import BlockDim, ThreadIdx, BlockIdx
from gpu import AddressSpace
from gpu.sync import barrier

from kernel_utils._utils import ManagedLayoutTensor, gpu_managed_alloc, gpu_free

from kernel_utils.layout_tensor import LayoutTensor, stack_allocation_like
from kernel_utils.layout import Layout
from kernel_utils.int_tuple import IntTuple

from builtin.io import _printf


fn naive_matmul[
    layout_dst: Layout,
    layout_lhs: Layout,
    layout_rhs: Layout,
    BM: Int,
    BN: Int,
](
    dst: LayoutTensor[layout_dst, DType.float32],
    lhs: LayoutTensor[layout_dst, DType.float32],
    rhs: LayoutTensor[layout_dst, DType.float32],
):
    var dst_tile = dst.tile[BM, BN](BlockIdx.y(), BlockIdx.x())
    dst_tile[ThreadIdx.y(), ThreadIdx.x()] = 0
    for k in range(dst.shape[0]()):
        var lhs_tile = rhs.tile[BM, 1](BlockIdx.y(), k)
        var rhs_tile = lhs.tile[1, BN](k, BlockIdx.x())
        dst_tile[ThreadIdx.y(), ThreadIdx.x()] += (
            lhs_tile[ThreadIdx.y(), k] * rhs_tile[k, ThreadIdx.x()]
        )


fn test_naive_matmul_kernel() raises:
    print("=== test_naive_matmul_kernel")
    alias M = 8
    alias N = 8
    alias K = 8
    alias BM = 4
    alias BN = 4

    alias layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    var mat_a = ManagedLayoutTensor[
        layout_a, DType.float32, gpu_managed_alloc, gpu_free
    ]()
    var mat_b = ManagedLayoutTensor[
        layout_b, DType.float32, gpu_managed_alloc, gpu_free
    ]()
    var mat_c = ManagedLayoutTensor[
        layout_c, DType.float32, gpu_managed_alloc, gpu_free
    ]()

    mat_a.tensor.linspace()
    mat_b.tensor.linspace()
    mat_c.tensor.fill(0)

    alias naive_matmul_kernel = naive_matmul[
        layout_c, layout_a, layout_b, BM, BN
    ]

    var kernel = Function[__type_of(naive_matmul_kernel), naive_matmul_kernel]()
    kernel(
        mat_c,
        mat_a,
        mat_b,
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    synchronize()
    mat_c.tensor.print()


fn sram_blocked_matmul[
    layout_dst: Layout,
    layout_lhs: Layout,
    layout_rhs: Layout,
    thread_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    dst: LayoutTensor[layout_dst, DType.float32],
    lhs: LayoutTensor[layout_dst, DType.float32],
    rhs: LayoutTensor[layout_dst, DType.float32],
):
    # Allocate an SRAM tile of (BM, BK) size with row-major layout for the l.h.s.
    var lhs_sram_tile = LayoutTensor[
        Layout(IntTuple(BM, BK)),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Allocate an SRAM tile of (BK, BN) size with row-major layout for
    # the r.h.s.
    var rhs_sram_tile = LayoutTensor[
        Layout(IntTuple(BK, BN)),
        DType.float32,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Block the dst matrix with [BM, BN] tile size.
    var dst_tile = dst.tile[BM, BN](BlockIdx.y(), BlockIdx.x())

    # Distribute thread layout into a block of size [BM, BN], It repeats the
    # layout accross the BMxBN block, e.g row major layout will repeate as the
    # the following:
    # +---------------------------------BN-+----------------------------------+-------------
    # |  TH_(0, 0) TH_(0, 1) ... TH_(M, N) | TH_(0, 0) TH_(0, 1) ... TH_(M, N)| TH_(0, 0)...
    # |  TH_(1, 0) TH_(1, 1) ... TH_(1, N) | TH_(1, 0) TH_(1, 1) ... TH_(1, N)| TH_(1, 0)...
    # |      .               .             |                     .            |      .
    # |      .                 .           |                       .          |      .
    # BN TH_(M, 0) TH_(M, 1) ... TH_(M, N) | TH_(M, 0) TH_(M, 1) ... TH_(M, N)| TH_(M, 0)
    # +------------------------------------+----------------------------------+------------
    # |  TH_(0, 0) TH_(0, 1) ... TH_(M, N) | TH_(0, 0) TH_(0, 1) ... TH_(M, N)| TH_(0, 0)...
    # |      .        .      ...     .     |     .        .      ...      .          .
    # |      .        .      ...     .     |     .        .      ...      .          .
    var dst_local_tile = dst_tile.distribute[thread_layout](
        ThreadIdx.y(), ThreadIdx.x()
    )

    # Allocate a register tile for the dst matrix with the same layout.
    var dst_register_tile = stack_allocation_like(dst_local_tile)
    dst_register_tile.fill(0)

    for k in range(BK):
        # Block both l.h.s and r.h.s DRAM tensors.
        var lhs_tile = lhs.tile[BM, BK](BlockIdx.y(), k)
        var rhs_tile = rhs.tile[BK, BN](k, BlockIdx.x())

        # Distribute layout of threads into DRAM and SRAM to perform the copy.
        var lhs_tile_local = lhs_tile.distribute[thread_layout](
            ThreadIdx.y(), ThreadIdx.x()
        )
        var rhs_tile_local = rhs_tile.distribute[thread_layout](
            ThreadIdx.y(), ThreadIdx.x()
        )
        var lhs_sram_tile_local = lhs_sram_tile.distribute[thread_layout](
            ThreadIdx.y(), ThreadIdx.x()
        )
        var rhs_sram_tile_local = rhs_sram_tile.distribute[thread_layout](
            ThreadIdx.y(), ThreadIdx.x()
        )
        lhs_sram_tile_local.copy_from(lhs_tile_local)
        rhs_sram_tile_local.copy_from(rhs_tile_local)

        barrier()

        # Distribute thread layout into rows of l.h.s and cols of r.h.s
        # to perform dot mma instruction.
        alias TH_ROWS = thread_layout.shape[0]
        alias TH_COLS = thread_layout.shape[1]
        var lhs_sram_local = lhs_sram_tile.distribute[
            Layout(IntTuple(TH_ROWS, 1), IntTuple(1, 1))
        ](ThreadIdx.y(), 0)
        var rhs_sram_local = rhs_sram_tile.distribute[
            Layout(IntTuple(1, TH_COLS), IntTuple(1, 1))
        ](0, ThreadIdx.x())

        # Iterate over fragments of each thread.
        for m_i in range(dst_register_tile.shape[0]()):
            for n_i in range(dst_register_tile.shape[1]()):
                # Accumlate into the register tile.
                for ki in range(BK):
                    dst_register_tile[m_i, n_i] += (
                        lhs_sram_local[m_i, ki] * rhs_sram_local[ki, n_i]
                    )

    # Move data from register tile to DRAM
    dst_local_tile.copy_from(dst_register_tile)


fn test_sram_blocked_matmul() raises:
    print("=== test_sram_blocked_matmul")
    alias M = 8
    alias N = 8
    alias K = 8
    alias BM = 4
    alias BN = 4
    alias BK = 4

    alias TH_M = 2
    alias TH_N = 2

    alias layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    alias thread_layout = Layout(IntTuple(TH_M, TH_N), IntTuple(TH_N, 1))

    var mat_a = ManagedLayoutTensor[
        layout_a, DType.float32, gpu_managed_alloc, gpu_free
    ]()
    var mat_b = ManagedLayoutTensor[
        layout_b, DType.float32, gpu_managed_alloc, gpu_free
    ]()
    var mat_c = ManagedLayoutTensor[
        layout_c, DType.float32, gpu_managed_alloc, gpu_free
    ]()

    mat_a.tensor.linspace()
    mat_b.tensor.linspace()
    mat_c.tensor.fill(0)

    alias sram_blocked_matmul_kernel = sram_blocked_matmul[
        layout_c, layout_a, layout_b, thread_layout, BM, BN, BK
    ]

    var kernel = Function[
        __type_of(sram_blocked_matmul_kernel), sram_blocked_matmul_kernel
    ]()
    kernel(
        mat_c,
        mat_a,
        mat_b,
        grid_dim=(M // BM, N // BN),
        block_dim=(TH_M, TH_N),
    )

    synchronize()
    mat_c.tensor.print()

    _ = mat_a ^
    _ = mat_b ^
    _ = mat_c ^


fn main() raises:
    with Context() as ctx:
        # CHECK: === test_naive_matmul_kernel
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        test_naive_matmul_kernel()
        # CHECK: === test_sram_blocked_matmul
        # CHECK: 1120.0   2912.0   1176.0   3096.0   1232.0   3280.0   1288.0   3464.0
        # CHECK: 1148.0   3004.0   1204.0   3188.0   1260.0   3372.0   1316.0   3556.0
        # CHECK: 4704.0   6496.0   5016.0   6936.0   5328.0   7376.0   5640.0   7816.0
        # CHECK: 4860.0   6716.0   5172.0   7156.0   5484.0   7596.0   5796.0   8036.0
        # CHECK: 8288.0   10080.0   8856.0   10776.0   9424.0   11472.0   9992.0   12168.0
        # CHECK: 8572.0   10428.0   9140.0   11124.0   9708.0   11820.0   10276.0   12516.0
        # CHECK: 11872.0   13664.0   12696.0   14616.0   13520.0   15568.0   14344.0   16520.0
        # CHECK: 12284.0   14140.0   13108.0   15092.0   13932.0   16044.0   14756.0   16996.0
        test_sram_blocked_matmul()
