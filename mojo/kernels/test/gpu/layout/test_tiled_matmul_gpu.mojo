# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu.memory import AddressSpace
from gpu.mma import mma
from gpu.sync import barrier
from layout import *
from layout._utils import ManagedLayoutTensor
from layout.fillers import arange
from layout.math import outer_product_acc
from layout.nd_buffer_stub import copy_from_nd_buffer, copy_to_nd_buffer
from memory import UnsafePointer

from utils import IndexList
from utils.index import Index


fn naive_matmul[
    layout_dst: Layout,
    layout_lhs: Layout,
    layout_rhs: Layout,
    BM: Int,
    BN: Int,
](
    dst: LayoutTensor[DType.float32, layout_dst, MutableAnyOrigin],
    lhs: LayoutTensor[DType.float32, layout_dst, MutableAnyOrigin],
    rhs: LayoutTensor[DType.float32, layout_dst, MutableAnyOrigin],
):
    var dst_tile = dst.tile[BM, BN](block_idx.y, block_idx.x)
    dst_tile[thread_idx.y, thread_idx.x] = 0
    for k in range(dst.shape[0]()):
        var lhs_tile = rhs.tile[BM, 1](block_idx.y, k)
        var rhs_tile = lhs.tile[1, BN](k, block_idx.x)
        dst_tile[thread_idx.y, thread_idx.x] += (
            lhs_tile[thread_idx.y, k] * rhs_tile[k, thread_idx.x]
        )


fn test_naive_matmul_kernel(ctx: DeviceContext) raises:
    print("=== test_naive_matmul_kernel")
    alias M = 8
    alias N = 8
    alias K = 8
    alias BM = 4
    alias BN = 4

    alias layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    alias layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    alias layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    var mat_a = ManagedLayoutTensor[DType.float32, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[DType.float32, layout_b](ctx)
    var mat_c = ManagedLayoutTensor[DType.float32, layout_c](ctx)

    arange(mat_a.tensor())
    arange(mat_b.tensor())
    _ = mat_c.tensor().fill(0)

    alias naive_matmul_kernel = naive_matmul[
        layout_c, layout_a, layout_b, BM, BN
    ]

    ctx.enqueue_function[naive_matmul_kernel](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(M // BM, N // BN),
        block_dim=(BM, BN),
    )

    ctx.synchronize()
    print(mat_c.tensor())
    _ = mat_a^
    _ = mat_b^
    _ = mat_c^


fn sram_blocked_matmul[
    layout_dst: Layout,
    layout_lhs: Layout,
    layout_rhs: Layout,
    thread_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    dst: LayoutTensor[DType.float32, layout_dst, MutableAnyOrigin],
    lhs: LayoutTensor[DType.float32, layout_lhs, MutableAnyOrigin],
    rhs: LayoutTensor[DType.float32, layout_rhs, MutableAnyOrigin],
):
    # Allocate an SRAM tile of (BM, BK) size with row-major layout for the l.h.s.
    var lhs_sram_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BK)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Allocate an SRAM tile of (BK, BN) size with row-major layout for
    # the r.h.s.
    var rhs_sram_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BK, BN)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Block the dst matrix with [BM, BN] tile size.
    var dst_tile = dst.tile[BM, BN](block_idx.y, block_idx.x)

    # Distribute thread layout into a block of size [BM, BN], It repeats the
    # layout accross the BMxBN block, e.g row major layout will repeate as the
    # the following:
    # +---------------------------------BN-+----------------------------------+-------------
    # |  TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N
    # |  TH_3 TH_4     ... TH_5    | TH_3 TH_4     ... TH_5    | TH_3 TH_4     ... TH_5
    # |    .            .          |   .           .           |   .            .
    # |    .             .         |   .           .           |   .             .
    # BN TH_M TH_(M+1) ... TH_(MN) | TH_M TH_(M+1) ... TH_(MN) | TH_M TH_(M+1) ... TH_(MN)
    # +------------------------------------+----------------------------------+------------
    # |  TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N    |  TH_0 TH_1     ... TH_N
    # |      .        .      ...   |     .         ...   .     |    .
    # |      .        .      ...   |     .         ...   .     |    .
    var dst_local_tile = dst_tile.distribute[thread_layout](thread_idx.x)

    # Allocate a register tile for the dst matrix with the same layout.
    var dst_register_tile = stack_allocation_like(dst_local_tile).fill(0)

    # Loop over tiles in K dim.
    for k in range(lhs.shape[1]() // BK):
        # Block both l.h.s and r.h.s DRAM tensors.
        var lhs_tile = lhs.tile[BM, BK](block_idx.y, k)
        var rhs_tile = rhs.tile[BK, BN](k, block_idx.x)

        # Distribute layout of threads into DRAM and SRAM to perform the copy.
        var lhs_tile_local = lhs_tile.distribute[thread_layout](thread_idx.x)
        var rhs_tile_local = rhs_tile.distribute[thread_layout](thread_idx.x)
        var lhs_sram_tile_local = lhs_sram_tile.distribute[thread_layout](
            thread_idx.x
        )
        var rhs_sram_tile_local = rhs_sram_tile.distribute[thread_layout](
            thread_idx.x
        )
        lhs_sram_tile_local.copy_from(lhs_tile_local)
        rhs_sram_tile_local.copy_from(rhs_tile_local)

        barrier()

        @parameter
        for kk in range(BK):
            var lhs_row = lhs_sram_tile.slice[:, kk : kk + 1]().coalesce()
            var rhs_row = rhs_sram_tile.slice[kk : kk + 1, :]().coalesce()
            var lhs_frags = lhs_row.distribute[thread_layout, axis=0](
                thread_idx.x
            )
            var rhs_frags = rhs_row.distribute[thread_layout, axis=1](
                thread_idx.x
            )
            outer_product_acc(dst_register_tile, lhs_frags, rhs_frags)

    # Move data from register tile to DRAM
    # FIXME: unrolled copy loop doesn't produce the correct results for some
    # tiles!!
    # dst_local_tile.copy_from(dst_register_tile)
    for m in range(dst_local_tile.shape[0]()):
        for n in range(dst_local_tile.shape[1]()):
            dst_local_tile[m, n] = dst_register_tile[m, n]


fn test_sram_blocked_matmul(ctx: DeviceContext) raises:
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

    var mat_a = ManagedLayoutTensor[DType.float32, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[DType.float32, layout_b](ctx)
    var mat_c = ManagedLayoutTensor[DType.float32, layout_c](ctx)

    arange(mat_a.tensor())
    arange(mat_b.tensor())
    _ = mat_c.tensor().fill(0)

    alias sram_blocked_matmul_kernel = sram_blocked_matmul[
        layout_c, layout_a, layout_b, thread_layout, BM, BN, BK
    ]

    ctx.enqueue_function[sram_blocked_matmul_kernel](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(N // BN, M // BM),
        block_dim=(thread_layout.size()),
    )

    ctx.synchronize()
    print(mat_c.tensor())

    _ = mat_a^
    _ = mat_b^
    _ = mat_c^


fn single_warp_mma_sync_m16n8k8[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
    layout_c_mma: Layout,
    layout_a_mma: Layout,
    layout_b_mma: Layout,
](
    mat_c: LayoutTensor[DType.float32, layout_c, MutableAnyOrigin],
    mat_a: LayoutTensor[DType.float32, layout_a, MutableAnyOrigin],
    mat_b: LayoutTensor[DType.float32, layout_b, MutableAnyOrigin],
):
    var mat_a_mma = mat_a.composition[layout_a_mma]()
    # Note: CUTLASS layout above assumes the same layout as the instruction itself, l.h.s row-major and r.h.s col-major.
    var mat_b_mma = mat_b.transpose().composition[layout_b_mma]()
    var mat_c_mma = mat_c.composition[layout_c_mma]()

    var thread_y: UInt = thread_idx.x // 4
    var thread_x: UInt = thread_idx.x % 4

    var vec_a_layout = SIMD[DType.float32, 4](
        rebind[Float32](mat_a_mma[thread_x, thread_y, 0, 0]),
        rebind[Float32](mat_a_mma[thread_x, thread_y, 1, 0]),
        rebind[Float32](mat_a_mma[thread_x, thread_y, 0, 1]),
        rebind[Float32](mat_a_mma[thread_x, thread_y, 1, 1]),
    )
    var vec_b_layout = SIMD[DType.float32, 2](
        rebind[Float32](mat_b_mma[thread_x, thread_y, 0]),
        rebind[Float32](mat_b_mma[thread_x, thread_y, 1]),
    )

    var vec_d = SIMD[DType.float32, 4](0)
    var vec_c = SIMD[DType.float32, 4](0)

    mma(vec_d, vec_a_layout, vec_b_layout, vec_c)

    mat_c_mma[thread_x, thread_y, 0, 0] = vec_d[0]
    mat_c_mma[thread_x, thread_y, 1, 0] = vec_d[1]
    mat_c_mma[thread_x, thread_y, 0, 1] = vec_d[2]
    mat_c_mma[thread_x, thread_y, 1, 1] = vec_d[3]


fn test_single_warp_tf32_m16n8k8_matmul(ctx: DeviceContext) raises:
    print("=== single_warp_tf32_m16n8k8_matmul")
    alias M = 16
    alias N = 8
    alias K = 8

    alias TH_M = 4
    alias TH_N = 8

    alias layout_a = Layout.row_major(M, K)
    alias layout_b = Layout.col_major(K, N)
    alias layout_c = Layout.row_major(M, N)

    var mat_a = ManagedLayoutTensor[DType.float32, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[DType.float32, layout_b](ctx)
    var mat_c = ManagedLayoutTensor[DType.float32, layout_c](ctx)

    arange(mat_a.tensor())
    arange(mat_b.tensor())
    _ = mat_c.tensor().fill(0)

    # MMA layout are copied from CUTLASS:
    # https://sourcegraph.com/github.com/NVIDIA/cutlass@ffa34e70756b0bc744e1dfcc115b5a991a68f132/-/blob/include/cute/atom/mma_traits_sm80.hpp?L167
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-a-tf32
    alias layout_a_mma = Layout(
        IntTuple(IntTuple(4, 8), IntTuple(2, 2)),
        IntTuple(IntTuple(16, 1), IntTuple(8, 64)),
    )
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-b-tf32
    alias layout_b_mma = Layout(
        IntTuple(IntTuple(4, 8), 2), IntTuple(IntTuple(8, 1), 32)
    )
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-b-tf32
    alias layout_c_mma = Layout(
        IntTuple(IntTuple(4, 8), IntTuple(2, 2)),
        IntTuple(IntTuple(32, 1), IntTuple(16, 8)),
    )

    alias single_warp_mma_sync_m16n8k8_kernel_kernel = single_warp_mma_sync_m16n8k8[
        layout_c, layout_a, layout_b, layout_c_mma, layout_a_mma, layout_b_mma
    ]

    ctx.enqueue_function[single_warp_mma_sync_m16n8k8_kernel_kernel](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(32),
    )

    ctx.synchronize()
    print(mat_c.tensor())

    _ = mat_a^
    _ = mat_b^
    _ = mat_c^


fn sram_blocked_matmul_dynamic_nd_buffer[
    thread_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    dst: NDBuffer[DType.float32, 2, DimList.create_unknown[2]()],
    lhs: NDBuffer[DType.float32, 2, DimList.create_unknown[2]()],
    rhs: NDBuffer[DType.float32, 2, DimList.create_unknown[2]()],
):
    # Allocate an SRAM tile of (BM, BK) size with row-major layout for the l.h.s.
    var lhs_sram_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BM, BK)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Allocate an SRAM tile of (BK, BN) size with row-major layout for
    # the r.h.s.
    var rhs_sram_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(BK, BN)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Block the dst matrix with [BM, BN] tile size.
    var dst_tile = dst.tile[BM, BN](IndexList[2](block_idx.y, block_idx.x))

    # Distribute thread layout into a block of size [BM, BN], It repeats the
    # layout accross the BMxBN block, e.g row major layout will repeate as the
    # the following:
    # +---------------------------------BN-+----------------------------------+-------------
    # |  TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N
    # |  TH_3 TH_4     ... TH_5    | TH_3 TH_4     ... TH_5    | TH_3 TH_4     ... TH_5
    # |    .            .          |   .           .           |   .            .
    # |    .             .         |   .           .           |   .             .
    # BN TH_M TH_(M+1) ... TH_(MN) | TH_M TH_(M+1) ... TH_(MN) | TH_M TH_(M+1) ... TH_(MN)
    # +------------------------------------+----------------------------------+------------
    # |  TH_0 TH_1     ... TH_N    | TH_0 TH_1     ... TH_N    |  TH_0 TH_1     ... TH_N
    # |      .        .      ...   |     .         ...   .     |    .
    # |      .        .      ...   |     .         ...   .     |    .

    # Allocate a register tile for the dst matrix with the same layout.
    # TODO: Is it useful to have stack_allocation_like[thread_layout](nd_buffer) ? We can do this if needed.
    var dst_register_tile = LayoutTensor[
        DType.float32, Layout.row_major(2, 2), MutableAnyOrigin
    ].stack_allocation().fill(0)

    # Loop over tiles in K dim.
    for k in range(lhs.dim(1) // BK):
        # Block both l.h.s and r.h.s DRAM tensors.
        var lhs_tile = lhs.tile[BM, BK](IndexList[2](block_idx.y, k))
        var rhs_tile = rhs.tile[BK, BN](IndexList[2](k, block_idx.x))

        # Distribute layout of threads into DRAM and SRAM to perform the copy.
        var lhs_sram_tile_local = lhs_sram_tile.distribute[thread_layout](
            thread_idx.x
        )
        copy_from_nd_buffer[thread_layout=thread_layout](
            lhs_sram_tile_local, lhs_tile, thread_idx.x
        )

        var rhs_sram_tile_local = rhs_sram_tile.distribute[thread_layout](
            thread_idx.x
        )
        copy_from_nd_buffer[thread_layout=thread_layout](
            rhs_sram_tile_local, rhs_tile, thread_idx.x
        )

        barrier()

        @parameter
        for kk in range(BK):
            var lhs_row = lhs_sram_tile.slice[:, kk : kk + 1]().coalesce()
            var rhs_row = rhs_sram_tile.slice[kk : kk + 1, :]().coalesce()
            var lhs_frags = lhs_row.distribute[thread_layout, axis=0](
                thread_idx.x
            )
            var rhs_frags = rhs_row.distribute[thread_layout, axis=1](
                thread_idx.x
            )
            outer_product_acc(dst_register_tile, lhs_frags, rhs_frags)

    # Move data from register tile to DRAM
    copy_to_nd_buffer[thread_layout=thread_layout](
        dst_tile, dst_register_tile, thread_idx.x
    )


fn test_sram_blocked_matmul_dynamic_nd_buffer(ctx: DeviceContext) raises:
    print("=== test_sram_blocked_matmul_dynamic_nd_buffer")
    alias M = 8
    alias N = 8
    alias K = 8
    alias BM = 4
    alias BN = 4
    alias BK = 4

    alias TH_M = 2
    alias TH_N = 2

    alias thread_layout = Layout(IntTuple(TH_M, TH_N), IntTuple(TH_N, 1))

    var mat_c_ptr = UnsafePointer[Float32].alloc(M * N)
    var mat_a_ptr = UnsafePointer[Float32].alloc(M * K)
    var mat_b_ptr = UnsafePointer[Float32].alloc(K * N)

    for i in range(M * K):
        mat_a_ptr[i] = i
    for i in range(K * N):
        mat_b_ptr[i] = i
    for i in range(M * N):
        mat_c_ptr[i] = 0

    var mat_c_dev = ctx.enqueue_create_buffer[DType.float32](M * N)
    var mat_a_dev = ctx.enqueue_create_buffer[DType.float32](M * K)
    var mat_b_dev = ctx.enqueue_create_buffer[DType.float32](K * N)

    ctx.enqueue_copy(mat_c_dev, mat_c_ptr)
    ctx.enqueue_copy(mat_a_dev, mat_a_ptr)
    ctx.enqueue_copy(mat_b_dev, mat_b_ptr)

    var mat_c = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        mat_c_dev.unsafe_ptr(), dynamic_shape=Index(M, N)
    )
    var mat_a = NDBuffer[DType.float32, 2, DimList(M, K)](
        mat_a_dev.unsafe_ptr(), dynamic_shape=Index(M, K)
    )
    var mat_b = NDBuffer[DType.float32, 2, DimList(K, N)](
        mat_b_dev.unsafe_ptr(), dynamic_shape=Index(K, N)
    )

    alias sram_blocked_matmul_dynamic_nd_buffer_kernel = sram_blocked_matmul_dynamic_nd_buffer[
        thread_layout, BM, BN, BK
    ]

    ctx.enqueue_function[sram_blocked_matmul_dynamic_nd_buffer_kernel](
        mat_c,
        mat_a,
        mat_b,
        grid_dim=(N // BN, M // BM),
        block_dim=(thread_layout.size()),
    )

    ctx.enqueue_copy(mat_c_ptr, mat_c_dev)

    for m in range(M):
        for n in range(N):
            print(mat_c_ptr[m * N + n], end=" ")
        print("")


fn main() raises:
    with DeviceContext() as ctx:
        # CHECK: === test_naive_matmul_kernel
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        test_naive_matmul_kernel(ctx)

        # CHECK: === test_sram_blocked_matmul
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        test_sram_blocked_matmul(ctx)

        # CHECK: === single_warp_tf32_m16n8k8_matmul
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        # CHECK: 15456.0   15996.0   16536.0   17076.0   17616.0   18156.0   18696.0   19236.0
        # CHECK: 17248.0   17852.0   18456.0   19060.0   19664.0   20268.0   20872.0   21476.0
        # CHECK: 19040.0   19708.0   20376.0   21044.0   21712.0   22380.0   23048.0   23716.0
        # CHECK: 20832.0   21564.0   22296.0   23028.0   23760.0   24492.0   25224.0   25956.0
        # CHECK: 22624.0   23420.0   24216.0   25012.0   25808.0   26604.0   27400.0   28196.0
        # CHECK: 24416.0   25276.0   26136.0   26996.0   27856.0   28716.0   29576.0   30436.0
        # CHECK: 26208.0   27132.0   28056.0   28980.0   29904.0   30828.0   31752.0   32676.0
        # CHECK: 28000.0   28988.0   29976.0   30964.0   31952.0   32940.0   33928.0   34916.0
        test_single_warp_tf32_m16n8k8_matmul(ctx)

        # CHECK: === test_sram_blocked_matmul_dynamic_nd_buffer
        # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
        # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
        # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
        # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
        # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
        # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
        # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
        # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
        test_sram_blocked_matmul_dynamic_nd_buffer(ctx)
