# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.testing import assert_equal

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.sync import barrier
from layout import *
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.math import outer_product_acc
from layout.tile_io import copy_dram_to_sram as copy_tile_dram_to_sram


def naive_matmul[
    layout_dst: TensorLayout,
    layout_lhs: TensorLayout,
    layout_rhs: TensorLayout,
    BM: Int,
    BN: Int,
](
    dst: TileTensor[.float32, layout_dst, MutAnyOrigin],
    lhs: TileTensor[.float32, layout_lhs, ImmutAnyOrigin],
    rhs: TileTensor[.float32, layout_rhs, ImmutAnyOrigin],
):
    var dst_tile = dst.tile[BM, BN](block_idx.y, block_idx.x)
    dst_tile[thread_idx.y, thread_idx.x] = 0
    for k in range(Int(lhs.dim[1]())):
        var lhs_tile = lhs.tile[BM, 1](block_idx.y, k)
        var rhs_tile = rhs.tile[1, BN](k, block_idx.x)
        dst_tile[thread_idx.y, thread_idx.x] += (
            lhs_tile[thread_idx.y, 0] * rhs_tile[0, thread_idx.x]
        )


def test_naive_matmul_kernel[
    M: Int = 8, N: Int = 8, K: Int = 8
](ctx: DeviceContext) raises:
    print("=== test_naive_matmul_kernel")
    comptime BM = 4
    comptime BN = 4

    comptime layout_a = row_major[M, K]()
    comptime layout_b = row_major[K, N]()
    comptime layout_c = row_major[M, N]()

    var mat_a = HostDeviceTileTensor[.float32](layout_a, ctx)
    var mat_b = HostDeviceTileTensor[.float32](layout_b, ctx)
    var mat_c = HostDeviceTileTensor[.float32](layout_c, ctx)

    arange(mat_a.host_tensor())
    arange(mat_b.host_tensor())
    _ = mat_c.host_tensor().fill(0)
    mat_a.to_device()
    mat_b.to_device()
    mat_c.to_device()

    comptime naive_matmul_kernel = naive_matmul[
        type_of(layout_c), type_of(layout_a), type_of(layout_b), BM, BN
    ]

    ctx.enqueue_function[naive_matmul_kernel](
        mat_c.device_tensor().as_unsafe_any_origin(),
        mat_a.device_tensor().as_imm().as_unsafe_any_origin(),
        mat_b.device_tensor().as_imm().as_unsafe_any_origin(),
        grid_dim=(N // BN, M // BM),
        block_dim=(BN, BM),
    )

    mat_c.to_host()
    var a = mat_a.host_tensor()
    var b = mat_b.host_tensor()
    var c = mat_c.host_tensor()
    for m in range(M):
        for n in range(N):
            var expected = Float32(0)
            for k in range(K):
                expected += a[m, k] * b[k, n]
            assert_equal(c[m, n], expected)
            print(c[m, n], end=" ")
        print()


def sram_blocked_matmul[
    layout_dst: TensorLayout,
    layout_lhs: TensorLayout,
    layout_rhs: TensorLayout,
    thread_layout: MixedLayout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    dst: TileTensor[.float32, layout_dst, MutAnyOrigin],
    lhs: TileTensor[.float32, layout_lhs, ImmutAnyOrigin],
    rhs: TileTensor[.float32, layout_rhs, ImmutAnyOrigin],
):
    comptime assert dst.flat_rank == 2
    comptime assert lhs.flat_rank == 2
    comptime assert rhs.flat_rank == 2
    comptime thread_m_size = type_of(thread_layout).static_shape[0]
    comptime thread_n_size = type_of(thread_layout).static_shape[1]
    var thread_m, thread_n = divmod(thread_idx.x, thread_n_size)

    # Preserve the original shape-only Layout's column-major SRAM strides.
    var lhs_sram_tile = stack_allocation[.float32, address_space=.SHARED](
        col_major[BM, BK]()
    )

    # Allocate an SRAM tile of (BK, BN) size with column-major layout for
    # the r.h.s.
    var rhs_sram_tile = stack_allocation[.float32, address_space=.SHARED](
        col_major[BK, BN]()
    )

    # Block the dst matrix with [BM, BN] tile size.
    var dst_tile = dst.tile[BM, BN](block_idx.y, block_idx.x)

    # Distribute thread layout into a block of size [BM, BN]. It repeats the
    # layout across the BMxBN block, e.g. row major layout will repeat as the
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
    comptime assert dst_local_tile.flat_rank == 2

    var dst_register_tile = stack_allocation[.float32, address_space=.LOCAL](
        row_major[BM // thread_m_size, BN // thread_n_size]()
    ).fill(0)

    # Loop over tiles in K dim.
    for k in range(Int(lhs.dim[1]()) // BK):
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

        comptime for kk in range(BK):
            var lhs_row = lhs_sram_tile.slice[:, kk]()
            var rhs_row = rhs_sram_tile.slice[kk, :]()
            var lhs_frags = lhs_row.distribute[row_major[thread_m_size]()](
                thread_m
            )
            var rhs_frags = rhs_row.distribute[row_major[thread_n_size]()](
                thread_n
            )
            outer_product_acc(dst_register_tile, lhs_frags, rhs_frags)

    # Move data from register tile to DRAM
    # FIXME: unrolled copy loop doesn't produce the correct results for some
    # tiles!!
    # dst_local_tile.copy_from(dst_register_tile)
    for m in range(Int(dst_local_tile.dim[0]())):
        for n in range(Int(dst_local_tile.dim[1]())):
            dst_local_tile[m, n] = dst_register_tile[m, n]


def test_sram_blocked_matmul(ctx: DeviceContext) raises:
    print("=== test_sram_blocked_matmul")
    comptime M = 8
    comptime N = 8
    comptime K = 8
    comptime BM = 4
    comptime BN = 4
    comptime BK = 4

    comptime TH_M = 2
    comptime TH_N = 2

    comptime layout_a = row_major[M, K]()
    comptime layout_b = row_major[K, N]()
    comptime layout_c = row_major[M, N]()

    comptime thread_layout = row_major[TH_M, TH_N]()

    var mat_a = HostDeviceTileTensor[.float32](layout_a, ctx)
    var mat_b = HostDeviceTileTensor[.float32](layout_b, ctx)
    var mat_c = HostDeviceTileTensor[.float32](layout_c, ctx)

    arange(mat_a.host_tensor())
    arange(mat_b.host_tensor())
    _ = mat_c.host_tensor().fill(0)
    mat_a.to_device()
    mat_b.to_device()
    mat_c.to_device()

    comptime sram_blocked_matmul_kernel = sram_blocked_matmul[
        type_of(layout_c),
        type_of(layout_a),
        type_of(layout_b),
        thread_layout,
        BM,
        BN,
        BK,
    ]

    ctx.enqueue_function[sram_blocked_matmul_kernel](
        mat_c.device_tensor().as_unsafe_any_origin(),
        mat_a.device_tensor().as_imm().as_unsafe_any_origin(),
        mat_b.device_tensor().as_imm().as_unsafe_any_origin(),
        grid_dim=(N // BN, M // BM),
        block_dim=(comptime (thread_layout.size())),
    )

    ctx.synchronize()
    mat_c.to_host()
    var c = mat_c.host_tensor()
    comptime assert c.flat_rank == 2
    for m in range(M):
        for n in range(N):
            print(c[m, n], end=" ")
        print()


def single_warp_mma_sync_m16n8k8[
    layout_c: TensorLayout,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
](
    mat_c: TileTensor[.float32, layout_c, MutAnyOrigin],
    mat_a: TileTensor[.float32, layout_a, ImmutAnyOrigin],
    mat_b: TileTensor[.float32, layout_b, ImmutAnyOrigin],
):
    comptime assert (
        layout_a.flat_rank == 2
        and layout_b.flat_rank == 2
        and layout_c.flat_rank == 2
    )
    comptime assert (
        layout_a.static_shape[0] == 16
        and layout_a.static_shape[1] == 8
        and layout_a.static_stride[0] == 8
        and layout_a.static_stride[1] == 1
    ), "A storage must be row-major 16x8"
    comptime assert (
        layout_b.static_shape[0] == 8
        and layout_b.static_shape[1] == 8
        and layout_b.static_stride[0] == 1
        and layout_b.static_stride[1] == 8
    ), "B storage must be column-major 8x8"
    comptime assert (
        layout_c.static_shape[0] == 16
        and layout_c.static_shape[1] == 8
        and layout_c.static_stride[0] == 8
        and layout_c.static_stride[1] == 1
    ), "C storage must be row-major 16x8"

    # MMA fragments address the row-major A and column-major B storage
    # directly; each axis describes a lane or a value owned by that lane.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-a-tf32
    var mat_a_mma = TileTensor(
        ptr=mat_a.unsafe_ptr(),
        layout=MixedLayout(
            Coord(Idx[4], Idx[8], Idx[2], Idx[2]),
            Coord(Idx[1], Idx[8], Idx[64], Idx[4]),
        ),
    )
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-b-tf32
    # These strides address column-major KxN storage without a transpose.
    var mat_b_mma = TileTensor(
        ptr=mat_b.unsafe_ptr(),
        layout=MixedLayout(
            Coord(Idx[4], Idx[8], Idx[2]),
            Coord(Idx[1], Idx[8], Idx[4]),
        ),
    )
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-1688-c
    var mat_c_mma = TileTensor(
        ptr=mat_c.unsafe_ptr(),
        layout=MixedLayout(
            Coord(Idx[4], Idx[8], Idx[2], Idx[2]),
            Coord(Idx[2], Idx[8], Idx[1], Idx[64]),
        ),
    )

    var thread_y, thread_x = divmod(thread_idx.x, 4)

    var vec_a_layout = SIMD[.float32, 4](
        mat_a_mma[thread_x, thread_y, 0, 0],
        mat_a_mma[thread_x, thread_y, 1, 0],
        mat_a_mma[thread_x, thread_y, 0, 1],
        mat_a_mma[thread_x, thread_y, 1, 1],
    )
    var vec_b_layout = SIMD[.float32, 2](
        mat_b_mma[thread_x, thread_y, 0],
        mat_b_mma[thread_x, thread_y, 1],
    )

    var vec_d = SIMD[.float32, 4](0)
    var vec_c = SIMD[.float32, 4](0)

    mma(vec_d, vec_a_layout, vec_b_layout, vec_c)

    mat_c_mma[thread_x, thread_y, 0, 0] = vec_d[0]
    mat_c_mma[thread_x, thread_y, 1, 0] = vec_d[1]
    mat_c_mma[thread_x, thread_y, 0, 1] = vec_d[2]
    mat_c_mma[thread_x, thread_y, 1, 1] = vec_d[3]


def test_single_warp_tf32_m16n8k8_matmul(ctx: DeviceContext) raises:
    print("=== single_warp_tf32_m16n8k8_matmul")
    comptime M = 16
    comptime N = 8
    comptime K = 8

    comptime layout_a = row_major[M, K]()
    comptime layout_b = col_major[K, N]()
    comptime layout_c = row_major[M, N]()

    var mat_a = HostDeviceTileTensor[.float32](layout_a, ctx)
    var mat_b = HostDeviceTileTensor[.float32](layout_b, ctx)
    var mat_c = HostDeviceTileTensor[.float32](layout_c, ctx)

    arange(mat_a.host_tensor())
    arange(mat_b.host_tensor())
    _ = mat_c.host_tensor().fill(0)
    mat_a.to_device()
    mat_b.to_device()
    mat_c.to_device()

    comptime single_warp_mma_sync_m16n8k8_kernel = single_warp_mma_sync_m16n8k8[
        type_of(layout_c), type_of(layout_a), type_of(layout_b)
    ]

    ctx.enqueue_function[single_warp_mma_sync_m16n8k8_kernel](
        mat_c.device_tensor().as_unsafe_any_origin(),
        mat_a.device_tensor().as_imm().as_unsafe_any_origin(),
        mat_b.device_tensor().as_imm().as_unsafe_any_origin(),
        grid_dim=(1, 1),
        block_dim=(32),
    )

    mat_c.to_host()
    var c = mat_c.host_tensor()
    for m in range(M):
        for n in range(N):
            print(c[m, n], end=" ")
        print()


def sram_blocked_matmul_dynamic_nd_buffer[
    thread_layout: MixedLayout,
    DstLayoutType: TensorLayout,
    LhsLayoutType: TensorLayout,
    RhsLayoutType: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
](
    dst: TileTensor[.float32, DstLayoutType, MutAnyOrigin],
    lhs: TileTensor[.float32, LhsLayoutType, ImmutAnyOrigin],
    rhs: TileTensor[.float32, RhsLayoutType, ImmutAnyOrigin],
):
    var lhs_sram_tile = stack_allocation[.float32, address_space=.SHARED](
        col_major[BM, BK]()
    )
    var rhs_sram_tile = stack_allocation[.float32, address_space=.SHARED](
        col_major[BK, BN]()
    )

    # Block the dst matrix with [BM, BN] tile size.
    var dst_tile = dst.tile[BM, BN]((block_idx.y, block_idx.x))

    # Distribute thread layout into a block of size [BM, BN]. It repeats the
    # layout across the BMxBN block, e.g. row major layout will repeat as the
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

    var dst_register_tile = stack_allocation[.float32, address_space=.LOCAL](
        row_major[2, 2]()
    ).fill(0)

    # Loop over tiles in K dim.
    for k in range(Int(lhs.dim(1)) // BK):
        # Block both l.h.s and r.h.s DRAM tensors.
        var lhs_tile = lhs.tile[BM, BK]((block_idx.y, k))
        var rhs_tile = rhs.tile[BK, BN]((k, block_idx.x))

        # Copy DRAM tiles to SRAM, distributing work across threads.
        copy_tile_dram_to_sram[thread_layout=thread_layout](
            lhs_sram_tile, lhs_tile
        )
        copy_tile_dram_to_sram[thread_layout=thread_layout](
            rhs_sram_tile, rhs_tile
        )

        barrier()

        comptime for kk in range(BK):
            var lhs_row = lhs_sram_tile.slice[:, kk]()
            var rhs_row = rhs_sram_tile.slice[kk, :]()
            comptime thread_m_size = type_of(thread_layout).static_shape[0]
            comptime thread_n_size = type_of(thread_layout).static_shape[1]
            var thread_m, thread_n = divmod(thread_idx.x, thread_n_size)
            var lhs_frags = lhs_row.distribute[row_major[thread_m_size]()](
                thread_m
            )
            var rhs_frags = rhs_row.distribute[row_major[thread_n_size]()](
                thread_n
            )
            outer_product_acc(dst_register_tile, lhs_frags, rhs_frags)

    # Move data from register tile to DRAM.
    comptime thread_shape_m = type_of(thread_layout).static_shape[0]
    comptime thread_shape_n = type_of(thread_layout).static_shape[1]
    var thread_m, thread_n = divmod(thread_idx.x, thread_shape_n)
    for m in range(Int(dst_register_tile.dim[0]())):
        for n in range(Int(dst_register_tile.dim[1]())):
            dst_tile[
                thread_m + m * thread_shape_m, thread_n + n * thread_shape_n
            ] = dst_register_tile[m, n]


def test_sram_blocked_matmul_dynamic_nd_buffer(ctx: DeviceContext) raises:
    print("=== test_sram_blocked_matmul_dynamic_nd_buffer")
    comptime M = 8
    comptime N = 8
    comptime K = 8
    comptime BM = 4
    comptime BN = 4
    comptime BK = 4

    comptime TH_M = 2
    comptime TH_N = 2

    comptime thread_layout = row_major[TH_M, TH_N]()

    var mat_c_ptr = alloc[Float32](M * N)
    var mat_a_ptr = alloc[Float32](M * K)
    var mat_b_ptr = alloc[Float32](K * N)

    for i in range(M * K):
        mat_a_ptr[i] = Float32(i)
    for i in range(K * N):
        mat_b_ptr[i] = Float32(i)
    for i in range(M * N):
        mat_c_ptr[i] = 0

    var mat_c_dev = ctx.enqueue_create_buffer[.float32](M * N)
    var mat_a_dev = ctx.enqueue_create_buffer[.float32](M * K)
    var mat_b_dev = ctx.enqueue_create_buffer[.float32](K * N)

    ctx.enqueue_copy(mat_c_dev, mat_c_ptr)
    ctx.enqueue_copy(mat_a_dev, mat_a_ptr)
    ctx.enqueue_copy(mat_b_dev, mat_b_ptr)

    var mat_c = TileTensor(mat_c_dev, row_major(M, N))
    var mat_a = TileTensor(mat_a_dev, row_major[M, K]())
    var mat_b = TileTensor(mat_b_dev, row_major[K, N]())

    comptime sram_blocked_matmul_dynamic_nd_buffer_kernel = sram_blocked_matmul_dynamic_nd_buffer[
        thread_layout,
        mat_c.LayoutType,
        mat_a.LayoutType,
        mat_b.LayoutType,
        BM,
        BN,
        BK,
    ]

    ctx.enqueue_function[sram_blocked_matmul_dynamic_nd_buffer_kernel](
        mat_c.as_unsafe_any_origin(),
        mat_a.as_imm().as_unsafe_any_origin(),
        mat_b.as_imm().as_unsafe_any_origin(),
        grid_dim=(N // BN, M // BM),
        block_dim=(comptime (thread_layout.size())),
    )

    ctx.enqueue_copy(mat_c_ptr, mat_c_dev)
    ctx.synchronize()

    for m in range(M):
        for n in range(N):
            print(mat_c_ptr[m * N + n], end=" ")
        print("")


def main() raises:
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
        test_naive_matmul_kernel[8, 12, 4](ctx)

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
