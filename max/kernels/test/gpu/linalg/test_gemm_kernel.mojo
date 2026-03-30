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

from std.math import ceildiv, isclose
from std.sys import argv

from std.gpu import WARP_SIZE
from std.gpu.host import DeviceContext
from std.gpu import block_idx_uint as block_idx, warp_id_uint as warp_id
from std.gpu.memory import async_copy_wait_all
from std.gpu.sync import barrier
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.layout_tensor import (
    copy_sram_to_local,
    copy_dram_to_sram_async,
    copy_local_to_dram,
)
from layout.math import outer_product_acc
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_almost_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    NUM_THREADS: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
](
    mat_c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    mat_a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    mat_b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
):
    var M = mat_c.dim(0)
    var N = mat_c.dim(1)
    var K = mat_a.dim(1)

    var a_tile_sram = LayoutTensor[
        a_type,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_tile_sram = LayoutTensor[
        b_type,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var num_warps = NUM_THREADS // WARP_SIZE
    var n_warp_n = BN // WN
    var n_warp_m = BM // WM
    var warp_m = Int(warp_id()) // n_warp_n
    var warp_n = Int(warp_id()) % n_warp_n

    # Allocate register tiles.
    var a_reg = LayoutTensor[
        a_type,
        Layout.row_major(TN),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var b_reg = LayoutTensor[
        b_type,
        Layout.row_major(TN),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var c_reg = (
        LayoutTensor[
            c_type,
            Layout.row_major(TM, TN),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    comptime warp_layout = Layout.row_major(8, 4)

    for k_i in range(ceildiv(K, BK)):
        var a_tile_dram = mat_a.tile[BM, BK](Int(block_idx.y), k_i)

        copy_dram_to_sram_async[
            thread_layout=Layout.row_major(NUM_THREADS // BK, BK)
        ](a_tile_sram, a_tile_dram)

        var b_tile_dram = mat_b.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[
            thread_layout=Layout.row_major(NUM_THREADS // BN, BN)
        ](b_tile_sram, b_tile_dram)

        async_copy_wait_all()
        barrier()

        comptime for k_i in range(BK):
            var a_smem_warp_row = a_tile_sram.tile[WM, BK](warp_m, 0).slice[
                :, k_i : k_i + 1
            ]()

            var b_smem_warp_row = b_tile_sram.tile[BK, WN](0, warp_n).slice[
                k_i : k_i + 1, :
            ]()
            copy_sram_to_local[src_warp_layout=warp_layout, axis=0](
                a_reg, a_smem_warp_row
            )
            copy_sram_to_local[src_warp_layout=warp_layout, axis=1](
                b_reg, b_smem_warp_row
            )
            outer_product_acc(c_reg, a_reg, b_reg)

        # Otherwise a data race, faster threads will modify shared memory.
        barrier()

    var c_warp_tile = mat_c.tile[BM, BN](
        Int(block_idx.y), Int(block_idx.x)
    ).tile[WM, WN](warp_m, warp_n)

    copy_local_to_dram[dst_thread_layout=warp_layout](c_warp_tile, c_reg)


def test_gemm_kernel_dynamic(ctx: DeviceContext) raises:
    comptime NUM_THREADS = 256
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime WM = 32
    comptime WN = 16
    comptime TM = 4
    comptime TN = 4

    comptime M = 1024
    comptime N = 1024
    comptime K = 128

    var a_host = alloc[Float32](M * K)
    var b_host = alloc[Float32](K * N)
    var c_host = alloc[Float32](M * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i)

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(K, M)
    comptime c_layout = Layout.row_major(M, N)

    var a_tensor = LayoutTensor[DType.float32, a_layout, MutAnyOrigin](
        a_device.unsafe_ptr()
    )
    var b_tensor = LayoutTensor[DType.float32, b_layout, MutAnyOrigin](
        b_device.unsafe_ptr()
    )
    var c_tensor = LayoutTensor[DType.float32, c_layout, MutAnyOrigin](
        c_device.unsafe_ptr()
    )

    comptime kernel = gemm_kernel[
        DType.float32,
        c_tensor.layout,
        DType.float32,
        a_tensor.layout,
        DType.float32,
        b_tensor.layout,
        NUM_THREADS,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
    ]

    ctx.enqueue_function_experimental[kernel](
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(NUM_THREADS),
    )

    ctx.enqueue_copy(c_host, c_device)

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function_experimental
    # requires exact type matches).
    from std.memory import UnsafePointer

    var c_ref_tt = TileTensor(
        c_device_ref.unsafe_ptr(),
        row_major(Coord(Idx(M), Idx(N))),
    )
    var a_tt = TileTensor(
        UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(Idx(M), Idx(K))),
    )
    var b_tt = TileTensor(
        UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(Idx(K), Idx(M))),
    )

    # Naive gemm.
    comptime BLOCK_DIM = 16
    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,
        DType.float32,
        DType.float32,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
    ]

    ctx.enqueue_function_experimental[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()
    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        @always_inline
        @parameter
        def run_func(ctx: DeviceContext) raises:
            ctx.enqueue_function_experimental[kernel](
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )

        # Warmup
        for i in range(nwarmup):
            ctx.enqueue_function_experimental[kernel](
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(NUM_THREADS),
            )
        var nstime = Float64(ctx.execution_time[run_func](nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()


def main() raises:
    with DeviceContext() as ctx:
        test_gemm_kernel_dynamic(ctx)
