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
from std.sys import align_of, argv, simd_width_of

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import async_copy_wait_all
from layout import (
    Coord,
    Idx,
    MixedLayout,
    TensorLayout,
    TileTensor,
    col_major,
    row_major,
    stack_allocation,
)
from layout.math import outer_product_acc
from layout.tile_io import copy_dram_to_sram_async
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_almost_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def sgemm_double_buffer[
    c_type: DType,
    CLayoutType: TensorLayout,
    a_type: DType,
    ALayoutType: TensorLayout,
    b_type: DType,
    BLayoutType: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    c: TileTensor[c_type, CLayoutType, MutUntrackedOrigin],
    a: TileTensor[a_type, ALayoutType, ImmUntrackedOrigin],
    b: TileTensor[b_type, BLayoutType, ImmUntrackedOrigin],
):
    comptime assert a.rank == 2 and b.rank == 2 and c.rank == 2
    comptime assert a.all_dims_known and b.all_dims_known and c.all_dims_known

    comptime simd_size = simd_width_of[c_type]()
    # The B copies and the fragment loads move whole SIMD vectors, so every
    # buffer is aligned to one.
    comptime alignment = align_of[SIMD[c_type, simd_size]]()

    var K = Int(a.dim[1]())

    comptime num_warps_n = BN // WN

    var tid = Int(thread_idx.x)
    var warp_id, lane_id = divmod(tid, WARP_SIZE)

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, num_warps_n)

    # Warp shape in 2D.
    comptime warp_dim_x = WN // TN
    comptime warp_dim_y = WM // TM
    comptime assert (
        warp_dim_x * warp_dim_y == WARP_SIZE
    ), "Warp 2d shape doesn't match the warp size"

    # Thread swizzling. The warp is a [warp_dim_y, warp_dim_x] grid and each
    # lane is placed in it as follows (the number is the lane id), so that
    # the lanes of a quad touch neighbouring rows and columns:
    # 0  2  4  6  8  10 12 14
    # 1  3  5  7  9  11 13 15
    # 16 18 20 22 24 26 28 30
    # 17 19 21 23 25 27 29 31
    # Row `q` and column `p` of the lane in that grid. The row is the nested
    # (2, 2) mode with strides (1, 2 * warp_dim_x); the column has stride 2.
    var lane_q = (lane_id % 2) + 2 * ((lane_id // (2 * warp_dim_x)) % 2)
    var lane_p = (lane_id // 2) % warp_dim_x

    # Pad BM to avoid bank conflicts.
    comptime pad_avoid_bank_conflict = 4
    comptime BM_padded = BM + pad_avoid_bank_conflict

    # Double buffer in shared memory. A is stored transposed, [BK, BM], with a
    # padded row stride.
    var a_smem = stack_allocation[
        a_type, address_space=.SHARED, alignment=alignment
    ](row_major[2 * BK, BM_padded]())
    comptime a_smem_layout = MixedLayout(
        Coord(Idx[BK], Idx[BM]), Coord(Idx[BM_padded], Idx[1])
    )
    var a_smem_tiles: Array[
        TileTensor[
            a_type,
            type_of(a_smem_layout),
            MutUntrackedOrigin,
            address_space=.SHARED,
        ],
        2,
    ] = [
        TileTensor(a_smem.ptr, a_smem_layout),
        TileTensor(a_smem.ptr.unsafe_offset(BK * BM_padded), a_smem_layout),
    ]

    var b_smem = stack_allocation[
        b_type, address_space=.SHARED, alignment=alignment
    ](row_major[2 * BK, BN]())
    var b_smem_tiles: Array[
        TileTensor[
            b_type,
            type_of(row_major[BK, BN]()),
            MutUntrackedOrigin,
            address_space=.SHARED,
        ],
        2,
    ] = [
        TileTensor(b_smem.ptr, row_major[BK, BN]()),
        TileTensor(b_smem.ptr.unsafe_offset(BK * BN), row_major[BK, BN]()),
    ]

    # Row major thread layout over the [BM, BK] global tile for coalesced
    # loads, column major over the [BK, BM] shared tile so the copy
    # transposes.
    comptime thread_loada_gmem_layout = row_major[NUM_THREADS // BK, BK]()
    comptime thread_storea_smem_layout = col_major[BK, NUM_THREADS // BK]()
    comptime thread_layout_loadb = row_major[
        (NUM_THREADS // BN) * simd_size, BN // simd_size
    ]()

    @inline(.always)
    def load_k_tile(k_tile_id: Int, buffer_id: Int) {imm}:
        var a_gmem_tile = a.tile[BM, BK]((Int(block_idx.y), k_tile_id))
        a_smem_tiles[buffer_id].distribute[thread_storea_smem_layout](
            tid
        ).copy_from_async(a_gmem_tile.distribute[thread_loada_gmem_layout](tid))

        var b_gmem_tile = b.tile[BK, BN]((k_tile_id, Int(block_idx.x)))
        copy_dram_to_sram_async[thread_layout=thread_layout_loadb](
            b_smem_tiles[buffer_id].vectorize[1, simd_size](),
            b_gmem_tile.vectorize[1, simd_size](),
        )

    load_k_tile(0, 0)
    async_copy_wait_all()
    barrier()

    # Double buffer in registers (fragments in nvidia terms).
    var a_reg: Array[
        TileTensor[
            a_type,
            type_of(row_major[TM]()),
            MutUntrackedOrigin,
            address_space=.LOCAL,
        ],
        2,
    ] = [
        stack_allocation[a_type, address_space=.LOCAL, alignment=alignment](
            row_major[TM]()
        ),
        stack_allocation[a_type, address_space=.LOCAL, alignment=alignment](
            row_major[TM]()
        ),
    ]
    var b_reg: Array[
        TileTensor[
            b_type,
            type_of(row_major[TN]()),
            MutUntrackedOrigin,
            address_space=.LOCAL,
        ],
        2,
    ] = [
        stack_allocation[b_type, address_space=.LOCAL, alignment=alignment](
            row_major[TN]()
        ),
        stack_allocation[b_type, address_space=.LOCAL, alignment=alignment](
            row_major[TN]()
        ),
    ]
    var c_reg = stack_allocation[
        c_type, address_space=.LOCAL, alignment=alignment
    ](row_major[TM, TN]()).fill(0)

    # Loads row `k` of the warp's [BK, WM] shared A tile into a fragment
    # buffer: lane row `q` takes vectors `q`, `q + warp_dim_y`, ...
    @inline(.always)
    def load_a_frag(smem_id: Int, k: Int, reg_id: Int) {imm}:
        var a_smem_warp_row = TileTensor(
            a_smem_tiles[smem_id].ptr.unsafe_offset(
                k * BM_padded + warp_y * WM
            ),
            row_major[WM](),
        )
        a_reg[reg_id].vectorize[simd_size]().copy_from(
            a_smem_warp_row.vectorize[simd_size]().distribute[
                row_major[warp_dim_y]()
            ](lane_q)
        )

    # Same for row `k` of the warp's [BK, WN] shared B tile and lane column
    # `p`.
    @inline(.always)
    def load_b_frag(smem_id: Int, k: Int, reg_id: Int) {imm}:
        var b_smem_warp_row = TileTensor(
            b_smem_tiles[smem_id].ptr.unsafe_offset(k * BN + warp_x * WN),
            row_major[WN](),
        )
        b_reg[reg_id].vectorize[simd_size]().copy_from(
            b_smem_warp_row.vectorize[simd_size]().distribute[
                row_major[warp_dim_x]()
            ](lane_p)
        )

    # Load the first fragments.
    var frag_smem_id = 0
    load_a_frag(frag_smem_id, 0, 0)
    load_b_frag(frag_smem_id, 0, 0)

    var num_k_tiles = ceildiv(K, BK)

    for k_tile_id in range(num_k_tiles):
        # The shared memory buffer to be prefetched.
        var prefetch_id = 1 if k_tile_id % 2 == 0 else 0

        comptime for k in range(BK):
            comptime next_k = (k + 1) % BK

            # Buffer id for the double register buffers. They alternate.
            comptime buffer_id = k % 2
            comptime next_buffer_id = (k + 1) % 2

            if k == BK - 1:
                async_copy_wait_all()
                barrier()
                frag_smem_id = prefetch_id

            # Fill the other fragment buffers using the next row.
            load_a_frag(frag_smem_id, next_k, next_buffer_id)
            load_b_frag(frag_smem_id, next_k, next_buffer_id)

            # Load the next k tile from global memory to shared memory.
            if k == 0 and k_tile_id < num_k_tiles - 1:
                load_k_tile(k_tile_id + 1, prefetch_id)

            outer_product_acc(c_reg, a_reg[buffer_id], b_reg[buffer_id])

    # Map the global memory tile down to the thread. The outer product
    # results are organized as simd_size x simd_size blocks: block row
    # `rv` of this lane holds warp rows `simd_size * (lane_q + warp_dim_y *
    # rv)` onwards, and its columns are distributed like the B fragments.
    var c_gmem_warp_tile = c.tile[BM, BN](
        (Int(block_idx.y), Int(block_idx.x))
    ).tile[WM, WN]((warp_y, warp_x))

    comptime for rv in range(TM // simd_size):
        comptime for ii in range(simd_size):
            var warp_row = simd_size * (lane_q + warp_dim_y * rv) + ii
            c_gmem_warp_tile.tile[1, WN]((warp_row, 0)).vectorize[
                1, simd_size
            ]().distribute[row_major[1, warp_dim_x]()](lane_p).copy_from(
                c_reg.tile[1, TN]((rv * simd_size + ii, 0)).vectorize[
                    1, simd_size
                ]()
            )


def test(ctx: DeviceContext) raises:
    comptime NUM_THREADS = 256
    comptime M = 8192
    comptime N = 8192
    comptime K = 128
    comptime BM = 128
    comptime BN = 128
    comptime BK = 16
    comptime WM = 32
    comptime WN = 64 if ctx.target.is_nvidia_gpu() else 128
    comptime TM = 8
    comptime TN = 8

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.synchronize()

    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i)

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var c_tensor = TileTensor(c_device, row_major[M, N]())
    var a_tensor = TileTensor(a_device, row_major[M, K]())
    var b_tensor = TileTensor(b_device, row_major[K, N]())

    comptime gemm = sgemm_double_buffer[
        .float32,
        c_tensor.LayoutType,
        .float32,
        a_tensor.LayoutType,
        .float32,
        b_tensor.LayoutType,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
        NUM_THREADS,
    ]

    @inline(.always)
    def run_func(ctx: DeviceContext) raises {imm}:
        ctx.enqueue_function[gemm](
            c_tensor,
            a_tensor.as_imm(),
            b_tensor.as_imm(),
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
            block_dim=(NUM_THREADS, 1, 1),
        )

    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        # Warmup
        for _ in range(nwarmup):
            run_func(ctx)

        var nstime = Float64(ctx.execution_time(run_func, nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    run_func(ctx)

    ctx.enqueue_copy(c_host, c_device)

    # Naive gemm.
    comptime BLOCK_DIM = 16

    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).
    var c_ref_tt = TileTensor(
        c_device_ref,
        row_major(M, N),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(M, K),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(K, N),
    )

    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,
        DType.float32,
        DType.float32,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
    ]
    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def main() raises:
    with DeviceContext() as ctx:
        test(ctx)
