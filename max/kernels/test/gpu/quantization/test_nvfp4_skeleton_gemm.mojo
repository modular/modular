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
"""Correctness test for the NVFP4 (E2M1) path of `multistage_gemm_q`.

Validates the `is_nvfp4=True` route (which dispatches `TensorCore.load_b_nvfp4`)
against a CPU/GPU reference that dequantizes the SAME packed bytes with the
canonical E2M1 decode. First milestone uses group_size=128 / BK=32 (the Q4 tile)
to validate the kernel machinery (decode + pipeline + fragment layout) with
maximal reuse; the real NVFP4 group=16 + FP8 scales + global scale come later in
the weight adapter. Both the kernel and this reference interpret each 4-bit code
as E2M1, so a match confirms the kernel reads/decodes/MMAs correctly.
"""

from std.math import ceildiv
from std.math.uutils import udivmod
from std.random import rand, randint
from std.time import perf_counter_ns

from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.host import DeviceContext

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    CoordLike,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from layout.layout import *
from linalg.matmul.gpu import multistage_gemm
from linalg.utils_gpu import MatmulConfig, MatmulKernels
from quantization.qmatmul_gpu import multistage_gemm_q
from std.memory.unsafe import bitcast

from std.utils.index import Index, IndexList


@always_inline
def _e2m1_pair_to_bf16(
    code_pair: Int32, scale: BFloat16
) -> SIMD[DType.bfloat16, 2]:
    """Decode the two E2M1 nibbles at bits[3:0] and bits[19:16] (same positions
    as int4tobf16's MASK 0x000F000F) via the Marlin bit-positioning trick,
    scaled. Identical formula to `TensorCore.load_b_nvfp4`'s `e2m1tobf16`."""
    var n = SIMD[DType.uint16, 2](
        (code_pair & 0xF).cast[DType.uint16](),
        ((code_pair >> 16) & 0xF).cast[DType.uint16](),
    )
    var q = n << 12
    var bits = (q & 0x8000) | ((q & 0x7000) >> 3)
    var vals = bitcast[DType.float16, 2](bits).cast[DType.float32]()
    var s = scale.cast[DType.float32]() * 16384.0  # fold 2^14
    return (vals * s).cast[DType.bfloat16]()


def create_ref_b_nvfp4[
    type_q: DType,
    type_b: DType,
    b_q_layout: Layout,
    b_layout: Layout,
    group_size: Int,
    pack_factor: Int,
    BLOCK_K: Int = 32,
](
    b_packed: LayoutTensor[type_q, b_q_layout, ImmutAnyOrigin],
    b_out: LayoutTensor[type_b, b_layout, MutAnyOrigin],
):
    """Dequantize the packed NVFP4 weights to a dense [N, K] bf16 reference.

    Mirrors `create_ref_b` in test_multistage_gemm_q.mojo exactly, but decodes
    each 4-bit code as E2M1 instead of symmetric int4 (same nibble->(n,k)
    mapping, so the layout interpretation matches the kernel). BLOCK_K must be
    >= group_size; set BLOCK_K == group_size for fine groups (one scale group
    per reference block)."""
    comptime WARP_SIZE = 32
    comptime BLOCK_N = 128
    comptime repack_tile = Index(64, 16)
    comptime TILE_N = 64
    comptime TILE_K = 16
    comptime num_k_warps = BLOCK_K // repack_tile[1]

    var tid = thread_idx.x
    var warp_id, lane_id = udivmod(tid, WARP_SIZE)
    var block_idx = Index(block_idx.x, block_idx.y)
    var warp_x, warp_y = udivmod(warp_id, num_k_warps)

    comptime group_bytes = group_size // 2 + 2
    comptime N = Int(b_q_layout.shape[0])
    comptime K = Int(b_q_layout.shape[1]) // group_bytes * group_size

    comptime scales_type = DType.bfloat16
    comptime b_type = DType.uint32
    comptime b_weight_layout = Layout.row_major(N // 64, K * 64 // pack_factor)
    var b_q = LayoutTensor[b_type, b_weight_layout, _](
        b_packed.ptr.bitcast[Scalar[b_type]](),
    )

    comptime b_scales_layout = Layout.row_major(K // group_size, N)
    var b_scales_ptr = b_packed.ptr + N * K // 2
    var scales = LayoutTensor[scales_type, b_scales_layout, _](
        b_scales_ptr.bitcast[Scalar[scales_type]](),
    )

    var b_q_gmem_tile = b_q.tile[
        BLOCK_N // repack_tile[0], (BLOCK_K * repack_tile[0]) // pack_factor
    ](block_idx[0], block_idx[1])
    var warp_q_tile = b_q_gmem_tile.tile[
        1, (repack_tile[0] * repack_tile[1]) // pack_factor
    ](warp_x, warp_y)

    var scales_tile = scales.tile[ceildiv(BLOCK_K, group_size), BLOCK_N](
        (block_idx[1] * BLOCK_K) // group_size, block_idx[0]
    )
    var warp_scales_tile = scales_tile.tile[
        ceildiv(BLOCK_K, group_size), repack_tile[0]
    ](0, warp_x)
    comptime smem_reg_scales_layout = Layout.row_major(8, 4)
    var scales_reg_tiles = (
        LayoutTensor[
            scales_type,
            Layout.row_major(repack_tile[0] // 8, 1),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, 1]()
    )
    scales_reg_tiles.vectorize[8, 1]().copy_from(
        warp_scales_tile.vectorize[1, 8]().distribute[
            smem_reg_scales_layout, axis=0
        ](lane_id)
    )

    var b_out_tile = b_out.tile[BLOCK_N, BLOCK_K](block_idx[0], block_idx[1])
    var warp_out_tile = b_out_tile.tile[repack_tile[0], repack_tile[1]](
        warp_x, warp_y
    )
    var mma_tile_iter_1 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 0)
    var mma_tile_iter_2 = warp_out_tile.tiled_iterator[8, 8, axis=0](0, 1)

    var vec = bitcast[DType.int32, 4](warp_q_tile.vectorize[1, 4]()[0, lane_id])

    comptime write_back_type = type_of(mma_tile_iter_1[].vectorize[1, 2]()[0, 0])
    var lane_row, lane_col = udivmod(lane_id, 4)

    comptime for i in range(0, TILE_N // 8, 2):
        var q_int = vec[i // 2]

        var v1 = _e2m1_pair_to_bf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i, 0])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        var v2 = _e2m1_pair_to_bf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i, 0])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v2)
        q_int >>= 4
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()

        v1 = _e2m1_pair_to_bf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1, 0])
        )
        mma_tile_iter_1[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v1)
        q_int >>= 4
        v2 = _e2m1_pair_to_bf16(
            q_int, bitcast[DType.bfloat16, 1](scales_reg_tiles[i + 1, 0])
        )
        mma_tile_iter_2[].vectorize[1, 2]()[lane_row, lane_col] = rebind[
            write_back_type
        ](v2)
        mma_tile_iter_1._incr()
        mma_tile_iter_2._incr()


def test_nvfp4[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    group_size: Int = 128,
    BK: Int = 32,
    ref_block_k: Int = 32,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime pack_factor = 8
    comptime group_bytes = group_size // 2 + 2
    comptime repack_tile = Index(64, 16)

    print("test nvfp4 multistage matmul")
    comptime a_type = DType.bfloat16

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    comptime _b_dim0 = NType.static_value
    comptime _b_dim1 = (KType.static_value // group_size) * group_bytes

    var dynamic_b_shape = IndexList[2](N, (K // group_size) * group_bytes)
    var dynamic_b_ref_shape = IndexList[2](N, K)

    var a_size = M * K
    var b_size = N * ((K // group_size) * group_bytes)
    var b_ref_size = N * K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[a_type](c_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)

    var b_scales_ptr = (b_host_ptr.unsafe_ptr() + N * K // 2).bitcast[
        Scalar[a_type]
    ]()
    var b_scales_size = (K // group_size) * N
    rand(b_scales_ptr, b_scales_size, min=0, max=0.125)
    randint(
        b_host_ptr.unsafe_ptr().bitcast[UInt32](),
        N * (K // pack_factor),
        Int(UInt32.MIN),
        Int(UInt32.MAX),
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var b_device_ref = ctx.enqueue_create_buffer[a_type](b_ref_size)
    var c_device = ctx.enqueue_create_buffer[a_type](c_size)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime b_layout = Layout.row_major(_b_dim0, _b_dim1)
    comptime b_ref_layout = Layout.row_major(
        NType.static_value, KType.static_value
    )
    comptime b_tensor_type = LayoutTensor[dtype, b_layout, _]
    comptime b_ref_tensor_type = LayoutTensor[a_type, b_ref_layout, _]

    var b_tensor = b_tensor_type(
        b_device.unsafe_ptr(),
        RuntimeLayout[
            b_layout,
            element_type=b_tensor_type.layout_int_type,
            linear_idx_type=b_tensor_type.linear_idx_type,
        ].row_major(dynamic_b_shape.cast[b_tensor_type.layout_int_type]()),
    )
    var b_ref_tensor = b_ref_tensor_type(
        b_device_ref.unsafe_ptr(),
        RuntimeLayout[
            b_ref_layout,
            element_type=b_ref_tensor_type.layout_int_type,
            linear_idx_type=b_ref_tensor_type.linear_idx_type,
        ].row_major(
            dynamic_b_ref_shape.cast[b_ref_tensor_type.layout_int_type]()
        ),
    )

    var c_device_ref = ctx.enqueue_create_buffer[a_type](c_size)

    comptime config = MatmulConfig[a_type, dtype, a_type, True](
        block_tile_shape=Index(128, 128, BK),
        warp_tile_shape=Index(64, 64, BK),
        num_pipeline_stages=4,
    )

    var c_tt_shape = row_major(Coord(m, Idx[NType.static_value]))
    var a_tt_shape = row_major(Coord(m, Idx[KType.static_value]))
    var c_dev_tt = TileTensor(c_device, c_tt_shape)
    var a_dev_tt = TileTensor(a_device, a_tt_shape)
    var c_dev_lt = c_dev_tt.to_layout_tensor()
    var a_dev_lt = a_dev_tt.to_layout_tensor()
    var b_dev_lt = b_tensor

    multistage_gemm_q[
        group_size=group_size,
        pack_factor=pack_factor,
        config=config,
        is_nvfp4=True,
    ](
        c_dev_lt,
        a_dev_lt,
        b_dev_lt,
        config,
        ctx,
    )

    comptime dequan = create_ref_b_nvfp4[
        dtype,
        a_type,
        b_tensor.layout,
        b_ref_tensor.layout,
        group_size,
        pack_factor,
        BLOCK_K=ref_block_k,
    ]

    # warps = (BLOCK_N // 64) * (BLOCK_K // repack_tile_K=16)
    comptime ref_threads = 2 * (ref_block_k // 16) * 32
    ctx.enqueue_function[dequan](
        b_tensor,
        b_ref_tensor,
        grid_dim=(ceildiv(N, 128), ceildiv(K, ref_block_k), 1),
        block_dim=(ref_threads, 1, 1),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    comptime kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    comptime config_ref = kernels_ref.ampere_128x128_4
    var c_ref_tt_shape = row_major(Coord(m, Idx[NType.static_value]))
    var b_ref_tt_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var c_ref_tt = TileTensor(c_device_ref, c_ref_tt_shape)
    var b_ref_tt = TileTensor(b_device_ref, b_ref_tt_shape)
    multistage_gemm[transpose_b=True, config=config_ref](
        c_ref_tt,
        a_dev_tt,
        b_ref_tt,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=1e-2,
    )
    print("NVFP4 skeleton GEMM correctness OK: M=", M, " N=", N, " K=", K)

    # Throughput
    comptime nrun = 200

    @always_inline
    @parameter
    def run_func(ctx: DeviceContext) raises:
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c_dev_lt, a_dev_lt, b_dev_lt, config, ctx)

    var nstime = Float64(ctx.execution_time[run_func](nrun)) / Float64(nrun)
    var sectime = nstime * 1e-9
    var tflops = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12 / sectime
    print("  NVFP4 M=", M, " N=", N, " K=", K, ": ", tflops, " TFLOP/s")

    _ = a_device^
    _ = b_device^
    _ = b_device_ref^
    _ = c_device^
    _ = c_device_ref^
    _ = b_tensor


def bench_nvfp4_g16[
    NType: CoordLike,
    KType: CoordLike, //,
    group_size: Int,
    BK: Int,
    stages: Int,
](ctx: DeviceContext, M: Int, n: NType, k: KType) raises:
    """Throughput-only de-risk over (group_size, BK, stages). Timing is
    value-independent so random packed weights/scales are fine. Only valid
    when group_size >= BK (the int4 scale cadence assumption)."""
    comptime a_type = DType.bfloat16
    comptime dtype = DType.uint8
    comptime pack_factor = 8
    comptime group_bytes = group_size // 2 + 2

    var N = Int(n.value())
    var K = Int(k.value())
    comptime _b_dim1 = (KType.static_value // group_size) * group_bytes

    var b_size = N * ((K // group_size) * group_bytes)
    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[a_type](M * N)
    ctx.synchronize()

    comptime b_layout = Layout.row_major(NType.static_value, _b_dim1)
    comptime b_tt_type = LayoutTensor[dtype, b_layout, _]
    var b_lt = b_tt_type(
        b_dev.unsafe_ptr(),
        RuntimeLayout[
            b_layout,
            element_type = b_tt_type.layout_int_type,
            linear_idx_type = b_tt_type.linear_idx_type,
        ].row_major(
            IndexList[2](N, (K // group_size) * group_bytes).cast[
                b_tt_type.layout_int_type
            ]()
        ),
    )
    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[KType.static_value])))
    var c_tt = TileTensor(c_dev, row_major(Coord(M, Idx[NType.static_value])))
    var a2 = a_tt.to_layout_tensor()
    var c2 = c_tt.to_layout_tensor()

    comptime config = MatmulConfig[a_type, dtype, a_type, True](
        block_tile_shape=Index(128, 128, BK),
        warp_tile_shape=Index(64, 64, BK),
        num_pipeline_stages=stages,
    )

    comptime nrun = 200
    # Warmup
    for _ in range(5):
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c2, a2, b_lt, config, ctx)
    ctx.synchronize()
    # Manual wall-clock over nrun back-to-back launches (steady state).
    var t0 = perf_counter_ns()
    for _ in range(nrun):
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c2, a2, b_lt, config, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var nstime = Float64(t1 - t0) / Float64(nrun)
    var tflops = (
        2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12 / (nstime * 1e-9)
    )
    print(
        "  NVFP4 group=",
        group_size,
        " BK=",
        BK,
        " stages=",
        stages,
        " M=",
        M,
        ": ",
        tflops,
        " TFLOP/s",
    )
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Correctness at cadence-1 (group == BK == 32). NOTE: BK=16 is invalid
        # (num_k_mmas=1 breaks the mainloop prefetch); group=16 needs BK>=32 with
        # per-subgroup scales (work in progress).
        test_nvfp4[DType.uint8, group_size=32](ctx, Int(482), Idx[4096], Idx[4096])
        print("--- manual wall-clock timing ---")
        bench_nvfp4_g16[128, 32, 4](ctx, Int(482), Idx[4096], Idx[4096])
        bench_nvfp4_g16[32, 32, 4](ctx, Int(482), Idx[4096], Idx[4096])
        bench_nvfp4_g16[16, 16, 4](ctx, Int(482), Idx[4096], Idx[4096])
