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
"""CONFIG-SWEEP microbenchmark for the NVFP4 multistage-skeleton GEMM.

For each gemma4 (M, N, K) shape and a fixed LIST of candidate MatmulConfigs,
this:

  1. builds canonical NVFP4 tensors + an INDEPENDENT dense E2M1-dequant
     reference (config-independent, built once per shape),
  2. repacks the canonical tensors via `repack_nvfp4_for_sm8x`
     (config-independent, once per shape),
  3. for each config: runs `multistage_gemm_q[is_nvfp4=True]`, verifies the
     result against the dense reference (atol 1e-2 / rtol 2e-2), and reports a
     manual wall-clock TFLOP/s (warmup + N enqueues + ONE synchronize -- never
     ctx.execution_time, which adds ~900us/call and fabricates cliffs).

Output is one line per (shape, config):

    M N K | BM BN WM WN stages warpk | <OK/WRONG/COMPILE_SKIP> | <TFLOP/s>

Correctness GATES the number: a WRONG config still prints its line (flagged
WRONG) so that "split-K breaks NVFP4 scale staging" findings stay visible, but
its TFLOP/s must not be trusted for dispatch.
"""

from std.math import ceildiv
from std.random import rand, randint, seed
from std.sys import size_of
from std.time import perf_counter_ns

from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute

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
from quantization.qmatmul_gpu import multistage_gemm_q, repack_nvfp4_for_sm8x

from std.utils.index import Index, IndexList

comptime a_type = DType.bfloat16
comptime group_size = 16
comptime pack_factor = 8
comptime group_bytes = group_size // 2 + 2


# E2M1 canonical magnitude LUT: codes 0..7 -> {0,0.5,1,1.5,2,3,4,6}; bit 3 is
# the sign.
@always_inline
def _e2m1_decode(code: UInt8) -> Float32:
    var c = Int(code & 0x7)
    var mags = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    var v = mags[c]
    return -v if (code & 0x8) != 0 else v


# Build the dense [N, K] bf16 reference B from the CANONICAL tensors.
def dequant_ref_kernel[
    weights_layout: Layout,
    scales_layout: Layout,
    out_layout: Layout,
](
    weights: LayoutTensor[DType.uint8, weights_layout, ImmutAnyOrigin],
    block_scales: LayoutTensor[
        DType.float8_e4m3fn, scales_layout, ImmutAnyOrigin
    ],
    global_scale: Float32,
    b_out: LayoutTensor[DType.bfloat16, out_layout, MutAnyOrigin],
):
    comptime N = Int(out_layout.shape[0])
    comptime K = Int(out_layout.shape[1])
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if Int(idx) >= N * K:
        return
    var n = Int(idx) // K
    var k = Int(idx) % K

    var byte = weights[n, k // 2]
    var nibble = (byte & 0x0F) if (k % 2 == 0) else ((byte >> 4) & 0x0F)
    var code_val = _e2m1_decode(rebind[UInt8](nibble))

    var fp8 = block_scales[n, k // group_size]
    var scale = fp8.cast[DType.float32]() * global_scale

    b_out[n, k] = (code_val * scale).cast[DType.bfloat16]()


# Run ONE comptime MatmulConfig at a given shape: skeleton GEMM, correctness
# check vs. the prebuilt dense reference, and a manual wall-clock TFLOP/s.
# a_dev / b_kernel buffer / c_ref are prebuilt by the caller and shared across
# configs for this shape.
def run_one_config[
    NType: CoordLike,
    KType: CoordLike, //,
    config: MatmulConfig[a_type, DType.uint8, a_type, True],
](
    ctx: DeviceContext,
    M: Int,
    n: NType,
    k: KType,
    a_dev: DeviceBuffer[a_type],
    combined_dev: DeviceBuffer[DType.uint8],
    c_ref_dev: DeviceBuffer[a_type],
) raises:
    comptime _N = NType.static_value
    comptime _K = KType.static_value
    var N = Int(n.value())
    var K = Int(k.value())

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime WM = config.warp_tile_shape[0]
    comptime WN = config.warp_tile_shape[1]
    comptime stages = config.num_pipeline_stages
    comptime warpk = config.num_warp_k_partitions

    var c_kernel_dev = ctx.enqueue_create_buffer[a_type](M * N)

    comptime b_kernel_layout = Layout.row_major(
        _N, (_K // group_size) * group_bytes
    )
    var b_kernel_lt = LayoutTensor[
        DType.uint8, b_kernel_layout, ImmutAnyOrigin
    ](
        combined_dev.unsafe_ptr(),
        RuntimeLayout[
            b_kernel_layout,
            element_type = LayoutTensor[
                DType.uint8, b_kernel_layout, ImmutAnyOrigin
            ].layout_int_type,
            linear_idx_type = LayoutTensor[
                DType.uint8, b_kernel_layout, ImmutAnyOrigin
            ].linear_idx_type,
        ].row_major(
            IndexList[2](N, (K // group_size) * group_bytes).cast[
                LayoutTensor[
                    DType.uint8, b_kernel_layout, ImmutAnyOrigin
                ].layout_int_type
            ]()
        ),
    )

    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[_K])))
    var c_kernel_tt = TileTensor(c_kernel_dev, row_major(Coord(M, Idx[_N])))
    var a_lt = a_tt.to_layout_tensor()
    var c_kernel_lt = c_kernel_tt.to_layout_tensor()

    # ---- correctness ----
    multistage_gemm_q[
        group_size=group_size,
        pack_factor=pack_factor,
        config=config,
        is_nvfp4=True,
    ](c_kernel_lt, a_lt, b_kernel_lt, config, ctx)

    var c_kernel_host = ctx.enqueue_create_host_buffer[a_type](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[a_type](M * N)
    ctx.enqueue_copy(c_kernel_host, c_kernel_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    var correct = True
    try:
        assert_almost_equal(
            c_kernel_host.unsafe_ptr(),
            c_ref_host.unsafe_ptr(),
            M * N,
            atol=1e-2,
            rtol=2e-2,
        )
    except e:
        correct = False

    # ---- manual wall-clock timing ----
    comptime nrun = 200
    for _ in range(10):
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c_kernel_lt, a_lt, b_kernel_lt, config, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(nrun):
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c_kernel_lt, a_lt, b_kernel_lt, config, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var nstime = Float64(t1 - t0) / Float64(nrun)
    var tflops = (
        2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12 / (nstime * 1e-9)
    )

    print(
        M,
        N,
        K,
        "|",
        BM,
        BN,
        WM,
        WN,
        stages,
        warpk,
        "|",
        "OK   " if correct else "WRONG",
        "|",
        tflops,
        "TFLOP/s",
    )

    _ = c_kernel_dev^


# Build the per-shape, config-independent fixture (canonical tensors, dense ref
# B, repacked kernel buffer, reference C = A @ B^T), then sweep every candidate
# config against it.
def sweep_shape[
    NType: CoordLike,
    KType: CoordLike, //,
](ctx: DeviceContext, M: Int, n: NType, k: KType) raises:
    comptime _N = NType.static_value
    comptime _K = KType.static_value
    var N = Int(n.value())
    var K = Int(k.value())

    var global_scale = Float32(0.5)

    var weights_size = N * (K // 2)
    var scales_size = N * (K // group_size)
    var combined_size = N * K // 2 + (K // group_size) * N * size_of[a_type]()

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var weights_host = ctx.enqueue_create_host_buffer[DType.uint8](weights_size)
    var scales_host = ctx.enqueue_create_host_buffer[DType.uint8](scales_size)

    rand(a_host.unsafe_ptr(), M * K)
    randint(weights_host.unsafe_ptr(), weights_size, 0, 255)
    var scales_f32 = ctx.enqueue_create_host_buffer[DType.float32](scales_size)
    rand(scales_f32.unsafe_ptr(), scales_size, min=0.125, max=1.0)
    var scales_fp8 = scales_host.unsafe_ptr().bitcast[
        Scalar[DType.float8_e4m3fn]
    ]()
    for i in range(scales_size):
        scales_fp8[i] = scales_f32.unsafe_ptr()[i].cast[DType.float8_e4m3fn]()

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var weights_dev = ctx.enqueue_create_buffer[DType.uint8](weights_size)
    var scales_dev = ctx.enqueue_create_buffer[DType.uint8](scales_size)
    var combined_dev = ctx.enqueue_create_buffer[DType.uint8](combined_size)
    var b_ref_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var c_ref_dev = ctx.enqueue_create_buffer[a_type](M * N)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(weights_dev, weights_host)
    ctx.enqueue_copy(scales_dev, scales_host)

    comptime weights_layout = Layout.row_major(_N, _K // 2)
    comptime scales_layout = Layout.row_major(_N, _K // group_size)
    comptime combined_layout = Layout.row_major(1, _N * _K // 2)

    var weights_lt = LayoutTensor[DType.uint8, weights_layout, ImmutAnyOrigin](
        weights_dev.unsafe_ptr()
    )
    var scales_fp8_lt = LayoutTensor[
        DType.float8_e4m3fn, scales_layout, ImmutAnyOrigin
    ](scales_dev.unsafe_ptr().bitcast[Scalar[DType.float8_e4m3fn]]())
    var combined_lt = LayoutTensor[
        DType.uint8, combined_layout, MutAnyOrigin
    ](combined_dev.unsafe_ptr())

    # ---- dense bf16 reference B ----
    comptime b_ref_layout = Layout.row_major(_N, _K)
    var b_ref_lt = LayoutTensor[a_type, b_ref_layout, MutAnyOrigin](
        b_ref_dev.unsafe_ptr()
    )
    comptime dequant = dequant_ref_kernel[
        weights_layout, scales_layout, b_ref_layout
    ]
    comptime threads = 256
    ctx.enqueue_function[dequant](
        weights_lt,
        scales_fp8_lt,
        global_scale,
        b_ref_lt,
        grid_dim=(ceildiv(N * K, threads), 1, 1),
        block_dim=(threads, 1, 1),
    )

    # ---- repack canonical -> kernel buffer ----
    comptime repack = repack_nvfp4_for_sm8x[
        weights_layout, scales_layout, combined_layout, group_size
    ]
    var repack_smem = 128 * 2 * 64
    ctx.enqueue_function[repack](
        weights_lt,
        scales_fp8_lt,
        global_scale,
        combined_lt,
        grid_dim=(ceildiv(N, 128), ceildiv(K, 1024), 1),
        block_dim=(128, 1, 1),
        shared_mem_bytes=repack_smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(repack_smem)
        ),
    )

    # ---- reference matmul C_ref = A @ B_dense^T ----
    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[_K])))
    comptime kernels_ref = MatmulKernels[a_type, a_type, a_type, True]()
    comptime config_ref = kernels_ref.ampere_128x128_4
    var c_ref_tt = TileTensor(c_ref_dev, row_major(Coord(M, Idx[_N])))
    var b_ref_tt = TileTensor(b_ref_dev, row_major(Coord(Idx[_N], Idx[_K])))
    multistage_gemm[transpose_b=True, config=config_ref](
        c_ref_tt, a_tt, b_ref_tt, ctx
    )
    ctx.synchronize()

    # Free fixture buffers we no longer need before the sweep allocates more.
    _ = weights_dev^
    _ = scales_dev^
    _ = b_ref_dev^
    _ = scales_f32^
    _ = a_host^
    _ = weights_host^
    _ = scales_host^

    # ----------------------------------------------------------------- #
    # SWEEP: one comptime MatmulConfig per call. All BK=32, group=16.    #
    # num_warps = (BM/WM)*(BN/WN); total threads = num_warps*warpk*32    #
    # must be <= 1024.                                                   #
    # ----------------------------------------------------------------- #
    comptime C = MatmulConfig[a_type, DType.uint8, a_type, True]

    # --- block(64,64,32) warp(64,64,32): num_warps=1 ---
    # stages 3/4/5 x warpk 1/2/4  (threads = 32 / 64 / 128)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=3,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=3,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=3,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=5,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=5,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=5,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)

    # --- block(64,64,32) warp(32,64,32): num_warps=2 ---
    # stages4 x warpk 2/4  (threads = 128 / 256)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)

    # --- block(32,64,32) warp(32,64,32): num_warps=1 ---
    # stages 3/4 x warpk 2/4  (threads = 64 / 128)
    run_one_config[
        C(
            block_tile_shape=Index(32, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=3,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(32, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=3,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(32, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(32, 64, 32),
            warp_tile_shape=Index(32, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)

    # --- block(64,128,32) warp(64,64,32): num_warps=2 ---
    # stages4 x warpk 1/2  (threads = 64 / 128)
    run_one_config[
        C(
            block_tile_shape=Index(64, 128, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)
    run_one_config[
        C(
            block_tile_shape=Index(64, 128, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=2,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)

    # --- block(128,128,32) warp(64,64,32) stages4 warpk1: CURRENT DEFAULT ---
    run_one_config[
        C(
            block_tile_shape=Index(128, 128, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
        )
    ](ctx, M, n, k, a_dev, combined_dev, c_ref_dev)

    _ = a_dev^
    _ = combined_dev^
    _ = c_ref_dev^


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        print(
            "M N K | BM BN WM WN stages warpk | correct | TFLOP/s   "
            "(group=16, BK=32; correctness GATES the number)"
        )
        # (N, K) gemma4 shapes x M buckets. 482 confirms the harness matches
        # the ~167 TFLOP/s large-M baseline.
        for M in [Int(64), Int(128), Int(256), Int(482)]:
            print("==== shape (N=4096, K=4096) ====")
            sweep_shape(ctx, M, Idx[4096], Idx[4096])
            print("==== shape (N=15360, K=3840) ====")
            sweep_shape(ctx, M, Idx[15360], Idx[3840])
            print("==== shape (N=3840, K=15360) ====")
            sweep_shape(ctx, M, Idx[3840], Idx[15360])
