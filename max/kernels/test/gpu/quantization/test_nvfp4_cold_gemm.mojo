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
"""COLD-WEIGHT NVFP4 decode-GEMM microbench: the foundational HBM-bandwidth tool.

nsys showed the FP4 decode GEMM is ~79% of batch-64 decode and runs at only
~16% of L40S HBM bandwidth on COLD weights (each of 60 layers has a distinct
weight, so weights are streamed from HBM, never reused). A prior microbench
replayed the SAME weight ~200x; that weight stayed resident in L2 (L40S has
48 MB L2) so the measured TFLOP/s were inflated and did NOT predict serving.

THE COLD METHODOLOGY (the whole point):

  * Allocate a POOL of P DISTINCT weight buffers (distinct device allocations at
    distinct addresses, so they cannot alias in L2) and round-robin through them
    in the timing loop: iteration i uses pool[i % P]. P is chosen per shape so
    that P * weight_bytes >> 48 MB AND the round-robin reuse distance evicts L2
    before a buffer is touched again (here P=16: even the smallest shape's FP4
    weight pool cycles >=352 MB, ~7x L2, so every reuse is a true HBM miss;
    confirmed empirically -- doubling P to 32 moves every COLD number <0.5%).
  * COLD-THROUGHPUT timing = wall-clock: warmup cycles the whole pool once, then
    T iterations (T >= 4*P) back-to-back round-robin with a SINGLE synchronize
    at the end. cold_tput_ms = wall / T. Because launches overlap, the GPU keeps
    many in-flight memory requests across kernel boundaries -> this is PIPELINED
    THROUGHPUT. It is NOT what real decode serving sees.
  * COLD-LATENCY timing = the SAME cold round-robin pool, but ctx.synchronize()
    AFTER EACH GEMM so kernels never overlap -- each runs ALONE. This is the
    single-GEMM cold latency a DEPENDENT decode chain sees (layer N+1 needs
    layer N, so layers cannot pipeline). cold_lat_ms = wall / LATRUN.

WHY THIS MATTERS: skeleton-wk4 measured 70-76% HBM in COLD-THROUGHPUT yet served
SLOWER than the bespoke (395 vs 445 tok/s), and wk1==wk4 in serving. Pipelined
throughput is not the serving bottleneck; the per-GEMM latency of one
non-pipelined GEMM is. A single GEMM may not keep enough memory requests in
flight to saturate HBM, so LATENCY bandwidth can collapse far below THROUGHPUT.

REPORTED per (shape, M, kernel/config), side by side:
  * COLD-THROUGHPUT: per-GEMM time (ms), bandwidth = weight_bytes/time (GB/s),
    % of 864 GB/s peak.
  * COLD-LATENCY: per-GEMM time (ms), bandwidth (GB/s), % of 864 GB/s peak.
  * lat/tput ratio: how much slower one non-pipelined GEMM is vs the pipelined
    per-GEMM time (>=1).

weight_bytes counts the bytes the kernel actually READS from the weight:
  * bespoke nvfp4_gemm: packed E2M1 [N, K//2] uint8 (= N*K/2) + f32 scales
    [N, K//16] (= N*K/16 * 4),
  * skeleton multistage_gemm_q: the combined buffer = N*K/2 packed +
    (K//16)*N*2 bf16 scales.
(A is M*K*2 bytes but for M=1/64 that is tiny vs the weight and, more
importantly, A IS reused across layers in serving, so the cold story is about
the weight; A is excluded from weight_bytes but noted.)

A correctness gate (assert vs an independent dense E2M1-dequant reference) runs
for one config per shape so we know the kernels are correct in this harness.

Output: one block per (shape, M), with a BESPOKE line and skeleton lines for
block(64,64,32)/warp(64,64,32) st4 wk1 and st4 wk4. Each line reports
COLD-THROUGHPUT next to COLD-LATENCY.
"""

from std.math import ceildiv
from std.random import rand, randint, seed
from std.sys import size_of
from std.time import perf_counter_ns

from std.gpu import block_dim, block_idx, thread_idx
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
from linalg.nvfp4_gemm import nvfp4_gemm
from linalg.utils_gpu import MatmulConfig, MatmulKernels

from quantization.qmatmul_gpu import multistage_gemm_q, repack_nvfp4_for_sm8x

from std.utils.index import Index, IndexList

comptime a_type = DType.bfloat16
comptime group_size = 16
comptime pack_factor = 8
comptime group_bytes = group_size // 2 + 2

# L40S spec constants for the bandwidth report.
comptime HBM_PEAK_GBPS = 864.0
comptime L2_BYTES = 48 * 1024 * 1024  # 48 MB

# Pool depth. P * weight_bytes must be >> 48 MB so the round-robin reuse
# distance evicts L2 between reuses. Smallest shape here (q_proj) has a
# 22 MB FP4 weight, so P=16 cycles ~352 MB (~7x L2) -> every reuse misses L2.
comptime POOL = 16
# Timing iteration count; T >= 4*POOL so each pool buffer is read >= 4x cold.
comptime NRUN = 200
# LATENCY-mode iteration count: each iter does its own synchronize so a smaller
# count (~100, still cycling the pool >=6x) is enough and keeps runtime sane.
comptime LATRUN = 100
comptime WARMUP = POOL  # one full pass over the pool


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


# Build the FLOAT32 scales buffer the bespoke nvfp4_gemm consumes: the fp8
# block scale cast to f32 with the per-tensor global scale folded in.
def fold_scales_kernel[
    scales_layout: Layout,
    out_layout: Layout,
](
    block_scales: LayoutTensor[
        DType.float8_e4m3fn, scales_layout, ImmutAnyOrigin
    ],
    global_scale: Float32,
    s_out: LayoutTensor[DType.float32, out_layout, MutAnyOrigin],
):
    comptime N = Int(out_layout.shape[0])
    comptime SC = Int(out_layout.shape[1])
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if Int(idx) >= N * SC:
        return
    var n = Int(idx) // SC
    var s = Int(idx) % SC
    s_out[n, s] = block_scales[n, s].cast[DType.float32]() * global_scale


# Compare a kernel C against the prebuilt dense reference C (host pointers).
def _is_correct(
    c_kernel_host: UnsafePointer[Scalar[a_type], _],
    c_ref_host: UnsafePointer[Scalar[a_type], _],
    count: Int,
) -> Bool:
    try:
        assert_almost_equal(
            c_kernel_host, c_ref_host, count, atol=1e-2, rtol=2e-2
        )
        return True
    except:
        return False


# Build the combined-buffer LayoutTensor the skeleton kernel consumes, from a
# given device buffer (one pool entry).
@always_inline
def _combined_lt[
    _N: Int, _K: Int
](buf: DeviceBuffer[DType.uint8], N: Int, K: Int) -> LayoutTensor[
    DType.uint8,
    Layout.row_major(_N, (_K // group_size) * group_bytes),
    ImmutAnyOrigin,
]:
    comptime b_kernel_layout = Layout.row_major(
        _N, (_K // group_size) * group_bytes
    )
    return LayoutTensor[DType.uint8, b_kernel_layout, ImmutAnyOrigin](
        buf.unsafe_ptr(),
        RuntimeLayout[
            b_kernel_layout,
            element_type=LayoutTensor[
                DType.uint8, b_kernel_layout, ImmutAnyOrigin
            ].layout_int_type,
            linear_idx_type=LayoutTensor[
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


# Pretty-print one measurement line. Reports the side-by-side
# COLD-THROUGHPUT (pipelined, one sync at end) vs COLD-LATENCY (non-pipelined,
# sync after each GEMM = what a dependent decode chain sees).
def _report(
    tag: String,
    N: Int,
    K: Int,
    M: Int,
    correct: Bool,
    weight_bytes: Int,
    cold_tput_ms: Float64,
    cold_lat_ms: Float64,
):
    var tput_s = cold_tput_ms * 1e-3
    var lat_s = cold_lat_ms * 1e-3
    var wb = Float64(weight_bytes)
    var tput_gbps = wb * 1e-9 / tput_s
    var lat_gbps = wb * 1e-9 / lat_s
    var tput_pct = 100.0 * tput_gbps / HBM_PEAK_GBPS
    var lat_pct = 100.0 * lat_gbps / HBM_PEAK_GBPS
    # ratio of latency-bound time to throughput-bound time (>=1; how much
    # the non-pipelined single-GEMM run loses vs the pipelined run).
    var lat_over_tput = cold_lat_ms / cold_tput_ms
    print(
        tag,
        " N=",
        N,
        " K=",
        K,
        " M=",
        M,
        "|",
        "OK  " if correct else "WRONG",
        "| COLD-TPUT",
        cold_tput_ms,
        "ms",
        tput_gbps,
        "GB/s",
        tput_pct,
        "% peak | COLD-LAT",
        cold_lat_ms,
        "ms",
        lat_gbps,
        "GB/s",
        lat_pct,
        "% peak | lat/tput x",
        lat_over_tput,
    )


# --------------------------------------------------------------------------- #
# BESPOKE nvfp4_gemm: cold (round-robin pool) + warm (single weight) timing.  #
# --------------------------------------------------------------------------- #
def bench_bespoke[
    NType: CoordLike,
    KType: CoordLike,
    //,
](
    ctx: DeviceContext,
    M: Int,
    n: NType,
    k: KType,
    a_dev: DeviceBuffer[a_type],
    weight_pool: List[DeviceBuffer[DType.uint8]],
    scale_pool: List[DeviceBuffer[DType.float32]],
    c_ref_dev: DeviceBuffer[a_type],
) raises:
    comptime _N = NType.static_value
    comptime _K = KType.static_value
    var N = Int(n.value())
    var K = Int(k.value())

    var c_dev = ctx.enqueue_create_buffer[a_type](M * N)
    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[_K])))
    var c_tt = TileTensor(c_dev, row_major(Coord(M, Idx[_N])))

    # weight bytes the kernel reads: packed FP4 + f32 scales.
    var weight_bytes = N * (K // 2) + N * (K // group_size) * 4

    # Per-pool-entry weight/scale tensors (built inline at each launch site so
    # the round-robin pool index selects a DISTINCT device allocation).
    @parameter
    @always_inline
    def launch(i: Int) raises:
        var w_tt = TileTensor(
            weight_pool[i], row_major(Coord(Idx[_N], Idx[_K // 2]))
        )
        var s_tt = TileTensor(
            scale_pool[i], row_major(Coord(Idx[_N], Idx[_K // group_size]))
        )
        nvfp4_gemm(ctx, c_tt, a_tt, w_tt, s_tt, M, N, K)

    # ---- correctness (pool[0]) ----
    launch(0)
    var c_kernel_host = ctx.enqueue_create_host_buffer[a_type](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[a_type](M * N)
    ctx.enqueue_copy(c_kernel_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()
    var correct = _is_correct(
        c_kernel_host.unsafe_ptr(), c_ref_host.unsafe_ptr(), M * N
    )

    # ---- COLD-THROUGHPUT: round-robin pool, ONE sync at the end (pipelined) ----
    for i in range(WARMUP):
        launch(i % POOL)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for i in range(NRUN):
        launch(i % POOL)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var cold_tput_ms = Float64(t1 - t0) / Float64(NRUN) * 1e-6

    # ---- COLD-LATENCY: round-robin pool, sync AFTER EACH GEMM (non-pipelined,
    # single-GEMM cold latency = what a dependent decode chain sees) ----
    for i in range(WARMUP):
        launch(i % POOL)
        ctx.synchronize()
    var l0 = perf_counter_ns()
    for i in range(LATRUN):
        launch(i % POOL)
        ctx.synchronize()
    var l1 = perf_counter_ns()
    var cold_lat_ms = Float64(l1 - l0) / Float64(LATRUN) * 1e-6

    _report(
        "BESPOKE ", N, K, M, correct, weight_bytes, cold_tput_ms, cold_lat_ms
    )
    _ = c_dev^


# --------------------------------------------------------------------------- #
# SKELETON multistage_gemm_q[is_nvfp4]: cold (pool) + warm (single) timing.    #
# --------------------------------------------------------------------------- #
def bench_skeleton[
    NType: CoordLike,
    KType: CoordLike,
    //,
    config: MatmulConfig[a_type, DType.uint8, a_type, True],
](
    ctx: DeviceContext,
    M: Int,
    n: NType,
    k: KType,
    a_dev: DeviceBuffer[a_type],
    combined_pool: List[DeviceBuffer[DType.uint8]],
    c_ref_dev: DeviceBuffer[a_type],
) raises:
    comptime _N = NType.static_value
    comptime _K = KType.static_value
    var N = Int(n.value())
    var K = Int(k.value())

    comptime stages = config.num_pipeline_stages
    comptime warpk = config.num_warp_k_partitions

    var c_kernel_dev = ctx.enqueue_create_buffer[a_type](M * N)
    var a_tt = TileTensor(a_dev, row_major(Coord(M, Idx[_K])))
    var c_kernel_tt = TileTensor(c_kernel_dev, row_major(Coord(M, Idx[_N])))
    var a_lt = a_tt.to_layout_tensor()
    var c_kernel_lt = c_kernel_tt.to_layout_tensor()

    # combined buffer the skeleton reads: packed FP4 + bf16 scales.
    var weight_bytes = N * (K // 2) + N * (K // group_size) * size_of[a_type]()

    @parameter
    @always_inline
    def launch(i: Int) raises:
        var b_lt = _combined_lt[_N, _K](combined_pool[i], N, K)
        multistage_gemm_q[
            group_size=group_size,
            pack_factor=pack_factor,
            config=config,
            is_nvfp4=True,
        ](c_kernel_lt, a_lt, b_lt, config, ctx)

    # ---- correctness (pool[0]) ----
    var correct = True
    try:
        launch(0)
        var c_kernel_host = ctx.enqueue_create_host_buffer[a_type](M * N)
        var c_ref_host = ctx.enqueue_create_host_buffer[a_type](M * N)
        ctx.enqueue_copy(c_kernel_host, c_kernel_dev)
        ctx.enqueue_copy(c_ref_host, c_ref_dev)
        ctx.synchronize()
        correct = _is_correct(
            c_kernel_host.unsafe_ptr(), c_ref_host.unsafe_ptr(), M * N
        )
    except:
        correct = False

    # ---- COLD-THROUGHPUT: round-robin pool, ONE sync at the end (pipelined) ----
    for i in range(WARMUP):
        launch(i % POOL)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for i in range(NRUN):
        launch(i % POOL)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var cold_tput_ms = Float64(t1 - t0) / Float64(NRUN) * 1e-6

    # ---- COLD-LATENCY: round-robin pool, sync AFTER EACH GEMM (non-pipelined,
    # single-GEMM cold latency = what a dependent decode chain sees) ----
    for i in range(WARMUP):
        launch(i % POOL)
        ctx.synchronize()
    var l0 = perf_counter_ns()
    for i in range(LATRUN):
        launch(i % POOL)
        ctx.synchronize()
    var l1 = perf_counter_ns()
    var cold_lat_ms = Float64(l1 - l0) / Float64(LATRUN) * 1e-6

    var tag = String("skel st", stages, " wk", warpk, " ")
    _report(tag, N, K, M, correct, weight_bytes, cold_tput_ms, cold_lat_ms)

    _ = c_kernel_dev^


# Build canonical tensors ONCE, then fill a pool of P DISTINCT device buffers
# (each its own allocation -> distinct address -> cannot alias in L2). We repack
# the SAME canonical weights into every pool slot; distinct allocations are what
# force cold HBM reads, not distinct values.
def sweep_shape[
    NType: CoordLike,
    KType: CoordLike,
    //,
](ctx: DeviceContext, M: Int, n: NType, k: KType) raises:
    comptime _N = NType.static_value
    comptime _K = KType.static_value
    var N = Int(n.value())
    var K = Int(k.value())

    var global_scale = Float32(0.5)

    var weights_size = N * (K // 2)
    var scales_size = N * (K // group_size)
    var combined_size = N * K // 2 + (K // group_size) * N * size_of[a_type]()

    # ---- canonical host tensors ----
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

    # ---- BESPOKE POOL: P distinct weight + f32 scale buffers ----
    var weight_pool = List[DeviceBuffer[DType.uint8]]()
    var scale_pool = List[DeviceBuffer[DType.float32]]()
    comptime fold = fold_scales_kernel[scales_layout, scales_layout]
    for _ in range(POOL):
        var wbuf = ctx.enqueue_create_buffer[DType.uint8](weights_size)
        var sbuf = ctx.enqueue_create_buffer[DType.float32](scales_size)
        ctx.enqueue_copy(wbuf, weights_host)
        var s_lt = LayoutTensor[DType.float32, scales_layout, MutAnyOrigin](
            sbuf.unsafe_ptr()
        )
        ctx.enqueue_function[fold](
            scales_fp8_lt,
            global_scale,
            s_lt,
            grid_dim=(ceildiv(scales_size, threads), 1, 1),
            block_dim=(threads, 1, 1),
        )
        weight_pool.append(wbuf^)
        scale_pool.append(sbuf^)
    ctx.synchronize()

    bench_bespoke(ctx, M, n, k, a_dev, weight_pool, scale_pool, c_ref_dev)

    # Free the bespoke pool before allocating the (larger) skeleton pool.
    weight_pool.clear()
    scale_pool.clear()

    # ---- SKELETON POOL: P distinct combined buffers (repacked) ----
    comptime repack = repack_nvfp4_for_sm8x[
        weights_layout, scales_layout, combined_layout, group_size
    ]
    var repack_smem = 128 * 2 * 64
    var combined_pool = List[DeviceBuffer[DType.uint8]]()
    for _ in range(POOL):
        var cbuf = ctx.enqueue_create_buffer[DType.uint8](combined_size)
        var c_lt = LayoutTensor[DType.uint8, combined_layout, MutAnyOrigin](
            cbuf.unsafe_ptr()
        )
        ctx.enqueue_function[repack](
            weights_lt,
            scales_fp8_lt,
            global_scale,
            c_lt,
            grid_dim=(ceildiv(N, 128), ceildiv(K, 1024), 1),
            block_dim=(128, 1, 1),
            shared_mem_bytes=repack_smem,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(repack_smem)
            ),
        )
        combined_pool.append(cbuf^)
    ctx.synchronize()

    comptime C = MatmulConfig[a_type, DType.uint8, a_type, True]

    # block(64,64,32)/warp(64,64,32) st4 wk1
    bench_skeleton[
        config=C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
        )
    ](ctx, M, n, k, a_dev, combined_pool, c_ref_dev)

    # block(64,64,32)/warp(64,64,32) st4 wk4
    bench_skeleton[
        config=C(
            block_tile_shape=Index(64, 64, 32),
            warp_tile_shape=Index(64, 64, 32),
            num_pipeline_stages=4,
            num_warp_k_partitions=4,
        )
    ](ctx, M, n, k, a_dev, combined_pool, c_ref_dev)

    combined_pool.clear()

    _ = a_dev^
    _ = weights_dev^
    _ = scales_dev^
    _ = b_ref_dev^
    _ = c_ref_dev^
    _ = scales_f32^
    _ = a_host^
    _ = weights_host^
    _ = scales_host^


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        var pool_mb_min = Float64(POOL) * Float64(8192 * 5376 // 2) / 1e6
        print(
            "COLD-WEIGHT NVFP4 decode-GEMM microbench (L40S: 864 GB/s HBM, 48"
            " MB L2)"
        )
        print(
            "POOL =",
            POOL,
            " NRUN(tput) =",
            NRUN,
            " LATRUN(lat) =",
            LATRUN,
            " (smallest-shape pool >=",
            pool_mb_min,
            "MB ~",
            pool_mb_min * 1e6 / Float64(L2_BYTES),
            "x L2 -> reuse distance evicts L2)",
        )
        print(
            "weight_bytes excludes A (A is reused across layers in serving);"
            " bandwidth = weight_bytes / time"
        )
        print(
            "COLD-TPUT = pipelined (1 sync/200 GEMMs); COLD-LAT = non-pipelined"
            " (sync after each GEMM) = dependent decode chain"
        )
        # Real gemma4-31B decode GEMM shapes (weight [N, K]); decode batches.
        for M in [Int(1), Int(64)]:
            print("######## M =", M, "########")
            print("==== gate_up_proj (N=21504, K=5376) ====")
            sweep_shape(ctx, M, Idx[21504], Idx[5376])
            print("==== down_proj    (N=5376, K=21504) ====")
            sweep_shape(ctx, M, Idx[5376], Idx[21504])
            print("==== q_proj       (N=8192, K=5376)  ====")
            sweep_shape(ctx, M, Idx[8192], Idx[5376])
