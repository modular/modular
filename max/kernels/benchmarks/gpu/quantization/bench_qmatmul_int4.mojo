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

"""Benchmarks the GPU INT4 (GPTQ / Q4) weight-only quantized GEMM.

The kernel under test is `multistage_gemm_q` in `quantization/qmatmul_gpu.mojo`
through its production dispatcher `matmul_gpu_qint4`, so the measured shape
takes the same `MatmulConfig` and the same `group_size`/`pack_factor` the
inference stack uses. `B` is supplied in the repacked layout the dispatcher
expects (64x16 GPTQ tiles, eight 4-bit weights per `uint32`, `bf16` group
scales appended after the packed weights), which is the layout
`gpu_qint4_repack_GPTQ` produces.

Two shape families cover the two regimes of the kernel:

* Decode/latency: `M` in 1..64 against a 7B-class `N = K = 4096` projection.
  Here the kernel streams `N * K / 2` bytes of weights per call, so the
  reported bytes are the weight bytes and the roofline is HBM bandwidth.
* Prefill/tensor-core: `M` in 1024..4096 against the same tiles, where the
  reported flops are `2 * M * N * K` and the roofline is the tensor-core rate.

`M` is a runtime argument so one binary sweeps the whole decode-to-prefill
range; `N`, `K` and `group_size` are compile-time defines because the
dispatcher picks its configuration from their static values.

Each shape reports two rows, because the launch policy this kernel now uses
(programmatic dependent launch) can only shorten a call by overlapping it with
a *neighbouring* call, and one number cannot tell that apart from a shorter
kernel:

* `queued` times `num_iters` consecutive launches inside one timing window, so a
  launch whose prologue starts before the launch ahead of it finished is counted
  as faster here. Each iteration streams its own cache-busting window and no
  launch reads the previous one's output, so the arms are independent GEMMs, not
  a producer-consumer layer chain: this number is per-call stream throughput for
  the op when it is queued behind other work, and it does not measure the
  overlap a real layer chain would expose.
* `isolated` times one launch per stream drain: `DeviceContext.execution_time`
  brackets a single launch with GPU events and blocks on the closing event, so
  the next sample starts with an empty stream. This number is the latency of
  one call and cannot benefit from overlap.

Both numbers are GPU event elapsed, not host wall clock. The isolated arm still
contains the host submission gap between its opening event and the kernel, so it
reads as `floor + kernel` against the launch floor printed below.

A gain that appears in `queued` and not in `isolated` is overlap between
neighbouring launches; a gain in both is a shorter kernel.

Output parity between two builds is checked by the `digest` line: a bitwise
fold of the first `_DIGEST_PROBE` elements of one explicitly produced window,
reduced on the host in a fixed order (sequential FNV-1a over the raw bf16
bits, so it only matches when those elements are bitwise identical), so
identical output prints an identical digest regardless of the launch
configuration each build chooses.

Usage:
    ./bazelw run //max/kernels/benchmarks:gpu/quantization/bench_qmatmul_int4 -- \
        --M=1 -D N=4096 -D K=4096
"""

from std.math import ceildiv
from std.sys import get_defined_bool, get_defined_int
from std.memory.unsafe import bitcast

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import Idx, TileTensor, row_major
from quantization.qmatmul_gpu import matmul_gpu_qint4


# Elements of one window folded into the digest. Position-weighted, so a
# permutation or a value change both move it.
comptime _DIGEST_PROBE = 4096


def _init_qint4_weights_gpu(
    buf: MutPointer[UInt8, MutAnyOrigin], num_bytes: Int32
):
    """Deterministically fills the packed int4 weight bytes of one window.

    The packed row the dispatcher expects is `N * K / 2` weight bytes followed
    by a contiguous `bf16` scale region (`qmatmul_gpu.mojo` slices the scales
    at `N * K // 2`), so the two regions are filled separately: hashing every
    byte, scale bytes included, can produce NaN or infinite scales.
    """
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(num_bytes):
        var h = (UInt32(i) * 2654435761) ^ (UInt32(i >> 7) * 40503)
        h = h ^ (h >> 13)
        buf[i] = UInt8(h & 0xFF)
        i += stride


def _init_qint4_scales_gpu(
    buf: MutPointer[UInt8, MutAnyOrigin],
    scale_byte_offset: Int32,
    num_scales: Int32,
):
    """Writes finite, exactly representable bf16 group scales.

    `1 + 0.125 * (i % 7)` needs at most four significant bits, so it is exact
    in bf16 and can never decode as NaN or infinity.
    """
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(num_scales):
        var scale = Scalar[DType.bfloat16](1.0 + Float32(i % 7) * 0.125)
        var bits = bitcast[DType.uint16](scale)
        var at = Int(scale_byte_offset) + 2 * i
        buf[at] = UInt8(UInt32(bits) & 0xFF)
        buf[at + 1] = UInt8((UInt32(bits) >> 8) & 0xFF)
        i += stride


def _init_qint4_packed[
    group_size: Int, K: Int, N: Int
](b: CacheBustingBuffer[DType.uint8], ctx: DeviceContext[]) raises:
    """Fills every cache-busting window with packed GPTQ weights and scales."""
    comptime weight_bytes = N * (K // 2)
    comptime scale_bytes = (K // group_size) * N * 2
    comptime window_bytes = weight_bytes + scale_bytes
    var num_windows = b.alloc_size() // window_bytes
    comptime block_dim = 256
    for w in range(num_windows):
        ctx.enqueue_function[_init_qint4_weights_gpu](
            b.offset_ptr(w),
            Int32(weight_bytes),
            grid_dim=ceildiv(weight_bytes, block_dim),
            block_dim=block_dim,
        )
        ctx.enqueue_function[_init_qint4_scales_gpu](
            b.offset_ptr(w),
            Int32(weight_bytes),
            Int32(scale_bytes // 2),
            grid_dim=ceildiv(scale_bytes // 2, block_dim),
            block_dim=block_dim,
        )


def _window_digest(
    ctx: DeviceContext[], c_device: DeviceBuffer[DType.bfloat16], size: Int
) raises -> UInt32:
    """Folds the first `_DIGEST_PROBE` elements of `c_device` positionally.

    Each element contributes its raw `bf16` bits, so two buffers hash equal only
    when every probed element is bitwise identical: no arithmetic between the
    load and the fold can round two distinct outputs onto the same value. The
    fold is a sequential FNV-1a, so a value change and a swap of two probed
    elements both move the digest, unlike a plain XOR or an absolute sum. Used
    only on an explicitly produced window (see the caller), never on the
    buffer as the timed loop left it.

    Args:
        ctx: The device context used for the readback.
        c_device: The bf16 output buffer of one kernel invocation.
        size: The number of elements in `c_device`.

    Returns:
        A 32-bit FNV-style fold of the probed window.
    """
    var probe = min(size, _DIGEST_PROBE)
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](probe)
    ctx.enqueue_copy(host, c_device)
    ctx.synchronize()
    var acc: UInt32 = 2166136261
    for i in range(probe):
        acc = (acc ^ UInt32(bitcast[DType.uint16](host[i]))) * 16777619
    return acc


def bench_qgemm[
    N: Int,
    K: Int,
    group_size: Int,
    *,
    cache_busting: Bool,
](
    ctx: DeviceContext[],
    mut bench: Bench,
    M: Int,
    init_type: InitializationType,
    run_benchmark: Bool,
) raises:
    """Benchmarks one `(M, N, K)` cell of the quantized GEMM.

    Parameters:
        N: The static N dimension of the weight matrix.
        K: The static K dimension of the weight matrix.
        group_size: The number of K elements sharing one quantized scale.
        cache_busting: Whether to spread each tensor over a buffer larger
            than the GPU cache so every iteration streams it from HBM.

    Args:
        ctx: The device context used to launch the kernel.
        bench: The benchmark suite receiving the timing report.
        M: The runtime M dimension of the activation matrix.
        init_type: The distribution used to fill the activation and weight
            buffers.
        run_benchmark: Whether to time the kernel or run it once.
    """
    comptime pack_factor = 8
    comptime group_bytes = group_size // 2 + 2
    comptime packed_row_bytes = (K // group_size) * group_bytes
    comptime a_type = DType.bfloat16
    comptime b_type = DType.uint8
    comptime c_type = DType.bfloat16

    var a_size = M * K
    var b_size = N * packed_row_bytes
    var c_size = M * N

    var cb_a = CacheBustingBuffer[a_type](a_size, ctx, cache_busting)
    var cb_b = CacheBustingBuffer[b_type](b_size, ctx, cache_busting)
    var cb_c = CacheBustingBuffer[c_type](c_size, ctx, cache_busting)

    # Every window of every cache-busting buffer gets the same distribution, so
    # each iteration streams a well-conditioned problem and the values do not
    # change between the two arms.
    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    # The packed weights carry two bf16 scale bytes per group; a generic byte
    # fill can make them NaN, so re-fill them deterministically.
    _init_qint4_packed[group_size, K, N](cb_b, ctx)
    ctx.synchronize()

    @inline(.always)
    def launch_window(
        ctx: DeviceContext[], window: Int
    ) raises {var cb_a, var cb_b, var cb_c, imm}:
        var a_tt = TileTensor(cb_a.offset_ptr(window), row_major(M, Idx[K]))
        var b_tt = TileTensor(
            cb_b.offset_ptr(window), row_major[N, packed_row_bytes]()
        )
        var c_tt = TileTensor(cb_c.offset_ptr(window), row_major(M, Idx[N]))
        matmul_gpu_qint4[group_size=group_size](c_tt, a_tt, b_tt, ctx)

    @inline(.always)
    def kernel_launch(ctx: DeviceContext[], iteration: Int) raises {imm}:
        launch_window(ctx, iteration)

    # Window the isolated arm reads next. It advances per sample, so that arm
    # streams a cold copy of the weights exactly like the queued one: pinning
    # window 0 would leave the weights in L2 and report cache bandwidth instead.
    var solo_window = 0

    @inline(.always)
    def kernel_launch_solo(ctx: DeviceContext[]) raises {mut solo_window, imm}:
        launch_window(ctx, solo_window)
        solo_window += 1

    @inline(.always)
    def bench_func_queued(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    @inline(.always)
    def launch_floor(c: DeviceContext[]) raises {var cb_b, imm}:
        c.enqueue_function[_init_qint4_weights_gpu](
            cb_b.offset_ptr(0), Int32(1), grid_dim=1, block_dim=32
        )

    # What one launch costs through this submission path when the stream is
    # empty. `execution_time` opens its window before the launch reaches the
    # GPU, so that gap sits inside every isolated sample; printing it lets the
    # isolated number be read as `floor + kernel` and keeps it comparable to
    # the queued one.
    var floor_samples = 64
    var floor_ns = 0
    for _ in range(floor_samples):
        floor_ns += ctx.execution_time(launch_floor, 1)
    print(
        t"launch floor (M=",
        M,
        t", N=",
        N,
        t", K=",
        K,
        "): ",
        Float64(floor_ns) / Float64(floor_samples) / 1000.0,
        " us",
    )

    @inline(.always)
    def bench_func_isolated(mut b: Bencher) raises {mut solo_window, imm}:
        # One launch per stream drain. `execution_time(f, 1)` brackets a single
        # launch with GPU events and blocks on the closing event, so the next
        # sample starts with an empty stream and cannot overlap the one before
        # it. That makes the per-iteration time this arm reports the duration
        # of one launch, with no prologue of a following launch folded in.
        var total = 0
        for _ in range(b.num_iters):
            total += ctx.execution_time(kernel_launch_solo, 1)
        b.elapsed = total

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.flops, 2 * M * N * K))
    measures.append(
        ThroughputMeasure(
            BenchMetric.bytes, N * K // 2 + (K // group_size) * N * 2
        )
    )
    # Two arms, two rows: `queued` times a stream of consecutive launches, so a
    # launch that starts before the one ahead of it finished counts as faster,
    # and `isolated` times one launch at a time. Both stream independent
    # cache-busted windows, so the first is throughput for the op queued behind
    # other work and not a producer-consumer chain; the second is the per-call
    # latency, and a gain that only shows up in the first is overlap, not a
    # shorter kernel.
    var queued_name = String(
        t"qmatmul_int4(gptq,queued)/M",
        M,
        t"_N",
        N,
        t"_K",
        K,
        t"_g",
        group_size,
    )
    var isolated_name = String(
        t"qmatmul_int4(gptq,isolated)/M",
        M,
        t"_N",
        N,
        t"_K",
        K,
        t"_g",
        group_size,
    )
    if run_benchmark:
        bench.bench_function(
            bench_func_queued, BenchId(queued_name), measures.copy()
        )
        bench.bench_function(
            bench_func_isolated, BenchId(isolated_name), measures.copy()
        )
    else:
        kernel_launch(ctx, 0)
        ctx.synchronize()

    # Digest an explicitly produced window rather than the buffer the timed
    # loop left behind: kbench rotates cache-busting windows per iteration, so
    # the surviving buffer is not one kernel's output. Running window 0 twice
    # and requiring the same digest also pins determinism.
    kernel_launch(ctx, 0)
    ctx.synchronize()
    var digest_first = _window_digest(ctx, cb_c.device_buffer(), c_size)
    kernel_launch(ctx, 0)
    ctx.synchronize()
    var digest_second = _window_digest(ctx, cb_c.device_buffer(), c_size)
    if digest_first != digest_second:
        raise "non-deterministic output: window 0 changed between two runs"
    print("digest window0: ", Int(digest_first))


def main() raises:
    comptime N = get_defined_int["N", 4096]()
    comptime K = get_defined_int["K", 4096]()
    comptime group_size = get_defined_int["group_size", 128]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var M = arg_parse("M", 1)
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var run_benchmark = arg_parse("run_benchmark", True)
    if M <= 0:
        raise "M must be positive"

    var m = Bench()
    with DeviceContext() as ctx:
        bench_qgemm[N, K, group_size, cache_busting=cache_busting](
            ctx, m, M, init_type, run_benchmark
        )
    if run_benchmark:
        m.dump_report()
