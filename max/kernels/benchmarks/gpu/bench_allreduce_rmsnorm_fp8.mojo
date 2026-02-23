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

"""Benchmark for allreduce + RMSNorm + FP8 quantization pipeline.

Measures three variants:
1. allreduce only
2. allreduce + fused RMSNorm+FP8 (two kernel launches)
3. fully fused allreduce+RMSNorm+FP8 (single kernel launch)
"""

from math import align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    is_amd_gpu,
    size_of,
    simd_width_of,
)

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.allreduce import allreduce
from comm.allreduce_rmsnorm_fp8 import allreduce_rmsnorm_fp8
from comm.sync import can_enable_p2p
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target

from layout._coord import Coord
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.normalization import rms_norm_fused_fp8
from runtime.asyncrt import DeviceContextPtr
from utils import IndexList
from utils.index import Index
from utils.numerics import max_finite


# Cache busting helpers: 512 MiB > 2x the infinity cache on MI300x.
fn _calculate_stride(tensor_size: Int, alignment: Int) -> Int:
    return align_up(tensor_size, alignment)


fn _calculate_buffer_size[
    dtype: DType
](tensor_size: Int, alignment: Int) -> Int:
    comptime k512m = 512 * 1024 * 1024
    var stride = _calculate_stride(tensor_size, alignment)
    return align_up(k512m, stride * size_of[dtype]()) // size_of[dtype]()


fn _calculate_offset(iteration: Int, stride: Int, buffer_size: Int) -> Int:
    return (iteration * stride) % buffer_size


@always_inline
fn _get_offset[
    cache_busting: Bool
](cache_iter: Int, data_stride: Int, buf_size: Int) -> Int:
    @parameter
    if cache_busting:
        return _calculate_offset(cache_iter, data_stride, buf_size)
    return 0


@always_inline
fn _repoint_input_bufs[
    in_dtype: DType,
    ngpus: Int,
    num_rows: Int,
    num_cols: Int,
](
    mut in_bufs: InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus],
    in_dev: List[DeviceBuffer[in_dtype]],
    offset: Int,
):
    @parameter
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[in_dtype, 2](
            in_dev[i].unsafe_ptr() + offset, DimList(num_rows, num_cols)
        )


fn _verify_results[
    in_dtype: DType,
    out_dtype: DType,
    ngpus: Int,
    num_rows: Int,
    num_cols: Int,
](
    list_of_ctx: List[DeviceContext],
    signal_buffers: List[DeviceBuffer[DType.uint8]],
    mut in_bufs: InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus],
    in_dev: List[DeviceBuffer[in_dtype]],
    ar_out_bufs: InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus],
    ar_out_dev: List[DeviceBuffer[in_dtype]],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    fused_fp8_out_dev: DeviceBuffer[out_dtype],
    fully_fused_fp8_out_dev: DeviceBuffer[out_dtype],
    fused_scales_dev: DeviceBuffer[DType.float32],
    fully_fused_scales_dev: DeviceBuffer[DType.float32],
    gamma_dev: DeviceBuffer[in_dtype],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
) raises:
    """Verify fused vs fully-fused kernel paths produce matching results."""
    comptime length = num_rows * num_cols

    var gamma_tensor = TileTensor(gamma_dev, row_major(Coord(Index(num_cols))))

    # Reset signal buffers.
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    _repoint_input_bufs[num_rows=num_rows, num_cols=num_cols](
        in_bufs, in_dev, 0
    )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Run allreduce.
    group_start()

    @parameter
    for i in range(ngpus):
        allreduce[ngpus=ngpus](
            in_bufs, ar_out_bufs[i], rank_sigs, list_of_ctx[i]
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var ctx0 = list_of_ctx[0]

    # Fused path: allreduce + fused RMSNorm+FP8.
    var ar_ptr_v = ar_out_dev[0].unsafe_ptr()

    @__copy_capture(ar_ptr_v)
    @always_inline
    @parameter
    fn v_fused_in[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var li = idx[0] * num_cols + idx[1]
        return ar_ptr_v.load[width=width, alignment=width](li)

    var v_fused_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        fused_fp8_out_dev.unsafe_ptr(), Index(num_rows, num_cols)
    )
    var v_fused_scale_shape = IndexList[2](num_rows, 1)
    var v_fused_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        fused_scales_dev.unsafe_ptr(), v_fused_scale_shape
    )
    comptime shape = IndexList[2](num_rows, num_cols)
    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        2,
        v_fused_in,
    ](
        shape,
        v_fused_ndbuf,
        gamma_tensor,
        epsilon,
        weight_offset,
        DeviceContextPtr(ctx0),
        scale_ub,
        v_fused_scales_ndbuf,
    )

    # Fully-fused kernel path.
    # Reset signal buffers for the fully-fused kernel run.
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var v_ff_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        fully_fused_fp8_out_dev.unsafe_ptr(), DimList(num_rows, num_cols)
    )
    var v_ff_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        fully_fused_scales_dev.unsafe_ptr(), DimList(num_rows, 1)
    )

    group_start()

    @parameter
    for i in range(ngpus):
        allreduce_rmsnorm_fp8(
            in_bufs,
            v_ff_ndbuf,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            v_ff_scales_ndbuf,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    ctx0.synchronize()

    # Compare fused (allreduce + fused RMSNorm+FP8) vs fully-fused kernel.
    # The fused path rounds the allreduce sum to bfloat16 before RMSNorm,
    # while the fully-fused kernel keeps the sum in float32 throughout. This
    # can cause ±1 FP8 ULP differences at rounding boundaries, so we compare
    # the cast-to-float32 values with exact equality and allow a small
    # mismatch rate.
    var fused_h = alloc[Scalar[out_dtype]](length)
    var ff_h = alloc[Scalar[out_dtype]](length)
    ctx0.enqueue_copy(fused_h, fused_fp8_out_dev)
    ctx0.enqueue_copy(ff_h, fully_fused_fp8_out_dev)
    ctx0.synchronize()

    var num_errors = 0
    for i in range(length):
        var fv = fused_h[i].cast[DType.float32]()
        var ffv = ff_h[i].cast[DType.float32]()
        if fv != ffv:
            num_errors += 1
            if num_errors <= 5:
                print(
                    "  Mismatch at",
                    i,
                    ": fused=",
                    fv,
                    ", fully_fused=",
                    ffv,
                )

    var error_rate = Float32(num_errors) / Float32(length)
    fused_h.free()
    ff_h.free()
    if error_rate > 0.03:
        raise Error(
            String(
                "Too many mismatches: ",
                num_errors,
                " / ",
                length,
                " (",
                error_rate * 100.0,
                "%)",
            )
        )

    # Compare per-row scale factors (fused vs fully-fused).
    var fused_scales_h = alloc[Scalar[DType.float32]](num_rows)
    var ff_scales_h = alloc[Scalar[DType.float32]](num_rows)
    ctx0.enqueue_copy(fused_scales_h, fused_scales_dev)
    ctx0.enqueue_copy(ff_scales_h, fully_fused_scales_dev)
    ctx0.synchronize()

    var scale_errors = 0
    for i in range(num_rows):
        var fused_s = fused_scales_h[i]
        var ff_s = ff_scales_h[i]
        var denom = max(abs(fused_s), Float32(1e-12))
        var rel_diff = abs(fused_s - ff_s) / denom
        if rel_diff > 0.01:
            scale_errors += 1
            if scale_errors <= 5:
                print(
                    "  Scale mismatch at row",
                    i,
                    ": fused=",
                    fused_s,
                    ", fully_fused=",
                    ff_s,
                    ", rel_diff=",
                    rel_diff,
                )

    fused_scales_h.free()
    ff_scales_h.free()
    if scale_errors > 0:
        raise Error(
            String(
                "Scale factor mismatches: ",
                scale_errors,
                " / ",
                num_rows,
            )
        )

    print("Verification PASSED")


fn bench_allreduce_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    ngpus: Int,
    num_rows: Int,
    num_cols: Int,
    cache_busting: Bool = True,
](mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    comptime length = num_rows * num_cols

    # --- Shared buffer setup ---
    comptime simd_size = simd_width_of[in_dtype, target = get_gpu_target()]()
    var data_stride = _calculate_stride(length, simd_size)
    var buf_in_size = _calculate_buffer_size[in_dtype](length, simd_size)
    var buf_out_fp8_size = _calculate_buffer_size[out_dtype](length, simd_size)
    var alloc_in = buf_in_size if cache_busting else length
    var alloc_out_fp8 = buf_out_fp8_size if cache_busting else length

    # Per-GPU input buffers (for allreduce).
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var ar_out_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[UnsafePointer[Scalar[in_dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    # Signal buffers.
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )
    var temp_bytes = ngpus * size_of[in_dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_in))
        ar_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](length)
        )

        var h = alloc[Scalar[in_dtype]](alloc_in)
        host_bufs.append(h)
        # Initialize all cache-busted positions.
        for j in range(alloc_in):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

    # NDBuffer views for allreduce.
    var in_bufs = InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus](
        fill={}
    )
    var ar_out_bufs = InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus](
        fill={}
    )
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[in_dtype, 2](
            in_dev[i].unsafe_ptr(), DimList(num_rows, num_cols)
        )
        ar_out_bufs[i] = NDBuffer[in_dtype, 2](
            ar_out_dev[i].unsafe_ptr(), DimList(num_rows, num_cols)
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # FP8 output buffers (on GPU 0).
    var fused_fp8_out_dev = list_of_ctx[0].enqueue_create_buffer[out_dtype](
        alloc_out_fp8
    )
    var fused_scales_dev = list_of_ctx[0].enqueue_create_buffer[DType.float32](
        num_rows
    )
    var fully_fused_fp8_out_dev = list_of_ctx[0].enqueue_create_buffer[
        out_dtype
    ](alloc_out_fp8)
    var fully_fused_scales_dev = list_of_ctx[0].enqueue_create_buffer[
        DType.float32
    ](num_rows)

    # Gamma weights.
    var gamma_dev = list_of_ctx[0].enqueue_create_buffer[in_dtype](num_cols)
    var gamma_host = alloc[Scalar[in_dtype]](num_cols)
    for i in range(num_cols):
        gamma_host[i] = (Float64(i + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()
    list_of_ctx[0].enqueue_copy(gamma_dev, gamma_host)

    var gamma_tensor = TileTensor(gamma_dev, row_major(Coord(Index(num_cols))))
    var epsilon = Scalar[in_dtype](0.001)
    var weight_offset = Scalar[in_dtype](0.0)
    var scale_ub = max_finite[out_dtype]().cast[DType.float32]()

    list_of_ctx[0].synchronize()

    # --- Benchmark parameters ---
    var total_bytes = ngpus * length * size_of[in_dtype]()
    var bench_name_prefix = String(
        "allreduce_rmsnorm_fp8/",
        in_dtype,
        "/",
        out_dtype,
        "/",
        ngpus,
        "gpu/",
        num_rows,
        "x",
        num_cols,
    )

    # Capture base pointers for closures.
    var fused_fp8_out_ptr_base = fused_fp8_out_dev.unsafe_ptr()
    var fully_fused_fp8_out_ptr_base = fully_fused_fp8_out_dev.unsafe_ptr()
    var fused_scales_ptr_base = fused_scales_dev.unsafe_ptr()
    var fully_fused_scales_ptr_base = fully_fused_scales_dev.unsafe_ptr()

    # ===== Benchmark 1: allreduce only =====

    @parameter
    @always_inline
    fn bench_allreduce_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            var offset = _get_offset[cache_busting](
                cache_iter, data_stride, buf_in_size
            )
            _repoint_input_bufs[num_rows=num_rows, num_cols=num_cols](
                in_bufs, in_dev, offset
            )
            allreduce[ngpus=ngpus](
                in_bufs, ar_out_bufs[ctx_idx], rank_sigs, ctx_inner
            )

        bench.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_allreduce_iter](
        list_of_ctx,
        BenchId("allreduce_only", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Benchmark 2: allreduce + fused RMSNorm+FP8 =====

    @parameter
    @always_inline
    fn bench_ar_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            var offset_in = _get_offset[cache_busting](
                cache_iter, data_stride, buf_in_size
            )
            _repoint_input_bufs[num_rows=num_rows, num_cols=num_cols](
                in_bufs, in_dev, offset_in
            )

            # Allreduce.
            allreduce[ngpus=ngpus](
                in_bufs, ar_out_bufs[ctx_idx], rank_sigs, ctx_inner
            )

            # Fused RMSNorm + FP8.
            var ar_ptr = ar_out_dev[ctx_idx].unsafe_ptr()

            @__copy_capture(ar_ptr)
            @always_inline
            @parameter
            fn fused_in[
                width: Int, _rank: Int
            ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
                var li = idx[0] * num_cols + idx[1]
                return ar_ptr.load[width=width, alignment=width](li)

            comptime shape = IndexList[2](num_rows, num_cols)
            var fused_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fused_fp8_out_ptr_base, DimList(num_rows, num_cols)
            )
            var fused_scale_shape = IndexList[2](num_rows, 1)
            var fused_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fused_scales_ptr_base, fused_scale_shape
            )

            rms_norm_fused_fp8[
                in_dtype,
                out_dtype,
                DType.float32,
                2,
                fused_in,
            ](
                shape,
                fused_ndbuf,
                gamma_tensor,
                epsilon,
                weight_offset,
                DeviceContextPtr(ctx_inner),
                scale_ub,
                fused_scales_ndbuf,
            )

        bench.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_ar_fused_iter](
        list_of_ctx,
        BenchId(
            "allreduce_then_fused_rmsnorm_fp8",
            input_id=bench_name_prefix,
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Benchmark 3: fully fused allreduce+RMSNorm+FP8 (single kernel) =====

    @parameter
    @always_inline
    fn bench_fully_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            var offset_in = _get_offset[cache_busting](
                cache_iter, data_stride, buf_in_size
            )
            _repoint_input_bufs[num_rows=num_rows, num_cols=num_cols](
                in_bufs, in_dev, offset_in
            )

            var ff_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fully_fused_fp8_out_ptr_base, DimList(num_rows, num_cols)
            )
            var ff_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fully_fused_scales_ptr_base, DimList(num_rows, 1)
            )

            allreduce_rmsnorm_fp8(
                in_bufs,
                ff_ndbuf,
                gamma_tensor,
                epsilon,
                weight_offset,
                scale_ub,
                ff_scales_ndbuf,
                rank_sigs,
                ctx_inner,
            )

        bench.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_fully_fused_iter](
        list_of_ctx,
        BenchId(
            "fused_allreduce_rmsnorm_fp8",
            input_id=bench_name_prefix,
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    b.dump_report()

    # --- Verification: compare fused vs fully-fused on GPU 0 ---
    _verify_results[in_dtype, out_dtype, ngpus, num_rows, num_cols](
        list_of_ctx,
        signal_buffers,
        in_bufs,
        in_dev,
        ar_out_bufs,
        ar_out_dev,
        rank_sigs,
        fused_fp8_out_dev,
        fully_fused_fp8_out_dev,
        fused_scales_dev,
        fully_fused_scales_dev,
        gamma_dev,
        epsilon,
        weight_offset,
        scale_ub,
    )

    # Cleanup.
    gamma_host.free()
    for i in range(ngpus):
        host_bufs[i].free()
    _ = signal_buffers^
    _ = in_dev^
    _ = ar_out_dev^
    _ = fused_fp8_out_dev^
    _ = fully_fused_fp8_out_dev^
    _ = fused_scales_dev^
    _ = fully_fused_scales_dev^
    _ = gamma_dev^


def main():
    comptime in_dtype = env_get_dtype["in_dtype", DType.bfloat16]()
    comptime out_dtype = env_get_dtype[
        "out_dtype",
        DType.float8_e4m3fnuz if is_amd_gpu() else DType.float8_e4m3fn,
    ]()
    comptime num_gpus = env_get_int["num_gpus", 4]()
    comptime num_rows = env_get_int["num_rows", 1]()
    comptime num_cols = env_get_int["num_cols", 8192]()
    comptime cache_busting = env_get_bool["cache_busting", True]()

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < num_gpus:
        print(
            "Need",
            num_gpus,
            "GPUs but only found",
            num_devices,
            "- skipping.",
        )
        return

    if not can_enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_gpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "Benchmarking allreduce + RMSNorm + FP8:",
        num_gpus,
        "GPUs,",
        in_dtype,
        "->",
        out_dtype,
        ",",
        num_rows,
        "x",
        num_cols,
    )

    var m = Bench(BenchConfig(num_repetitions=1))
    bench_allreduce_rmsnorm_fp8[
        in_dtype, out_dtype, num_gpus, num_rows, num_cols, cache_busting
    ](m, list_of_ctx)
