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

Measures five variants:
1. allreduce only
2. allreduce + fused RMSNorm+FP8 (two kernel launches)
3. fully fused allreduce+RMSNorm+FP8 (single kernel launch)
4. allreduce (with add epilogue) + fused RMSNorm+FP8 (two kernel launches)
5. fully fused allreduce+add+RMSNorm+FP8 (single kernel launch)
"""

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
from comm.allreduce import allreduce, elementwise_epilogue_type
from nn.allreduce_residual_rmsnorm_fp8 import (
    allreduce_residual_rmsnorm_fp8,
    allreduce_rmsnorm_fp8,
)
from collections import Optional
from comm.sync import is_p2p_enabled
from gpu import block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import CacheBustingBuffer, arg_parse

from layout import Coord, TileTensor
from layout._layout import row_major
from nn.normalization import rms_norm_fused_fp8
from runtime.asyncrt import DeviceContextPtr
from utils import IndexList
from utils.index import Index
from utils.numerics import max_finite


@always_inline
fn _repoint_input_bufs[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    mut in_bufs: InlineArray[NDBuffer[in_dtype, 2, ImmutAnyOrigin], ngpus],
    cb_inputs: List[CacheBustingBuffer[in_dtype]],
    iteration: Int,
    num_rows: Int,
):
    @parameter
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[in_dtype, 2](
            cb_inputs[i].offset_ptr(iteration), DimList(num_rows, num_cols)
        )


fn _verify_results[
    in_dtype: DType,
    out_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    signal_buffers: List[DeviceBuffer[DType.uint8]],
    mut in_bufs: InlineArray[NDBuffer[in_dtype, 2, ImmutAnyOrigin], ngpus],
    cb_inputs: List[CacheBustingBuffer[in_dtype]],
    ar_out_bufs: InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus],
    ar_out_dev: List[DeviceBuffer[in_dtype]],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    gamma_dev: DeviceBuffer[in_dtype],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
) raises:
    """Verify fused vs fully-fused kernel paths produce matching results.

    Uses fresh DeviceBuffers (not CacheBustingBuffers) for verification
    outputs to ensure D2H copies transfer exactly `length` elements.
    """
    var length = num_rows * num_cols
    var ctx0 = list_of_ctx[0]

    var gamma_tensor = TileTensor(gamma_dev, row_major(Coord(Index(num_cols))))

    # Fresh output buffers for verification (avoid CacheBustingBuffer
    # size mismatch during D2H copy).
    var v_fused_fp8_dev = ctx0.enqueue_create_buffer[out_dtype](length)
    var v_fused_scales_dev = ctx0.enqueue_create_buffer[DType.float32](num_rows)
    var v_ff_fp8_dev = ctx0.enqueue_create_buffer[out_dtype](length)
    var v_ff_scales_dev = ctx0.enqueue_create_buffer[DType.float32](num_rows)

    # Reset signal buffers.
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    _repoint_input_bufs[num_cols=num_cols](in_bufs, cb_inputs, 0, num_rows)
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
        v_fused_fp8_dev.unsafe_ptr(), Index(num_rows, num_cols)
    )
    var v_fused_scale_shape = IndexList[2](num_rows, 1)
    var v_fused_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        v_fused_scales_dev.unsafe_ptr(), v_fused_scale_shape
    )
    var shape = IndexList[2](num_rows, num_cols)
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
        v_ff_fp8_dev.unsafe_ptr(), DimList(num_rows, num_cols)
    )
    var v_ff_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        v_ff_scales_dev.unsafe_ptr(), DimList(num_rows, 1)
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
    ctx0.enqueue_copy(fused_h, v_fused_fp8_dev)
    ctx0.enqueue_copy(ff_h, v_ff_fp8_dev)
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
    ctx0.enqueue_copy(fused_scales_h, v_fused_scales_dev)
    ctx0.enqueue_copy(ff_scales_h, v_ff_scales_dev)
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

    _ = v_fused_fp8_dev^
    _ = v_fused_scales_dev^
    _ = v_ff_fp8_dev^
    _ = v_ff_scales_dev^

    print("Verification PASSED")


fn _verify_add_results[
    in_dtype: DType,
    out_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    signal_buffers: List[DeviceBuffer[DType.uint8]],
    mut in_bufs: InlineArray[NDBuffer[in_dtype, 2, ImmutAnyOrigin], ngpus],
    cb_inputs: List[CacheBustingBuffer[in_dtype]],
    ar_out_bufs: InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus],
    ar_out_dev: List[DeviceBuffer[in_dtype]],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    gamma_dev: DeviceBuffer[in_dtype],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    residual_dev: DeviceBuffer[in_dtype],
    residual_output_dev: DeviceBuffer[in_dtype],
) raises:
    """Verify epilogue-add path (Benchmark 4) vs fully-fused path (Benchmark 5).

    Uses fresh DeviceBuffers for verification outputs to ensure D2H copies
    transfer exactly `length` elements.
    """
    var length = num_rows * num_cols
    var ctx0 = list_of_ctx[0]

    var gamma_tensor = TileTensor(gamma_dev, row_major(Coord(Index(num_cols))))

    # Fresh output buffers for verification.
    var v_ep_fp8_dev = ctx0.enqueue_create_buffer[out_dtype](length)
    var v_ep_scales_dev = ctx0.enqueue_create_buffer[DType.float32](num_rows)
    var v_ff_fp8_dev = ctx0.enqueue_create_buffer[out_dtype](length)
    var v_ff_scales_dev = ctx0.enqueue_create_buffer[DType.float32](num_rows)
    var v_res_out_dev = ctx0.enqueue_create_buffer[in_dtype](length)

    # --- Epilogue path: allreduce (with add epilogue) + fused RMSNorm+FP8 ---
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    _repoint_input_bufs[num_cols=num_cols](in_bufs, cb_inputs, 0, num_rows)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var residual_ptr = residual_dev.unsafe_ptr()

    group_start()

    @parameter
    for i in range(ngpus):
        var ar_ptr_i = ar_out_dev[i].unsafe_ptr()

        @__copy_capture(ar_ptr_i, residual_ptr)
        @always_inline
        @parameter
        fn add_epilogue_v[
            _dtype: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_dtype, size=_width],) -> None:
            var li = coords[0] * num_cols + coords[1]
            var res = residual_ptr.load[width=_width, alignment=_alignment](li)
            ar_ptr_i.store[width=_width, alignment=_alignment](
                li,
                rebind[SIMD[in_dtype, _width]](
                    val + rebind[SIMD[_dtype, _width]](res)
                ),
            )

        allreduce[
            ngpus=ngpus,
            output_lambda = Optional[elementwise_epilogue_type](add_epilogue_v),
        ](in_bufs, ar_out_bufs[i], rank_sigs, list_of_ctx[i])
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Fused RMSNorm + FP8 on GPU 0 (reads from ar_out which has
    # allreduce + residual).
    var ar_ptr_v = ar_out_dev[0].unsafe_ptr()

    @__copy_capture(ar_ptr_v)
    @always_inline
    @parameter
    fn v_ep_fused_in[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var li = idx[0] * num_cols + idx[1]
        return ar_ptr_v.load[width=width, alignment=width](li)

    var v_ep_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        v_ep_fp8_dev.unsafe_ptr(), Index(num_rows, num_cols)
    )
    var v_ep_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        v_ep_scales_dev.unsafe_ptr(), IndexList[2](num_rows, 1)
    )
    var shape = IndexList[2](num_rows, num_cols)
    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        2,
        v_ep_fused_in,
    ](
        shape,
        v_ep_ndbuf,
        gamma_tensor,
        epsilon,
        weight_offset,
        DeviceContextPtr(ctx0),
        scale_ub,
        v_ep_scales_ndbuf,
    )

    # --- Fully-fused path: allreduce+add+RMSNorm+FP8 (single kernel) ---
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var v_ff_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        v_ff_fp8_dev.unsafe_ptr(), DimList(num_rows, num_cols)
    )
    var v_ff_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        v_ff_scales_dev.unsafe_ptr(), DimList(num_rows, 1)
    )
    var v_res_ndbuf = NDBuffer[in_dtype, 2, MutAnyOrigin](
        residual_dev.unsafe_ptr(), DimList(num_rows, num_cols)
    )
    var v_res_out_ndbuf = NDBuffer[in_dtype, 2, MutAnyOrigin](
        v_res_out_dev.unsafe_ptr(), DimList(num_rows, num_cols)
    )

    group_start()

    @parameter
    for i in range(ngpus):
        allreduce_residual_rmsnorm_fp8(
            in_bufs,
            v_res_ndbuf,
            v_ff_ndbuf,
            v_res_out_ndbuf,
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

    # Compare epilogue path vs fully-fused kernel (FP8 values).
    var ep_h = alloc[Scalar[out_dtype]](length)
    var ff_h = alloc[Scalar[out_dtype]](length)
    ctx0.enqueue_copy(ep_h, v_ep_fp8_dev)
    ctx0.enqueue_copy(ff_h, v_ff_fp8_dev)
    ctx0.synchronize()

    var num_errors = 0
    for i in range(length):
        var ev = ep_h[i].cast[DType.float32]()
        var ffv = ff_h[i].cast[DType.float32]()
        if ev != ffv:
            num_errors += 1
            if num_errors <= 5:
                print(
                    "  Add-path mismatch at",
                    i,
                    ": epilogue=",
                    ev,
                    ", fully_fused=",
                    ffv,
                )

    var error_rate = Float32(num_errors) / Float32(length)
    ep_h.free()
    ff_h.free()
    if error_rate > 0.03:
        raise Error(
            String(
                "Add-path too many mismatches: ",
                num_errors,
                " / ",
                length,
                " (",
                error_rate * 100.0,
                "%)",
            )
        )

    # Compare per-row scale factors.
    var ep_scales_h = alloc[Scalar[DType.float32]](num_rows)
    var ff_scales_h = alloc[Scalar[DType.float32]](num_rows)
    ctx0.enqueue_copy(ep_scales_h, v_ep_scales_dev)
    ctx0.enqueue_copy(ff_scales_h, v_ff_scales_dev)
    ctx0.synchronize()

    var scale_errors = 0
    for i in range(num_rows):
        var ep_s = ep_scales_h[i]
        var ff_s = ff_scales_h[i]
        var denom = max(abs(ep_s), Float32(1e-12))
        var rel_diff = abs(ep_s - ff_s) / denom
        if rel_diff > 0.01:
            scale_errors += 1
            if scale_errors <= 5:
                print(
                    "  Add-path scale mismatch at row",
                    i,
                    ": epilogue=",
                    ep_s,
                    ", fully_fused=",
                    ff_s,
                    ", rel_diff=",
                    rel_diff,
                )

    ep_scales_h.free()
    ff_scales_h.free()
    if scale_errors > 0:
        raise Error(
            String(
                "Add-path scale factor mismatches: ",
                scale_errors,
                " / ",
                num_rows,
            )
        )

    _ = v_ep_fp8_dev^
    _ = v_ep_scales_dev^
    _ = v_ff_fp8_dev^
    _ = v_ff_scales_dev^
    _ = v_res_out_dev^

    print("Add-path verification PASSED")


fn bench_allreduce_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    cache_busting: Bool = True,
](num_rows: Int, mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    var length = num_rows * num_cols

    # --- Shared buffer setup ---
    comptime simd_size = simd_width_of[in_dtype, target = get_gpu_target()]()

    # Per-GPU input CacheBustingBuffers (for allreduce).
    var cb_inputs = List[CacheBustingBuffer[in_dtype]]()
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
        cb_inputs.append(
            CacheBustingBuffer[in_dtype](
                length, simd_size, list_of_ctx[i], cache_busting
            )
        )
        ar_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](length)
        )

        var h = alloc[Scalar[in_dtype]](cb_inputs[0].alloc_size())
        host_bufs.append(h)
        # Initialize all cache-busted positions.
        for j in range(cb_inputs[0].alloc_size()):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(cb_inputs[i].device_buffer(), h)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

    # NDBuffer views for allreduce.
    var in_bufs = InlineArray[NDBuffer[in_dtype, 2, ImmutAnyOrigin], ngpus](
        fill={}
    )
    var ar_out_bufs = InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus](
        fill={}
    )
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[in_dtype, 2](
            cb_inputs[i].unsafe_ptr(), DimList(num_rows, num_cols)
        )
        ar_out_bufs[i] = NDBuffer[in_dtype, 2](
            ar_out_dev[i].unsafe_ptr(), DimList(num_rows, num_cols)
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Per-GPU FP8 output and scale buffers — each GPU must write to local
    # memory to avoid P2P traffic that inflates benchmark latency.
    # Bench 2: allreduce + fused RMSNorm+FP8
    var fused_fp8_out_dev = List[DeviceBuffer[out_dtype]](capacity=ngpus)
    var fused_scales_dev = List[DeviceBuffer[DType.float32]](capacity=ngpus)
    # Bench 3: fully fused allreduce+RMSNorm+FP8
    var fully_fused_fp8_out_dev = List[DeviceBuffer[out_dtype]](capacity=ngpus)
    var fully_fused_scales_dev = List[DeviceBuffer[DType.float32]](
        capacity=ngpus
    )
    # Bench 4 & 5: residual variants
    var fused_add_fp8_out_dev = List[DeviceBuffer[out_dtype]](capacity=ngpus)
    var fused_add_scales_dev = List[DeviceBuffer[DType.float32]](capacity=ngpus)
    var residual_out_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    for i in range(ngpus):
        fused_fp8_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[out_dtype](length)
        )
        fused_scales_dev.append(
            list_of_ctx[i].enqueue_create_buffer[DType.float32](num_rows)
        )
        fully_fused_fp8_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[out_dtype](length)
        )
        fully_fused_scales_dev.append(
            list_of_ctx[i].enqueue_create_buffer[DType.float32](num_rows)
        )
        fused_add_fp8_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[out_dtype](length)
        )
        fused_add_scales_dev.append(
            list_of_ctx[i].enqueue_create_buffer[DType.float32](num_rows)
        )
        residual_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](length)
        )

    # Residual input (read-only, GPU 0 is fine — all GPUs read same data).
    var cb_residual = CacheBustingBuffer[in_dtype](
        length, simd_size, list_of_ctx[0], cache_busting
    )

    # Initialize residual buffer with deterministic values.
    var residual_host = alloc[Scalar[in_dtype]](cb_residual.alloc_size())
    for i in range(cb_residual.alloc_size()):
        residual_host[i] = Scalar[in_dtype](i % 127 + 1) / Scalar[in_dtype](127)
    list_of_ctx[0].enqueue_copy(cb_residual.device_buffer(), residual_host)

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

    # Capture per-GPU pointers for closures.
    var residual_ptr_base = cb_residual.unsafe_ptr()
    var fused_fp8_out_ptrs = InlineArray[
        UnsafePointer[Scalar[out_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var fused_scales_ptrs = InlineArray[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var fully_fused_fp8_out_ptrs = InlineArray[
        UnsafePointer[Scalar[out_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var fully_fused_scales_ptrs = InlineArray[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var fused_add_fp8_out_ptrs = InlineArray[
        UnsafePointer[Scalar[out_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var fused_add_scales_ptrs = InlineArray[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var residual_output_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    for i in range(ngpus):
        fused_fp8_out_ptrs[i] = fused_fp8_out_dev[i].unsafe_ptr()
        fused_scales_ptrs[i] = fused_scales_dev[i].unsafe_ptr()
        fully_fused_fp8_out_ptrs[i] = fully_fused_fp8_out_dev[i].unsafe_ptr()
        fully_fused_scales_ptrs[i] = fully_fused_scales_dev[i].unsafe_ptr()
        fused_add_fp8_out_ptrs[i] = fused_add_fp8_out_dev[i].unsafe_ptr()
        fused_add_scales_ptrs[i] = fused_add_scales_dev[i].unsafe_ptr()
        residual_output_ptrs[i] = residual_out_dev[i].unsafe_ptr()

    # ===== Benchmark 1: allreduce only =====

    @parameter
    @always_inline
    fn bench_allreduce_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            _repoint_input_bufs[num_cols=num_cols](
                in_bufs, cb_inputs, cache_iter, num_rows
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
            _repoint_input_bufs[num_cols=num_cols](
                in_bufs, cb_inputs, cache_iter, num_rows
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

            var shape = IndexList[2](num_rows, num_cols)
            var fused_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fused_fp8_out_ptrs[ctx_idx], DimList(num_rows, num_cols)
            )
            var fused_scale_shape = IndexList[2](num_rows, 1)
            var fused_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fused_scales_ptrs[ctx_idx], fused_scale_shape
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
            _repoint_input_bufs[num_cols=num_cols](
                in_bufs, cb_inputs, cache_iter, num_rows
            )

            var ff_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fully_fused_fp8_out_ptrs[ctx_idx], DimList(num_rows, num_cols)
            )
            var ff_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fully_fused_scales_ptrs[ctx_idx], DimList(num_rows, 1)
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
    # ===== Benchmark 4: allreduce (with add epilogue) + fused RMSNorm+FP8 =====

    @parameter
    @always_inline
    fn bench_ar_add_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            _repoint_input_bufs[num_cols=num_cols](
                in_bufs, cb_inputs, cache_iter, num_rows
            )

            # Step 1: Allreduce with add epilogue (fuses allreduce + add).
            var ar_ptr = ar_out_dev[ctx_idx].unsafe_ptr()

            @__copy_capture(ar_ptr, residual_ptr_base)
            @always_inline
            @parameter
            fn add_epilogue[
                _dtype: DType,
                _rank: Int,
                _width: Int,
                *,
                _alignment: Int,
            ](
                coords: IndexList[_rank],
                val: SIMD[_dtype, size=_width],
            ) -> None:
                var li = coords[0] * num_cols + coords[1]
                var res = residual_ptr_base.load[
                    width=_width, alignment=_alignment
                ](li)
                ar_ptr.store[width=_width, alignment=_alignment](
                    li,
                    rebind[SIMD[in_dtype, _width]](
                        val + rebind[SIMD[_dtype, _width]](res)
                    ),
                )

            allreduce[
                ngpus=ngpus,
                output_lambda = Optional[elementwise_epilogue_type](
                    add_epilogue
                ),
            ](in_bufs, ar_out_bufs[ctx_idx], rank_sigs, ctx_inner)

            # Step 2: Fused RMSNorm + FP8 (reads from ar_out which has
            # allreduce + residual).
            @__copy_capture(ar_ptr)
            @always_inline
            @parameter
            fn add_fused_in[
                width: Int, _rank: Int
            ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
                var li = idx[0] * num_cols + idx[1]
                return ar_ptr.load[width=width, alignment=width](li)

            var shape = IndexList[2](num_rows, num_cols)
            var ea_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fused_add_fp8_out_ptrs[ctx_idx], DimList(num_rows, num_cols)
            )
            var ea_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fused_add_scales_ptrs[ctx_idx], IndexList[2](num_rows, 1)
            )

            rms_norm_fused_fp8[
                in_dtype,
                out_dtype,
                DType.float32,
                2,
                add_fused_in,
            ](
                shape,
                ea_ndbuf,
                gamma_tensor,
                epsilon,
                weight_offset,
                DeviceContextPtr(ctx_inner),
                scale_ub,
                ea_scales_ndbuf,
            )

        bench.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_ar_add_fused_iter](
        list_of_ctx,
        BenchId(
            "allreduce_epilogue_add_then_fused_rmsnorm_fp8",
            input_id=bench_name_prefix,
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Benchmark 5: fused allreduce+add+RMSNorm+FP8 (single kernel) =====

    @parameter
    @always_inline
    fn bench_fused_add_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            _repoint_input_bufs[num_cols=num_cols](
                in_bufs, cb_inputs, cache_iter, num_rows
            )

            var fa_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                fused_add_fp8_out_ptrs[ctx_idx], DimList(num_rows, num_cols)
            )
            var fa_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                fused_add_scales_ptrs[ctx_idx], DimList(num_rows, 1)
            )
            var res_ndbuf = NDBuffer[in_dtype, 2, MutAnyOrigin](
                residual_ptr_base, DimList(num_rows, num_cols)
            )
            var res_out_ndbuf = NDBuffer[in_dtype, 2, MutAnyOrigin](
                residual_output_ptrs[ctx_idx], DimList(num_rows, num_cols)
            )

            allreduce_residual_rmsnorm_fp8(
                in_bufs,
                res_ndbuf,
                fa_ndbuf,
                res_out_ndbuf,
                gamma_tensor,
                epsilon,
                weight_offset,
                scale_ub,
                fa_scales_ndbuf,
                rank_sigs,
                ctx_inner,
            )

        bench.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_fused_add_iter](
        list_of_ctx,
        BenchId(
            "fused_allreduce_residual_rmsnorm_fp8",
            input_id=bench_name_prefix,
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    b.dump_report()

    # --- Optional verification: compare fused vs fully-fused on GPU 0 ---
    # Uses fresh DeviceBuffers (not CacheBustingBuffers) for D2H copies
    # to avoid buffer size mismatch that crashed the HIP driver.
    comptime verify = env_get_bool["verify", True]()
    comptime if verify:
        _verify_results[in_dtype, out_dtype, ngpus, num_cols](
            num_rows,
            list_of_ctx,
            signal_buffers,
            in_bufs,
            cb_inputs,
            ar_out_bufs,
            ar_out_dev,
            rank_sigs,
            gamma_dev,
            epsilon,
            weight_offset,
            scale_ub,
        )

    comptime if verify:
        _verify_add_results[in_dtype, out_dtype, ngpus, num_cols](
            num_rows,
            list_of_ctx,
            signal_buffers,
            in_bufs,
            cb_inputs,
            ar_out_bufs,
            ar_out_dev,
            rank_sigs,
            gamma_dev,
            epsilon,
            weight_offset,
            scale_ub,
            cb_residual.device_buffer(),
            residual_out_dev[0],
        )

    # Cleanup.
    residual_host.free()
    gamma_host.free()
    for i in range(ngpus):
        host_bufs[i].free()
    _ = signal_buffers^
    _ = cb_inputs^
    _ = ar_out_dev^
    _ = fused_fp8_out_dev^
    _ = fused_scales_dev^
    _ = fully_fused_fp8_out_dev^
    _ = fully_fused_scales_dev^
    _ = cb_residual^
    _ = fused_add_fp8_out_dev^
    _ = fused_add_scales_dev^
    _ = residual_out_dev^
    _ = gamma_dev^


def main() raises:
    comptime in_dtype = env_get_dtype["in_dtype", DType.bfloat16]()
    comptime out_dtype = env_get_dtype[
        "out_dtype",
        DType.float8_e4m3fnuz if is_amd_gpu() else DType.float8_e4m3fn,
    ]()
    comptime num_gpus = env_get_int["num_gpus", 4]()
    var num_rows = Int(arg_parse("num_rows", 1))
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

    if not is_p2p_enabled():
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
        in_dtype, out_dtype, num_gpus, num_cols, cache_busting
    ](num_rows, m, list_of_ctx)
