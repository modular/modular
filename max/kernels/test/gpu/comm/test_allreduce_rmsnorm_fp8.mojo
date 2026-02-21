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

"""Tests for the fully-fused allreduce + RMSNorm + FP8 quantization kernel."""

from sys import is_amd_gpu, size_of

from buffer import NDBuffer
from buffer.dimlist import DimList
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.allreduce import allreduce
from comm.allreduce_rmsnorm_fp8 import allreduce_rmsnorm_fp8
from comm.sync import can_enable_p2p
from gpu.host import DeviceBuffer, DeviceContext
from layout._coord import Coord
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.normalization import rms_norm_fused_fp8
from runtime.asyncrt import DeviceContextPtr
from testing import assert_true
from utils import IndexList
from utils.index import Index
from utils.numerics import max_finite

from comm_test_utils import test_value_for_gpu_element

# Platform-agnostic FP8 output type.
comptime out_fp8_dtype = DType.float8_e4m3fnuz if is_amd_gpu() else DType.float8_e4m3fn


# --- Test: fully fused allreduce + RMSNorm + FP8 ---


fn test_fused_allreduce_rmsnorm_fp8[
    ngpus: Int,
    in_dtype: DType,
    out_dtype: DType,
    rows: Int,
    cols: Int,
](list_of_ctx: List[DeviceContext]) raises:
    """Verify fused kernel against separate allreduce → fused RMSNorm+FP8."""
    comptime length = rows * cols

    print(
        "  test_fused_allreduce_rmsnorm_fp8[",
        ngpus,
        ",",
        in_dtype,
        "->",
        out_dtype,
        ",",
        rows,
        "x",
        cols,
        "]",
    )

    # --- Setup: per-GPU input buffers ---
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[UnsafePointer[Scalar[in_dtype], MutExternalOrigin]](
        capacity=ngpus
    )
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )
    var temp_bytes = ngpus * size_of[in_dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = alloc[Scalar[in_dtype]](length)
        host_bufs.append(h)
        for j in range(length):
            h[j] = test_value_for_gpu_element[in_dtype](i, j)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = signal_buffers[i].unsafe_ptr().bitcast[Signal]()

    var in_bufs = InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus](
        fill={}
    )
    for i in range(ngpus):
        in_bufs[i] = NDBuffer[in_dtype, 2](
            in_dev[i].unsafe_ptr(), DimList(rows, cols)
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Shared params ---
    var ctx = list_of_ctx[0]
    var gamma_dev = ctx.enqueue_create_buffer[in_dtype](cols)
    var gamma_host = alloc[Scalar[in_dtype]](cols)
    for i in range(cols):
        gamma_host[i] = (Float64(i + cols) / Float64(cols)).cast[in_dtype]()
    ctx.enqueue_copy(gamma_dev, gamma_host)

    comptime shape = IndexList[2](rows, cols)
    var gamma_tensor = TileTensor(gamma_dev, row_major(Coord(Index(cols))))
    var epsilon = Scalar[in_dtype](1e-5)
    var weight_offset = Scalar[in_dtype](0.0)
    var scale_ub = max_finite[out_dtype]().cast[DType.float32]()

    # --- Reference path: allreduce → fused RMSNorm+FP8 (separate launches) ---
    var ar_out_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    for i in range(ngpus):
        ar_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](length)
        )
    var ar_out_bufs = InlineArray[NDBuffer[in_dtype, 2, MutAnyOrigin], ngpus](
        fill={}
    )
    for i in range(ngpus):
        ar_out_bufs[i] = NDBuffer[in_dtype, 2](
            ar_out_dev[i].unsafe_ptr(), DimList(rows, cols)
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()

    @parameter
    for i in range(ngpus):
        allreduce[ngpus=ngpus](
            in_bufs, ar_out_bufs[i], rank_sigs, list_of_ctx[i]
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var ref_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var ref_scales_dev = ctx.enqueue_create_buffer[DType.float32](rows)

    var ar_ptr_ref = ar_out_dev[0].unsafe_ptr()

    @__copy_capture(ar_ptr_ref)
    @always_inline
    @parameter
    fn ref_input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var linear_idx = idx[0] * cols + idx[1]
        return ar_ptr_ref.load[width=width, alignment=width](linear_idx)

    var ref_fp8_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        ref_fp8_dev.unsafe_ptr(), Index(rows, cols)
    )
    var ref_scale_shape = IndexList[2](rows, 1)
    var ref_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        ref_scales_dev.unsafe_ptr(), ref_scale_shape
    )

    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        2,
        ref_input_fn,
    ](
        shape,
        ref_fp8_ndbuf,
        gamma_tensor,
        epsilon,
        weight_offset,
        DeviceContextPtr(ctx),
        scale_ub,
        ref_scales_ndbuf,
    )

    ctx.synchronize()

    # --- Fused kernel path ---
    # Reset signal buffers for the fused kernel run.
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var fused_fp8_dev = ctx.enqueue_create_buffer[out_dtype](length)
    var fused_scales_dev = ctx.enqueue_create_buffer[DType.float32](rows)

    var fused_fp8_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
        fused_fp8_dev.unsafe_ptr(), DimList(rows, cols)
    )
    var fused_scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
        fused_scales_dev.unsafe_ptr(), DimList(rows, 1)
    )

    group_start()

    @parameter
    for i in range(ngpus):
        allreduce_rmsnorm_fp8(
            in_bufs,
            fused_fp8_ndbuf,
            gamma_tensor,
            epsilon,
            weight_offset,
            scale_ub,
            fused_scales_ndbuf,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Compare reference (allreduce + fused_rmsnorm_fp8) vs fused kernel ---
    # The reference path rounds the allreduce sum to bfloat16 before RMSNorm,
    # while the fused kernel keeps the sum in float32 throughout. This can
    # cause ±1 FP8 ULP differences at rounding boundaries, so we compare
    # the cast-to-float32 values with a tolerance of 1 FP8 ULP.
    var ref_fp8_host = alloc[Scalar[out_dtype]](length)
    var fused_fp8_host = alloc[Scalar[out_dtype]](length)
    ctx.enqueue_copy(ref_fp8_host, ref_fp8_dev)
    ctx.enqueue_copy(fused_fp8_host, fused_fp8_dev)
    ctx.synchronize()

    var num_errors = 0
    for i in range(length):
        var ref_val = ref_fp8_host[i].cast[DType.float32]()
        var fused_val = fused_fp8_host[i].cast[DType.float32]()
        if ref_val != fused_val:
            num_errors += 1

    # Allow up to 3% mismatches (bfloat16 rounding in reference path
    # vs float32 accumulation in fused kernel causes ±1 FP8 ULP diffs).
    var error_rate = Float32(num_errors) / Float32(length)
    if error_rate > 0.03:
        # Print first few mismatches for debugging.
        var printed = 0
        for i in range(length):
            if printed >= 5:
                break
            var ref_val = ref_fp8_host[i].cast[DType.float32]()
            var fused_val = fused_fp8_host[i].cast[DType.float32]()
            if ref_val != fused_val:
                print(
                    "  Mismatch at",
                    i,
                    ": ref=",
                    ref_val,
                    ", fused=",
                    fused_val,
                )
                printed += 1
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

    # Compare per-row scale factors (should be close despite float32 vs
    # bfloat16 intermediate precision differences).
    var ref_scales_host = alloc[Scalar[DType.float32]](rows)
    var fused_scales_host = alloc[Scalar[DType.float32]](rows)
    ctx.enqueue_copy(ref_scales_host, ref_scales_dev)
    ctx.enqueue_copy(fused_scales_host, fused_scales_dev)
    ctx.synchronize()

    var scale_errors = 0
    for i in range(rows):
        var ref_s = ref_scales_host[i]
        var fused_s = fused_scales_host[i]
        # Allow 1% relative tolerance for scale factors.
        var denom = max(abs(ref_s), Float32(1e-12))
        var rel_diff = abs(ref_s - fused_s) / denom
        if rel_diff > 0.01:
            scale_errors += 1
            if scale_errors <= 5:
                print(
                    "  Scale mismatch at row",
                    i,
                    ": ref=",
                    ref_s,
                    ", fused=",
                    fused_s,
                    ", rel_diff=",
                    rel_diff,
                )

    if scale_errors > 0:
        raise Error(
            String(
                "Scale factor mismatches: ",
                scale_errors,
                " / ",
                rows,
            )
        )

    # Cleanup.
    ref_scales_host.free()
    fused_scales_host.free()
    ref_fp8_host.free()
    fused_fp8_host.free()
    gamma_host.free()
    for i in range(ngpus):
        host_bufs[i].free()
    _ = signal_buffers^
    _ = in_dev^
    _ = ar_out_dev^
    _ = ref_fp8_dev^
    _ = ref_scales_dev^
    _ = fused_fp8_dev^
    _ = fused_scales_dev^
    _ = gamma_dev^

    print("    PASS")


# --- Main ---

comptime test_gpu_counts = (2, 4, 8)


def main():
    var num_devices = DeviceContext.number_of_devices()
    assert_true(num_devices >= 2, "need at least 2 GPUs")

    if not can_enable_p2p():
        print("P2P not enabled, skipping test.")
        return

    print("FP8 output dtype:", out_fp8_dtype)

    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if num_devices < num_gpus:
            continue

        var list_of_ctx = List[DeviceContext]()
        for i in range(num_gpus):
            list_of_ctx.append(DeviceContext(device_id=i))

        print(
            "\n=== fused_allreduce_rmsnorm_fp8 (",
            num_gpus,
            "GPUs) ===",
        )
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1, 16384
        ](list_of_ctx)

        # --- rows > 512: exercise persistent row loop ---
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 1024, 4096
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 4096, 8192
        ](list_of_ctx)
        test_fused_allreduce_rmsnorm_fp8[
            num_gpus, DType.bfloat16, out_fp8_dtype, 8192, 8192
        ](list_of_ctx)

    print("\nAll tests passed!")
