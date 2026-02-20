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

from collections import InlineArray
from math import align_up
from sys import env_get_bool, env_get_dtype, env_get_int, size_of, simd_width_of
from utils.numerics import get_accum_type

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import NDBuffer
from buffer.dimlist import DimList
from comm.sync import can_enable_p2p
from comm.allreduce import allreduce
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
)
from gpu.primitives.grid_controls import PDLLevel
from internal_utils import (
    InitializationType,
    arg_parse,
    pytorch_like_tolerances_for,
    human_readable_size,
)

from testing import assert_almost_equal, assert_true

from utils.index import IndexList, StaticTuple


@always_inline
@parameter
fn _per_gpu_value[
    dtype: DType,
](gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


# TODO: convert 'ngpus' to runtime variable
fn bench_reduce[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    use_quickreduce: Bool,
    cache_busting: Bool,
    use_vendor_ccl: Bool = False,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
    ragged: Bool,
) raises:
    comptime assert ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"
    comptime assert rank == 1, "this test code currently assumes rank 1"

    var name = String(
        _get_test_str[dtype, use_multimem, use_vendor_ccl, cache_busting](
            ngpus, num_bytes, ragged
        )
    )

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype], MutExternalOrigin]](
        capacity=ngpus
    )

    comptime num_buffers = 1 if use_multimem else ngpus

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        fill={}
    )

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // size_of[dtype]()

    comptime simd_size = simd_width_of[dtype, target = get_gpu_target()]()
    var stride = align_up(length, simd_size)
    comptime m512 = 512 * 1024 * 1024
    var cache_elems = (
        align_up(m512, stride * size_of[dtype]()) // size_of[dtype]()
    )

    # Initialize buffers for each GPU
    comptime for gpu_idx in range(ngpus):
        # Create and store device buffers (outputs)
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = alloc[Scalar[dtype]](cache_elems)
        host_buffers.append(host_buffer)

        for i in range(cache_elems // stride):
            for j in range(length):
                host_buffer[i * stride + j] = _per_gpu_value[dtype](gpu_idx, j)

        comptime if not use_multimem:
            # Create per-GPU input buffers on device and copy from host
            in_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](cache_elems)
            )
            list_of_ctx[gpu_idx].enqueue_copy(
                in_bufs_list[gpu_idx], host_buffer
            )

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[DType.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[gpu_idx].enqueue_memset[DType.uint8](
            signal_buffers[gpu_idx], 0
        )
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx].unsafe_ptr().bitcast[Signal]()
        )

    # Create and initialize input and output buffers.
    var in_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], num_buffers](
        fill={}
    )
    var out_bufs = InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus](
        fill={}
    )

    var multi_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    comptime if use_multimem:
        multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), cache_elems
        )

        comptime for i in range(ngpus):
            var unicast_buf = multicast_buf.unicast_buffer_for(list_of_ctx[i])
            list_of_ctx[i].enqueue_copy(unicast_buf, host_buffers[i])

        # All GPUs use the same multicast pointer
        in_bufs[0] = NDBuffer[dtype, rank](
            multicast_buf.multicast_buffer_for(list_of_ctx[0]).unsafe_ptr(),
            DimList(length),
        )
        multi_ptr = multicast_buf.multicast_buffer_for(
            list_of_ctx[0]
        ).unsafe_ptr()
    else:
        comptime for i in range(ngpus):
            in_bufs[i] = NDBuffer[dtype, rank](
                in_bufs_list[i].unsafe_ptr(), DimList(length)
            )

    for i in range(ngpus):
        out_bufs[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        # Ensure setup has propagated.
        list_of_ctx[i].synchronize()

    # Zero device output buffers once before benchmarking so verification isn't
    # affected by any stale data in case a kernel path doesn't overwrite fully.
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_memset(out_bufs_list[i], 0)

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, rank, MutAnyOrigin], ngpus
    ](NDBuffer[dtype, rank, MutAnyOrigin]())

    comptime for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    # Monotonic counter to color quickreduce flags across launches.
    var quickreduce_iter = 0

    # Pre-initialize vendor CCL communicators from the main thread.
    # ncclCommInitAll is not thread-safe, so we must initialize before
    # spawning worker threads.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_allreduce_available():
            raise "Vendor CCL not available; skipping vendor path."
        vendor_ccl.init_comms(ngpus)

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher, ctx: DeviceContext, ctx_idx: Int) raises:
        @parameter
        @always_inline
        fn call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises:
            # Offset the input buffer if cache_busting
            var offset = 0

            comptime if cache_busting:
                offset = (cache_iter * stride) % cache_elems

            comptime if not use_multimem:
                comptime for i in range(ngpus):
                    in_bufs[i] = NDBuffer[dtype, rank](
                        in_bufs_list[i].unsafe_ptr() + offset,
                        DimList(length),
                    )
            else:
                in_bufs[0] = NDBuffer[dtype, rank](
                    multi_ptr + offset, DimList(length)
                )
            # Run allreduce
            comptime allreduce_kernel = vendor_ccl.allreduce if use_vendor_ccl else allreduce
            allreduce_kernel[
                ngpus=ngpus,
                use_multimem=use_multimem,
                use_quickreduce=use_quickreduce,
            ](
                in_bufs,
                out_bufs[ctx_idx],
                rank_sigs,
                ctx_inner,
                max_num_blocks,
                quickreduce_iter,
            )

        b.iter_custom[call_fn](ctx)

    b.bench_multicontext[bench_iter](
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    var max_time = b.info_vec[0].result.mean(unit="ms")
    var gbps = Float64(num_bytes) / (max_time * 1000 * 1000)
    # algbw and busbw are explain in the following link:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    var busbw = 2 * gbps * Float64((ngpus - 1)) / Float64(ngpus)
    print(
        "|",
        name,
        "| slowest mean time",
        max_time,
        "ms |",
        "algbw:",
        gbps,
        "GB/s |",
        "busbw:",
        busbw,
        "GB/s |",
    )

    # Copy results back and verify
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    # Verify results
    # For low-precision dtypes (e.g., bfloat16), inputs were quantized to `dtype`
    # before reduction on device. Mirror the device path here by:
    #  - quantizing each per-GPU term to `dtype` by calling _per_gpu_value[dtype](...)
    #  - accumulating in Float32
    #  - finally casting to `dtype` for the expected value
    comptime for i in range(ngpus):
        for j in range(length):
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)

            comptime for k in range(ngpus):
                var term_dtype = _per_gpu_value[dtype](k, j)
                accum += Scalar[accum_t](term_dtype)
            var expected_sum = Scalar[dtype](accum)
            try:
                var rtol, atol = pytorch_like_tolerances_for[dtype]()
                assert_almost_equal(
                    host_buffers[i][j], expected_sum, atol=atol, rtol=rtol
                )
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e^

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
    _ = signal_buffers^


fn _get_test_str[
    dtype: DType, use_multimem: Bool, use_vendorccl: Bool, cache_busting: Bool
](ngpus: Int, num_bytes: Int, ragged: Bool) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var vendorccl_tag = "-vendorccl" if use_vendorccl else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    var ragged_tag = "-ragged" if ragged else ""
    return String(
        "allreduce-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        vendorccl_tag,
        cache_tag,
        ragged_tag,
        "-",
        human_readable_size(num_bytes),
    )


def main():
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = env_get_int["num_gpus", 2]()
    comptime rank = env_get_int["rank", 1]()
    comptime ragged = env_get_bool["ragged", False]()
    # Force passing `max_num_blocks` explicitly.
    var max_nb = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)
    comptime use_multimem = env_get_bool["use_multimem", False]()
    comptime use_quickreduce = env_get_bool["use_quickreduce", False]()
    comptime use_vendor_ccl = env_get_bool["use_vendor_ccl", False]()
    comptime cache_busting = True

    # When ragged, add (ngpus/2) * simd_width elements to create uneven partitions
    comptime simd_size = simd_width_of[dtype, target = get_gpu_target()]()

    comptime if ragged:
        num_bytes += (num_gpus // 2) * simd_size * size_of[dtype]()

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not can_enable_p2p():
        # Don't benchmark the naive allreduce.
        print("P2P not enabled, skipping benchmark.")
        return

    bench_reduce[
        dtype=dtype,
        rank=rank,
        ngpus=num_gpus,
        use_multimem=use_multimem,
        use_quickreduce=use_quickreduce,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes, max_num_blocks, ragged)
