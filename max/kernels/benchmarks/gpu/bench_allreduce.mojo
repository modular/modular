# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from math import floor, align_up
from sys import env_get_bool, env_get_dtype, env_get_int, size_of, simd_width_of
from utils.numerics import get_accum_type

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from comm.allreduce import MAX_GPUS, Signal, allreduce, can_enable_p2p
import comm.vendor.ccl as vendor_ccl
from gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
)
from internal_utils import InitializationType, arg_parse
from memory import LegacyUnsafePointer as UnsafePointer
from testing import assert_almost_equal, assert_true
from algorithm import sync_parallelize

from utils.index import IndexList, StaticTuple


@always_inline
fn _pytorch_like_tolerances_for[dtype: DType]() -> Tuple[Float64, Float64]:
    # Returns (rtol, atol) modeled after PyTorch defaults.
    @parameter
    if dtype is DType.float16:
        return (1e-3, 1e-5)
    elif dtype is DType.bfloat16:
        return (1.6e-2, 1e-5)
    elif dtype is DType.float32:
        return (1.3e-6, 1e-5)
    elif dtype is DType.float64:
        return (1e-7, 1e-7)
    else:
        return (0.0, 0.0)


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


fn _human_memory(size: Int) -> String:
    comptime KB = 1024
    comptime MB = KB * KB
    comptime GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"

    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"

    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"

    return String(size) + "B"


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
    mut m: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
) raises:
    constrained[ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"]()
    constrained[rank == 1, "this test code currently assumes rank 1"]()

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[dtype]]](capacity=ngpus)

    comptime num_buffers = 1 if use_multimem else ngpus

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](fill={})

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // size_of[dtype]()

    alias simd_size = simd_width_of[dtype, target = get_gpu_target()]()
    var stride = align_up(length, simd_size)
    alias m512 = 512 * 1024 * 1024
    var cache_elems = (
        align_up(m512, stride * size_of[dtype]()) // size_of[dtype]()
    )

    # Initialize buffers for each GPU
    @parameter
    for gpu_idx in range(ngpus):
        # Create and store device buffers (outputs)
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = UnsafePointer[Scalar[dtype]].alloc(cache_elems)
        host_buffers.append(host_buffer)

        for i in range(cache_elems // stride):
            for j in range(length):
                host_buffer[i * stride + j] = _per_gpu_value[dtype](gpu_idx, j)

        @parameter
        if not use_multimem:
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

    var multi_ptr = UnsafePointer[Scalar[dtype]]()

    @parameter
    if use_multimem:
        multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), cache_elems
        )

        @parameter
        for i in range(ngpus):
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

        @parameter
        for i in range(ngpus):
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
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].enqueue_memset(out_bufs_list[i], 0)

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_bufs_capture = StaticTuple[
        NDBuffer[dtype, rank, MutAnyOrigin], ngpus
    ](NDBuffer[dtype, rank, MutAnyOrigin]())

    @parameter
    for i in range(ngpus):
        out_bufs_capture[i] = NDBuffer[dtype, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    # Monotonic iteration counter to color quickreduce flags across launches.
    var iter = 0

    # Pre-initialize vendor CCL communicators from the main thread.
    # ncclCommInitAll is not thread-safe, so we must initialize before
    # spawning worker threads.
    @parameter
    if use_vendor_ccl:
        if not vendor_ccl.is_allreduce_available():
            raise "Vendor CCL not available; skipping vendor path."
        vendor_ccl.init_comms(ngpus)

    var results = InlineArray[Float64, ngpus](fill={})

    @parameter
    fn per_gpu(i: Int) raises:
        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn call_fn(ctx: DeviceContext, iteration: Int) raises:
                # Offset the input buffer if cache_busting
                var offset = 0

                @parameter
                if cache_busting:
                    offset = (iteration * stride) % cache_elems

                @parameter
                if not use_multimem:

                    @parameter
                    for i in range(ngpus):
                        in_bufs[i] = NDBuffer[dtype, rank](
                            in_bufs_list[i].unsafe_ptr() + offset,
                            DimList(length),
                        )
                else:
                    in_bufs[0] = NDBuffer[dtype, rank](
                        multi_ptr + offset, DimList(length)
                    )

                # Run allreduce
                @parameter
                if use_vendor_ccl:
                    constrained[
                        not use_multimem,
                        "vendor CCL does not support multimem path",
                    ]()
                    vendor_ccl.allreduce[dtype=dtype, rank=rank, ngpus=ngpus](
                        in_bufs[i],
                        out_bufs[i],
                        i,
                        list_of_ctx[i],
                    )
                else:
                    allreduce[
                        ngpus=ngpus,
                        use_multimem=use_multimem,
                        use_quickreduce=use_quickreduce,
                    ](
                        in_bufs,
                        out_bufs[i],
                        rank_sigs,
                        list_of_ctx[i],
                        max_num_blocks,
                        iter,
                    )

            b.iter_custom[call_fn](list_of_ctx[i])

        var b = Bench()
        b.config.show_progress = False
        b.bench_function[bench_iter](
            BenchId(String("")),
            [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
        )
        results[i] = b.info_vec[0].result.mean(unit="ms")

    sync_parallelize[per_gpu](ngpus)

    var max_time = 0.0
    for i in range(ngpus):
        if results[i] > max_time:
            max_time = results[i]

    var gbps = num_bytes / (max_time * 1000 * 1000)
    print("")
    var name = String(
        _get_test_str[dtype, use_multimem](ngpus, num_bytes),
        "-vendor_ccl" if use_vendor_ccl else "",
    )
    # algbw and busbw are explain in the following link:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    var busbw = 2 * gbps * (ngpus - 1) / ngpus
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
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    # Verify results
    # For low-precision dtypes (e.g., bfloat16), inputs were quantized to `dtype`
    # before reduction on device. Mirror the device path here by:
    #  - quantizing each per-GPU term to `dtype` by calling _per_gpu_value[dtype](...)
    #  - accumulating in Float32
    #  - finally casting to `dtype` for the expected value
    @parameter
    for i in range(ngpus):
        for j in range(length):
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)

            @parameter
            for k in range(ngpus):
                var term_dtype = _per_gpu_value[dtype](k, j)
                accum += Scalar[accum_t](term_dtype)
            var expected_sum = Scalar[dtype](accum)
            try:
                var rtol, atol = _pytorch_like_tolerances_for[dtype]()
                assert_almost_equal(
                    host_buffers[i][j], expected_sum, atol=atol, rtol=rtol
                )
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
    _ = signal_buffers^


fn _get_test_str[
    dtype: DType, use_multimem: Bool
](ngpus: Int, num_bytes: Int) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    return String(
        "allreduce-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        "-",
        _human_memory(num_bytes),
    )


def main():
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime num_gpus = env_get_int["num_gpus", 2]()
    comptime rank = env_get_int["rank", 1]()
    # Force passing `max_num_blocks` explicitly.
    var max_nb = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)
    comptime use_multimem = env_get_bool["multimem", False]()
    comptime use_quickreduce = env_get_bool["quickreduce", False]()
    comptime use_vendor_ccl = env_get_bool["use_vendor_ccl", False]()
    comptime cache_busting = True

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

    # Generate descriptive test name.
    print(_get_test_str[dtype, use_multimem](num_gpus, num_bytes))

    var m = Bench()

    if use_quickreduce:
        bench_allreduce_push[
            dtype=dtype,
            rank=rank,
            ngpus=num_gpus,
            cache_busting=cache_busting,
            use_vendor_ccl=use_vendor_ccl,
        ](m, ctx, num_bytes, max_num_blocks)
    else:
        bench_allreduce_pull[
            dtype=dtype,
            rank=rank,
            ngpus=num_gpus,
            use_multimem=use_multimem,
            cache_busting=cache_busting,
            use_vendor_ccl=use_vendor_ccl,
        ](m, ctx, num_bytes, max_num_blocks)


# Convenience wrappers matching reviewer terminology.
fn bench_allreduce_pull[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool = False,
    cache_busting: Bool = True,
    use_vendor_ccl: Bool = False,
](
    mut m: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
) raises:
    # Pull path: default allreduce (use_quickreduce=False)
    bench_reduce[
        dtype=dtype,
        rank=rank,
        ngpus=ngpus,
        use_multimem=use_multimem,
        use_quickreduce=False,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, list_of_ctx, num_bytes, max_num_blocks)


fn bench_allreduce_push[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    cache_busting: Bool = True,
    use_vendor_ccl: Bool = False,
](
    mut m: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
) raises:
    # Push path: quickreduce (use_quickreduce=True)
    bench_reduce[
        dtype=dtype,
        rank=rank,
        ngpus=ngpus,
        use_multimem=False,
        use_quickreduce=True,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, list_of_ctx, num_bytes, max_num_blocks)
