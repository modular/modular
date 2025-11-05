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
from math import floor
from sys import env_get_bool, env_get_dtype, env_get_int, size_of
from utils.numerics import get_accum_type

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from comm.allreduce import MAX_GPUS, Signal, allreduce, can_enable_p2p
import comm.vendor.ccl as vendor_ccl
from gpu.host import DeviceBuffer, DeviceContext, DeviceMulticastBuffer
from internal_utils import InitializationType, arg_parse
from memory import LegacyUnsafePointer as UnsafePointer
from testing import assert_almost_equal, assert_true

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
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

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

    alias num_buffers = 1 if use_multimem else ngpus

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](fill={})

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // size_of[dtype]()

    # Initialize buffers for each GPU
    @parameter
    for gpu_idx in range(ngpus):
        # Create and store device buffers (outputs)
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = UnsafePointer[Scalar[dtype]].alloc(length)
        host_buffers.append(host_buffer)

        for j in range(length):
            host_buffer[j] = _per_gpu_value[dtype](gpu_idx, j)

        @parameter
        if not use_multimem:
            # Create per-GPU input buffers on device and copy from host
            in_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
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

    if use_multimem:
        var multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), length
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

    @always_inline
    @parameter
    @__copy_capture(out_bufs_capture)
    fn outputs_lambda[
        input_index: Int,
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
        out_bufs_capture[input_index].store[width=_width, alignment=_alignment](
            rebind[IndexList[rank]](coords), rebind[SIMD[dtype, _width]](val)
        )

    # Monotonic iteration counter to color quickreduce flags across launches.
    var iter = 0

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn call_fn() raises:
            @parameter
            if use_vendor_ccl:
                constrained[
                    not use_multimem,
                    "vendor CCL does not support multimem path",
                ]()
                if not vendor_ccl.is_allreduce_available():
                    raise "Vendor CCL not available; skipping vendor path."
                vendor_ccl.allreduce[dtype=dtype, rank=rank, ngpus=ngpus](
                    rebind[
                        InlineArray[NDBuffer[dtype, rank, MutAnyOrigin], ngpus]
                    ](in_bufs),
                    out_bufs,
                    list_of_ctx,
                )
            else:

                @parameter
                for i in range(ngpus):
                    allreduce[
                        ngpus=ngpus,
                        output_lambda = outputs_lambda[input_index=i],
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
                iter += 1

        b.iter_custom_multicontext[call_fn](list_of_ctx)

    var vendor_tag = "-vendor_ccl" if use_vendor_ccl else ""
    var name = String(
        _get_test_str[dtype, use_multimem](ngpus, num_bytes), vendor_tag
    )
    m.bench_function[bench_iter](
        BenchId(name),
        # add data movement to measures
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
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
            alias accum_t = get_accum_type[dtype]()
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

    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias num_gpus = env_get_int["num_gpus", 2]()
    alias rank = env_get_int["rank", 1]()
    # Force passing `max_num_blocks` explicitly.
    var max_nb = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)
    alias use_multimem = env_get_bool["multimem", False]()
    alias use_quickreduce = env_get_bool["quickreduce", False]()
    alias use_vendor_ccl = env_get_bool["use_vendor_ccl", False]()

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
            use_vendor_ccl=use_vendor_ccl,
        ](m, ctx, num_bytes, max_num_blocks)
    else:
        bench_allreduce_pull[
            dtype=dtype,
            rank=rank,
            ngpus=num_gpus,
            use_multimem=use_multimem,
            use_vendor_ccl=use_vendor_ccl,
        ](m, ctx, num_bytes, max_num_blocks)

    m.dump_report()


# Convenience wrappers matching reviewer terminology.
fn bench_allreduce_pull[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool = False,
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
        use_vendor_ccl=use_vendor_ccl,
    ](m, list_of_ctx, num_bytes, max_num_blocks)


fn bench_allreduce_push[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
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
        use_vendor_ccl=use_vendor_ccl,
    ](m, list_of_ctx, num_bytes, max_num_blocks)
