# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug %s

from collections import OptionalReg
from math import ceildiv, iota
from os import abort
from random import random_float64

from algorithm.functional import parallelize_over_rows
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from memory import UnsafePointer
from nn.softmax import softmax
from nn.toppminp_gpu import min_p_sampling_gpu, top_p_sampling_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList

alias DEBUG_BENCH = False
alias PRINT_OUTPUT = False


@value
struct TestCase[_type: DType, _out_idx_type: DType, _is_top_p: Bool]:
    alias is_top_p = _is_top_p
    alias type = _type
    alias out_idx_type = _out_idx_type
    var batch_size: Int
    var vocab_size: Int
    var temperature: Scalar[_type]
    var p_threshold: Scalar[_type]

    fn __init__(
        mut self,
        batch_size: Int,
        vocab_size: Int,
        temperature: Scalar[_type] = Scalar[_type](1.0),
        p_threshold: Scalar[_type] = Scalar[_type](0.9),
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.p_threshold = p_threshold


fn time_kernel[
    func: fn (DeviceContext) raises capturing -> None
](mut m: Bench, ctx: DeviceContext, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(mut m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            func(ctx)

        m.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](BenchId(kernel_name))


@parameter
fn fill_random[rank: Int, dtype: DType](mut buffer: NDBuffer[dtype, rank]):
    alias min_val = -1e6
    alias max_val = 1e6
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[rank: Int, type: DType](mut buf: NDBuffer[type, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


fn merge[
    type: DType, rank: Int
](mut buf: NDBuffer[type, rank], start: Int, mid: Int, end: Int):
    """Merge two sorted subarrays into one sorted array."""
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_ptr = UnsafePointer[Scalar[type]].alloc(left_size)
    var right_ptr = UnsafePointer[Scalar[type]].alloc(right_size)

    # Copy data to temporary arrays
    for i in range(left_size):
        left_ptr[i] = buf.data[start + i]
    for i in range(right_size):
        right_ptr[i] = buf.data[mid + i]

    # Merge back into original array
    var i = 0  # Index for left subarray
    var j = 0  # Index for right subarray
    var k = start  # Index for merged array

    while i < left_size and j < right_size:
        if left_ptr[i] >= right_ptr[j]:  # Use >= for descending order
            buf.data[k] = left_ptr[i]
            i += 1
        else:
            buf.data[k] = right_ptr[j]
            j += 1
        k += 1

    # Copy remaining elements if any
    while i < left_size:
        buf.data[k] = left_ptr[i]
        i += 1
        k += 1

    while j < right_size:
        buf.data[k] = right_ptr[j]
        j += 1
        k += 1

    # Free temporary arrays
    left_ptr.free()
    right_ptr.free()


fn merge_sort_recursive[
    type: DType, rank: Int
](mut buf: NDBuffer[type, rank], start: Int, end: Int):
    """Recursive merge sort implementation."""
    if end - start > 1:
        var mid = start + (end - start) // 2
        merge_sort_recursive(buf, start, mid)
        merge_sort_recursive(buf, mid, end)
        merge(buf, start, mid, end)


fn sort_buf_descending[
    type: DType, rank: Int
](mut buf: NDBuffer[type, rank], vocab_size: Int):
    """Sort each batch separately in descending order using parallel merge sort.
    """
    constrained[rank == 2, "rank must be 2"]()
    var batch_size = buf.num_elements() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf, start, end)


fn test_is_sorted_descending[
    type: DType, rank: Int
](mut buf: NDBuffer[type, rank], vocab_size: Int) -> Bool:
    constrained[rank == 2, "rank must be 2"]()
    var batch_size = buf.num_elements() // vocab_size
    var sorted_flag = UnsafePointer[Bool].alloc(batch_size)

    # Initialize all flags to True
    for i in range(batch_size):
        sorted_flag[i] = True

    @parameter
    fn process_rows(start_batch: Int, end_batch: Int):
        # Process a chunk of batches
        for batch_id in range(start_batch, end_batch):
            var offset = batch_id * vocab_size
            for i in range(vocab_size - 1):
                if buf.data[offset + i] < buf.data[offset + i + 1]:
                    print(
                        "[",
                        batch_id,
                        "][",
                        i,
                        "]: ",
                        buf.data[offset + i],
                        " < ",
                        buf.data[offset + i + 1],
                    )
                    sorted_flag[batch_id] = False
                    break

    alias parallelism_grain_size = 1
    # Create shape with batch_size as the second dimension
    var shape = IndexList[1](
        batch_size,
    )
    parallelize_over_rows[process_rows](shape, 0, parallelism_grain_size)

    # Check if all batches are sorted by AND-ing all flags
    var all_sorted = True
    for i in range(batch_size):
        all_sorted = all_sorted and sorted_flag[i]

    # Free the temporary array
    sorted_flag.free()

    return all_sorted


fn print_test_case(test_case: TestCase):
    print(
        "==== Running",
        "Top-P" if test_case.is_top_p else "Min-P",
        ", type=",
        test_case.type,
        ", out_idx_type=",
        test_case.out_idx_type,
        "sampling with batch_size=",
        test_case.batch_size,
        ", vocab_size=",
        test_case.vocab_size,
        ", temperature=",
        test_case.temperature,
        ", p_threshold=",
        test_case.p_threshold,
    )


fn test_case_sampling[
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[type, rank]
    ) capturing -> None,
](ctx: DeviceContext, test_case: TestCase) raises:
    print_test_case(test_case)
    alias rank = 2
    alias type = test_case.type
    alias out_idx_type = test_case.out_idx_type
    alias is_top_p = test_case.is_top_p
    var batch_size = test_case.batch_size
    var vocab_size = test_case.vocab_size
    var temperature = rebind[Scalar[type]](test_case.temperature)
    var p_threshold = rebind[Scalar[type]](test_case.p_threshold)

    var m: Bench

    @parameter
    if DEBUG_BENCH:
        m = Bench()

    # Create input tensors
    var in_logits_ptr = UnsafePointer[Scalar[type]].alloc(
        batch_size * vocab_size
    )
    var in_logits = NDBuffer[type, rank](
        in_logits_ptr, DimList(batch_size, vocab_size)
    )
    var token_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        batch_size * 1
    )
    var token_ids = NDBuffer[out_idx_type, rank](
        token_ids_ptr, DimList(batch_size, 1)
    )
    var p_thresholds_ptr = UnsafePointer[Scalar[type]].alloc(batch_size)
    var p_thresholds = NDBuffer[type, 1](p_thresholds_ptr, DimList(batch_size))

    # Fill tensors
    fill_fn(in_logits)
    for i in range(batch_size):
        p_thresholds.data[i] = p_threshold

    # Create device buffers
    var device_in = DeviceNDBuffer[type, rank](
        DimList(batch_size, vocab_size), ctx=ctx
    )
    var device_token_ids = DeviceNDBuffer[out_idx_type, rank](
        DimList(batch_size, 1), ctx=ctx
    )
    var device_p_thresholds = DeviceNDBuffer[type, 1](
        DimList(batch_size), ctx=ctx
    )

    # Copy to device
    ctx.enqueue_copy(device_in.buffer, in_logits.data)
    ctx.enqueue_copy(device_p_thresholds.buffer, p_thresholds.data)

    # Copy to CPU and perform softmax & sort for correctness testing
    var in_logits_cpu_test_ptr = UnsafePointer[Scalar[type]].alloc(
        batch_size * vocab_size
    )
    var in_logits_cpu_test = NDBuffer[type, rank](
        in_logits_cpu_test_ptr, DimList(batch_size, vocab_size)
    )
    var probs_cpu_test_ptr = UnsafePointer[Scalar[type]].alloc(
        batch_size * vocab_size
    )
    var probs_cpu_test = NDBuffer[type, rank](
        probs_cpu_test_ptr, DimList(batch_size, vocab_size)
    )
    for i in range(in_logits.num_elements()):
        in_logits_cpu_test.data[i] = in_logits.data[i] / temperature

    softmax[simd_width=1](in_logits_cpu_test, probs_cpu_test, axis=1)
    in_logits_cpu_test_ptr.free()
    sort_buf_descending(probs_cpu_test, vocab_size)

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            if is_top_p:
                top_p_sampling_gpu(
                    ctx,
                    device_p_thresholds.tensor,
                    device_in.tensor,
                    device_token_ids.tensor,
                    temperature=temperature,
                )
            else:
                min_p_sampling_gpu(
                    ctx,
                    device_p_thresholds.tensor,
                    device_in.tensor,
                    device_token_ids.tensor,
                    temperature=temperature,
                )
            ctx.synchronize()

        time_kernel[run_func](
            m, ctx, "top-p-sampling" if is_top_p else "min-p-sampling"
        )

    # Run sampling
    @parameter
    if is_top_p:
        top_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds.tensor,
            device_in.tensor,
            device_token_ids.tensor,
            temperature=temperature,
        )
    else:
        min_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds.tensor,
            device_in.tensor,
            device_token_ids.tensor,
            temperature=temperature,
        )
    # Copy results back
    ctx.enqueue_copy(token_ids.data, device_token_ids.buffer)
    ctx.enqueue_copy(in_logits.data, device_in.buffer)  # for testing
    ctx.synchronize()

    # Check if the probs are sorted in descending order, this validates the
    # softmax, and the sort. The random sampling is much simpler compared
    # to the softmax & sort kernels so this is a good check.
    assert_equal(test_is_sorted_descending(in_logits, vocab_size), True)
    # (More rigorous) Check if the sorted probs are the same as the cpu test
    for i in range(in_logits.num_elements()):
        try:
            assert_almost_equal(
                in_logits.data[i], probs_cpu_test.data[i], atol=5e-3
            )
        except e:
            print(
                "i: ",
                i,
                "in_logits: ",
                in_logits.data[i],
                "probs_cpu_test: ",
                probs_cpu_test.data[i],
            )
            raise e

    @parameter
    if PRINT_OUTPUT:
        print("Sampled token indices:", token_ids)

    @parameter
    if DEBUG_BENCH:
        m.dump_report()
    _ = device_token_ids.buffer^
    _ = device_in.buffer^
    _ = device_p_thresholds.buffer^
    # free all pointers
    in_logits_ptr.free()
    token_ids_ptr.free()
    p_thresholds_ptr.free()
    probs_cpu_test_ptr.free()


fn test_toppminp_gpu[
    type: DType,
    out_idx_type: DType,
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[type, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    alias test_case1 = TestCase[type, out_idx_type, _is_top_p=True](
        batch_size=1, vocab_size=1024, temperature=1.0, p_threshold=0.9
    )
    alias test_case2 = TestCase[type, out_idx_type, _is_top_p=True](
        batch_size=16, vocab_size=32000, temperature=10.0, p_threshold=0.95
    )
    alias test_case3 = TestCase[type, out_idx_type, _is_top_p=False](
        batch_size=64,
        vocab_size=128256,
        temperature=0.7,
        p_threshold=0.1,
    )

    test_case_sampling[fill_fn](ctx, test_case1)
    test_case_sampling[fill_fn](ctx, test_case2)
    test_case_sampling[fill_fn](ctx, test_case3)


fn test_all_out_idx_types[
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[type, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    test_toppminp_gpu[type, DType.int32, fill_fn](ctx)
    test_toppminp_gpu[type, DType.int64, fill_fn](ctx)
    test_toppminp_gpu[type, DType.uint64, fill_fn](ctx)


fn test_all_types[
    fill_fn: fn[rank: Int, type: DType] (
        mut NDBuffer[type, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    print("\n=== Testing Float32 ===")
    test_all_out_idx_types[DType.float32, fill_fn](ctx)
    print("\n=== Testing BFloat16 ===")
    test_all_out_idx_types[DType.bfloat16, fill_fn](ctx)


fn main() raises:
    with DeviceContext() as ctx:
        print("\n====== Testing Fill Iota ======\n")
        test_all_types[fill_iota](ctx)
        print("\n====== Testing Fill Random ======\n")
        test_all_types[fill_random](ctx)
