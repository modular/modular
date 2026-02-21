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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from math import iota
from random import random_float64

from algorithm.functional import parallelize_over_rows
from benchmark import Bench, Bencher, BenchId
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from layout._coord import Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.softmax import softmax
from nn.toppminp_gpu import min_p_sampling_gpu, top_p_sampling_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False


struct TestCase[_dtype: DType, _out_idx_type: DType, _is_top_p: Bool](
    ImplicitlyCopyable
):
    comptime is_top_p = Self._is_top_p
    comptime dtype = Self._dtype
    comptime out_idx_type = Self._out_idx_type
    var batch_size: Int
    var vocab_size: Int
    var temperature: Scalar[Self.dtype]
    var p_threshold: Scalar[Self.dtype]

    fn __init__(
        out self,
        batch_size: Int,
        vocab_size: Int,
        temperature: Scalar[Self.dtype] = Scalar[Self.dtype](1.0),
        p_threshold: Scalar[Self.dtype] = Scalar[Self.dtype](0.9),
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.p_threshold = p_threshold


fn time_kernel[
    func: fn(DeviceContext) raises capturing -> None
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
fn fill_random[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    comptime min_val = -1e6
    comptime max_val = 1e6
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[rank: Int, dtype: DType](mut buf: NDBuffer[mut=True, dtype, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


fn merge[
    dtype: DType, rank: Int
](mut buf: NDBuffer[mut=True, dtype, rank], start: Int, mid: Int, end: Int):
    """Merge two sorted subarrays into one sorted array."""
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_ptr = UnsafePointer[Scalar[dtype]].alloc(left_size)
    var right_ptr = UnsafePointer[Scalar[dtype]].alloc(right_size)

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
    dtype: DType, rank: Int
](mut buf: NDBuffer[mut=True, dtype, rank], start: Int, end: Int):
    """Recursive merge sort implementation."""
    if end - start > 1:
        var mid = start + (end - start) // 2
        merge_sort_recursive(buf, start, mid)
        merge_sort_recursive(buf, mid, end)
        merge(buf, start, mid, end)


fn sort_buf_descending[
    dtype: DType, rank: Int
](mut buf: NDBuffer[mut=True, dtype, rank], vocab_size: Int):
    """Sort each batch separately in descending order using parallel merge sort.
    """
    comptime assert rank == 2, "rank must be 2"
    var batch_size = buf.num_elements() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf, start, end)


fn test_is_sorted_descending[
    dtype: DType, rank: Int
](mut buf: NDBuffer[mut=True, dtype, rank], vocab_size: Int) -> Bool:
    comptime assert rank == 2, "rank must be 2"
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

    comptime parallelism_grain_size = 1
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
        ", dtype=",
        test_case.dtype,
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
    fill_fn: fn[rank: Int, dtype: DType](
        mut NDBuffer[mut=True, dtype, rank]
    ) capturing -> None,
](ctx: DeviceContext, test_case: TestCase) raises:
    print_test_case(test_case)
    comptime rank = 2
    comptime dtype = test_case.dtype
    comptime out_idx_type = test_case.out_idx_type
    comptime is_top_p = test_case.is_top_p
    var batch_size = test_case.batch_size
    var vocab_size = test_case.vocab_size
    var temperature = rebind[Scalar[dtype]](test_case.temperature)
    var p_threshold = rebind[Scalar[dtype]](test_case.p_threshold)

    var m: Bench

    comptime if DEBUG_BENCH:
        m = Bench()

    # Create input tensors
    var in_logits_ptr = UnsafePointer[Scalar[dtype]].alloc(
        batch_size * vocab_size
    )
    var in_logits = NDBuffer[dtype, rank](
        in_logits_ptr, DimList(batch_size, vocab_size)
    )
    var token_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        batch_size * 1
    )
    var token_ids = NDBuffer[out_idx_type, rank](
        token_ids_ptr, DimList(batch_size, 1)
    )
    var p_thresholds_ptr = UnsafePointer[Scalar[dtype]].alloc(batch_size)
    var p_thresholds = NDBuffer[dtype, 1](p_thresholds_ptr, DimList(batch_size))

    # Fill tensors
    fill_fn(in_logits)
    for i in range(batch_size):
        p_thresholds.data[i] = p_threshold

    # Create device buffers
    var device_in_buf = ctx.enqueue_create_buffer[dtype](
        batch_size * vocab_size
    )
    var device_token_ids_buf = ctx.enqueue_create_buffer[out_idx_type](
        batch_size * 1
    )
    var device_p_thresholds_buf = ctx.enqueue_create_buffer[dtype](batch_size)

    # Copy to device
    ctx.enqueue_copy(device_in_buf, in_logits.data)
    ctx.enqueue_copy(device_p_thresholds_buf, p_thresholds.data)

    # Copy to CPU and perform softmax & sort for correctness testing
    var in_logits_cpu_test_ptr = UnsafePointer[Scalar[dtype]].alloc(
        batch_size * vocab_size
    )
    var in_logits_cpu_test = NDBuffer[dtype, rank](
        in_logits_cpu_test_ptr, DimList(batch_size, vocab_size)
    )
    var probs_cpu_test_ptr = UnsafePointer[Scalar[dtype]].alloc(
        batch_size * vocab_size
    )
    var probs_cpu_test = NDBuffer[dtype, rank](
        probs_cpu_test_ptr, DimList(batch_size, vocab_size)
    )
    for i in range(in_logits.num_elements()):
        in_logits_cpu_test.data[i] = in_logits.data[i] / temperature

    softmax[simd_width=1, rank=rank](
        LayoutTensor[
            in_logits_cpu_test.type,
            Layout.row_major[in_logits_cpu_test.rank](in_logits_cpu_test.shape),
        ](
            in_logits_cpu_test.data,
            RuntimeLayout[
                Layout.row_major[in_logits_cpu_test.rank](
                    in_logits_cpu_test.shape
                )
            ].row_major(in_logits_cpu_test.get_shape().canonicalize()),
        ),
        LayoutTensor[
            probs_cpu_test.type,
            Layout.row_major[probs_cpu_test.rank](probs_cpu_test.shape),
        ](
            probs_cpu_test.data,
            RuntimeLayout[
                Layout.row_major[probs_cpu_test.rank](probs_cpu_test.shape)
            ].row_major(probs_cpu_test.get_shape().canonicalize()),
        ),
        axis=1,
    )
    in_logits_cpu_test_ptr.free()
    sort_buf_descending(probs_cpu_test, vocab_size)

    var device_in_tensor = TileTensor(
        device_in_buf.unsafe_ptr(),
        row_major((Idx(batch_size), Idx(vocab_size))),
    )
    var device_token_ids_tensor = TileTensor(
        device_token_ids_buf.unsafe_ptr(),
        row_major((Idx(batch_size), Idx(1))),
    )
    var device_p_thresholds_tensor = TileTensor(
        device_p_thresholds_buf.unsafe_ptr(),
        row_major((Idx(batch_size),)),
    )

    comptime if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            if is_top_p:
                top_p_sampling_gpu(
                    ctx,
                    device_p_thresholds_tensor,
                    device_in_tensor,
                    device_token_ids_tensor,
                    temperature=temperature,
                )
            else:
                min_p_sampling_gpu(
                    ctx,
                    device_p_thresholds_tensor,
                    device_in_tensor,
                    device_token_ids_tensor,
                    temperature=temperature,
                )
            ctx.synchronize()

        time_kernel[run_func](
            m,
            ctx,
            "top-p-sampling" if is_top_p else "min-p-sampling",
        )

    # Run sampling
    comptime if is_top_p:
        top_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds_tensor,
            device_in_tensor,
            device_token_ids_tensor,
            temperature=temperature,
        )
    else:
        min_p_sampling_gpu[_test_sort=True](
            ctx,
            device_p_thresholds_tensor,
            device_in_tensor,
            device_token_ids_tensor,
            temperature=temperature,
        )
    # Copy results back
    ctx.enqueue_copy(token_ids.data, device_token_ids_buf)
    ctx.enqueue_copy(in_logits.data, device_in_buf)  # for testing
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
            raise e^

    comptime if PRINT_OUTPUT:
        print("Sampled token indices:", token_ids)

    comptime if DEBUG_BENCH:
        m.dump_report()
    _ = device_token_ids_buf^
    _ = device_in_buf^
    _ = device_p_thresholds_buf^
    # free all pointers
    in_logits_ptr.free()
    token_ids_ptr.free()
    p_thresholds_ptr.free()
    probs_cpu_test_ptr.free()


fn test_toppminp_gpu[
    dtype: DType,
    out_idx_type: DType,
    fill_fn: fn[rank: Int, dtype: DType](
        mut NDBuffer[mut=True, dtype, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    comptime test_case1 = TestCase[dtype, out_idx_type, _is_top_p=True](
        batch_size=1, vocab_size=1024, temperature=1.0, p_threshold=0.9
    )
    comptime test_case2 = TestCase[dtype, out_idx_type, _is_top_p=True](
        batch_size=16, vocab_size=32000, temperature=10.0, p_threshold=0.95
    )
    comptime test_case3 = TestCase[dtype, out_idx_type, _is_top_p=False](
        batch_size=64,
        vocab_size=128256,
        temperature=0.7,
        p_threshold=0.1,
    )

    test_case_sampling[fill_fn](ctx, test_case1)
    test_case_sampling[fill_fn](ctx, test_case2)
    test_case_sampling[fill_fn](ctx, test_case3)


fn test_all_out_idx_types[
    dtype: DType,
    fill_fn: fn[rank: Int, dtype: DType](
        mut NDBuffer[mut=True, dtype, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    test_toppminp_gpu[dtype, DType.int32, fill_fn](ctx)
    test_toppminp_gpu[dtype, DType.int64, fill_fn](ctx)
    test_toppminp_gpu[dtype, DType.uint64, fill_fn](ctx)


fn test_all_types[
    fill_fn: fn[rank: Int, dtype: DType](
        mut NDBuffer[mut=True, dtype, rank]
    ) capturing -> None,
](ctx: DeviceContext) raises:
    print("\n=== Testing Float32 ===")
    test_all_out_idx_types[DType.float32, fill_fn](ctx)
    print("\n=== Testing BFloat16 ===")
    test_all_out_idx_types[DType.bfloat16, fill_fn](ctx)


def main():
    with DeviceContext() as ctx:
        print("\n====== Testing Fill Iota ======\n")
        test_all_types[fill_iota](ctx)
        print("\n====== Testing Fill Random ======\n")
        test_all_types[fill_random](ctx)
