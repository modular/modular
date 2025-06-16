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

from math import ceildiv, iota
from random import random_float64

from algorithm.functional import parallelize_over_rows
from benchmark import Bench, Bencher, BenchId
from layout import LayoutTensor, Layout, RuntimeLayout
from nn.softmax import softmax
from nn.toppminp import min_p_sampling, top_p_sampling
from testing import assert_equal

from utils import IndexList

alias DEBUG_BENCH = False
alias PRINT_OUTPUT = False


struct TestCase[_type: DType, _out_idx_type: DType, _is_top_p: Bool](
    Copyable, Movable
):
    alias is_top_p = _is_top_p
    alias type = _type
    alias out_idx_type = _out_idx_type
    var batch_size: Int
    var vocab_size: Int
    var temperature: Scalar[_type]
    var p_threshold: Scalar[_type]

    fn __init__(
        out self,
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
    func: fn () raises capturing -> None
](mut m: Bench, kernel_name: String) raises:
    @parameter
    @always_inline
    fn bench_func(mut m: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch() raises:
            func()

        m.iter[kernel_launch]()

    m.bench_function[bench_func](BenchId(kernel_name))


@parameter
fn fill_random[dtype: DType](mut buffer: LayoutTensor[mut=True, dtype, **_]):
    alias min_val = -1e6
    alias max_val = 1e6
    var total_elements = buffer.size()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.ptr[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[type: DType](mut buf: LayoutTensor[mut=True, type, **_]):
    iota(buf.ptr, buf.size())


fn test_is_sorted_descending[
    type: DType
](mut buf: LayoutTensor[type, **_], vocab_size: Int) -> Bool:
    constrained[buf.rank == 2, "rank must be 2"]()
    var batch_size = buf.size() // vocab_size
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
                if buf.ptr[offset + i] < buf.ptr[offset + i + 1]:
                    print(
                        "[",
                        batch_id,
                        "][",
                        i,
                        "]: ",
                        buf.ptr[offset + i],
                        " < ",
                        buf.ptr[offset + i + 1],
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
    fill_fn: fn[type: DType] (
        mut LayoutTensor[mut=True, type, **_]
    ) capturing -> None,
](test_case: TestCase) raises:
    print_test_case(test_case)
    alias rank = 2
    alias type = test_case.type
    alias out_idx_type = test_case.out_idx_type
    alias is_top_p = test_case.is_top_p
    var batch_size = test_case.batch_size
    var vocab_size = test_case.vocab_size
    var temperature = rebind[Scalar[type]](test_case.temperature)
    var p_threshold = rebind[Scalar[type]](test_case.p_threshold)

    var m = Bench()

    # Create input tensors
    var in_logits_ptr = UnsafePointer[Scalar[type]].alloc(
        batch_size * vocab_size
    )
    var in_logits = LayoutTensor[type, Layout.row_major[rank]()](
        in_logits_ptr,
        RuntimeLayout[Layout.row_major[rank]()].row_major(
            IndexList[rank](batch_size, vocab_size)
        ),
    )
    var token_ids_ptr = UnsafePointer[Scalar[out_idx_type]].alloc(
        batch_size * 1
    )
    var token_ids = LayoutTensor[out_idx_type, Layout.row_major[1]()](
        token_ids_ptr,
        RuntimeLayout[Layout.row_major[1]()].row_major(
            IndexList[1](batch_size)
        ),
    )
    var p_thresholds_ptr = UnsafePointer[Scalar[type]].alloc(batch_size)
    var p_thresholds = LayoutTensor[type, Layout.row_major[1]()](
        p_thresholds_ptr,
        RuntimeLayout[Layout.row_major[1]()].row_major(
            IndexList[1](batch_size)
        ),
    )

    # Fill tensors
    fill_fn(in_logits)
    for i in range(batch_size):
        p_thresholds.ptr[i] = p_threshold

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func() raises:
            if is_top_p:
                top_p_sampling(
                    p_thresholds,
                    in_logits,
                    token_ids,
                    temperature=temperature,
                )
            else:
                min_p_sampling(
                    p_thresholds,
                    in_logits,
                    token_ids,
                    temperature=temperature,
                )

        time_kernel[run_func](
            m, "top-p-sampling" if is_top_p else String("min-p-sampling")
        )

    # Run sampling
    @parameter
    if is_top_p:
        top_p_sampling[_test_sort=True](
            p_thresholds,
            in_logits,
            token_ids,
            temperature=temperature,
        )
    else:
        min_p_sampling[_test_sort=True](
            p_thresholds,
            in_logits,
            token_ids,
            temperature=temperature,
        )

    # Check if the probs are sorted in descending order, this validates the
    # softmax, and the sort. The random sampling is much simpler compared
    # to the softmax & sort kernels so this is a good check.
    assert_equal(test_is_sorted_descending(in_logits, vocab_size), True)

    @parameter
    if PRINT_OUTPUT:
        print("Sampled token indices:", token_ids)

    @parameter
    if DEBUG_BENCH:
        m.dump_report()

    # free all pointers
    in_logits_ptr.free()
    token_ids_ptr.free()
    p_thresholds_ptr.free()


fn test_toppminp[
    type: DType,
    out_idx_type: DType,
    fill_fn: fn[type: DType] (
        mut LayoutTensor[mut=True, type, **_]
    ) capturing -> None,
]() raises:
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

    test_case_sampling[fill_fn](test_case1)
    test_case_sampling[fill_fn](test_case2)
    test_case_sampling[fill_fn](test_case3)


fn test_all_out_idx_types[
    type: DType,
    fill_fn: fn[type: DType] (
        mut LayoutTensor[mut=True, type, **_]
    ) capturing -> None,
]() raises:
    test_toppminp[type, DType.int32, fill_fn]()
    test_toppminp[type, DType.int64, fill_fn]()
    test_toppminp[type, DType.uint64, fill_fn]()


fn test_all_types[
    fill_fn: fn[type: DType] (
        mut LayoutTensor[mut=True, type, **_]
    ) capturing -> None,
]() raises:
    print("\n=== Testing Float32 ===")
    test_all_out_idx_types[DType.float32, fill_fn]()


fn main() raises:
    print("\n====== Testing Fill Iota ======\n")
    test_all_types[fill_iota]()
    print("\n====== Testing Fill Random ======\n")
    test_all_types[fill_random]()
