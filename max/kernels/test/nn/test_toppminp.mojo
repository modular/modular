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

from std.math import iota
from std.random import random_float64

from std.algorithm.functional import parallelize_over_rows
from std.benchmark import Bench, Bencher, BenchId
from layout import Coord, Idx, TileTensor, row_major
from nn.toppminp import min_p_sampling, top_p_sampling
from std.testing import assert_equal

from std.utils import IndexList

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

    def __init__(
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


def time_kernel[
    func: def() raises capturing -> None
](mut m: Bench, kernel_name: String) raises:
    @parameter
    @always_inline
    def bench_func(mut m: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch() raises:
            func()

        m.iter[kernel_launch]()

    m.bench_function[bench_func](BenchId(kernel_name))


@parameter
def fill_random[dtype: DType](mut buffer: TileTensor[mut=True, dtype, ...]):
    comptime min_val = -1e6
    comptime max_val = 1e6
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.raw_store(i, random_value.cast[dtype]())


@parameter
def fill_iota[dtype: DType](mut buf: TileTensor[mut=True, dtype, ...]):
    iota(buf.ptr, buf.num_elements())


def test_is_sorted_descending[
    dtype: DType
](mut buf: TileTensor[dtype, ...], vocab_size: Int) -> Bool:
    comptime assert buf.rank == 2, "rank must be 2"
    var batch_size = buf.num_elements() // vocab_size
    var sorted_flag = List(length=batch_size, fill=True)

    @parameter
    def process_rows(start_batch: Int, end_batch: Int):
        # Process a chunk of batches
        for batch_id in range(start_batch, end_batch):
            var offset = batch_id * vocab_size
            for i in range(vocab_size - 1):
                if buf.raw_load(offset + i) < buf.raw_load(offset + i + 1):
                    print(
                        "[",
                        batch_id,
                        "][",
                        i,
                        "]: ",
                        buf.raw_load(offset + i),
                        " < ",
                        buf.raw_load(offset + i + 1),
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

    return all_sorted


def print_test_case(test_case: TestCase):
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


def test_case_sampling[
    fill_fn: def[dtype: DType](
        mut TileTensor[mut=True, dtype, ...]
    ) capturing -> None,
](test_case: TestCase) raises:
    print_test_case(test_case)
    comptime rank = 2
    comptime dtype = test_case.dtype
    comptime out_idx_type = test_case.out_idx_type
    comptime is_top_p = test_case.is_top_p
    var batch_size = test_case.batch_size
    var vocab_size = test_case.vocab_size
    var temperature = rebind[Scalar[dtype]](test_case.temperature)
    var p_threshold = rebind[Scalar[dtype]](test_case.p_threshold)

    var m = Bench()

    # Create input tensors
    var in_logits_ptr = List(
        length=batch_size * vocab_size, fill=Scalar[dtype](0)
    )
    var in_logits = TileTensor(
        in_logits_ptr,
        row_major(Coord(Idx(batch_size), Idx(vocab_size))),
    )
    var token_ids_ptr = List(
        length=batch_size * 1, fill=Scalar[out_idx_type](0)
    )
    var token_ids = TileTensor(
        token_ids_ptr,
        row_major(Coord(Idx(batch_size), Idx(1))),
    )
    var p_thresholds_ptr = List(length=batch_size, fill=Scalar[dtype](0))
    var p_thresholds = TileTensor(
        p_thresholds_ptr,
        row_major(Idx(batch_size)),
    )

    # Fill tensors
    fill_fn(in_logits)
    for i in range(batch_size):
        p_thresholds.raw_store(i, p_threshold)

    comptime if DEBUG_BENCH:

        @always_inline
        @parameter
        def run_func() raises:
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
            m, "top-p-sampling" if is_top_p else "min-p-sampling"
        )

    # Run sampling
    comptime if is_top_p:
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

    comptime if PRINT_OUTPUT:
        print("Sampled token indices:", token_ids)

    comptime if DEBUG_BENCH:
        m.dump_report()


def test_toppminp[
    dtype: DType,
    out_idx_type: DType,
    fill_fn: def[dtype: DType](
        mut TileTensor[mut=True, dtype, ...]
    ) capturing -> None,
]() raises:
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

    test_case_sampling[fill_fn](test_case1)
    test_case_sampling[fill_fn](test_case2)
    test_case_sampling[fill_fn](test_case3)


def test_all_out_idx_types[
    dtype: DType,
    fill_fn: def[dtype: DType](
        mut TileTensor[mut=True, dtype, ...]
    ) capturing -> None,
]() raises:
    test_toppminp[dtype, DType.int32, fill_fn]()
    test_toppminp[dtype, DType.int64, fill_fn]()
    test_toppminp[dtype, DType.uint64, fill_fn]()


def test_all_types[
    fill_fn: def[dtype: DType](
        mut TileTensor[mut=True, dtype, ...]
    ) capturing -> None,
]() raises:
    print("\n=== Testing Float32 ===")
    test_all_out_idx_types[DType.float32, fill_fn]()


def main() raises:
    print("\n====== Testing Fill Iota ======\n")
    test_all_types[fill_iota]()
    print("\n====== Testing Fill Random ======\n")
    test_all_types[fill_random]()
