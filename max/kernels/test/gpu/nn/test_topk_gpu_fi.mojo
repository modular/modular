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

from collections import OptionalReg
from math import ceildiv, iota, exp, log
from random import random_float64, seed
from nn.topk_fi import topk_mask_logits, topk_sampling_from_prob
from utils.numerics import max_or_inf, min_or_neg_inf

from algorithm.reduction import max as reduce_max
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from nn.topk import _top_k_cpu, _topk_gpu
from testing import assert_almost_equal, assert_equal

from utils import IndexList

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False
comptime NUM_VALIDATION_TRIALS = 50


@parameter
fn fill_random_normalized[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    """Fill buffer with random values and normalize to valid probabilities.

    Each row will sum to 1.0, creating valid probability distributions.
    """
    var shape = buffer.get_shape()
    var batch_size = shape[0]
    var vocab_size = shape[1]

    for b in range(batch_size):
        var row_sum = Scalar[dtype](0.0)

        for i in range(vocab_size):
            var random_value = random_float64(0.01, 10.0)
            buffer.data[b * vocab_size + i] = random_value.cast[dtype]()
            row_sum += buffer.data[b * vocab_size + i]

        # Normalize to sum to 1.0.
        for i in range(vocab_size):
            buffer.data[b * vocab_size + i] = (
                buffer.data[b * vocab_size + i] / row_sum
            )


fn compute_topk_mask_from_probs[
    dtype: DType,
    rank: Int,
](
    probs: HostNDBuffer[dtype, rank],
    K: Int,
    batch_size: Int,
    N: Int,
) raises -> HostNDBuffer[DType.bool, rank]:
    """
    Compute a boolean mask indicating which tokens are in the top-K.
    Marks all tokens whose probability is >= K-th largest value.
    """
    var mask = HostNDBuffer[DType.bool, rank](DimList(batch_size, N))

    for b in range(batch_size):
        var values = List[Scalar[dtype]]()
        for i in range(N):
            values.append(probs.tensor.data[b * N + i])

        @parameter
        fn _greater_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
            return lhs > rhs

        sort[_greater_than](values)

        # K-th largest value.
        var kth_value = values[K - 1]

        # Mark all tokens >= kth_value.
        for i in range(N):
            mask.tensor.data[b * N + i] = (
                probs.tensor.data[b * N + i] >= kth_value
            )

    return mask


fn validate_sampling_results_from_probs[
    out_idx_type: DType,
    rank: Int,
](
    sampled_idxs: HostNDBuffer[out_idx_type, 1],
    mask: HostNDBuffer[DType.bool, rank],
    batch_size: Int,
    N: Int,
    trial_num: Int,
) raises:
    """
    Validate that all sampled indices are within the valid top-K set.

    Args:
        sampled_idxs: Sampled token indices [batch_size].
        mask: Boolean mask indicating valid top-K tokens [batch_size, N].
        batch_size: Batch size.
        N: Vocabulary size.
        trial_num: Current trial number (for error messages).
    """
    for b in range(batch_size):
        var idx = Int(sampled_idxs.tensor.data[b])

        # Check 1: Index is within valid range.
        if idx < 0 or idx >= N:
            raise Error(
                "Trial "
                + String(trial_num)
                + ", Batch "
                + String(b)
                + ": Sampled index "
                + String(idx)
                + " is out of range [0, "
                + String(N)
                + ")"
            )

        # Check 2: Index is in the top-K set.
        var is_valid = mask.tensor.data[b * N + idx]
        if not is_valid:
            raise Error(
                "Trial "
                + String(trial_num)
                + ", Batch "
                + String(b)
                + ": Sampled index "
                + String(idx)
                + " is NOT in the top-K set! This indicates a bug in the"
                " sampling kernel."
            )


fn test_topk_sampling_from_prob[
    dtype: DType,
    out_idx_type: DType = DType.int32,
    rank: Int = 2,
    block_size: Int = 1024,
](ctx: DeviceContext, test_case: TestCase) raises:
    """
    Test topk_sampling_from_prob kernel with property-based validation.

    This test validates that the sampling kernel correctly samples from the
    top-K probability distribution by:
    1. Computing the ground truth top-K set
    2. Running sampling multiple times
    3. Verifying each sample is from the valid top-K set
    """

    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling

    constrained[sampling, "topk_sampling_from_prob requires sampling=True"]()

    # Create probability distributions (normalized).
    var probs_host = HostNDBuffer[dtype, rank](DimList(batch_size, N))
    fill_random_normalized[rank, dtype](probs_host.tensor)

    @parameter
    if PRINT_OUTPUT:
        print("Sample probabilities (first batch):")
        for i in range(min(10, N)):
            print("  Token", i, ":", probs_host.tensor.data[i])

    var device_probs = DeviceNDBuffer[dtype, rank](
        DimList(batch_size, N), ctx=ctx
    )
    var device_output = DeviceNDBuffer[out_idx_type, 1](
        DimList(batch_size), ctx=ctx
    )

    ctx.enqueue_copy(device_probs.buffer, probs_host.tensor.data)
    ctx.synchronize()

    # STEP 1: Compute ground truth mask.
    var mask = compute_topk_mask_from_probs[dtype, rank](
        probs_host, K, batch_size, N
    )

    @parameter
    if PRINT_OUTPUT:
        print("  Valid top-K indices for first batch:")
        for i in range(N):
            if mask.tensor.data[i]:
                print("    Token", i, "with prob", probs_host.tensor.data[i])

    # STEP 2: Run sampling validation.
    @parameter
    if PRINT_OUTPUT:
        print(
            "  [Validation] Running",
            NUM_VALIDATION_TRIALS,
            "sampling trials...",
        )

    var sampled_idxs = HostNDBuffer[out_idx_type, 1](DimList(batch_size))
    var num_passed = 0
    var num_failed = 0

    for trial in range(NUM_VALIDATION_TRIALS):
        var trial_seed = UInt64(42 + trial)

        topk_sampling_from_prob[dtype, out_idx_type, block_size](
            ctx,
            device_probs.to_layout_tensor(),
            device_output.to_layout_tensor(),
            K,
            deterministic=False,
            rng_seed=trial_seed,
            rng_offset=0,
        )

        ctx.enqueue_copy(sampled_idxs.tensor.data, device_output.buffer)
        ctx.synchronize()

        try:
            validate_sampling_results_from_probs[out_idx_type, rank](
                sampled_idxs, mask, batch_size, N, trial
            )
            num_passed += 1
        except e:
            print("  [ERROR] Trial", trial, "failed:", e)
            raise Error("Sampling validation failed. Kernel is incorrect.")

    @parameter
    if PRINT_OUTPUT:
        print(
            "  [Validation] ✓ All",
            num_passed,
            "/",
            NUM_VALIDATION_TRIALS,
            "trials passed!",
        )

    # STEP 3: Benchmark the kernel (separate from validation)
    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            topk_sampling_from_prob[dtype, out_idx_type, block_size](
                ctx,
                device_probs.to_layout_tensor(),
                device_output.to_layout_tensor(),
                K,
                deterministic=False,
                rng_seed=UInt64(42),
                rng_offset=0,
            )
            ctx.enqueue_copy(sampled_idxs.tensor.data, device_output.buffer)
            ctx.synchronize()

        time_kernel[run_func](m, ctx, "topk-sampling-from-prob")
        m.dump_report()

    _ = probs_host
    _ = device_probs
    _ = device_output
    _ = sampled_idxs
    _ = mask


fn extract_topk_from_masked[
    dtype: DType,
    out_idx_type: DType,
    rank: Int = 2,
](
    masked_logits: NDBuffer[dtype, rank],
    K: Int,
    mut topk_vals_out: NDBuffer[mut=True, dtype, rank],
    mut topk_idxs_out: NDBuffer[mut=True, out_idx_type, rank],
) raises:
    """Extract top-K values and indices from masked logits tensor.

    Masked logits tensor has top-K values at their original positions,
    and rest are set to -inf. This function extracts the K non-inf values
    and their indices.

    Args:
        masked_logits: Input masked logits tensor (batch_size, N).
        K: Number of top elements to extract.
        topk_vals_out: Output buffer for top-K values (batch_size, K).
        topk_idxs_out: Output buffer for top-K indices (batch_size, K).
    """
    var batch_size = masked_logits.get_shape()[0]
    var N = masked_logits.get_shape()[1]

    for b in range(batch_size):
        var values = List[Scalar[dtype]]()
        var indices = List[Int]()

        for i in range(N):
            var val = masked_logits.data[b * N + i]
            if val != min_or_neg_inf[dtype]():
                values.append(val)
                indices.append(i)

        # Sort by value (descending).
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[j] > values[i]:
                    # Swap values
                    var temp_val = values[i]
                    values[i] = values[j]
                    values[j] = temp_val
                    # Swap indices
                    var temp_idx = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp_idx

        # Copy top-K values and indices to output.
        for k in range(K):
            if k < len(values):
                topk_vals_out.data[b * K + k] = values[k]
                topk_idxs_out.data[b * K + k] = Scalar[out_idx_type](indices[k])
            else:
                # If we have fewer than K non-inf values, fill with -inf and -1.
                topk_vals_out.data[b * K + k] = min_or_neg_inf[dtype]()
                topk_idxs_out.data[b * K + k] = Scalar[out_idx_type](-1)


fn test_case_batched[
    dtype: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        mut NDBuffer[mut=True, dtype, rank]
    ) capturing [_] -> None,
    out_idx_type: DType = DType.int,
    rank: Int = 2,
](ctx: DeviceContext, test_case: TestCase) raises:
    """Test topk_mask_logits kernel by comparing with CPU reference."""

    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling
    comptime block_size = test_case.block_size

    # sampling must be False for mask_logits kernel
    constrained[not sampling, "topk_mask_logits only supports sampling=False"]()

    var in_buffer = HostNDBuffer[dtype, rank](DimList(batch_size, N))
    var masked_logits_host = HostNDBuffer[dtype, rank](DimList(batch_size, N))

    fill_fn(in_buffer.tensor)

    var device_in = DeviceNDBuffer[dtype, rank](DimList(batch_size, N), ctx=ctx)
    var device_masked_logits = DeviceNDBuffer[dtype, rank](
        DimList(batch_size, N), ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)
    ctx.synchronize()

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            topk_mask_logits[dtype, out_idx_type, block_size](
                ctx,
                device_in.to_layout_tensor(),
                device_masked_logits.to_layout_tensor(),
                K,
            )
            ctx.enqueue_copy(
                masked_logits_host.tensor.data, device_masked_logits.buffer
            )
            ctx.synchronize()

        time_kernel[run_func](m, ctx, "topk-mask-logits")

    topk_mask_logits[dtype, out_idx_type, block_size](
        ctx,
        device_in.to_layout_tensor(),
        device_masked_logits.to_layout_tensor(),
        K,
    )

    ctx.enqueue_copy(
        masked_logits_host.tensor.data, device_masked_logits.buffer
    )
    ctx.synchronize()

    @parameter
    if PRINT_OUTPUT:
        print("Masked logits:", masked_logits_host.tensor)

    var topk_vals_extracted = HostNDBuffer[dtype, rank](DimList(batch_size, K))
    var topk_idxs_extracted = HostNDBuffer[out_idx_type, rank](
        DimList(batch_size, K)
    )

    extract_topk_from_masked[dtype, out_idx_type, rank](
        masked_logits_host.tensor,
        K,
        topk_vals_extracted.tensor,
        topk_idxs_extracted.tensor,
    )

    @parameter
    if PRINT_OUTPUT:
        print("Extracted top-K values:", topk_vals_extracted.tensor)
        print("Extracted top-K indices:", topk_idxs_extracted.tensor)

    var topk_vals_cpu = HostNDBuffer[dtype, rank](DimList(batch_size, K))
    var topk_idxs_cpu = HostNDBuffer[DType.int64, rank](DimList(batch_size, K))

    @parameter
    if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func_cpu(ctx: DeviceContext) raises:
            _top_k_cpu[
                dtype=dtype,
                out_idx_type = DType.int64,
                largest=largest,
            ](
                in_buffer.to_layout_tensor(),
                K,
                rank - 1,
                topk_vals_cpu.to_layout_tensor(),
                topk_idxs_cpu.to_layout_tensor(),
                1,
                True,
            )

        time_kernel[run_func_cpu](m, ctx, "topk-cpu")

    _top_k_cpu[dtype=dtype, out_idx_type = DType.int64, largest=largest](
        in_buffer.to_layout_tensor(),
        K,
        rank - 1,
        topk_vals_cpu.to_layout_tensor(),
        topk_idxs_cpu.to_layout_tensor(),
        1,
        True,
    )

    @parameter
    if PRINT_OUTPUT:
        print("CPU top-K values:", topk_vals_cpu.tensor)
        print("CPU top-K indices:", topk_idxs_cpu.tensor)

    for i in range(topk_vals_extracted.tensor.num_elements()):
        assert_almost_equal(
            topk_vals_extracted.tensor.data[i],
            topk_vals_cpu.tensor.data[i],
            msg="Top-K values mismatch at index " + String(i),
        )

        # Note: We don't check exact index equality because different implementations
        # may break ties differently when values are equal or very close. As long as
        # the top-K values match, the indices can differ for tied values.

    _ = in_buffer
    _ = masked_logits_host
    _ = topk_vals_extracted
    _ = topk_idxs_extracted
    _ = topk_vals_cpu
    _ = topk_idxs_cpu
    _ = device_in
    _ = device_masked_logits

    @parameter
    if DEBUG_BENCH:
        m.dump_report()


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

    m.bench_function[bench_func](
        BenchId(
            kernel_name
        ),  # ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


@parameter
fn fill_random[
    rank: Int, dtype: DType
](mut buffer: NDBuffer[mut=True, dtype, rank]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


struct TestCase[_sampling: Bool, _largest: Bool = True, _block_size: Int = 256](
    ImplicitlyCopyable, Movable
):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var N: Int
    var K: Int
    comptime block_size: Int = Self._block_size
    var batch_size: Int
    var num_blocks_per_input: OptionalReg[Int]

    fn __init__(
        out self,
        N: Int,
        K: Int,
        batch_size: Int,
        num_blocks_per_input: OptionalReg[Int] = None,
    ):
        self.N = N
        self.K = K
        self.batch_size = batch_size
        self.num_blocks_per_input = num_blocks_per_input


fn print_test_case(test_case: TestCase):
    var num_blocks_per_in_msg = "auto"
    if test_case.num_blocks_per_input:
        num_blocks_per_in_msg = String(test_case.num_blocks_per_input.value())
    print(
        "==== Running Top-K sampling=",
        test_case.sampling,
        ", N=",
        test_case.N,
        ", K=",
        test_case.K,
        ", block_size=",
        test_case.block_size,
        ", batch_size=",
        test_case.batch_size,
        ", num_blocks_per_input=",
        num_blocks_per_in_msg,
    )


def main():
    """Test suite for topk_mask_logits kernel.

    This function tests the topk_mask_logits kernel by comparing its output
    (after extraction) with the CPU reference implementation.
    """
    seed(42)
    comptime llama3_vocab_size = 128256
    with DeviceContext() as ctx:
        comptime float32_dtype = DType.float32
        comptime bf16_type = DType.bfloat16

        print("\n" + "=" * 80)
        print("Testing topk_mask_logits kernel")
        print("=" * 80 + "\n")

        comptime default_block_size = 1024

        comptime test_case0 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=1024,
            K=256,
            batch_size=1,
        )
        print_test_case(test_case0)
        test_case_batched[
            float32_dtype,
            fill_random,
            out_idx_type = DType.uint64,
        ](ctx, test_case0)

        comptime test_case1 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=1024,
            K=1,
            batch_size=1,
        )
        print_test_case(test_case1)
        test_case_batched[
            float32_dtype,
            fill_random,
            out_idx_type = DType.uint64,
        ](ctx, test_case1)

        comptime test_case2 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=32000,
            K=5,
            batch_size=16,
        )
        print_test_case(test_case2)
        test_case_batched[float32_dtype, fill_random](ctx, test_case2)

        comptime test_case3 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=10,
            batch_size=64,
        )
        print_test_case(test_case3)
        test_case_batched[float32_dtype, fill_random](ctx, test_case3)

        comptime test_case4 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=1024,
            K=5,
            batch_size=16,
        )
        print_test_case(test_case4)
        test_case_batched[float32_dtype, fill_random](ctx, test_case4)

        comptime test_case5 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=32000,
            K=25,
            batch_size=64,
        )
        print_test_case(test_case5)
        test_case_batched[float32_dtype, fill_random](ctx, test_case5)

        comptime test_case6 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=1,
            batch_size=256,
        )
        print_test_case(test_case6)
        test_case_batched[float32_dtype, fill_random](ctx, test_case6)

        comptime test_case7 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=1024,
            K=10,
            batch_size=256,
        )
        print_test_case(test_case7)
        test_case_batched[
            bf16_type,
            fill_random,
            out_idx_type = DType.uint64,
        ](ctx, test_case7)

        comptime test_case8 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=32000,
            K=1,
            batch_size=1,
        )
        print_test_case(test_case8)
        test_case_batched[bf16_type, fill_random](ctx, test_case8)

        comptime test_case9 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=1,
            batch_size=16,
        )
        print_test_case(test_case9)
        test_case_batched[bf16_type, fill_random](ctx, test_case9)

        comptime test_case10 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=5,
            batch_size=16,
        )
        print_test_case(test_case10)
        test_case_batched[bf16_type, fill_random](ctx, test_case10)

        comptime test_case11 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=1024,
            K=5,
            batch_size=64,
        )
        print_test_case(test_case11)
        test_case_batched[bf16_type, fill_random](ctx, test_case11)

        comptime test_case12 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=50,
            K=25,
            batch_size=2,
        )
        print_test_case(test_case12)
        test_case_batched[float32_dtype, fill_random](ctx, test_case12)

        comptime test_case13 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=75,
            batch_size=2,
        )
        print_test_case(test_case13)
        test_case_batched[float32_dtype, fill_random](ctx, test_case13)

        comptime test_case14 = TestCase[
            _sampling=False, _block_size=default_block_size
        ](
            N=50,
            K=25,
            batch_size=1,
        )
        print_test_case(test_case14)
        test_case_batched[float32_dtype, fill_random](ctx, test_case14)

        print("\n" + "=" * 80)
        print("All topk_mask_logits tests passed! ✓")
        print("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("Testing topk_sampling_from_prob kernel")
        print("=" * 80 + "\n")

        comptime sampling_test_case1 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=100,
            K=10,
            batch_size=1,
        )
        print_test_case(sampling_test_case1)
        test_topk_sampling_from_prob[
            float32_dtype, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case1)

        comptime sampling_test_case2 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=1024,
            K=50,
            batch_size=16,
        )
        print_test_case(sampling_test_case2)
        test_topk_sampling_from_prob[
            float32_dtype, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case2)

        comptime sampling_test_case3 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=32000,
            K=100,
            batch_size=8,
        )
        print_test_case(sampling_test_case3)
        test_topk_sampling_from_prob[
            float32_dtype, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case3)

        comptime sampling_test_case4 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=32000,
            K=5,
            batch_size=32,
        )
        print_test_case(sampling_test_case4)
        test_topk_sampling_from_prob[
            float32_dtype, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case4)

        comptime sampling_test_case5 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=1024,
            K=20,
            batch_size=256,
        )
        print_test_case(sampling_test_case5)
        test_topk_sampling_from_prob[
            float32_dtype, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case5)

        comptime sampling_test_case6 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=50,
            batch_size=4,
        )
        print_test_case(sampling_test_case6)
        test_topk_sampling_from_prob[
            bf16_type, DType.int32, 2, default_block_size
        ](ctx, sampling_test_case6)

        print("\n" + "=" * 80)
        print("All topk_sampling_from_prob tests passed! ✓")
        print("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80 + "\n")
