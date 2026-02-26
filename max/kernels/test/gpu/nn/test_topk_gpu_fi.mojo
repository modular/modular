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


from algorithm.reduction import max as reduce_max
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from layout._fillers import random
from math import ceildiv, iota, exp, log
from nn.topk import _top_k_cpu, _topk_gpu, _topk_topp_sampling_fi
from nn.topk_fi import (
    topk_mask_logits,
    topk_sampling_from_prob,
    topk_topp_sampling_from_prob,
    topk_softmax_sample,
)
from random import random_float64, seed
from testing import assert_almost_equal, assert_equal
from utils import IndexList
from utils.numerics import max_or_inf, min_or_neg_inf

comptime DEBUG_BENCH = False
comptime PRINT_OUTPUT = False
comptime NUM_VALIDATION_TRIALS = 50


@parameter
fn fill_random_for_test[
    dtype: DType, normalized: Bool
](buffer: TileTensor[mut=True, dtype, ...]):
    """Fill buffer with random values, optionally normalizing to probabilities.

    Parameters:
        dtype: Data type of the buffer.
        normalized: If True, normalize each row to sum to 1.0 (probabilities).
                   If False, use raw random values (logits).
    """
    comptime assert buffer.flat_rank == 2, "expected rank-2 TileTensor"
    var shape = coord_to_index_list(buffer.layout.shape_coord())
    var batch_size = shape[0]
    var vocab_size = shape[1]

    for b in range(batch_size):
        comptime if normalized:
            var row_sum = Scalar[dtype](0.0)

            for i in range(vocab_size):
                var val = random_float64(0.01, 10.0).cast[dtype]()
                buffer[b, i] = val
                row_sum += val

            # Normalize to sum to 1.0.
            for i in range(vocab_size):
                buffer[b, i] = buffer[b, i] / row_sum
        else:
            # Raw logits (unnormalized) in given range [-5.0, 5.0].
            for i in range(vocab_size):
                var random_value = random_float64(-5.0, 5.0)
                buffer[b, i] = random_value.cast[dtype]()


fn compute_topk_mask[
    dtype: DType,
](
    values: TileTensor[dtype, ...],
    mask: TileTensor[mut=True, DType.bool, ...],
    K: Int,
    batch_size: Int,
    N: Int,
) raises:
    """
    Compute a boolean mask indicating which tokens are in the top-K.
    Marks all tokens whose value is >= K-th largest value.
    """
    comptime assert values.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert mask.flat_rank == 2, "expected rank-2 TileTensor"
    for b in range(batch_size):
        var values_list = List[Scalar[dtype]]()
        for i in range(N):
            values_list.append(values.load[width=1]((Idx(b), Idx(i))))

        @parameter
        fn _greater_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
            return lhs > rhs

        sort[_greater_than](values_list)

        # K-th largest value.
        var kth_value = values_list[K - 1]

        # Mark all tokens >= kth_value.
        for i in range(N):
            mask[b, i] = values[b, i] >= kth_value


fn validate_sampling_results[
    out_idx_type: DType,
](
    sampled_idxs: TileTensor[out_idx_type, ...],
    mask: TileTensor[DType.bool, ...],
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
    comptime assert sampled_idxs.flat_rank == 1, "expected rank-1 TileTensor"
    comptime assert mask.flat_rank == 2, "expected rank-2 TileTensor"
    for b in range(batch_size):
        var idx = Int(sampled_idxs.load[width=1]((Idx(b),)))

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
        var is_valid = mask[b, idx]
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


# ===----------------------------------------------------------------------=== #
# Top-K + Top-P (nucleus) sampling tests.
#
# The TopKTopPSamplingFromProbKernel is a copy of TopKSamplingFromProbKernel
# with one additional constraint in the acceptance check:
#
#   Top-K only:
#     accept if count_above_pivot < k
#
#   Top-K + Top-P:
#     accept if count_above_pivot < k AND sum_above_pivot < p
#
# The `.value` field (cumulative probability above the pivot) was already
# computed by the ValueCount block reduction — the top-p kernel just uses it.
#
# When top_p = 1.0, sum < 1.0 is always true so it degrades to top-k-only
# with zero overhead.
# ===----------------------------------------------------------------------=== #


fn compute_topp_mask[
    dtype: DType,
](
    probs: TileTensor[dtype, ...],
    mask: TileTensor[mut=True, DType.bool, ...],
    p: Float32,
    batch_size: Int,
    N: Int,
) raises:
    """Compute a boolean mask indicating which tokens are in the top-p nucleus.

    Tokens are sorted by descending probability and included until cumulative
    probability exceeds p. All included tokens are marked True.
    """
    comptime assert probs.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert mask.flat_rank == 2, "expected rank-2 TileTensor"
    for b in range(batch_size):
        # Collect (prob, index) pairs.
        var prob_idx = List[Tuple[Scalar[dtype], Int]]()
        for i in range(N):
            prob_idx.append((probs.load[width=1]((Idx(b), Idx(i))), i))

        # Sort descending by probability.
        @parameter
        fn _greater_than(
            lhs: Tuple[Scalar[dtype], Int], rhs: Tuple[Scalar[dtype], Int]
        ) -> Bool:
            return lhs[0] > rhs[0]

        sort[_greater_than](prob_idx)

        # Walk sorted list, include tokens until cumulative prob >= p.
        var cumsum = Float32(0.0)
        for i in range(N):
            var prob = prob_idx[i][0]
            var idx = prob_idx[i][1]
            if cumsum < p:
                mask[b, idx] = True
                cumsum += prob.cast[DType.float32]()
            else:
                mask[b, idx] = False


fn validate_topk_topp_sampling_results[
    out_idx_type: DType,
](
    sampled_idxs: TileTensor[out_idx_type, ...],
    topk_mask: TileTensor[DType.bool, ...],
    topp_mask: TileTensor[DType.bool, ...],
    batch_size: Int,
    N: Int,
    trial_num: Int,
) raises:
    """Validate that sampled indices satisfy both top-k AND top-p constraints.
    """
    comptime assert sampled_idxs.flat_rank == 1, "expected rank-1 TileTensor"
    comptime assert topk_mask.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert topp_mask.flat_rank == 2, "expected rank-2 TileTensor"
    for b in range(batch_size):
        var idx = Int(sampled_idxs.load[width=1]((Idx(b),)))

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

        var in_topk = topk_mask[b, idx]
        var in_topp = topp_mask[b, idx]

        if not in_topk:
            raise Error(
                "Trial "
                + String(trial_num)
                + ", Batch "
                + String(b)
                + ": Sampled index "
                + String(idx)
                + " is NOT in the top-K set!"
            )

        if not in_topp:
            raise Error(
                "Trial "
                + String(trial_num)
                + ", Batch "
                + String(b)
                + ": Sampled index "
                + String(idx)
                + " is NOT in the top-P nucleus!"
            )


fn test_topk_topp_sampling[
    dtype: DType,
    out_idx_type: DType = DType.int32,
    block_size: Int = 1024,
](ctx: DeviceContext, batch_size: Int, N: Int, K: Int, p: Float32) raises:
    """Test joint top-K + top-P sampling by validating samples are in both sets.

    Runs multiple trials with different seeds and verifies every sample is in
    the intersection of the top-K set and the top-P nucleus.
    """
    print(
        "==== Running Top-K+Top-P, N=",
        N,
        ", K=",
        K,
        ", p=",
        p,
        ", batch_size=",
        batch_size,
    )

    var input_shape = IndexList[2](batch_size, N)
    var input_runtime_layout = row_major(Coord(input_shape))
    var output_shape = IndexList[1](batch_size)
    var output_runtime_layout = row_major(Idx(batch_size))
    var mask_runtime_layout = row_major(Coord(input_shape))

    var device_input = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_output = ctx.enqueue_create_buffer[out_idx_type](
        output_shape.flattened_length()
    )
    var topk_mask_buffer = ctx.enqueue_create_buffer[DType.bool](
        input_shape.flattened_length()
    )
    var topp_mask_buffer = ctx.enqueue_create_buffer[DType.bool](
        input_shape.flattened_length()
    )

    var input_tensor = TileTensor(device_input, input_runtime_layout)
    var output_tensor = TileTensor(device_output, output_runtime_layout)

    # Fill with normalized probabilities.
    with device_input.map_to_host() as input_host:
        var input_host_tensor = TileTensor(input_host, input_runtime_layout)
        fill_random_for_test[dtype, normalized=True](input_host_tensor)

        # Compute ground truth masks on host.
        with topk_mask_buffer.map_to_host() as topk_host:
            var topk_mask_tensor = TileTensor(topk_host, mask_runtime_layout)
            compute_topk_mask[dtype](
                input_host_tensor, topk_mask_tensor, K, batch_size, N
            )

        with topp_mask_buffer.map_to_host() as topp_host:
            var topp_mask_tensor = TileTensor(topp_host, mask_runtime_layout)
            compute_topp_mask[dtype](
                input_host_tensor, topp_mask_tensor, p, batch_size, N
            )

    # Create a 1-element seed buffer on device.
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    var seed_layout = row_major(Idx(1))

    # Run sampling trials.
    var num_passed = 0
    for trial in range(NUM_VALIDATION_TRIALS):
        var trial_seed = UInt64(42 + trial)
        with seed_buf.map_to_host() as seed_host:
            seed_host[0] = trial_seed
        var seed_tt = (
            TileTensor(seed_buf, seed_layout).as_any_origin().as_immut()
        )

        topk_topp_sampling_from_prob[dtype, out_idx_type, block_size](
            ctx,
            input_tensor,
            output_tensor,
            K,
            top_p_val=p,
            deterministic=False,
            rng_seed=seed_tt,
            rng_offset=0,
        )

        with device_output.map_to_host() as output_host:
            with topk_mask_buffer.map_to_host() as topk_host:
                with topp_mask_buffer.map_to_host() as topp_host:
                    var output_host_tensor = TileTensor(
                        output_host, output_runtime_layout
                    )
                    var topk_mask_tensor = TileTensor(
                        topk_host, mask_runtime_layout
                    )
                    var topp_mask_tensor = TileTensor(
                        topp_host, mask_runtime_layout
                    )
                    validate_topk_topp_sampling_results[out_idx_type](
                        output_host_tensor,
                        topk_mask_tensor,
                        topp_mask_tensor,
                        batch_size,
                        N,
                        trial,
                    )
        num_passed += 1

    print("  All", num_passed, "trials passed!")


fn test_topk_sampling[
    dtype: DType,
    out_idx_type: DType = DType.int32,
    block_size: Int = 1024,
    sampling_from_prob: Bool = True,
](ctx: DeviceContext, test_case: TestCase) raises:
    """
    Test top-K sampling kernels with property-based validation.

    Parameters:
        dtype: Data type of the input.
        out_idx_type: Data type of the output indices.
        block_size: Block size for the kernel.
        sampling_from_prob: If True, test topk_sampling_from_prob with probabilities.
                           If False, test topk_softmax_sample with logits.

    This test validates that the sampling kernel correctly samples from the
    top-K distribution by:
    1. Computing the ground truth top-K set.
    2. Running sampling multiple times.
    3. Verifying each sample is from the valid top-K set.
    """

    var m = Bench()
    var batch_size = test_case.batch_size
    var N = test_case.N
    var K = test_case.K
    comptime largest = test_case.largest
    comptime sampling = test_case.sampling

    comptime assert sampling, "This test requires sampling=True"

    # Create layouts for input tensor [batch_size, N].
    var input_shape = IndexList[2](batch_size, N)
    var input_runtime_layout = row_major(Coord(input_shape))

    # Create layouts for output tensor [batch_size].
    var output_shape = IndexList[1](batch_size)
    var output_runtime_layout = row_major(Idx(batch_size))

    # Create layouts for mask tensor [batch_size, N].
    var mask_runtime_layout = row_major(Coord(input_shape))

    # Create device buffers.
    var device_input = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_output = ctx.enqueue_create_buffer[out_idx_type](
        output_shape.flattened_length()
    )
    var mask_buffer = ctx.enqueue_create_buffer[DType.bool](
        input_shape.flattened_length()
    )

    # Create layout tensors for GPU operations.
    var input_tensor = TileTensor(device_input, input_runtime_layout)
    var output_tensor = TileTensor(device_output, output_runtime_layout)

    # Initialize input data on host.
    with device_input.map_to_host() as input_host:
        var input_host_tensor = TileTensor(input_host, input_runtime_layout)

        comptime if sampling_from_prob:
            fill_random_for_test[dtype, normalized=True](input_host_tensor)
        else:
            fill_random_for_test[dtype, normalized=False](input_host_tensor)

        comptime if PRINT_OUTPUT:
            comptime if sampling_from_prob:
                print("Sample probabilities (first batch, first 10):")
            else:
                print("Sample logits (first batch, first 10):")
            for i in range(min(10, N)):
                print("  Token", i, ":", input_host_tensor.ptr[i])

        # STEP 1: Compute ground truth mask (while we have input on host).
        with mask_buffer.map_to_host() as mask_host:
            var mask_host_tensor = TileTensor(mask_host, mask_runtime_layout)
            compute_topk_mask[dtype](
                input_host_tensor, mask_host_tensor, K, batch_size, N
            )

            comptime if PRINT_OUTPUT:
                print("  Valid top-K indices for first batch:")
                for i in range(N):
                    if mask_host_tensor.ptr[i]:
                        comptime if sampling_from_prob:
                            print(
                                "    Token",
                                i,
                                "with prob",
                                input_host_tensor.ptr[i],
                            )
                        else:
                            print(
                                "    Token",
                                i,
                                "with logit",
                                input_host_tensor.ptr[i],
                            )

    # STEP 2: Run sampling validation.
    comptime if PRINT_OUTPUT:
        print(
            "  [Validation] Running",
            NUM_VALIDATION_TRIALS,
            "sampling trials...",
        )

    var num_passed = 0

    for trial in range(NUM_VALIDATION_TRIALS):
        var trial_seed = UInt64(42 + trial)

        comptime if sampling_from_prob:
            topk_sampling_from_prob[dtype, out_idx_type, block_size](
                ctx,
                input_tensor,
                output_tensor,
                K,
                deterministic=False,
                rng_seed=trial_seed,
                rng_offset=0,
            )
        else:
            topk_softmax_sample[dtype, out_idx_type, block_size](
                ctx,
                input_tensor,
                output_tensor,
                K,
                temperature_val=1.0,
                seed_val=trial_seed,
            )

        # Read back results and validate.
        with device_output.map_to_host() as output_host:
            with mask_buffer.map_to_host() as mask_host:
                var output_host_tensor = TileTensor(
                    output_host, output_runtime_layout
                )
                var mask_host_tensor = TileTensor(
                    mask_host, mask_runtime_layout
                )

                validate_sampling_results[out_idx_type](
                    output_host_tensor, mask_host_tensor, batch_size, N, trial
                )
        num_passed += 1

    comptime if PRINT_OUTPUT:
        print(
            "  [Validation] ✓ All",
            num_passed,
            "/",
            NUM_VALIDATION_TRIALS,
            "trials passed!",
        )

    # STEP 3: Benchmark the kernel (separate from validation).
    comptime if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            comptime if sampling_from_prob:
                topk_sampling_from_prob[dtype, out_idx_type, block_size](
                    ctx,
                    input_tensor,
                    output_tensor,
                    K,
                    deterministic=False,
                    rng_seed=UInt64(42),
                    rng_offset=0,
                )
            else:
                topk_softmax_sample[dtype, out_idx_type, block_size](
                    ctx,
                    input_tensor,
                    output_tensor,
                    K,
                    temperature_val=1.0,
                    seed_val=UInt64(42),
                )
            ctx.synchronize()

        comptime if sampling_from_prob:
            time_kernel[run_func](m, ctx, "topk-sampling-from-prob")
        else:
            time_kernel[run_func](m, ctx, "topk-softmax-sample")
        m.dump_report()


fn extract_topk_from_masked[
    dtype: DType,
    out_idx_type: DType,
](
    masked_logits: TileTensor[dtype, ...],
    K: Int,
    topk_vals_out: TileTensor[mut=True, dtype, ...],
    topk_idxs_out: TileTensor[mut=True, out_idx_type, ...],
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
    comptime assert masked_logits.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert topk_vals_out.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert topk_idxs_out.flat_rank == 2, "expected rank-2 TileTensor"
    var batch_size = masked_logits.layout.shape[0]().value()
    var N = masked_logits.layout.shape[1]().value()

    for b in range(batch_size):
        var values = List[Scalar[dtype]]()
        var indices = List[Int]()

        for i in range(N):
            var val = masked_logits.load[width=1]((Idx(b), Idx(i)))
            if val != min_or_neg_inf[dtype]():
                values.append(val)
                indices.append(i)

        # Sort by value (descending).
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[j] > values[i]:
                    # Swap values.
                    var temp_val = values[i]
                    values[i] = values[j]
                    values[j] = temp_val
                    # Swap indices.
                    var temp_idx = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp_idx

        # Copy top-K values and indices to output.
        for k in range(K):
            if k < len(values):
                topk_vals_out[b, k] = values[k]
                topk_idxs_out[b, k] = Scalar[out_idx_type](indices[k])
            else:
                # If we have fewer than K non-inf values, fill with -inf and -1.
                topk_vals_out[b, k] = min_or_neg_inf[dtype]()
                topk_idxs_out[b, k] = Scalar[out_idx_type](-1)


fn test_case_batched[
    dtype: DType,
    fill_fn: fn[rank: Int, dtype: DType](
        TileTensor[mut=True, dtype, ...]
    ) capturing[_] -> None,
    out_idx_type: DType = DType.int,
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
    comptime assert (
        not sampling
    ), "topk_mask_logits only supports sampling=False"

    # Create layouts for input/masked_logits tensors [batch_size, N].
    var input_shape = IndexList[2](batch_size, N)
    var input_runtime_layout = row_major(Coord(input_shape))

    # Create layouts for topk output tensors [batch_size, K].
    var topk_shape = IndexList[2](batch_size, K)
    var topk_runtime_layout = row_major(Coord(topk_shape))

    # Create device buffers.
    var device_in = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_masked_logits = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var topk_vals_extracted_buf = ctx.enqueue_create_buffer[dtype](
        topk_shape.flattened_length()
    )
    var topk_idxs_extracted_buf = ctx.enqueue_create_buffer[out_idx_type](
        topk_shape.flattened_length()
    )
    var topk_vals_cpu_buf = ctx.enqueue_create_buffer[dtype](
        topk_shape.flattened_length()
    )
    var topk_idxs_cpu_buf = ctx.enqueue_create_buffer[DType.int64](
        topk_shape.flattened_length()
    )

    # Create layout tensors for GPU operations.
    var in_tensor = TileTensor(device_in, input_runtime_layout)
    var masked_logits_tensor = TileTensor(
        device_masked_logits, input_runtime_layout
    )

    # Initialize input data on host.
    with device_in.map_to_host() as in_host:
        var in_host_tensor = TileTensor(in_host, input_runtime_layout)
        fill_fn[2, dtype](in_host_tensor)

    comptime if DEBUG_BENCH:

        @always_inline
        @parameter
        fn run_func(ctx: DeviceContext) raises:
            topk_mask_logits[dtype, out_idx_type, block_size](
                ctx,
                in_tensor,
                masked_logits_tensor,
                K,
            )
            ctx.synchronize()

        time_kernel[run_func](m, ctx, "topk-mask-logits")

    topk_mask_logits[dtype, out_idx_type, block_size](
        ctx,
        in_tensor,
        masked_logits_tensor,
        K,
    )
    ctx.synchronize()

    # Read back masked logits and extract top-K.
    with device_masked_logits.map_to_host() as masked_logits_host:
        var masked_logits_host_tensor = TileTensor(
            masked_logits_host, input_runtime_layout
        )

        comptime if PRINT_OUTPUT:
            print("Masked logits (first 10):")
            for i in range(min(10, input_shape.flattened_length())):
                print("  ", i, ":", masked_logits_host_tensor.ptr[i])

        with topk_vals_extracted_buf.map_to_host() as topk_vals_host:
            with topk_idxs_extracted_buf.map_to_host() as topk_idxs_host:
                var topk_vals_tensor = TileTensor(
                    topk_vals_host, topk_runtime_layout
                )
                var topk_idxs_tensor = TileTensor(
                    topk_idxs_host, topk_runtime_layout
                )

                extract_topk_from_masked[dtype, out_idx_type](
                    masked_logits_host_tensor,
                    K,
                    topk_vals_tensor,
                    topk_idxs_tensor,
                )

                comptime if PRINT_OUTPUT:
                    print("Extracted top-K values (first 10):")
                    for i in range(min(10, topk_shape.flattened_length())):
                        print("  ", i, ":", topk_vals_tensor.ptr[i])
                    print("Extracted top-K indices (first 10):")
                    for i in range(min(10, topk_shape.flattened_length())):
                        print("  ", i, ":", topk_idxs_tensor.ptr[i])

    # Run CPU reference.
    with device_in.map_to_host() as in_host:
        with topk_vals_cpu_buf.map_to_host() as topk_vals_cpu_host:
            with topk_idxs_cpu_buf.map_to_host() as topk_idxs_cpu_host:
                var in_host_tensor = TileTensor(in_host, input_runtime_layout)
                var topk_vals_cpu_tensor = TileTensor(
                    topk_vals_cpu_host, topk_runtime_layout
                )
                var topk_idxs_cpu_tensor = TileTensor(
                    topk_idxs_cpu_host, topk_runtime_layout
                )

                comptime if DEBUG_BENCH:

                    @always_inline
                    @parameter
                    fn run_func_cpu(ctx: DeviceContext) raises:
                        _top_k_cpu[
                            dtype=dtype,
                            out_idx_type = DType.int64,
                            largest=largest,
                        ](
                            in_host_tensor,
                            K,
                            1,  # rank - 1
                            topk_vals_cpu_tensor,
                            topk_idxs_cpu_tensor,
                            1,
                            True,
                        )

                    time_kernel[run_func_cpu](m, ctx, "topk-cpu")

                _top_k_cpu[
                    dtype=dtype, out_idx_type = DType.int64, largest=largest
                ](
                    in_host_tensor,
                    K,
                    1,  # rank - 1
                    topk_vals_cpu_tensor,
                    topk_idxs_cpu_tensor,
                    1,
                    True,
                )

                comptime if PRINT_OUTPUT:
                    print("CPU top-K values (first 10):")
                    for i in range(min(10, topk_shape.flattened_length())):
                        print("  ", i, ":", topk_vals_cpu_tensor.ptr[i])
                    print("CPU top-K indices (first 10):")
                    for i in range(min(10, topk_shape.flattened_length())):
                        print("  ", i, ":", topk_idxs_cpu_tensor.ptr[i])

    # Compare extracted values with CPU reference.
    with topk_vals_extracted_buf.map_to_host() as topk_vals_ext_host:
        with topk_vals_cpu_buf.map_to_host() as topk_vals_cpu_host:
            var topk_vals_ext_tensor = TileTensor(
                topk_vals_ext_host, topk_runtime_layout
            )
            var topk_vals_cpu_tensor = TileTensor(
                topk_vals_cpu_host, topk_runtime_layout
            )

            for i in range(topk_shape.flattened_length()):
                assert_almost_equal(
                    topk_vals_ext_tensor.ptr[i],
                    topk_vals_cpu_tensor.ptr[i],
                    msg="Top-K values mismatch at index " + String(i),
                )

                # Note: We don't check exact index equality because different
                # implementations may break ties differently when values are
                # equal or very close. As long as the top-K values match, the
                # indices can differ for tied values.

    comptime if DEBUG_BENCH:
        m.dump_report()


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

    m.bench_function[bench_func](
        BenchId(
            kernel_name
        ),  # ThroughputMeasure(BenchMetric.elements, 2 * size)
    )


@parameter
fn fill_random[
    rank: Int, dtype: DType
](buffer: TileTensor[mut=True, dtype, ...]):
    comptime min_val = -1e9
    comptime max_val = 1e9
    var total_elements = buffer.numel()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.ptr[i] = random_value.cast[dtype]()


struct TestCase[_sampling: Bool, _largest: Bool = True, _block_size: Int = 256](
    ImplicitlyCopyable
):
    comptime sampling = Self._sampling
    comptime largest = Self._largest
    var N: Int
    var K: Int
    comptime block_size: Int = Self._block_size
    var batch_size: Int
    var num_blocks_per_input: Optional[Int]

    fn __init__(
        out self,
        N: Int,
        K: Int,
        batch_size: Int,
        num_blocks_per_input: Optional[Int] = None,
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


fn _cpu_softmax[
    dtype: DType,
](
    logits: TileTensor[dtype, ...],
    probs_out: TileTensor[mut=True, DType.float32, ...],
    batch_size: Int,
    N: Int,
    T: Float32,
):
    """Compute softmax(logits/T) per row on CPU, writing float32 probs."""
    comptime assert logits.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert probs_out.flat_rank == 2, "expected rank-2 TileTensor"
    for b in range(batch_size):
        var max_val = logits.load[width=1]((Idx(b), Idx(0))).cast[
            DType.float32
        ]()
        for i in range(1, N):
            var v = logits.load[width=1]((Idx(b), Idx(i))).cast[DType.float32]()
            if v > max_val:
                max_val = v

        var exp_sum = Float32(0.0)
        for i in range(N):
            var e = exp(
                (
                    logits.load[width=1]((Idx(b), Idx(i))).cast[DType.float32]()
                    - max_val
                )
                / T
            )
            probs_out[b, i] = e
            exp_sum += e
        for i in range(N):
            probs_out[b, i] = probs_out[b, i] / exp_sum


fn test_topk_topp_sampling_fi[
    dtype: DType,
    out_idx_type: DType = DType.int32,
](
    ctx: DeviceContext,
    batch_size: Int,
    N: Int,
    K: Int,
    p: Float32 = 1.0,
    T: Float32 = 1.0,
) raises:
    """Test _topk_topp_sampling_fi (logits → softmax → top-k+top-p sampling).

    Reuses compute_topk_mask, compute_topp_mask, and
    validate_topk_topp_sampling_results from the existing test infrastructure.
    """
    print(
        "==== Testing _topk_topp_sampling_fi: dtype=",
        dtype,
        " batch=",
        batch_size,
        " N=",
        N,
        " K=",
        K,
        " p=",
        p,
        " T=",
        T,
    )

    var input_shape = IndexList[2](batch_size, N)
    var input_layout = row_major(Coord(input_shape))
    var output_layout = row_major(Idx(batch_size))
    var mask_layout = row_major(Coord(input_shape))

    # Device buffers.
    var logits_buf = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var out_buf = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    var temp_buf = ctx.enqueue_create_buffer[DType.float32](batch_size)

    # CPU reference buffers: probs after softmax, and masks.
    var probs_buf = ctx.enqueue_create_buffer[DType.float32](
        input_shape.flattened_length()
    )
    var topk_mask_buf = ctx.enqueue_create_buffer[DType.bool](
        input_shape.flattened_length()
    )
    var topp_mask_buf = ctx.enqueue_create_buffer[DType.bool](
        input_shape.flattened_length()
    )

    # Fill logits, compute CPU softmax → probs → masks.
    with logits_buf.map_to_host() as logits_host:
        var logits_tt = TileTensor(logits_host, input_layout)
        fill_random_for_test[dtype, normalized=False](logits_tt)

        with probs_buf.map_to_host() as probs_host:
            var probs_tt = TileTensor(probs_host, input_layout)
            _cpu_softmax[dtype](logits_tt, probs_tt, batch_size, N, T)

            with topk_mask_buf.map_to_host() as topk_host:
                var topk_tt = TileTensor(topk_host, mask_layout)
                compute_topk_mask[DType.float32](
                    probs_tt, topk_tt, K, batch_size, N
                )

            with topp_mask_buf.map_to_host() as topp_host:
                var topp_tt = TileTensor(topp_host, mask_layout)
                compute_topp_mask[DType.float32](
                    probs_tt, topp_tt, p, batch_size, N
                )

    # Fill temperature.
    with temp_buf.map_to_host() as temp_host:
        for i in range(batch_size):
            temp_host[i] = T

    # Create kernel input tensors.
    var logits_tt = TileTensor(logits_buf, input_layout)
    var out_tt = TileTensor(
        out_buf, row_major(Coord(IndexList[2](batch_size, 1)))
    )
    var temp_tt = TileTensor(temp_buf, row_major(Idx(batch_size)))

    # Create a 1-element seed buffer on device.
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    var seed_layout = row_major(Idx(1))

    # Run trials with different seeds.
    var num_passed = 0
    for trial in range(NUM_VALIDATION_TRIALS):
        with seed_buf.map_to_host() as seed_host:
            seed_host[0] = UInt64(42 + trial)
        var seed_tt = (
            TileTensor(seed_buf, seed_layout).as_any_origin().as_immut()
        )

        _topk_topp_sampling_fi(
            ctx,
            K,
            p,
            logits_tt,
            out_tt,
            temperature=temp_tt.as_any_origin().as_immut(),
            rng_seed=seed_tt,
        )

        with out_buf.map_to_host() as out_host:
            with topk_mask_buf.map_to_host() as topk_host:
                with topp_mask_buf.map_to_host() as topp_host:
                    validate_topk_topp_sampling_results[out_idx_type](
                        TileTensor(out_host, output_layout),
                        TileTensor(topk_host, mask_layout),
                        TileTensor(topp_host, mask_layout),
                        batch_size,
                        N,
                        trial,
                    )
        num_passed += 1

    print("  All", num_passed, "trials passed!")


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
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case1)

        comptime sampling_test_case2 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=1024,
            K=50,
            batch_size=16,
        )
        print_test_case(sampling_test_case2)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case2)

        comptime sampling_test_case3 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=32000,
            K=100,
            batch_size=8,
        )
        print_test_case(sampling_test_case3)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case3)

        comptime sampling_test_case4 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=32000,
            K=5,
            batch_size=32,
        )
        print_test_case(sampling_test_case4)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case4)

        comptime sampling_test_case5 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=1024,
            K=20,
            batch_size=256,
        )
        print_test_case(sampling_test_case5)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case5)

        comptime sampling_test_case6 = TestCase[
            _sampling=True, _block_size=default_block_size
        ](
            N=llama3_vocab_size,
            K=50,
            batch_size=4,
        )
        print_test_case(sampling_test_case6)
        test_topk_sampling[
            bf16_type,
            DType.int32,
            default_block_size,
            sampling_from_prob=True,
        ](ctx, sampling_test_case6)

        print("\n" + "=" * 80)
        print("All topk_sampling_from_prob tests passed! ✓")
        print("=" * 80 + "\n")

        # Tests topk_softmax_sample with logits
        print("\n" + "=" * 80)
        print("Testing topk_softmax_sample kernel")
        print("=" * 80 + "\n")

        print_test_case(sampling_test_case1)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=False,
        ](ctx, sampling_test_case1)

        print_test_case(sampling_test_case2)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=False,
        ](ctx, sampling_test_case2)

        print_test_case(sampling_test_case3)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=False,
        ](ctx, sampling_test_case3)

        print_test_case(sampling_test_case4)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=False,
        ](ctx, sampling_test_case4)

        print_test_case(sampling_test_case5)
        test_topk_sampling[
            float32_dtype,
            DType.int32,
            default_block_size,
            sampling_from_prob=False,
        ](ctx, sampling_test_case5)

        print_test_case(sampling_test_case6)
        test_topk_sampling[
            bf16_type,
            DType.int32,
            512,  # Note: 1024 is too large (out of resources)
            sampling_from_prob=False,
        ](ctx, sampling_test_case6)

        print("\n" + "=" * 80)
        print("All topk_softmax_sample tests passed! ✓")
        print("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("Testing topk_topp_sampling_from_prob kernel")
        print("=" * 80 + "\n")

        # top_p = 1.0 should behave identically to top-k only.
        test_topk_topp_sampling[float32_dtype, DType.int32, default_block_size](
            ctx, batch_size=1, N=100, K=10, p=1.0
        )

        # Small vocab, tight nucleus.
        test_topk_topp_sampling[float32_dtype, DType.int32, default_block_size](
            ctx, batch_size=4, N=1024, K=50, p=0.9
        )

        # Larger vocab, moderate nucleus.
        test_topk_topp_sampling[float32_dtype, DType.int32, default_block_size](
            ctx, batch_size=8, N=32000, K=100, p=0.95
        )

        # Tight nucleus (p=0.5) — should restrict more than k alone.
        test_topk_topp_sampling[float32_dtype, DType.int32, default_block_size](
            ctx, batch_size=16, N=32000, K=50, p=0.5
        )

        # bfloat16 test.
        test_topk_topp_sampling[bf16_type, DType.int32, default_block_size](
            ctx, batch_size=4, N=1024, K=20, p=0.9
        )

        print("\n" + "=" * 80)
        print("All topk_topp_sampling_from_prob tests passed! ✓")
        print("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("Testing _topk_topp_sampling_fi (logits → softmax → sampling)")
        print("=" * 80 + "\n")

        # Top-k only (p=1.0), default temperature.
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=1, N=1024, K=10
        )
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=4, N=32000, K=50
        )

        # Top-k + top-p.
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=4, N=32000, K=50, p=0.9
        )
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=8, N=1024, K=20, p=0.5, T=0.8
        )

        # bfloat16.
        test_topk_topp_sampling_fi[bf16_type](
            ctx, batch_size=4, N=1024, K=20, p=0.9
        )

        print("\n" + "=" * 80)
        print("All _topk_topp_sampling_fi tests passed! ✓")
        print("=" * 80 + "\n")
