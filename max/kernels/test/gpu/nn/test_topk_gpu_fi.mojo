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


from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major
from layout._fillers import random
from std.math import exp
from nn.topk import _topk_topp_sampling_fi
from nn.sampling import (
    topk_topp_sampling_from_prob,
)
from std.random import random_float64, seed
from std.utils import IndexList
from std.utils.index import Index

comptime NUM_VALIDATION_TRIALS = 50


def fill_random_for_test[
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


def compute_topk_mask[
    dtype: DType,
](
    values: TileTensor[dtype, ...],
    mask: TileTensor[mut=True, .bool, ...],
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
    comptime assert values.flat_rank >= 2
    for b in range(batch_size):
        # K == -1 means no top-k filtering; all tokens are valid.
        if K == -1:
            for i in range(N):
                mask[b, i] = True
            continue

        var values_list = List[Scalar[dtype]]()
        for i in range(N):
            values_list.append(values.load[width=1]((b, i)))

        def _greater_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
            return lhs > rhs

        sort(values_list, _greater_than)

        # K-th largest value.
        var kth_value = values_list[K - 1]

        # Mark all tokens >= kth_value.
        for i in range(N):
            mask[b, i] = values[b, i] >= kth_value


def compute_topp_mask[
    dtype: DType,
](
    probs: TileTensor[dtype, ...],
    mask: TileTensor[mut=True, .bool, ...],
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
    comptime assert probs.flat_rank >= 2
    for b in range(batch_size):
        # Collect (prob, index) pairs.
        var prob_idx = List[Tuple[Scalar[dtype], Int]]()
        for i in range(N):
            prob_idx.append((probs.load[width=1]((b, i)), i))

        # Sort descending by probability.
        def _greater_than(
            lhs: Tuple[Scalar[dtype], Int], rhs: Tuple[Scalar[dtype], Int]
        ) -> Bool:
            return lhs[0] > rhs[0]

        sort(prob_idx, _greater_than)

        # Walk sorted list, include tokens until cumulative prob >= p.
        var cumsum = Float32(0.0)
        for i in range(N):
            var prob = prob_idx[i][0]
            var idx = prob_idx[i][1]
            if cumsum < p:
                mask[b, idx] = True
                cumsum += prob.cast[.float32]()
            else:
                mask[b, idx] = False


def validate_topk_topp_sampling_results[
    out_idx_type: DType,
](
    sampled_idxs: TileTensor[out_idx_type, ...],
    topk_mask: TileTensor[.bool, ...],
    topp_mask: TileTensor[.bool, ...],
    batch_size: Int,
    N: Int,
    trial_num: Int,
) raises:
    """Validate that sampled indices satisfy both top-k AND top-p constraints.
    """
    comptime assert sampled_idxs.flat_rank == 1, "expected rank-1 TileTensor"
    comptime assert topk_mask.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert topp_mask.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert sampled_idxs.flat_rank >= 1
    for b in range(batch_size):
        var idx = Int(sampled_idxs[b])

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


def test_topk_topp_sampling[
    dtype: DType,
    out_idx_type: DType = .int32,
    block_size: Int = 1024,
](ctx: DeviceContext[], batch_size: Int, N: Int, K: Int, p: Float32) raises:
    """Test joint top-K + top-P sampling by validating samples are in both sets.

    Runs multiple trials with different seeds and verifies every sample is in
    the intersection of the top-K set and the top-P nucleus.
    """
    print(
        t"==== Running Top-K+Top-P, N= {N} , K= {K} , p= {p} , batch_size="
        t" {batch_size}"
    )

    var input_shape = Index(batch_size, N)
    var input_runtime_layout = row_major(Coord(input_shape))
    var output_shape = IndexList[1](batch_size)
    var output_runtime_layout = row_major(batch_size)
    var mask_runtime_layout = row_major(Coord(input_shape))

    var device_input = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_output = ctx.enqueue_create_buffer[out_idx_type](
        output_shape.flattened_length()
    )
    var topk_mask_buffer = ctx.enqueue_create_buffer[.bool](
        input_shape.flattened_length()
    )
    var topp_mask_buffer = ctx.enqueue_create_buffer[.bool](
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

    # Per-row seed buffer: the kernel indexes rng_seed by row_idx (the
    # request's logical row), so every row needs an entry even though all
    # rows share the same seed value here.
    var seed_buf = ctx.enqueue_create_buffer[.uint64](batch_size)
    var seed_layout = row_major(batch_size)

    # Run sampling trials.
    var num_passed = 0
    for trial in range(NUM_VALIDATION_TRIALS):
        var trial_seed = UInt64(42 + trial)
        with seed_buf.map_to_host() as seed_host:
            for b in range(batch_size):
                seed_host[b] = trial_seed
        var seed_tt = (
            TileTensor(seed_buf, seed_layout).as_unsafe_any_origin().as_imm()
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


def test_topk_topp_rng_offset_batch_invariant[
    dtype: DType,
    out_idx_type: DType = .int32,
    block_size: Int = 1024,
](ctx: DeviceContext[], N: Int, K: Int, p: Float32) raises:
    """Regression test: the same request samples the same token regardless of
    its physical batch slot.

    The per-row seed is the kernel's whole RNG key -- nothing derived from
    `block_idx.x` reaches the Philox counter -- so slots carrying the same
    distribution and the same seed must agree. Every row is given identical
    probabilities and the identical seed, leaving `block_idx.x` as the only
    difference between them. Keying the counter on the slot (as this kernel
    once did) makes them diverge. Uses a nucleus with several tokens so the
    draw actually selects among candidates rather than collapsing to the
    argmax.

    Decorrelating co-resident requests is the per-row seed's job, not the
    counter's: callers derive each row's seed from a stable per-request id.
    """
    comptime batch_size = 8
    print(
        t"==== Running RNG-offset batch-invariance, N= {N} , K= {K} , p= {p} ,"
        t" batch_size= {batch_size}"
    )

    var input_shape = Index(batch_size, N)
    var input_runtime_layout = row_major(Coord(input_shape))
    var output_runtime_layout = row_major(batch_size)

    var device_input = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var device_output = ctx.enqueue_create_buffer[out_idx_type](batch_size)

    var input_tensor = TileTensor(device_input, input_runtime_layout)
    var output_tensor = TileTensor(device_output, output_runtime_layout)

    # Every row gets row 0's distribution, so the rows differ only by slot.
    with device_input.map_to_host() as input_host:
        var input_host_tensor = TileTensor(input_host, input_runtime_layout)
        fill_random_for_test[dtype, normalized=True](input_host_tensor)
        for b in range(1, batch_size):
            for j in range(N):
                input_host[b * N + j] = input_host[j]

    # One seed value, repeated: the same request key in every slot.
    var seed_buf = ctx.enqueue_create_buffer[.uint64](batch_size)
    var seed_layout = row_major(batch_size)
    with seed_buf.map_to_host() as seed_host:
        for b in range(batch_size):
            seed_host[b] = UInt64(12345)
    var seed_tt = (
        TileTensor(seed_buf, seed_layout).as_unsafe_any_origin().as_imm()
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
        var output_host_tensor = TileTensor(output_host, output_runtime_layout)
        var expected = Int(output_host_tensor[0])
        for b in range(1, batch_size):
            var got = Int(output_host_tensor[b])
            if got != expected:
                raise Error(
                    "RNG offset is not batch-invariant: slot 0 sampled token "
                    + String(expected)
                    + " but slot "
                    + String(b)
                    + " sampled token "
                    + String(got)
                    + " for the same request (same probs + same seed)."
                )

    print("  All", batch_size, "slots sampled the same token!")


def _cpu_softmax[
    dtype: DType,
](
    logits: TileTensor[dtype, ...],
    probs_out: TileTensor[mut=True, .float32, ...],
    batch_size: Int,
    N: Int,
    T: Float32,
):
    """Compute softmax(logits/T) per row on CPU, writing float32 probs."""
    comptime assert logits.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert probs_out.flat_rank == 2, "expected rank-2 TileTensor"
    comptime assert logits.flat_rank >= 2
    for b in range(batch_size):
        var max_val = logits.load[width=1]((b, Idx[0])).cast[.float32]()
        for i in range(1, N):
            var v = logits.load[width=1]((b, i)).cast[.float32]()
            if v > max_val:
                max_val = v

        var exp_sum = Float32(0.0)
        for i in range(N):
            var e = exp(
                (logits.load[width=1]((b, i)).cast[.float32]() - max_val) / T
            )
            probs_out[b, i] = e
            exp_sum += e
        for i in range(N):
            probs_out[b, i] = probs_out[b, i] / exp_sum


def test_topk_topp_sampling_fi[
    dtype: DType,
    out_idx_type: DType = .int32,
](
    ctx: DeviceContext[],
    batch_size: Int,
    N: Int,
    K: Int,
    p: Float32 = 1.0,
    T: Float32 = 1.0,
    max_k: Int = 1,
) raises:
    """Test _topk_topp_sampling_fi (logits → softmax → top-k+top-p sampling).

    Reuses compute_topk_mask, compute_topp_mask, and
    validate_topk_topp_sampling_results from the existing test infrastructure.
    """
    print(
        t"==== Testing _topk_topp_sampling_fi: dtype= {dtype}  batch="
        t" {batch_size}  N= {N}  K= {K}  p= {p}  T= {T}"
    )

    var input_shape = Index(batch_size, N)
    var input_layout = row_major(Coord(input_shape))
    var output_layout = row_major(batch_size)
    var mask_layout = row_major(Coord(input_shape))

    # Device buffers.
    var logits_buf = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var out_buf = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    var temp_buf = ctx.enqueue_create_buffer[.float32](batch_size)

    # Per-row K, P, and seed arrays.
    var k_buf = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    var p_buf = ctx.enqueue_create_buffer[.float32](batch_size)
    var seed_buf = ctx.enqueue_create_buffer[.uint64](batch_size)
    var batch_layout = row_major(batch_size)

    # CPU reference buffers: probs after softmax, and masks.
    var probs_buf = ctx.enqueue_create_buffer[.float32](
        input_shape.flattened_length()
    )
    var topk_mask_buf = ctx.enqueue_create_buffer[.bool](
        input_shape.flattened_length()
    )
    var topp_mask_buf = ctx.enqueue_create_buffer[.bool](
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
                compute_topk_mask[.float32](probs_tt, topk_tt, K, batch_size, N)

            with topp_mask_buf.map_to_host() as topp_host:
                var topp_tt = TileTensor(topp_host, mask_layout)
                compute_topp_mask[.float32](probs_tt, topp_tt, p, batch_size, N)

    # Fill temperature.
    with temp_buf.map_to_host() as temp_host:
        for i in range(batch_size):
            temp_host[i] = T

    # Fill per-row K array (same value for all rows).
    with k_buf.map_to_host() as k_host:
        for i in range(batch_size):
            k_host[i] = Scalar[out_idx_type](K)

    # Fill per-row P array (same value for all rows).
    with p_buf.map_to_host() as p_host:
        for i in range(batch_size):
            p_host[i] = p

    # Create kernel input tensors.
    var logits_tt = TileTensor(logits_buf, input_layout)
    var out_tt = TileTensor(out_buf, row_major(batch_size, 1))
    var temp_tt = TileTensor(temp_buf, batch_layout)
    var k_tt = TileTensor(k_buf, batch_layout)
    var p_tt = TileTensor(p_buf, batch_layout)

    # Run trials with different seeds.
    var num_passed = 0
    for trial in range(NUM_VALIDATION_TRIALS):
        # Fill per-row seed array (different seed per row).
        with seed_buf.map_to_host() as seed_host:
            for i in range(batch_size):
                seed_host[i] = UInt64(42 + trial * batch_size + i)
        var seed_tt = (
            TileTensor(seed_buf, batch_layout).as_unsafe_any_origin().as_imm()
        )

        _topk_topp_sampling_fi(
            ctx,
            max_k,
            p,
            logits_tt,
            out_tt,
            k=k_tt.as_unsafe_any_origin().as_imm(),
            temperature=temp_tt.as_unsafe_any_origin().as_imm(),
            top_p=p_tt.as_unsafe_any_origin().as_imm(),
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


def main() raises:
    """Test suite for the top-k + top-p sampling kernels."""
    seed(42)
    with DeviceContext() as ctx:
        comptime float32_dtype = DType.float32
        comptime bf16_type = DType.bfloat16

        comptime default_block_size = 1024

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

        # Regression: nothing derived from the batch slot may reach the RNG
        # counter, so a request samples the same token wherever it lands.
        test_topk_topp_rng_offset_batch_invariant[
            float32_dtype, DType.int32, default_block_size
        ](ctx, N=1024, K=50, p=0.9)

        print("\n" + "=" * 80)
        print("All topk_topp_sampling_from_prob tests passed! ✓")
        print("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("Testing _topk_topp_sampling_fi (logits → softmax → sampling)")
        print("=" * 80 + "\n")

        # Top-k only (p=1.0), default temperature.
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=1, N=1024, K=10, max_k=10
        )
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=4, N=32000, K=50, max_k=50
        )

        # Top-k + top-p.
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=4, N=32000, K=50, p=0.9, max_k=50
        )
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=8, N=1024, K=20, p=0.5, T=0.8, max_k=20
        )

        # bfloat16.
        test_topk_topp_sampling_fi[bf16_type](
            ctx, batch_size=4, N=1024, K=20, p=0.9, max_k=20
        )

        # K=-1 in array (no top-k filtering), top-p only.
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=4, N=1024, K=-1, p=0.9, max_k=1024
        )
        test_topk_topp_sampling_fi[float32_dtype](
            ctx, batch_size=8, N=32000, K=-1, p=0.95, max_k=32000
        )

        print("\n" + "=" * 80)
        print("All _topk_topp_sampling_fi tests passed! ✓")
        print("=" * 80 + "\n")
