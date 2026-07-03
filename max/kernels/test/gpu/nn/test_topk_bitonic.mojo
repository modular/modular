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
"""Correctness tests for the block-wide bitonic sort top-k.

Tests `persistent_topk_block` (`topk_bitonic.mojo`) in isolation:
- Output indices are in *descending* score order (scores[idx[0]] ≥ … ≥ scores[idx[K-1]]).
- The set of selected indices exactly matches the CPU reference top-K set.
- Multiple independent batch rows are sorted correctly.
- Padding (N < 2048) does not introduce spurious -1 indices inside [0, K).
- Partial top-K (K < N) selects only the true K-largest elements.
- Duplicate scores are handled without producing duplicate indices.
"""

from std.collections import Set
from std.gpu.host import DeviceContext
from std.math import max
from std.random import seed
from std.testing import assert_equal, assert_true
from layout import TileTensor, row_major

from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    persistent_topk_block,
)


# ===----------------------------------------------------------------------=== #
# CPU reference
# ===----------------------------------------------------------------------=== #


def _cpu_topk_set(scores: List[Float32], K: Int) -> Set[Int]:
    """CPU reference: return the set of indices of the K largest values."""
    var N = len(scores)
    var order = List[Int](capacity=N)
    for i in range(N):
        order.append(i)

    for i in range(min(K, N)):
        var best = i
        for j in range(i + 1, N):
            if scores[order[j]] > scores[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var result = Set[Int]()
    for i in range(K):
        result.add(order[i])
    return result^


# ===----------------------------------------------------------------------=== #
# Core test helper
# ===----------------------------------------------------------------------=== #


def _run_and_check(
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    label: String,
) raises:
    """Run persistent_topk_block on a single row and compare to CPU reference.

    Verifies:
    1. All output indices are in [0, N) or -1 (no OOB).
    2. The output indices are in non-increasing score order.
    3. The set of output indices matches the CPU reference set.
    4. No duplicate indices appear in the output.
    """
    assert N <= PERSISTENT_TOPK_MAX_N, "N exceeds kernel limit"
    assert K <= N, "K must be <= N"
    assert len(scores_host) == N, "scores_host length mismatch"

    # GPU buffers: 1 row of N scores → 1 row of K indices.
    var scores_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var idxs_dev = ctx.enqueue_create_buffer[DType.int32](K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots

    with scores_dev.map_to_host() as buf:
        for i in range(N):
            buf[i] = Scalar[DType.float32](scores_host[i])

    persistent_topk_block(
        ctx,
        rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
            idxs_dev.unsafe_ptr()
        ),
        N,
        K,
        total_seq_len=1,
    )
    ctx.synchronize()

    # Copy back and validate.
    var idxs_host = ctx.enqueue_create_host_buffer[DType.int32](K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var seen = Set[Int]()
    for k in range(K):
        var idx = Int(idxs_host[k])

        # 1. In-bounds.
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] idx[", k, "]=", idx, " is OOB for N=", N),
        )

        # 2. Descending order (scores are non-increasing).
        if k > 0:
            var prev = Int(idxs_host[k - 1])
            assert_true(
                scores_host[idx] <= scores_host[prev],
                String(
                    "[",
                    label,
                    "] order violation at k=",
                    k,
                    ": scores[",
                    idx,
                    "]=",
                    scores_host[idx],
                    " > scores[",
                    prev,
                    "]=",
                    scores_host[prev],
                ),
            )

        # 3. No duplicates.
        assert_true(
            not (idx in seen),
            String("[", label, "] duplicate index ", idx, " at k=", k),
        )
        seen.add(idx)

    # 4. Output set matches the CPU reference set.
    var ref_set = _cpu_topk_set(scores_host, K)

    # GPU set == reference set (ties may order arbitrarily within equal values).
    for k in range(K):
        var idx = Int(idxs_host[k])
        assert_true(
            idx in ref_set,
            String(
                "[",
                label,
                "] idx[",
                k,
                "]=",
                idx,
                " not in reference top-K set",
            ),
        )
    for ref_idx in ref_set:
        assert_true(
            ref_idx in seen,
            String(
                "[", label, "] reference idx ", ref_idx, " missing from output"
            ),
        )

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


# ===----------------------------------------------------------------------=== #
# Test cases
# ===----------------------------------------------------------------------=== #


def test_full_sort_n2048(ctx: DeviceContext) raises:
    """K=N=2048 — full sort; the exact bottleneck shape from the issue."""
    comptime N = 2048
    comptime K = 2048
    seed(42)
    var scores = List[Float32](capacity=N)
    for i in range(N):
        # Unique values: score[i] = float(N - i) so index 0 should rank first.
        scores.append(Float32(N - i))
    _run_and_check(ctx, scores, N, K, "full_sort_n2048")
    print("PASS test_full_sort_n2048")


def test_random_full_sort_n2048(ctx: DeviceContext) raises:
    """K=N=2048 with random float32 scores (may have near-duplicates)."""
    comptime N = 2048
    comptime K = 2048

    # Use a deterministic pseudo-random sequence via a simple LCG.
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state: UInt32 = 0xDEADBEEF
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        # Map to [-10, 10]
        var f = Float32(Int32(state)) / Float32(2**31) * 10.0
        scores.append(f)

    _run_and_check(ctx, scores, N, K, "random_full_sort_n2048")
    print("PASS test_random_full_sort_n2048")


def test_partial_topk_k16(ctx: DeviceContext) raises:
    """K=16, N=2048 — sparse selection (matches MSA block-indexer k=16)."""
    comptime N = 2048
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))  # scores[N-1] is the max → should be idx 0
    _run_and_check(ctx, scores, N, K, "partial_topk_k16")
    print("PASS test_partial_topk_k16")


def test_partial_topk_k1024(ctx: DeviceContext) raises:
    """K=1024, N=2048 — half-sort."""
    comptime N = 2048
    comptime K = 1024
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i * 3 % 1000))  # non-trivial pattern
    _run_and_check(ctx, scores, N, K, "partial_topk_k1024")
    print("PASS test_partial_topk_k1024")


def test_small_n_padded(ctx: DeviceContext) raises:
    """N=64, K=16 — heavily padded (2048 - 64 = 1984 -inf slots)."""
    comptime N = 64
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(N - i) * 0.5)
    _run_and_check(ctx, scores, N, K, "small_n_padded")
    print("PASS test_small_n_padded")


def test_n_equals_k_small(ctx: DeviceContext) raises:
    """N=K=32 — small full sort; all padded slots must be ignored."""
    comptime N = 32
    comptime K = 32
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i * 7 % 97))  # scattered values
    _run_and_check(ctx, scores, N, K, "n_equals_k_small")
    print("PASS test_n_equals_k_small")


def test_n_equals_k_power_of_2(ctx: DeviceContext) raises:
    """N=K=512 — mid-size full sort; also a power of 2."""
    comptime N = 512
    comptime K = 512
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))
    _run_and_check(ctx, scores, N, K, "n_equals_k_512")
    print("PASS test_n_equals_k_power_of_2")


def test_duplicate_scores(ctx: DeviceContext) raises:
    """All scores identical — every index is a valid answer, no duplicates allowed.
    """
    comptime N = 256
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        scores.append(Float32(1.0))
    _run_and_check(ctx, scores, N, K, "duplicate_scores")
    print("PASS test_duplicate_scores")


def test_two_valued_scores(ctx: DeviceContext) raises:
    """Scores are 0.0 or 1.0 alternating — tests tie-breaking within a value class.
    """
    comptime N = 128
    comptime K = 32
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(1.0) if i % 2 == 0 else Float32(0.0))
    _run_and_check(ctx, scores, N, K, "two_valued_scores")
    print("PASS test_two_valued_scores")


def test_negative_scores(ctx: DeviceContext) raises:
    """All negative scores — top-K should select the least negative."""
    comptime N = 128
    comptime K = 16
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(
            Float32(-Float32(i + 1))
        )  # scores[0]=-1 is max, scores[127]=-128 is min
    _run_and_check(ctx, scores, N, K, "negative_scores")
    print("PASS test_negative_scores")


def test_multi_batch(ctx: DeviceContext) raises:
    """Multiple batch rows — each must be sorted independently and correctly."""
    comptime N = 256
    comptime K = 32
    comptime BATCH = 4

    var scores_dev = ctx.enqueue_create_buffer[DType.float32](BATCH * N)
    var idxs_dev = ctx.enqueue_create_buffer[DType.int32](BATCH * K)
    idxs_dev.enqueue_fill(Int32(-2))

    # Fill each row with a distinct pattern.
    with scores_dev.map_to_host() as buf:
        for b in range(BATCH):
            for i in range(N):
                # Row b: scores[i] = (b+1) * (N - i), so each row has the same
                # top-K structure (indices 0..K-1 are the top K).
                buf[b * N + i] = Float32((b + 1) * (N - i))

    persistent_topk_block(
        ctx,
        rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
            idxs_dev.unsafe_ptr()
        ),
        N,
        K,
        total_seq_len=BATCH,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[DType.int32](BATCH * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(BATCH):
        # For each row, build the score array for validation.
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(Float32((b + 1) * (N - i)))

        var seen = Set[Int]()
        for k in range(K):
            var idx = Int(idxs_host[b * K + k])
            assert_true(
                idx >= 0 and idx < N,
                String("multi_batch row ", b, " k=", k, " OOB idx=", idx),
            )
            assert_true(
                not (idx in seen),
                String(
                    "multi_batch row ", b, " duplicate idx=", idx, " at k=", k
                ),
            )
            seen.add(idx)

            # Descending order check.
            if k > 0:
                var prev = Int(idxs_host[b * K + k - 1])
                assert_true(
                    row_scores[idx] <= row_scores[prev],
                    String(
                        "multi_batch row ",
                        b,
                        " order violation at k=",
                        k,
                    ),
                )

        # Every index from [0, K) should appear (they are the K largest).
        for expected in range(K):
            assert_true(
                expected in seen,
                String(
                    "multi_batch row ", b, " missing expected idx ", expected
                ),
            )

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host
    print("PASS test_multi_batch")


def test_sorted_input_already_descending(ctx: DeviceContext) raises:
    """Input is already sorted descending — bitonic sort must not corrupt it."""
    comptime N = 2048
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(N - i))
    _run_and_check(ctx, scores, N, K, "sorted_desc")
    print("PASS test_sorted_input_already_descending")


def test_sorted_input_ascending(ctx: DeviceContext) raises:
    """Input is sorted ascending — the reverse of the desired output."""
    comptime N = 2048
    comptime K = 64
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))  # scores[2047] is the max
    _run_and_check(ctx, scores, N, K, "sorted_asc")
    print("PASS test_sorted_input_ascending")


def test_single_element(ctx: DeviceContext) raises:
    """N=K=1 — degenerate case."""
    comptime N = 1
    comptime K = 1
    var scores = List[Float32](capacity=N)
    scores.append(Float32(42.0))
    _run_and_check(ctx, scores, N, K, "single_element")
    print("PASS test_single_element")


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    with DeviceContext() as ctx:
        test_full_sort_n2048(ctx)
        test_random_full_sort_n2048(ctx)
        test_partial_topk_k16(ctx)
        test_partial_topk_k1024(ctx)
        test_small_n_padded(ctx)
        test_n_equals_k_small(ctx)
        test_n_equals_k_power_of_2(ctx)
        test_duplicate_scores(ctx)
        test_two_valued_scores(ctx)
        test_negative_scores(ctx)
        test_multi_batch(ctx)
        test_sorted_input_already_descending(ctx)
        test_sorted_input_ascending(ctx)
        test_single_element(ctx)
    print("ALL TESTS PASSED")
