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
    persistent_topk_block_split,
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
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
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
# Streaming path (N > 2048) — the GLM 5.x long-context / prefill regime.
# ===----------------------------------------------------------------------=== #


def test_streaming_n16384_random(ctx: DeviceContext) raises:
    """N=16384 (8 tiles), K=2048 random — the long-context decode shape."""
    comptime N = 16384
    comptime K = 2048
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state: UInt32 = 0x12345678
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * 100.0)
    _run_and_check(ctx, scores, N, K, "streaming_n16384_random")
    print("PASS test_streaming_n16384_random")


def test_streaming_n16006_nonmultiple(ctx: DeviceContext) raises:
    """N=16006 (not a multiple of the 2048 tile), K=2048 — partial last tile."""
    comptime N = 16006
    comptime K = 2048
    var a: UInt32 = 22695477
    var c: UInt32 = 1
    var state: UInt32 = 0xCAFEBABE
    var scores = List[Float32](capacity=N)
    for _ in range(N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * 7.0)
    _run_and_check(ctx, scores, N, K, "streaming_n16006_nonmultiple")
    print("PASS test_streaming_n16006_nonmultiple")


def test_streaming_n163840_ascending(ctx: DeviceContext) raises:
    """N=163840 (80 tiles, GLM max context), K=2048 — the max-shape stress."""
    comptime N = 163840
    comptime K = 2048
    var scores = List[Float32](capacity=N)
    for i in range(N):
        scores.append(Float32(i))
    _run_and_check(ctx, scores, N, K, "streaming_n163840_ascending")
    print("PASS test_streaming_n163840_ascending")


def test_streaming_masked_and_ties(ctx: DeviceContext) raises:
    """N=8192 with a masked (-1e30) prefix and heavy ties (causal-mask regime).
    """
    comptime N = 8192
    comptime K = 2048
    var scores = List[Float32](capacity=N)
    for i in range(N):
        if i < N // 2:
            scores.append(Float32(-1.0e30))
        else:
            scores.append(Float32(1.0) if (i % 3 == 0) else Float32(2.0))
    _run_and_check(ctx, scores, N, K, "streaming_masked_and_ties")
    print("PASS test_streaming_masked_and_ties")


# ===----------------------------------------------------------------------=== #
# Split path (`persistent_topk_block_split`) — the low-row / long-context
# decode regime that fans the streaming fold across `rows * S` blocks.
# ===----------------------------------------------------------------------=== #


def _check_topk_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Tie-robust top-K validation of one row.

    Verifies (1) indices in `[0, N)`, distinct, exactly `K`; (2) non-increasing
    score order; (3) every *non-selected* score is `<= min(selected scores)` —
    the exact top-K condition, which (unlike an exact index-set match) admits
    any valid tie-break at the K-th boundary.
    """
    var seen = Set[Int]()
    var min_sel = Float32.MAX
    for k in range(K):
        var idx = idxs[k]
        assert_true(
            idx >= 0 and idx < N,
            String("[", label, "] row ", row, " idx[", k, "]=", idx, " OOB"),
        )
        assert_true(
            not (idx in seen),
            String("[", label, "] row ", row, " duplicate idx=", idx),
        )
        seen.add(idx)
        if k > 0:
            var prev = idxs[k - 1]
            assert_true(
                scores_row[idx] <= scores_row[prev],
                String("[", label, "] row ", row, " order violation at k=", k),
            )
        min_sel = min(min_sel, scores_row[idx])

    assert_true(
        len(seen) == K,
        String("[", label, "] row ", row, " selected ", len(seen), " != K"),
    )
    for i in range(N):
        if not (i in seen):
            assert_true(
                scores_row[i] <= min_sel,
                String(
                    "[",
                    label,
                    "] row ",
                    row,
                    " non-selected idx ",
                    i,
                    " score ",
                    scores_row[i],
                    " exceeds selected min ",
                    min_sel,
                ),
            )


def _run_and_check_split(
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, row-major
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """Run `persistent_topk_block_split` over `B` rows and validate each row."""
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[DType.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[DType.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Scalar[DType.float32](scores_host[i])

    persistent_topk_block_split(
        ctx,
        rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
            idxs_dev.unsafe_ptr()
        ),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[DType.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])
        var row_idxs = List[Int](capacity=K)
        for k in range(K):
            row_idxs.append(Int(idxs_host[b * K + k]))
        _check_topk_row(row_scores, row_idxs, N, K, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


def _lcg_scores(B: Int, N: Int, sd: UInt32, scale: Float32) -> List[Float32]:
    var a: UInt32 = 1664525
    var c: UInt32 = 1013904223
    var state = sd
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        state = a * state + c
        scores.append(Float32(Int32(state)) / Float32(2**31) * scale)
    return scores^


def test_split_decode_long(ctx: DeviceContext) raises:
    """Rows=8, N=32769 (17 tiles, non-multiple), K=2048 — the decode-long shape.
    """
    comptime N = 32769
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x1234ABCD, 100.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_decode_long")
    print("PASS test_split_decode_long")


def test_split_decode_mtp(ctx: DeviceContext) raises:
    """Rows=16, N=8193 (5 tiles), K=2048 — the MTP decode shape."""
    comptime N = 8193
    comptime K = 2048
    comptime B = 16
    var scores = _lcg_scores(B, N, 0xCAFED00D, 50.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_decode_mtp")
    print("PASS test_split_decode_mtp")


def test_split_max_context(ctx: DeviceContext) raises:
    """Rows=2, N=163840 (80 tiles, GLM max context), K=2048."""
    comptime N = 163840
    comptime K = 2048
    comptime B = 2
    var scores = _lcg_scores(B, N, 0xBADF00D5, 100.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_max_context")
    print("PASS test_split_max_context")


def test_split_partial_k(ctx: DeviceContext) raises:
    """Rows=4, N=8193, K=512 — split with K < champion width."""
    comptime N = 8193
    comptime K = 512
    comptime B = 4
    var scores = _lcg_scores(B, N, 0x0FF1CE55, 30.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_partial_k")
    print("PASS test_split_partial_k")


def test_split_masked_and_ties(ctx: DeviceContext) raises:
    """Rows=4, N=8193 with a masked prefix + heavy ties — the causal regime.

    The tie-robust check accepts any valid boundary tie-break, so the split's
    per-slice tie order need not match a specific reference permutation.
    """
    comptime N = 8193
    comptime K = 2048
    comptime B = 4
    var scores = List[Float32](capacity=B * N)
    for _b in range(B):
        for i in range(N):
            if i < N // 2:
                scores.append(Float32(-1.0e30))
            else:
                scores.append(Float32(1.0) if (i % 3 == 0) else Float32(2.0))
    _run_and_check_split(ctx, scores, N, K, B, "split_masked_and_ties")
    print("PASS test_split_masked_and_ties")


# The tree phase-2 reduces S partials with fan-in 5; these rows=8 shapes pin S
# (= min(num_tiles, ceildiv(2*sm_count, rows))) at values that exercise the
# reduction at several fan-out patterns: 6 (fan-in+1 edge), 8 (power of two),
# 13 (prime, uneven final group), 17 (non-power-of-two). N = num_tiles*2048 - 1
# keeps the last tile partial too.


def test_split_tree_s6(ctx: DeviceContext) raises:
    """Rows=8, N=12287 (6 tiles) -> S=6, one reduce round (6 -> 2) + final."""
    comptime N = 12287
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00516006, 80.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s6")
    print("PASS test_split_tree_s6")


def test_split_tree_s8(ctx: DeviceContext) raises:
    """Rows=8, N=16383 (8 tiles) -> S=8, one reduce round (8 -> 2) + final."""
    comptime N = 16383
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00518008, 90.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s8")
    print("PASS test_split_tree_s8")


def test_split_tree_s13(ctx: DeviceContext) raises:
    """Rows=8, N=26623 (13 tiles) -> S=13, reduce (13 -> 3) + final of 3."""
    comptime N = 26623
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x00513013, 120.0)
    _run_and_check_split(ctx, scores, N, K, B, "split_tree_s13")
    print("PASS test_split_tree_s13")


# ===----------------------------------------------------------------------=== #
# Odd-N / odd-alignment coverage for the non-split kernels.
#
# The vectorized score load emits a 128-bit load only when the row base is
# 16B-aligned; odd `N` makes `token*N` non-aligned for most rows, so these
# multi-row odd-N shapes exercise the scalar fallback (a misaligned 128-bit
# load would fault). `test_split_decode_long` covers the same for the split
# kernels; these cover `persistent_topk_block`'s 2048 and streaming kernels,
# which every other test only drives at `total_seq_len=1` (row 0, always
# aligned).
# ===----------------------------------------------------------------------=== #


def _run_and_check_block_multirow(
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, row-major
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """Run `persistent_topk_block` over `B` rows and validate each row."""
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[DType.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[DType.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
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
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[DType.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])
        var row_idxs = List[Int](capacity=K)
        for k in range(K):
            row_idxs.append(Int(idxs_host[b * K + k]))
        _check_topk_row(row_scores, row_idxs, N, K, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = idxs_host


def test_block_odd_n_multirow(ctx: DeviceContext) raises:
    """N=1025 (odd, <=2048), rows=8, K=512 — 2048 kernel, misaligned bases.

    `token*1025` is non-16B-aligned for most rows, forcing the scalar-fallback
    score load in `_persistent_topk_2048_kernel`.
    """
    comptime N = 1025
    comptime K = 512
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x0DD00001, 40.0)
    _run_and_check_block_multirow(ctx, scores, N, K, B, "block_odd_n_multirow")
    print("PASS test_block_odd_n_multirow")


def test_block_streaming_odd_n_multirow(ctx: DeviceContext) raises:
    """N=8193 (odd, >2048), rows=8, K=2048 — streaming kernel, misaligned bases.

    Drives `persistent_topk_block` (not the split launcher), so the streaming
    fold runs one block per row with odd `token*N` bases -> scalar-fallback
    score load. Complements `test_split_decode_long` (split path's odd-N loads).
    """
    comptime N = 8193
    comptime K = 2048
    comptime B = 8
    var scores = _lcg_scores(B, N, 0x0DD08193, 60.0)
    _run_and_check_block_multirow(ctx, scores, N, K, B, "block_streaming_odd_n")
    print("PASS test_block_streaming_odd_n_multirow")


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
        test_streaming_n16384_random(ctx)
        test_streaming_n16006_nonmultiple(ctx)
        test_streaming_n163840_ascending(ctx)
        test_streaming_masked_and_ties(ctx)
        test_split_decode_long(ctx)
        test_split_decode_mtp(ctx)
        test_split_max_context(ctx)
        test_split_partial_k(ctx)
        test_split_masked_and_ties(ctx)
        test_split_tree_s6(ctx)
        test_split_tree_s8(ctx)
        test_split_tree_s13(ctx)
        test_block_odd_n_multirow(ctx)
        test_block_streaming_odd_n_multirow(ctx)
    print("ALL TESTS PASSED")
