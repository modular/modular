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

from std.builtin.device_passable import DeviceTypeEncoder
from std.collections import Set
from max.gpu.host import (
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
    get_gpu_target,
)
from max.gpu.host.compile import _compile_code
from std.math import max
from std.memory import bitcast
from std.random import seed
from std.testing import assert_equal, assert_true

from nn.attention.gpu.nvidia.topk_cluster import (
    _choose_topk_cluster_size,
    _ctopk_max_rows,
    _topk_cluster_kernel,
    topk_cluster_split,
)
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
    RowBoundsPtr,
    TopkSink,
    _phi,
    _phi_of,
    persistent_topk_block,
    persistent_topk_block_split,
)


# ===----------------------------------------------------------------------=== #
# CPU reference
# ===----------------------------------------------------------------------=== #


def _as_read[in_dtype: DType](scores: List[Float32]) -> List[Float32]:
    """The scores as the kernel will actually see them.

    The kernel reads `in_dtype` and widens exactly, so what it selects over is
    the f32 image of the ROUNDED value, not the value the generator produced.
    Every checker below has to be handed this list rather than the original:
    rounding to bf16 creates ties that did not exist in the input, and an
    oracle run on the unrounded scores reports each of them as a selection bug.
    """
    comptime if in_dtype == .float32:
        return scores.copy()
    var out = List[Float32](capacity=len(scores))
    for v in scores:
        out.append(Float32(v.cast[in_dtype]()))
    return out^


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
    var scores_dev = ctx.enqueue_create_buffer[.float32](N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots

    with scores_dev.map_to_host() as buf:
        for i in range(N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block(
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=1,
    )
    ctx.synchronize()

    # Copy back and validate.
    var idxs_host = ctx.enqueue_create_host_buffer[.int32](K)
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

    var scores_dev = ctx.enqueue_create_buffer[.float32](BATCH * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](BATCH * K)
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
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=BATCH,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](BATCH * K)
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

    var scores_dev = ctx.enqueue_create_buffer[.float32](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = Float32(scores_host[i])

    persistent_topk_block_split[ordered=True, deterministic=True](
        ctx,
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_dev.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
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


def _run_and_check_block_multirow[
    in_dtype: DType = .float32
](
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

    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores_host[i].cast[in_dtype]()

    persistent_topk_block(
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var seen_scores = _as_read[in_dtype](scores_host)
    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(seen_scores[b * N + i])
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
# `ordered=False`: the same set, without the ranking pass
# ===----------------------------------------------------------------------=== #


def _expected_topk_set(scores_row: List[Float32], K: Int) -> Set[Int]:
    """Exact top-`K` under the kernel's total order: descending score, then
    ascending column.

    Found by bisecting the order-preserving key rather than sorting, so it is
    O(N log R) and stays usable at the largest shapes -- a per-element rank would
    be O(N^2) and a sort drags in an ordering dependency this oracle exists to
    avoid. Ties are resolved exactly, which is the point: it pins which members of
    a plateau straddling `K` must be chosen, where `_cpu_topk_set` admits any of
    them.
    """
    var N = len(scores_row)
    var keys = List[UInt32](capacity=N)
    for i in range(N):
        keys.append(_phi(scores_row[i]))

    # Largest threshold `t` whose at-or-above count still reaches `K`.
    var lo = UInt32(0)
    var hi = UInt32(0xFFFFFFFF)
    while lo < hi:
        var mid = lo + (hi - lo) // 2 + 1
        var cnt = 0
        for i in range(N):
            if keys[i] >= mid:
                cnt += 1
        if cnt >= K:
            lo = mid
        else:
            hi = mid - 1
    var t = lo

    var res = Set[Int]()
    var n_gt = 0
    for i in range(N):
        if keys[i] > t:
            res.add(i)
            n_gt += 1
    # The rest come from the threshold value itself, lowest column first.
    var need = K - n_gt
    for i in range(N):
        if need == 0:
            break
        if keys[i] == t:
            res.add(i)
            need -= 1
    return res^


def _check_set_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Assert one row's indices are exactly the expected top-`K`, as a set.

    Order is deliberately not asserted here: `ordered=False` promises membership
    and determinism, and only shapes with a cheaper path take one -- the rest
    legitimately return the ordered result, which satisfies the weaker promise
    too. Ordering is checked separately, for the ordered contract.
    """
    var seen = Set[Int]()
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
    assert_equal(
        len(seen), K, String("[", label, "] row ", row, " wrong count")
    )

    var want = _expected_topk_set(scores_row, K)
    for c in want:
        assert_true(
            c in seen,
            String(
                "[",
                label,
                "] row ",
                row,
                " missing column ",
                c,
                " -- selection or tie-break differs from the oracle",
            ),
        )


def _check_score_valid_row(
    scores_row: List[Float32],
    idxs: List[Int],
    N: Int,
    K: Int,
    label: String,
    row: Int,
) raises:
    """Assert one row is *a* valid top-`K` by score, without pinning which one.

    Three things, which together are the whole contract and no more:

    1. `K` distinct columns, all in `[0, N)`;
    2. every column scoring **strictly above** the `K`-th largest is present --
       those are forced, no contract makes them optional;
    3. the rest tie the `K`-th largest exactly.

    Checking only 1 and 3 admits a row that drops higher-scoring columns outright,
    because the threshold is itself one of the selected scores -- so a row made
    entirely of columns tied at it satisfies "at least the K-th largest" while
    leaving out everything above. Which *tied* columns are chosen stays unasserted,
    because that is the part the mode really does leave free.
    """
    var seen = Set[Int]()
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
    assert_equal(
        len(seen), K, String("[", label, "] row ", row, " wrong count")
    )

    # The K-th largest score, taken from the oracle's own set.
    var want = _expected_topk_set(scores_row, K)
    var thresh = Float32.MAX
    for c in want:
        if scores_row[c] < thresh:
            thresh = scores_row[c]

    var n_at = 0
    for c in seen:
        assert_true(
            scores_row[c] >= thresh,
            String(
                "[",
                label,
                "] row ",
                row,
                " column ",
                c,
                " scores below the K-th largest",
            ),
        )
        if scores_row[c] == thresh:
            n_at += 1

    var n_above = 0
    for c in range(N):
        if scores_row[c] > thresh:
            n_above += 1
            assert_true(
                c in seen,
                String(
                    "[",
                    label,
                    "] row ",
                    row,
                    " column ",
                    c,
                    " scores above the K-th largest and was not selected",
                ),
            )
    assert_equal(
        n_at,
        K - n_above,
        String(
            "[",
            label,
            "] row ",
            row,
            " wrong count tied at the K-th largest",
        ),
    )


def _run_split[
    ordered: Bool = True,
    deterministic: Bool = True,
    in_dtype: DType = .float32,
](
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    B: Int,
) raises -> List[Int]:
    """One launch of the given contract, returned flat.

    Parameterized rather than written once per contract: the contracts differ only
    in what they promise about order, so a launch that differed anywhere else would
    be comparing two things at once.
    """
    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))
    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores_host[i].cast[in_dtype]()

    persistent_topk_block_split[ordered=ordered, deterministic=deterministic](
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
    )
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=host, src_buf=idxs_dev)
    ctx.synchronize()
    var out = List[Int](capacity=B * K)
    for i in range(B * K):
        out.append(Int(host[i]))
    _ = scores_dev
    _ = idxs_dev
    _ = host
    return out^


def _check_both(
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """The full battery on one input: both contracts, cross-checked.

    1. `ordered=False` selects exactly the oracle's set, every row.
    2. `ordered=True` selects the same set *and* is correctly ordered -- so the
       ordered path is gated at least as hard here as the new one.
    3. The two modes agree as sets, which is the claim that justifies the mode.
    4. `ordered=False` is byte-stable across launches, which an implementation
       that places tied keys by an atomic race cannot offer.
    5. `deterministic=False` -- which adopts that race deliberately -- returns *a*
       valid top-`K` by score on *both* of two launches. It is held to the
       strongest predicate that does not over-promise (see
       `_check_score_valid_row`): the columns above the threshold are forced, only
       the tied ones are free. The two launches are allowed to disagree, so that
       is reported rather than asserted -- forbidding it would forbid the mode.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"

    var unord = _run_split[False, True](ctx, scores_host, N, K, B)
    var ord = _run_split(ctx, scores_host, N, K, B)
    var unord_nd = _run_split[False, False](ctx, scores_host, N, K, B)
    var unord_nd2 = _run_split[False, False](ctx, scores_host, N, K, B)

    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(scores_host[b * N + i])

        var u_row = List[Int](capacity=K)
        var o_row = List[Int](capacity=K)
        var r_row = List[Int](capacity=K)
        for k in range(K):
            u_row.append(unord[b * K + k])
            o_row.append(ord[b * K + k])
            r_row.append(unord_nd[b * K + k])

        _check_set_row(row_scores, u_row, N, K, String(label, ":unord"), b)
        _check_set_row(row_scores, o_row, N, K, String(label, ":ord"), b)
        var r2_row = List[Int](capacity=K)
        for k in range(K):
            r2_row.append(unord_nd2[b * K + k])
        _check_score_valid_row(
            row_scores, r_row, N, K, String(label, ":unord_nd"), b
        )
        _check_score_valid_row(
            row_scores, r2_row, N, K, String(label, ":unord_nd2"), b
        )

        # The ordered contract: non-increasing score, ties by ascending column.
        for k in range(1, K):
            var prev = o_row[k - 1]
            var cur = o_row[k]
            var sp = row_scores[prev]
            var sc = row_scores[cur]
            assert_true(
                sp > sc or (sp == sc and prev < cur),
                String(
                    "[",
                    label,
                    ":ord] row ",
                    b,
                    " order violation at k=",
                    k,
                    ": col ",
                    prev,
                    " then ",
                    cur,
                ),
            )

        # Same membership from both modes.
        var uset = Set[Int]()
        for k in range(K):
            uset.add(u_row[k])
        for k in range(K):
            assert_true(
                o_row[k] in uset,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " ordered picked column ",
                    o_row[k],
                    " that unordered did not",
                ),
            )

    var again = _run_split[False, True](ctx, scores_host, N, K, B)
    for i in range(B * K):
        assert_equal(
            unord[i],
            again[i],
            String("[", label, "] slot ", i, " differs between launches"),
        )


def test_unordered_resident_band(ctx: DeviceContext) raises:
    """The resident band, with lengths either side of the `bin_digit` crossover.
    """
    comptime K = 2048
    comptime B = 48
    for N in [2049, 2432, 2560, 4097, 8192]:
        var scores = _lcg_scores(B, N, 0x51EED000 + UInt32(N), 100.0)
        _check_both(ctx, scores, N, K, B, String("band_n", N))
    print("PASS test_unordered_resident_band")


def test_phi16_identity() raises:
    """`_phi_of` at a 16-bit payload is the top half of the 32-bit one. All 65536.

    This is the whole basis for narrowing the payload, so it is proven rather
    than argued, and it is proven exhaustively because the domain is small
    enough to enumerate. Three claims, and each rules out a different way the
    narrowing could be wrong:

    - **The identity.** `phi16(b) == phi32(widen(b)) >> 16` for every pattern.
      Widening is a 16-bit shift, and `_phi` xors with `0x80000000` above a
      positive and `0xFFFFFFFF` above a negative, whose top halves are `0x8000`
      and `0xFFFF`.
    - **Bijectivity.** Two patterns never share a `phi16`. This is what makes
      the identity enough: an equality test on `phi16` decides the same thing an
      equality test on `phi32` would, so the plateau tests and the column half's
      `phi == t_phi` do not move.
    - **Monotonicity.** Walking `phi16` from 0 to 65535 walks the bf16 values
      non-decreasingly. Non-decreasing rather than increasing because `-0.0` and
      `+0.0` are two patterns of one value; bijectivity supplies the strictness
      of the map itself. NaN patterns are skipped -- `_phi` orders finite values
      and the f32 path makes no claim about NaN either.

    And the padding sentinel: `phi16(0xFFFF) == 0` is the same minimum the wide
    form gets from `0xFFFFFFFF`, which is what lets the select count padding as
    an ordinary candidate at either width.
    """
    comptime NPAT = 1 << 16
    var value = List[Float32](capacity=NPAT)
    var seen = List[Int](capacity=NPAT)
    var inverse = List[Int](capacity=NPAT)
    for _ in range(NPAT):
        value.append(0.0)
        seen.append(0)
        inverse.append(-1)

    var nan_pats = 0
    for b in range(NPAT):
        var pat = SIMD[.uint16, 1](UInt16(b))
        var score = bitcast[.bfloat16, 1](pat)
        var p16 = Int(_phi_of[.bfloat16, .uint16, 1](pat)[0])
        var p32 = Int(_phi_of[.bfloat16, .uint32, 1](pat)[0])

        assert_equal(p16, p32 >> 16, String("phi16 identity at pattern ", b))
        assert_equal(
            p32,
            Int(_phi(Float32(score[0]))),
            String("phi32 of a widened bf16 is _phi of its f32 at ", b),
        )
        assert_equal(seen[p16], 0, String("phi16 collision at pattern ", b))
        seen[p16] = 1
        inverse[p16] = b
        # An exponent field of all ones with a non-zero mantissa: bf16 shares
        # f32's 8-bit exponent, so the test is the f32 one on the top half.
        var is_nan = (b & 0x7F80) == 0x7F80 and (b & 0x007F) != 0
        if is_nan:
            nan_pats += 1
        value[b] = Float32(score[0])

    assert_equal(
        _phi_of[.bfloat16, .uint16, 1](SIMD[.uint16, 1](0xFFFF))[0],
        UInt16(0),
        "the padding sentinel's phi16 image must be the minimum",
    )

    # Bijectivity was checked as "no collision"; with 65536 patterns into 65536
    # slots that already forces surjectivity, but assert it rather than infer
    # it, because a mis-typed accumulator would break exactly this.
    var covered = 0
    for i in range(NPAT):
        covered += seen[i]
    assert_equal(covered, NPAT, "phi16 must be a bijection on 16 bits")

    var prev = Float32(0.0)
    var have_prev = False
    for p in range(NPAT):
        var b = inverse[p]
        if (b & 0x7F80) == 0x7F80 and (b & 0x007F) != 0:
            continue
        if have_prev:
            assert_true(
                prev <= value[b],
                String(
                    "phi16 order must follow score order: phi ",
                    p,
                    " gave ",
                    value[b],
                    " after ",
                    prev,
                ),
            )
        prev = value[b]
        have_prev = True

    # 2 * 127 patterns (both signs, mantissa 1..127) are NaN. Stated so the
    # skip above cannot quietly grow to swallow the finite values.
    assert_equal(nan_pats, 254, "bf16 has exactly 254 NaN patterns")
    print("PASS test_phi16_identity")


def test_unord_nd_predicate_bites() raises:
    """The unord_nd predicate must reject the row its weaker form used to accept.

    A checker that cannot fail is not a gate, and this one silently could not: see
    `_check_score_valid_row` for the row that satisfies "at least the K-th largest"
    while omitting every column above it. Host-only, so it costs nothing to keep.
    """
    comptime N = 8
    comptime K = 3
    var sc = List[Float32](capacity=N)
    sc.append(Float32(10.0))
    for _ in range(N - 1):
        sc.append(Float32(5.0))

    # Columns tied at the threshold only, with the forced one left out.
    var bad = List[Int](capacity=K)
    bad.append(1)
    bad.append(2)
    bad.append(3)
    var rejected = False
    try:
        _check_score_valid_row(sc, bad, N, K, "selftest", 0)
    except:
        rejected = True
    assert_true(
        rejected,
        (
            "unord_nd predicate accepted a row that drops a column above the"
            " K-th largest"
        ),
    )

    # The same row with the forced column present must pass.
    var good = List[Int](capacity=K)
    good.append(0)
    good.append(1)
    good.append(2)
    _check_score_valid_row(sc, good, N, K, "selftest", 0)
    print("PASS test_unord_nd_predicate_bites")


def test_unordered_wide_payload_band(ctx: DeviceContext) raises:
    """The wide register payload: both its edges, both row regimes, maximal ties.

    A row narrow enough is held in registers when the rows under-fill the GPU and
    streamed otherwise, so every case here runs two different kernels across the
    two row counts. The lengths straddle each payload width's ceiling from both
    sides, and one of them is not a multiple of the vector width, which forces the
    unaligned tail of the load rather than the vectorised path.
    """
    comptime K = 2048
    for N in [8192, 8193, 16384, 16385, 16388]:
        for B in [8, 160]:
            var scores = _lcg_scores(B, N, 0x5EED0000 + UInt32(N), 100.0)
            _check_both(ctx, scores, N, K, B, String("wide_n", N, "_b", B))

    # Every column ties, so the plateau cut alone decides the answer and a
    # miscounted `n_above` shows up as a short row rather than a wrong score.
    comptime NEQ = 12288
    comptime BEQ = 8
    var eq = List[Float32](capacity=BEQ * NEQ)
    for _ in range(BEQ * NEQ):
        eq.append(Float32(1.5))
    _check_both(ctx, eq, NEQ, K, BEQ, "wide_all_equal")

    # `K` at both extremes, on a row the wide payload holds.
    for KK in [1, 2048]:
        var sc = _lcg_scores(8, 14336, 0xC0DE0000 + UInt32(KK), 100.0)
        _check_both(ctx, sc, 14336, KK, 8, String("wide_k", KK))
    print("PASS test_unordered_wide_payload_band")


def test_unordered_prime_lengths(ctx: DeviceContext) raises:
    """Prime `N`, so no tile, vector or bin width divides the row.

    8191 is also not a multiple of four, which is what forces the scalar tail of
    the vectorised load rather than the `float4` path.
    """
    comptime K = 2048
    comptime B = 24
    # The last two sit inside the wide payload's band, which the rest of this list
    # brackets without entering, and one of them just below its ceiling.
    for N in [2053, 3079, 4099, 6151, 8191, 12289, 16381]:
        var scores = _lcg_scores(B, N, 0x9E3779B1 + UInt32(N), 100.0)
        _check_both(ctx, scores, N, K, B, String("prime_n", N))
    print("PASS test_unordered_prime_lengths")


def test_unordered_k_edges(ctx: DeviceContext) raises:
    """`K` at its extremes, including 1, 2 and `K = N - 1`."""
    comptime B = 12
    comptime N = 4099
    for K in [1, 2, 17, 999, 1024, 2047, 2048]:
        var scores = _lcg_scores(B, N, 0xC0FFEE00 + UInt32(K), 50.0)
        _check_both(ctx, scores, N, K, B, String("kedge_k", K))
    print("PASS test_unordered_k_edges")


def test_unordered_row_counts(ctx: DeviceContext) raises:
    """Row counts around the SM count, which is what picks resident vs streaming.
    """
    comptime N = 4097
    comptime K = 2048
    for B in [1, 2, 47, 147, 200]:
        var scores = _lcg_scores(B, N, 0xB105F00D + UInt32(B), 100.0)
        _check_both(ctx, scores, N, K, B, String("rows_b", B))
    print("PASS test_unordered_row_counts")


def test_unordered_all_equal(ctx: DeviceContext) raises:
    """Every score identical, so the whole row is one tie plateau.

    The answer is then forced to columns `0 .. K-1`: ties go to the lower column,
    and "the K largest" alone would not pin it.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        scores.append(Float32(0.25))
    _check_both(ctx, scores, N, K, B, "all_equal")

    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        var seen = Set[Int]()
        for k in range(K):
            seen.add(got[b * K + k])
        for c in range(K):
            assert_true(
                c in seen,
                String(
                    "all_equal row ",
                    b,
                    ": column ",
                    c,
                    " must be selected when every score is equal",
                ),
            )
    print("PASS test_unordered_all_equal")


def test_unordered_plateau_straddles_k(ctx: DeviceContext) raises:
    """A plateau of equal scores spanning the `K` boundary.

    `0 .. K-201` score high, `K-200 .. K+399` share one middle score, the rest
    are low. The cut therefore lands *inside* the plateau, so only the tie-break
    decides which 200 of its 600 members are in.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    comptime hi_end = K - 200
    comptime plateau_end = K + 400
    var scores = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            if c < hi_end:
                scores.append(Float32(10.0))
            elif c < plateau_end:
                scores.append(Float32(5.0))
            else:
                scores.append(Float32(-1.0))
    _check_both(ctx, scores, N, K, B, "plateau_straddle")

    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        var seen = Set[Int]()
        for k in range(K):
            seen.add(got[b * K + k])
        for c in range(K):
            assert_true(
                c in seen,
                String("plateau row ", b, ": column ", c, " must win"),
            )
        for c in range(K, plateau_end):
            assert_true(
                not (c in seen),
                String(
                    "plateau row ",
                    b,
                    ": column ",
                    c,
                    (
                        " ties the boundary but has a higher index, so it must"
                        " lose"
                    ),
                ),
            )
    print("PASS test_unordered_plateau_straddles_k")


def test_unordered_plateau_edges(ctx: DeviceContext) raises:
    """Plateaus that end exactly at `K`, and that start exactly at `K`.

    Both are off-by-one bait: the first needs the whole plateau in, the second
    needs all of it out.
    """
    comptime N = 4097
    comptime K = 2048
    comptime B = 4
    for start in [K - 300, K]:
        var scores = List[Float32](capacity=B * N)
        for _ in range(B):
            for c in range(N):
                if c < start:
                    scores.append(Float32(9.0))
                elif c < (K if start < K else K + 300):
                    scores.append(Float32(4.0))
                else:
                    scores.append(Float32(-2.0))
        _check_both(ctx, scores, N, K, B, String("plateau_edge", start))
    print("PASS test_unordered_plateau_edges")


def test_unordered_k1_all_equal(ctx: DeviceContext) raises:
    """`K = 1` on an all-equal row: the answer is exactly column 0."""
    comptime N = 3079
    comptime K = 1
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for _ in range(B * N):
        scores.append(Float32(-7.5))
    _check_both(ctx, scores, N, K, B, "k1_all_equal")
    var got = _run_split[False, True](ctx, scores, N, K, B)
    for b in range(B):
        assert_equal(
            got[b],
            0,
            String("k1_all_equal row ", b, ": must pick column 0"),
        )
    print("PASS test_unordered_k1_all_equal")


def test_unordered_two_valued(ctx: DeviceContext) raises:
    """Two distinct scores only, so the digit has almost nothing to split on."""
    comptime N = 2560
    comptime K = 2048
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            scores.append(Float32(1.0) if (c + b) % 3 == 0 else Float32(0.0))
    _check_both(ctx, scores, N, K, B, "two_valued")
    print("PASS test_unordered_two_valued")


def test_unordered_masked_tail(ctx: DeviceContext) raises:
    """A masked tail at `-3.0e38` over 17 score levels, as the indexer produces.

    Rows differ in valid length, and the lowest ones have fewer valid columns
    than `K`, so the selection must reach into the masked region.
    """
    comptime N = 3071
    comptime K = 2048
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        var valid = N - 150 * b
        for c in range(N):
            if c < valid:
                scores.append(Float32((c * 37 + b * 11) % 17) / Float32(16.0))
            else:
                scores.append(Float32(-3.0e38))
    _check_both(ctx, scores, N, K, B, "masked_tail")
    print("PASS test_unordered_masked_tail")


def test_unordered_negative_and_extremes(ctx: DeviceContext) raises:
    """All-negative rows, and rows mixing huge magnitudes with denormals.

    Exercises the sign-flip half of the key map and a very wide exponent range,
    which is where the rank's digit is weakest.
    """
    comptime N = 4099
    comptime K = 2048
    comptime B = 6
    var neg = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            neg.append(-Float32((c * 13 + b) % 1000) - Float32(1.0))
    _check_both(ctx, neg, N, K, B, "all_negative")

    var wide = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            var m = (c + b) % 6
            if m == 0:
                wide.append(Float32(3.0e38))
            elif m == 1:
                wide.append(Float32(-3.0e38))
            elif m == 2:
                wide.append(Float32(1.0e-38))
            elif m == 3:
                wide.append(Float32(-1.0e-38))
            elif m == 4:
                wide.append(Float32(0.0))
            else:
                wide.append(Float32(c))
    _check_both(ctx, wide, N, K, B, "wide_exponents")
    print("PASS test_unordered_negative_and_extremes")


def test_unordered_sorted_inputs(ctx: DeviceContext) raises:
    """Already-sorted rows, both directions.

    Descending input makes the answer the first `K` columns; ascending makes it
    the last `K`, in reverse. Both are degenerate for a histogram select.
    """
    comptime N = 6151
    comptime K = 2048
    comptime B = 6
    var desc = List[Float32](capacity=B * N)
    var asc = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            desc.append(Float32(N - c))
            asc.append(Float32(c))
    _check_both(ctx, desc, N, K, B, "sorted_desc")
    _check_both(ctx, asc, N, K, B, "sorted_asc")
    print("PASS test_unordered_sorted_inputs")


def test_unordered_beyond_resident(ctx: DeviceContext) raises:
    """`N` past the narrow resident cap, over every select the contract can reach.

    Both row counts, because they instantiate different kernels: below the SM
    count a row this wide is held in registers by the wide-payload resident
    select, or streamed with prefetch once it is wider than that; above the SM
    count the occupancy variant streams it. The cheap contract has to land on
    exactly `K` in each. Long-context decode and prefill sit on opposite sides of
    that line.

    Two of the lengths straddle the wide payload's ceiling, so together they pin
    that dispatch edge: one column more moves the row off registers and onto the
    streaming select, and both sides must agree with each other and with
    themselves across launches.
    """
    comptime K = 2048
    for N in [8193, 16006, 16384, 16388, 32769]:
        for B in [8, 160]:
            var scores = _lcg_scores(B, N, 0xFEED0000 + UInt32(N), 100.0)
            _check_both(ctx, scores, N, K, B, String("beyond_n", N, "_b", B))
    print("PASS test_unordered_beyond_resident")


def test_unordered_small_n(ctx: DeviceContext) raises:
    """`N` at or below the champion width, which routes to the block kernel."""
    comptime B = 8
    for cfg in [(2048, 2048), (2048, 16), (512, 512), (64, 16), (1, 1)]:
        var N = cfg[0]
        var K = cfg[1]
        var scores = _lcg_scores(B, N, 0x0DDBA11 + UInt32(N + K), 20.0)
        _check_both(ctx, scores, N, K, B, String("small_", N, "_", K))
    print("PASS test_unordered_small_n")


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# Bounded rows (`row_bounds`) — per-row live-column clamps, as driven by the
# MLA indexer under capture-frozen metadata where the row stride `N` is a
# worst-case bound far above any row's real length.
# ===----------------------------------------------------------------------=== #


struct _MirrorSink(TopkSink):
    """Writes each slot to `out` and a row-dependent image of it to `mirror`."""

    var out: MutPointer[Int32, MutUntrackedOrigin]
    var mirror: MutPointer[Int32, MutUntrackedOrigin]
    var row: Int

    @inline(.always)
    def __init__(
        out self,
        out_ptr: MutPointer[Int32, MutAnyOrigin],
        mirror: MutPointer[Int32, MutUntrackedOrigin],
        row: Int,
    ):
        self.out = out_ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        self.mirror = mirror
        self.row = row

    @inline(.always)
    def store(self, idx: Int, col: Int32):
        self.out[idx] = col
        self.mirror[idx] = _mirror_of(col, self.row)


@inline(.always)
def _mirror_of(col: Int32, row: Int) -> Int32:
    """What `_MirrorSink` writes for `col` on `row`; `-1` stays `-1`."""
    return col * 3 + Int32(row) if col >= 0 else Int32(-1)


struct _MirrorBound(RowBound):
    """`RowBoundsPtr`'s bounds with a sink that also fills `mirror`.

    Checks that every output slot of every arm, `-1` padding included, goes
    through the sink, and that the sink carries the per-row state `row`
    resolved once at entry.
    """

    comptime device_type: AnyType = Self
    comptime Sink = _MirrorSink

    var bounds: ImmPointer[Int32, ImmUntrackedOrigin]
    var mirror: MutPointer[Int32, MutUntrackedOrigin]

    def __init__(
        out self,
        bounds: ImmPointer[Int32, ImmutAnyOrigin],
        mirror: MutPointer[Int32, MutAnyOrigin],
    ):
        self.bounds = bounds.unsafe_origin_cast[ImmUntrackedOrigin]()
        self.mirror = mirror.unsafe_origin_cast[MutUntrackedOrigin]()

    @inline(.always)
    def row(
        self, row: Int, out_idxs: MutPointer[Int32, MutAnyOrigin]
    ) -> Tuple[Int, _MirrorSink]:
        return (Int(self.bounds[row]), _MirrorSink(out_idxs, self.mirror, row))

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "_MirrorBound"


def _run_and_check_bounded[
    in_dtype: DType = .float32,
    ordered: Bool = True,
    deterministic: Bool = True,
](
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, poisoned at/past each bound
    N: Int,
    K: Int,
    B: Int,
    bounds: List[Int],
    label: String,
) raises:
    """Run the bounded split launcher and validate each row against its bound.

    The caller poisons columns at or past each row's bound with values larger
    than any live score, so a kernel that reads past a bound cannot pass: a
    poisoned column would displace a live selection and surface as an
    out-of-bound index.

    Verifies per row, with `E = min(K, bound)` expected live selections:
    1. Slots `[0, E)` hold the exact top-`E` of the row's first `bound`
       columns (distinct, in-bounds, non-increasing score order, tie-robust).
    2. Slots `[E, K)` are `-1`.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds champion width"
    assert len(scores_host) == B * N, "scores_host length mismatch"
    assert len(bounds) == B, "bounds length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores_host[i].cast[in_dtype]()
    with bounds_dev.map_to_host() as buf:
        for b in range(B):
            buf[b] = Int32(bounds[b])

    persistent_topk_block_split[ordered=ordered, deterministic=deterministic](
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
        row_bounds=Optional(
            RowBoundsPtr(
                rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                    bounds_dev.unsafe_ptr()
                )
            )
        ),
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var seen_scores = _as_read[in_dtype](scores_host)
    for b in range(B):
        var bound = bounds[b]
        var live = min(K, bound)

        for k in range(live, K):
            var idx = Int(idxs_host[b * K + k])
            assert_true(
                idx == -1,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " tail slot ",
                    k,
                    " = ",
                    idx,
                    " (expected -1 past bound ",
                    bound,
                    ")",
                ),
            )

        if live == 0:
            continue
        var row_scores = List[Float32](capacity=bound)
        for i in range(bound):
            row_scores.append(seen_scores[b * N + i])
        var row_idxs = List[Int](capacity=live)
        for k in range(live):
            row_idxs.append(Int(idxs_host[b * K + k]))
        comptime if ordered:
            _check_topk_row(row_scores, row_idxs, bound, live, label, b)
        else:
            _check_set_row(row_scores, row_idxs, bound, live, label, b)

    # Same launch through a custom sink: every slot of `mirror` must be the
    # image of the slot the kernel wrote to the main output.
    var mirror_dev = ctx.enqueue_create_buffer[.int32](B * K)
    mirror_dev.enqueue_fill(Int32(-2))
    idxs_dev.enqueue_fill(Int32(-2))
    persistent_topk_block_split[
        ordered=ordered, deterministic=deterministic, RB=_MirrorBound
    ](
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
        row_bounds=Optional(
            _MirrorBound(
                rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                    bounds_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](
                    mirror_dev.unsafe_ptr()
                ),
            )
        ),
    )
    var mirror_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.enqueue_copy(dst_buf=mirror_host, src_buf=mirror_dev)
    ctx.synchronize()
    for b in range(B):
        for k in range(K):
            var col = idxs_host[b * K + k]
            assert_true(
                col != -2 and mirror_host[b * K + k] == _mirror_of(col, b),
                String(
                    "[",
                    label,
                    "] sink row ",
                    b,
                    " slot ",
                    k,
                    ": out ",
                    col,
                    ", mirror ",
                    mirror_host[b * K + k],
                ),
            )

    _ = scores_dev
    _ = idxs_dev
    _ = bounds_dev
    _ = mirror_dev
    _ = idxs_host
    _ = mirror_host


def _poisoned_bounded_scores(
    B: Int, N: Int, bounds: List[Int], sd: UInt32
) -> List[Float32]:
    """LCG scores with columns at/past each row's bound poisoned to +1e30."""
    var scores = _lcg_scores(B, N, sd, 100.0)
    for b in range(B):
        for i in range(bounds[b], N):
            scores[b * N + i] = Float32(1.0e30)
    return scores^


def test_bounded_2048_kernel(ctx: DeviceContext) raises:
    """N=K=2048 — the single-block bitonic kernel with per-row bounds."""
    comptime N = 2048
    comptime K = 2048
    var bounds: List[Int] = [2048, 1024, 100, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE2048)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_2048")
    print("PASS test_bounded_2048_kernel")


def test_bounded_resident(ctx: DeviceContext) raises:
    """N=4096 (register-resident histsel, plain digit) with per-row bounds."""
    comptime N = 4096
    comptime K = 2048
    var bounds: List[Int] = [4096, 3000, 2048, 500, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE4096)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_resident")
    print("PASS test_bounded_resident")


def test_bounded_resident_bin_digit(ctx: DeviceContext) raises:
    """N=2560 < K + K//2 — the bin-digit resident instantiation, bounded."""
    comptime N = 2560
    comptime K = 2048
    var bounds: List[Int] = [2560, 2100, 800, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE2560)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_bin_digit")
    print("PASS test_bounded_resident_bin_digit")


def test_bounded_histsel_prefetch(ctx: DeviceContext) raises:
    """N=32769, few rows — the prefetching streaming histsel, bounded.

    This is the capture-frozen decode shape: the stride is far above every
    bound, including bounds below K (select-all rows) and at exactly K.
    """
    comptime N = 32769
    comptime K = 2048
    var bounds: List[Int] = [32769, 20000, 8192, 2048, 2047, 1000, 1, 0]
    var B = len(bounds)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE7E7C)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_prefetch")
    print("PASS test_bounded_histsel_prefetch")


def test_bounded_histsel_fills_gpu(ctx: DeviceContext) raises:
    """Rows above the SM count — the non-prefetch histsel, bounded."""
    comptime N = 8193
    comptime K = 2048
    comptime B = 192
    var bounds = List[Int](capacity=B)
    for b in range(B):
        var m = b % 4
        if m == 0:
            bounds.append(N)
        elif m == 1:
            bounds.append(4096)
        elif m == 2:
            bounds.append(1500)
        else:
            bounds.append(0)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DEF111)
    _run_and_check_bounded(ctx, scores, N, K, B, bounds, "bounded_fills_gpu")
    print("PASS test_bounded_histsel_fills_gpu")


def test_bounded_histsel_select_all_tail(ctx: DeviceContext) raises:
    """A causal prefill whose rows mostly hold fewer live keys than `K`.

    The MLA indexer's shape and contract (`ordered=False`,
    `deterministic=False`): `N` just past the 2048 cutoff and every row's
    bound ramping from below `K` to above it, with rows above the SM count.
    Select-all rows append fewer than `K` keys, so their output tail must be
    written `-1` rather than read back from unwritten selection slots.

    Untouched shared memory reads as zero, which decodes to column `-1` and
    hides an unwritten tail, so a first launch with every row full leaves real
    keys in every SM's selection slots for the ramp launch to trip over.
    """
    comptime N = 2104
    comptime K = 2048
    comptime B = 200
    var full = List[Int](length=B, fill=N)
    _run_and_check_bounded[ordered=False, deterministic=False](
        ctx,
        _poisoned_bounded_scores(B, N, full, 0xB0DE7A10),
        N,
        K,
        B,
        full,
        "bounded_select_all_tail_dirty",
    )
    var bounds = List[Int](capacity=B)
    for b in range(B):
        bounds.append(1901 + b)
    var scores = _poisoned_bounded_scores(B, N, bounds, 0xB0DE7A11)
    _run_and_check_bounded[ordered=False, deterministic=False](
        ctx, scores, N, K, B, bounds, "bounded_select_all_tail"
    )
    print("PASS test_bounded_histsel_select_all_tail")


# ===----------------------------------------------------------------------=== #
# bf16 score input
# ===----------------------------------------------------------------------=== #
#
# Every read widens exactly to f32 on the way in, so the arithmetic below the
# load is dtype-blind. The load arms here only cover each load helper against
# its paths -- vector, scalar tail, past-`count` padding; mirroring the whole
# f32 suite would double this file's compile time to re-test dtype-blind logic.
#
# The ROUND SCHEDULE is the exception, and the last three arms cover it. A bf16
# score occupies 16 bits of the 32-bit key half, so `_hsel_sig_bits` gives it a
# two-round score half where f32 takes three, and its last round leaves the
# key's low 16 bits uniform. `_hsel_exact_key` recovers them from the sign, and
# only a NEGATIVE score needs it -- above a positive one those bits are already
# zero, so a positive plateau cannot fail. Every schedule arm therefore puts a
# negative value at the threshold, paired with the same data at f32 as a
# control: f32 resolves every bit, so a failure there means the case or the
# oracle is wrong, not the schedule.
#
# These arms go vacuous under a 16-bit payload, where `_hsel_exact_key`'s pad is
# zero and the helper folds to the identity. If `_hsel_phi_dtype` is ever
# narrowed, they stop testing anything while still passing.
#
# `[False, False]` is not missing from the load coverage -- `deterministic` is
# consumed after the append and never touches `in_scores`, so `[False, True]`
# covers the same loads. It has an arm anyway, `test_bf16_unord_nd_streaming`,
# because it is the only contract whose scan GROUP WIDTH follows the payload and
# the only one the fp8 indexer launches. It is held to the score-valid predicate
# rather than to the exact oracle, which is all that contract promises.


def _check_split_bf16[
    ordered: Bool = False, in_dtype: DType = .bfloat16
](
    ctx: DeviceContext,
    scores_host: List[Float32],
    N: Int,
    K: Int,
    B: Int,
    label: String,
) raises:
    """Run the split launcher at `in_dtype` and check the exact oracle set.

    The oracle runs on `_as_read`, not on the generated scores: bf16 rounding
    manufactures ties, and `_expected_topk_set` resolves ties exactly, so an
    unrounded oracle would disagree with a perfectly correct kernel.

    `ordered` picks the contract, because the two reach different kernels at the
    same shape and the score half's handoff feeds both. `deterministic` stays
    `True` throughout: the relaxed contract does not promise which members of a
    tie plateau survive, so it cannot hold an exact set -- and a plateau is
    exactly what the schedule arms below are made of.
    """
    var idxs = _run_split[ordered, True, in_dtype=in_dtype](
        ctx, scores_host, N, K, B
    )
    var seen = _as_read[in_dtype](scores_host)
    for b in range(B):
        var row_scores = List[Float32](capacity=N)
        for i in range(N):
            row_scores.append(seen[b * N + i])
        var row = List[Int](capacity=K)
        for k in range(K):
            row.append(idxs[b * K + k])
        _check_set_row(row_scores, row, N, K, label, b)


def test_bf16_block_multirow(ctx: DeviceContext) raises:
    """`_load4_scores` at bf16, both paths, across both single-block kernels.

    Odd `N` leaves most rows' bases off the 4-element multiple the width-4 load
    tests, so each row exercises the vector path or the scalar tail depending on
    its own base -- the same split the f32 twins rely on.
    """
    for shape in [(1025, 512), (8193, 2048)]:
        var N = shape[0]
        var K = shape[1]
        comptime B = 8
        var scores = _lcg_scores(B, N, 0xBF160000 + UInt32(N), 40.0)
        _run_and_check_block_multirow[.bfloat16](
            ctx, scores, N, K, B, String("bf16_block_n", N)
        )
    print("PASS test_bf16_block_multirow")


def test_bf16_resident(ctx: DeviceContext) raises:
    """`_phi4` at bf16: vector path, scalar tail, and the past-the-end branch.

    `B` under the SM count is what keeps the register-resident select chosen.
    2560 is below the `K + K // 2` crossover so it takes the `bin_digit`
    instantiation; 4093 is not a multiple of 4, so its tail rows reach both the
    scalar branch and the group-entirely-past-the-row-end branch.
    """
    comptime K = 2048
    comptime B = 48
    for N in [2560, 4093, 4096]:
        var scores = _lcg_scores(B, N, 0xBF16A000 + UInt32(N), 100.0)
        _check_split_bf16(ctx, scores, N, K, B, String("bf16_resident_n", N))
    print("PASS test_bf16_resident")


def test_bf16_histsel(ctx: DeviceContext) raises:
    """`_load_scan_group` at bf16 -- including its 16-byte-aligned fast path.

    These are the only shapes that reach `parked`, so they are also the only
    ones that run the four scalar `in_scores[row + col]` reads in the refine
    loop. The `N` values are chosen for the alignment test, which at bf16 asks
    for an 8-element multiple rather than f32's 4: 32769 and 8193 are `1 mod 8`
    and take the scalar tail, while 32768 is the one that takes the 16-byte
    vector path. Without it that path would be compiled and never run.
    """
    comptime K = 2048
    for shape in [(32769, 8), (32768, 8), (8193, 192)]:
        var N = shape[0]
        var B = shape[1]
        var scores = _lcg_scores(B, N, 0xBF16B000 + UInt32(N), 100.0)
        _check_split_bf16(
            ctx, scores, N, K, B, String("bf16_histsel_n", N, "_b", B)
        )
    print("PASS test_bf16_histsel")


def test_bf16_bounded(ctx: DeviceContext) raises:
    """The bound contract at bf16, across all three load helpers.

    `_poisoned_bounded_scores` puts values larger than any live score at and
    past each bound, so a read past a bound displaces a live selection and
    surfaces as an out-of-bound index. The poison is 1.0e30, which bf16
    represents (approximately, but far above every live score), so the trap
    still springs at this width.
    """
    comptime K = 2048
    for N in [2048, 4096, 32769]:
        var bounds: List[Int] = [N, N - 1, 4096, 2048, 1000, 1, 0, N]
        var B = len(bounds)
        for b in range(B):
            if bounds[b] > N:
                bounds[b] = N
        var scores = _poisoned_bounded_scores(
            B, N, bounds, 0xBF16C000 + UInt32(N)
        )
        _run_and_check_bounded[.bfloat16](
            ctx, scores, N, K, B, bounds, String("bf16_bounded_n", N)
        )
    print("PASS test_bf16_bounded")


def test_bf16_rounding_plateau(ctx: DeviceContext) raises:
    """Scores distinct in f32 that collapse to one bf16 value across `K`.

    A monotone ramp will not do this, which is worth stating because it is the
    obvious construction and it is inert: the total order breaks ties by
    ASCENDING column, so on a ramp the rounded selection reproduces the
    unrounded one exactly and the case tests nothing.

    What separates them is a group whose f32 order runs OPPOSITE to its column
    order. Sixteen columns straddling the cut hold `1.0 + j/16384` increasing
    with the column; bf16 keeps 8 significand bits, so all sixteen are inside
    one rounding interval of 1.0 and tie. Exactly eight are wanted. The exact
    order takes the eight LARGEST -- the last eight columns; the rounded order
    sees one plateau and takes the eight LOWEST columns. The two selections are
    disjoint, so the arms cannot coincide by luck.

    The second assertion is the point of the test: the oracle on the UNROUNDED
    scores must disagree with the kernel. Without it this case would pass just
    as happily if `_as_read` were the identity, and the rounding discipline the
    rest of these arms depend on would be untested.
    """
    comptime N = 4096
    comptime K = 2048
    comptime B = 4
    comptime group_lo = K - 8
    comptime group_hi = K + 8
    var scores = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            if c < group_lo:
                scores.append(Float32(10.0))
            elif c < group_hi:
                # Distinct in f32, one value in bf16: the step is 2**-14 and
                # bf16's rounding interval around 1.0 is 2**-9 wide.
                scores.append(
                    Float32(1.0) + Float32(c - group_lo) / Float32(16384.0)
                )
            else:
                scores.append(Float32(-1.0))

    _check_split_bf16(ctx, scores, N, K, B, "bf16_rounding_plateau")

    var row_raw = List[Float32](capacity=N)
    var row_seen = List[Float32](capacity=N)
    var seen = _as_read[.bfloat16](scores)
    for i in range(N):
        row_raw.append(scores[i])
        row_seen.append(seen[i])
    assert_true(
        _expected_topk_set(row_raw, K) != _expected_topk_set(row_seen, K),
        (
            "the plateau is supposed to exist only after bf16 rounding, but the"
            " rounded and unrounded oracles agree -- this case no longer tests"
            " that the reference is rounded, so widen the run or shrink the"
            " step"
        ),
    )
    print("PASS test_bf16_rounding_plateau")


def _wide_negative_plateau(
    B: Int, N: Int, n_above: Int, plateau: Int
) -> List[Float32]:
    """Rows whose threshold sits inside a wide NEGATIVE tie plateau.

    Three things make it the shape that reaches the score half's handoff, and
    all three are load-bearing:

    - The plateau value is NEGATIVE, which is the only sign for which
      `_hsel_exact_key` does any work.
    - It is WIDER than the select's slack. A round hands over as soon as the
      columns at or above its bracket fit `sel_cap` (`_HSEL_SEL_CAP`, twice
      `K`), and handing over sets the threshold from the bracket FLOOR, where
      the floor is the right answer at any width. Only a plateau too wide to
      hand over runs the score half out of rounds, which is the path that needs
      the key itself.
    - Every value is exactly representable in bf16, so `_as_read` is the
      identity and the tie is the f32 tie -- no rounding confound between the
      two arms of the control.
    """
    var scores = List[Float32](capacity=B * N)
    for _ in range(B):
        for c in range(N):
            if c < n_above:
                scores.append(Float32(-0.5))
            elif c < n_above + plateau:
                scores.append(Float32(-1.0))
            else:
                scores.append(Float32(-1.0e30))
    return scores^


def test_bf16_unord_nd_streaming(ctx: DeviceContext) raises:
    """The relaxed contract at bf16, on the arm production actually launches.

    `_check_split_bf16` pins `deterministic=True` because the relaxed contract
    cannot hold an exact set -- which left the ONE arm the fp8 indexer launches
    with no bf16 coverage at all. It is also the only arm whose scan group width
    follows the payload (see `_hsel_prefetch_scan_items`), so a change to the row
    scan lands here and nowhere a bf16 test looked.

    Held to `_check_score_valid_row`, the strongest predicate that does not
    over-promise: `deterministic=False` does not say WHICH members of a
    threshold plateau survive, only that everything strictly above it is in.

    The shapes pick the arm and then both of its load paths. `rows` below the SM
    count and `N` past the wide resident payload is the prefetching streaming
    select; `N = 32768` is a whole number of 8-element bf16 vectors so every
    thread takes the wide load, and `32769` and `20481` leave the last group
    short so the scalar tail runs too. Each is paired with the same data at f32,
    which reaches the same arm at the narrow group width -- so a failure there
    means the case or the checker is wrong rather than the group width.
    """
    comptime K = 2048
    comptime B = 8
    for N in [32768, 32769, 20481]:
        var scores = _lcg_scores(B, N, 0xC1000000 + UInt32(N), 100.0)
        var seen = _as_read[.bfloat16](scores)
        for arm in range(2):
            var idxs: List[Int]
            if arm == 0:
                idxs = _run_split[False, False, .bfloat16](ctx, scores, N, K, B)
            else:
                idxs = _run_split[False, False, .float32](ctx, scores, N, K, B)
            for b in range(B):
                var row_scores = List[Float32](capacity=N)
                for i in range(N):
                    if arm == 0:
                        row_scores.append(seen[b * N + i])
                    else:
                        row_scores.append(scores[b * N + i])
                var row = List[Int](capacity=K)
                for k in range(K):
                    row.append(idxs[b * K + k])
                _check_score_valid_row(
                    row_scores,
                    row,
                    N,
                    K,
                    String("unord_nd_bf16_n", N, "_arm", arm),
                    b,
                )
    print("PASS test_bf16_unord_nd_streaming")


def test_bf16_negative_plateau_streaming(ctx: DeviceContext) raises:
    """The two-round score half's handoff, on the streaming select.

    `N` past `_HSEL_RES_MAX` with `B` under the SM count is what streams; 32769
    is also `1 mod 8`, so the rows take the bf16 scalar tail as well. 2040
    columns sit above a 4096-wide plateau, so `need` falls to 8 and the plateau
    is 6136 at-or-above against a 4096 slack -- no round can hand over and the
    column half has to break the tie, which it can only do against the exact
    threshold key.
    """
    comptime N = 32769
    comptime K = 2048
    comptime B = 8
    var scores = _wide_negative_plateau(B, N, 2040, 4096)
    _check_split_bf16[ordered=False](ctx, scores, N, K, B, "bf16_negplat_str")
    _check_split_bf16[ordered=True](
        ctx, scores, N, K, B, "bf16_negplat_str_ord"
    )
    # f32 control: three rounds resolve the key outright, so this arm passes
    # with or without the sign fill. A failure here is the case, not the fix.
    _check_split_bf16[ordered=False, in_dtype=.float32](
        ctx, scores, N, K, B, "f32_negplat_str"
    )
    print("PASS test_bf16_negative_plateau_streaming")


def test_bf16_negative_plateau_resident(ctx: DeviceContext) raises:
    """The same handoff on the register-resident select, ordered contract.

    Ordered on purpose: under the cheap contract this kernel folds a
    single-valued bracket into its threshold directly (`ctl[6] == ctl[7]`) and
    never reaches the handoff, so the arm that looks most production-like is
    the one that cannot fail. `N` at `_HSEL_RES_MAX` and `B` under the SM count
    is what keeps the row in registers.
    """
    comptime N = 8192
    comptime K = 2048
    comptime B = 48
    var scores = _wide_negative_plateau(B, N, 2040, 4096)
    _check_split_bf16[ordered=True](ctx, scores, N, K, B, "bf16_negplat_res")
    _check_split_bf16[ordered=False](
        ctx, scores, N, K, B, "bf16_negplat_res_un"
    )
    _check_split_bf16[ordered=True, in_dtype=.float32](
        ctx, scores, N, K, B, "f32_negplat_res"
    )
    print("PASS test_bf16_negative_plateau_resident")


def test_bf16_negative_levels_masked(ctx: DeviceContext) raises:
    """The indexer's own shape with the levels pushed negative.

    A per-row valid prefix, a `-3.0e38` masked tail, and four coarse score
    levels -- coarse so the top level alone holds several times `K`, which is
    what runs the score half out of rounds on real data rather than on a
    hand-built plateau. The mask sentinel is negative and bf16-representable
    (~-2.996e38), and the past-`count` padding maps to a `phi` of 0, which sits
    in a different round-0 bin from every live score.
    """
    comptime N = 32769
    comptime K = 2048
    comptime B = 8
    var scores = List[Float32](capacity=B * N)
    var state = UInt32(0xBF16D000)
    for b in range(B):
        var valid = N - 1500 * b
        for c in range(N):
            state = 1664525 * state + 1013904223
            if c < valid:
                scores.append(
                    Float32(-1.0) - Float32(Int((state >> 16) % 4)) / 8.0
                )
            else:
                scores.append(Float32(-3.0e38))
    _check_split_bf16[ordered=False](ctx, scores, N, K, B, "bf16_neglevels")
    _check_split_bf16[ordered=True](ctx, scores, N, K, B, "bf16_neglevels_ord")
    _check_split_bf16[ordered=False, in_dtype=.float32](
        ctx, scores, N, K, B, "f32_neglevels"
    )
    print("PASS test_bf16_negative_levels_masked")


# ===----------------------------------------------------------------------=== #
# Short row beside a long one, unordered + bf16 — the `select_all` pad overrun.
# ===----------------------------------------------------------------------=== #


def _run_short_row_overrun[
    in_dtype: DType, ordered: Bool, deterministic: Bool
](
    ctx: DeviceContext,
    N: Int,
    K: Int,
    B: Int,
    bounds: List[Int],
    label: String,
) raises:
    """Assert a row with `bound < K` writes only into its own `K` slots.

    A row whose live count is below `K` takes the select-all path, which has
    nothing to resolve and appends every live column. The append's "greater
    than the threshold" branch writes without a slot bound -- only its "equal"
    sibling checks -- so whether the padding past the row end lands in the
    greater branch decides whether the write stays inside the row.

    That is dtype-dependent. `_score_pad` is all-ones in the BITS domain, whose
    `phi` image is 0 at f32 but `0x0000FFFF` once a bf16 pattern is widened.
    The select-all threshold is 0, so an f32 pad ties and is bounds-checked
    while a bf16 pad compares strictly greater and is not.

    The buffer is over-allocated and the slack poisoned, so an overrun is a
    failed assertion here rather than whatever it would corrupt in production.
    """
    assert len(bounds) == B, "bounds length mismatch"

    # Room for the worst case (the last row appending all `N` of its columns)
    # so the check reports the overrun instead of faulting on it.
    var slack = 2 * N
    var total_slots = B * K + slack

    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](total_slots)
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)

    var scores_host = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            # Descending within a row, so the top-`K` of a long row is a
            # prefix and any pad that displaces one is visible as a big index.
            scores_host.append(Float32(1.0) - Float32(c) * Float32(1e-4))
        _ = b

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores_host[i].cast[in_dtype]()
    with bounds_dev.map_to_host() as buf:
        for b in range(B):
            buf[b] = Int32(bounds[b])

    comptime GUARD = Int32(0x5A5A5A5A)
    with idxs_dev.map_to_host() as buf:
        for i in range(B * K):
            buf[i] = Int32(-2)  # catches an unwritten in-row slot
        for i in range(B * K, total_slots):
            buf[i] = GUARD

    persistent_topk_block_split[ordered=ordered, deterministic=deterministic](
        ctx,
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            scores_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
        N,
        K,
        total_seq_len=B,
        row_bounds=Optional(
            RowBoundsPtr(
                rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                    bounds_dev.unsafe_ptr()
                )
            )
        ),
    )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](total_slots)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var trespass = 0
    var first_bad = -1
    for i in range(B * K, total_slots):
        if idxs_host[i] != GUARD:
            trespass += 1
            if first_bad < 0:
                first_bad = i
    assert_true(
        trespass == 0,
        String(
            "[",
            label,
            "] wrote past the index buffer: ",
            trespass,
            " of ",
            slack,
            " guard slots clobbered, first at flat slot ",
            first_bad,
            " (buffer holds B*K = ",
            B * K,
            ")",
        ),
    )

    for b in range(B):
        var bound = bounds[b]
        var live = min(K, bound)
        for k in range(K):
            var idx = Int(idxs_host[b * K + k])
            if k < live:
                assert_true(
                    idx >= 0 and idx < bound,
                    String(
                        "[",
                        label,
                        "] row ",
                        b,
                        " (bound ",
                        bound,
                        ") slot ",
                        k,
                        " holds ",
                        idx,
                        ", which is not a live column",
                    ),
                )
            else:
                assert_true(
                    idx == -1,
                    String(
                        "[",
                        label,
                        "] row ",
                        b,
                        " (bound ",
                        bound,
                        ") tail slot ",
                        k,
                        " holds ",
                        idx,
                        ", expected -1",
                    ),
                )

        # A short row selects every live column exactly once, so its set is
        # forced and a displaced pad shows up as a missing column.
        if bound <= K:
            var seen = Set[Int]()
            for k in range(live):
                seen.add(Int(idxs_host[b * K + k]))
            assert_true(
                len(seen) == live,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " selected ",
                    len(seen),
                    " distinct columns, expected ",
                    live,
                ),
            )


def test_short_row_beside_long_row_bf16(ctx: DeviceContext) raises:
    """GLM's decode shape: a freshly admitted short request beside a long one.

    `N` sits in the wide register-resident window (`8192 < N <= 16384`, taken
    only by the unordered contract) and `B` is under the SM count, which is
    what routes these rows to the resident select at all. One row is shorter
    than `K`; the rest are full length.
    """
    comptime N = 12288
    comptime K = 2048
    comptime B = 4
    var bounds: List[Int] = [N, N, 500, N]

    _run_short_row_overrun[.bfloat16, ordered=False, deterministic=False](
        ctx, N, K, B, bounds, "bf16 unord nd short-row"
    )
    _run_short_row_overrun[.bfloat16, ordered=False, deterministic=True](
        ctx, N, K, B, bounds, "bf16 unord det short-row"
    )
    # f32 is the control: the pad ties at the threshold there, so this arm
    # passing while bf16 fails localizes the fault to the pad's image.
    _run_short_row_overrun[.float32, ordered=False, deterministic=False](
        ctx, N, K, B, bounds, "f32 unord nd short-row"
    )
    print("PASS test_short_row_beside_long_row_bf16")


# ===----------------------------------------------------------------------=== #
# Cluster-cooperative kernel (`topk_cluster.mojo`): the direct entry point the
# dispatch routes to, the dispatch gate itself, and device-graph capture.
#
# The kernel serves the `ordered=False, deterministic=False` contract only, so
# every check here runs on `_check_score_valid_row` -- the strongest predicate
# that contract carries: every column strictly above the K-th largest score is
# selected, everything else ties it exactly, the count is exact, and `-1` tails
# sit past each row's live columns. Nothing about the arrangement or which
# plateau members survive is asserted, because nothing about them is promised.
# ===----------------------------------------------------------------------=== #


def _run_and_check_cluster[
    in_dtype: DType = .float32,
    use_bounds: Bool = True,
    via_dispatch: Bool = False,
    use_tma: Bool = False,
    nan_rows: Bool = False,
](
    ctx: DeviceContext,
    scores_host: List[Float32],  # B * N flat, row-major
    N: Int,
    K: Int,
    B: Int,
    bounds: List[Int],  # live length per row (all `N` when `use_bounds=False`)
    cluster_size: Int,
    label: String,
) raises:
    """Launch the cluster top-k and validate the nd contract row by row.

    `cluster_size` is the caller's, not the dispatch's, so the direct-entry
    tests pin the kernel instantiation they mean to reach regardless of what
    `_choose_topk_cluster_size` would pick. `via_dispatch` instead goes through
    `persistent_topk_block_split`, whose gate picks the cluster (and both
    `cluster_size` and `use_tma` are then ignored); shapes the gate rejects
    keep the histogram selects, which serve the same contract, so they
    validate identically. `use_tma` pins the streaming flavor's load path --
    the launcher clamps it to the plain loads wherever the bulk copy cannot
    serve the shape, so an arm labeled TMA is only exercised as such on a
    target and stride that allow it. Columns at or past each row's bound are
    the caller's to poison; a read past a bound displaces a live selection
    and surfaces as an out-of-bound index. `nan_rows` swaps the contract for
    `_check_nan_row`'s weaker one, for rows outside the contract.
    """
    assert K <= PERSISTENT_TOPK_MAX_N, "K exceeds selection width"
    assert len(scores_host) == B * N, "scores_host length mismatch"
    assert len(bounds) == B, "bounds length mismatch"

    var scores_dev = ctx.enqueue_create_buffer[in_dtype](B * N)
    var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
    idxs_dev.enqueue_fill(Int32(-2))  # sentinel to catch unwritten slots
    var bounds_dev = ctx.enqueue_create_buffer[.int32](B)

    with scores_dev.map_to_host() as buf:
        for i in range(B * N):
            buf[i] = scores_host[i].cast[in_dtype]()
            comptime if nan_rows and in_dtype == .bfloat16:
                # A cast canonicalizes NaNs; keep the payload's top half so
                # an all-ones f32 NaN stays all ones.
                var v = scores_host[i]
                if v != v:
                    buf[i] = bitcast[in_dtype, 1](
                        UInt16(bitcast[.uint32, 1](v)[0] >> 16)
                    )[0]
    with bounds_dev.map_to_host() as buf:
        for b in range(B):
            buf[b] = Int32(bounds[b])

    if use_bounds:
        if via_dispatch:
            persistent_topk_block_split[ordered=False, deterministic=False](
                ctx,
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    scores_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
                N,
                K,
                total_seq_len=B,
                row_bounds=Optional(
                    RowBoundsPtr(
                        rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                            bounds_dev.unsafe_ptr()
                        )
                    )
                ),
            )
        else:
            topk_cluster_split[RB=RowBoundsPtr](
                ctx,
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    scores_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
                N,
                K,
                total_seq_len=B,
                row_bounds=Optional(
                    RowBoundsPtr(
                        rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                            bounds_dev.unsafe_ptr()
                        )
                    )
                ),
                cluster_size=cluster_size,
                use_tma=use_tma,
            )
    else:
        if via_dispatch:
            persistent_topk_block_split[ordered=False, deterministic=False](
                ctx,
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    scores_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
                N,
                K,
                total_seq_len=B,
            )
        else:
            topk_cluster_split(
                ctx,
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    scores_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
                N,
                K,
                total_seq_len=B,
                row_bounds=None,
                cluster_size=cluster_size,
                use_tma=use_tma,
            )
    ctx.synchronize()

    var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
    ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
    ctx.synchronize()

    var seen_scores = _as_read[in_dtype](scores_host)
    for b in range(B):
        var bound = min(bounds[b], N)
        var live = min(K, bound)
        for k in range(live, K):
            var idx = Int(idxs_host[b * K + k])
            assert_true(
                idx == -1,
                String(
                    "[",
                    label,
                    "] row ",
                    b,
                    " tail slot ",
                    k,
                    " = ",
                    idx,
                    " (expected -1 past bound ",
                    bound,
                    ")",
                ),
            )
        if live == 0:
            continue
        var row_scores = List[Float32](capacity=bound)
        for i in range(bound):
            row_scores.append(seen_scores[b * N + i])
        var row_idxs = List[Int](capacity=live)
        for k in range(live):
            row_idxs.append(Int(idxs_host[b * K + k]))
        comptime if nan_rows:
            _check_nan_row(row_scores, row_idxs, bound, label, b)
        else:
            _check_score_valid_row(row_scores, row_idxs, bound, live, label, b)

    _ = scores_dev
    _ = idxs_dev
    _ = bounds_dev
    _ = idxs_host


def _check_nan_row(
    scores: List[Float32],
    idxs: List[Int],
    bound: Int,
    label: String,
    row: Int,
) raises:
    """What a row holding NaN scores is still owed: every slot is `-1` or a
    distinct live column, and every non-NaN column is selected (the callers
    keep those fewer than `K`, so each lies strictly above the threshold)."""
    var seen = Set[Int]()
    for k in range(len(idxs)):
        var idx = idxs[k]
        if idx == -1:
            continue
        assert_true(
            idx >= 0 and idx < bound,
            String("[", label, "] row ", row, " slot ", k, " = ", idx),
        )
        assert_true(
            idx not in seen,
            String("[", label, "] row ", row, " repeats column ", idx),
        )
        seen.add(idx)
    for c in range(bound):
        if scores[c] == scores[c]:
            assert_true(
                c in seen,
                String("[", label, "] row ", row, " drops column ", c),
            )


def _cluster_bounds(N: Int, K: Int) -> List[Int]:
    """Bounds at the row gate's interesting edges: full, one short, half,
    `K` from both sides, a one-column row and a dead row."""
    var bounds: List[Int] = [
        N,
        N - 1,
        N // 2,
        K + 1,
        K,
        max(K - 1, 1),
        1,
        0,
    ]
    for b in range(len(bounds)):
        bounds[b] = min(bounds[b], N)
    return bounds^


def test_topk_cluster_band(ctx: DeviceContext) raises:
    """The kernel at both score dtypes, both flavors, and every cluster size.

    The streaming shapes are those whose slices exceed the single-pass
    staging at their dtype (51872 f32, 103744 bf16), so every round and the
    scatter re-read the slice: f32 at `N=163840, CS=2` (81920), at
    `N=550000, CS=10` (55000) and at `N=163840, CS=3` (54616); bf16 at
    `N=900000, CS=8` (112504), at `N=1100000, CS=10` (110000) and at
    `N=327680, CS=3` (109232). Each runs twice, once per streaming load path:
    the plain-load prefetch arm and the `cp.async.bulk` pipeline arm (the
    launcher clamps the latter to the former wherever it cannot serve the
    shape, so on a non-SM100 target the pair still passes, twice over the
    same code). One more run at one column wider than the first shape leaves
    most rows' slices off 16 B, which only the plain loads' per-element
    flavor serves. The single-pass shapes cover every other cluster size, the
    non-powers of two at strides that leave a remainder of 16 B units to
    spread over the ranks; round 0's one global read is all they pay.
    `N=103744, CS=2` fills the f32 staging exactly, and `N=103752` puts its
    full rows' slices one 16 B unit past it while `_cluster_bounds`' shorter
    rows stay single-pass, so one launch runs both flavors.
    `bf16, N=107232, CS=4` is the GLM decode shape. The bounds arm covers the
    trivial path (bounds at/below `K`), select-all tails and dead rows; the
    unbounded arm covers the plain entry point.
    """
    comptime K = 2048
    comptime for dt in [DType.float32, DType.bfloat16]:
        for shape in [
            (8193, 2),
            (32769, 4),
            (107232, 4),
            (122880, 8),
            (163840, 10),
            (49160, 3),
            (98312, 5),
            (107232, 6),
            (163848, 10),
            (103744, 2),
            (103752, 2),
        ]:
            var N = shape[0]
            var cs = shape[1]
            var B = 8
            var bounds = _cluster_bounds(N, K)
            var scores = _poisoned_bounded_scores(
                B, N, bounds, 0xC1E50000 + UInt32(N)
            )
            _run_and_check_cluster[dt](
                ctx,
                scores,
                N,
                K,
                B,
                bounds,
                cs,
                String("cluster_n", N),
            )
        var streaming: List[Tuple[Int, Int]]
        comptime if dt == DType.float32:
            streaming = [(163840, 2), (550000, 10), (163840, 3)]
        else:
            streaming = [(900000, 8), (1100000, 10), (327680, 3)]
        for si in range(len(streaming)):
            var N = streaming[si][0]
            var cs = streaming[si][1]
            var bounds = _cluster_bounds(N, K)
            var scores = _poisoned_bounded_scores(
                8, N, bounds, 0xC1E5F231 + UInt32(N + cs)
            )
            _run_and_check_cluster[dt](
                ctx,
                scores,
                N,
                K,
                8,
                bounds,
                cs,
                String("cluster_streaming_n", N, "_cs", cs),
            )
            _run_and_check_cluster[dt, use_tma=True](
                ctx,
                scores,
                N,
                K,
                8,
                bounds,
                cs,
                String("cluster_streaming_tma_n", N, "_cs", cs),
            )
            if si == 0:
                var Nu = N + 1
                var bounds_u = _cluster_bounds(Nu, K)
                _run_and_check_cluster[dt](
                    ctx,
                    _poisoned_bounded_scores(8, Nu, bounds_u, 0xC1E5F232),
                    Nu,
                    K,
                    8,
                    bounds_u,
                    cs,
                    String("cluster_streaming_unaligned_n", Nu, "_cs", cs),
                )
        # One unbounded arm: the plain kernel entry over the full stride.
        var full = List[Int](length=8, fill=32769)
        _run_and_check_cluster[dt, use_bounds=False](
            ctx,
            _lcg_scores(8, 32769, 0xC1EF0000, 100.0),
            32769,
            K,
            8,
            full,
            4,
            "cluster_unbounded",
        )
    print("PASS test_topk_cluster_band")


def test_topk_cluster_ties(ctx: DeviceContext) raises:
    """The distributions whose threshold lands inside a plateau.

    All-equal, a plateau straddling `K`, a masked tail over coarse levels, and
    a wide NEGATIVE plateau (the shape that reached `_hsel_exact_key` in the
    selects, kept here so the cluster path's own score-half resolution is held
    to the same stimulus). All exactly representable at bf16, so the f32 arm
    and the bf16 arm select over the same values. Each runs at a width rank 0
    selects alone (12289) and at one the cluster splits (40963).
    """
    comptime K = 2048
    comptime B = 4
    comptime for dt in [DType.float32, DType.bfloat16]:
        for N in [12289, 40963]:
            var eq = List[Float32](capacity=B * N)
            for _ in range(B * N):
                eq.append(Float32(1.5))
            _run_and_check_cluster[dt](
                ctx,
                eq,
                N,
                K,
                B,
                List[Int](length=B, fill=N),
                4,
                "clu_all_equal",
            )

            var plateau = List[Float32](capacity=B * N)
            for _b in range(B):
                for c in range(N):
                    if c < K - 200:
                        plateau.append(Float32(10.0))
                    elif c < K + 400:
                        plateau.append(Float32(5.0))
                    else:
                        plateau.append(Float32(-1.0))
            _run_and_check_cluster[dt](
                ctx,
                plateau,
                N,
                K,
                B,
                List[Int](length=B, fill=N),
                4,
                "clu_plateau",
            )

            var masked = List[Float32](capacity=B * N)
            for b in range(B):
                var valid = N - 1500 * b
                for c in range(N):
                    if c < valid:
                        masked.append(Float32((c * 37 + b * 11) % 17) / 16.0)
                    else:
                        masked.append(Float32(-3.0e38))
            var m_bounds = List[Int](capacity=B)
            for b in range(B):
                m_bounds.append(N - 1500 * b)
            _run_and_check_cluster[dt](
                ctx, masked, N, K, B, m_bounds, 4, "clu_masked"
            )

            _run_and_check_cluster[dt](
                ctx,
                _wide_negative_plateau(B, N, 2040, 4096),
                N,
                K,
                B,
                List[Int](length=B, fill=N),
                4,
                "clu_neg_plateau",
            )

        # All-equal at CS 10 with every rank's tie class capped at K: the
        # packed counts' tie half sums to 10 * K, its widest reach.
        comptime N10 = 10 * 4096
        _run_and_check_cluster[dt](
            ctx,
            List[Float32](length=2 * N10, fill=Float32(1.5)),
            N10,
            K,
            2,
            List[Int](length=2, fill=N10),
            10,
            "clu_all_equal_cs10",
        )
    print("PASS test_topk_cluster_ties")


def _cand_noise(row: Int, c: Int) -> Float32:
    """A deterministic value in `[0, 1)` per `(row, column)`."""
    return Float32((c * 2654435761 + row * 40503) % 1000003) / 1000003.0


def _cand_rows(kind: Int, B: Int, N: Int, bounds: List[Int]) -> List[Float32]:
    """Rows shaped to steer the cluster kernel's candidate path.

    Each `kind` places round 0's split bin (`_ctopk_key0`, 16 bins per
    octave) where one branch of the candidate protocol has to handle it:

    0. Dense: every third column ties at 2.0, with distinct values just
       above it in the same bin, so the bin holds far more than rank 0's
       list and the multi-round path refines it.
    1. One peer over its cap: 1500 distinct candidates in the last 1500 live
       columns (the last rank's slice), under the total cap but past one
       peer's, so the counts verdict sends the cluster to the multi-round
       path after the candidate scatter.
    2. A candidate path exact plateau on a peer: 1000 candidates in the last
       rank's slice, 200 distinct above a 300-wide plateau and 500 below,
       with the threshold cut inside the plateau.
    3. Kind 2 inside rank 0's slice, so rank 0's own list carries them.
    4. The saturated top bin: 1365 distinct scores past 2^16, so the
       refinement starts 30 bits wide and runs several rounds.
    5. The bin just below zero: 1560 distinct negative scores under 2^-16
       in magnitude, beside 410 positives and a deep negative floor.

    Kinds 4 and 5 are meant for `K = 1024`; the others for `K = 2048`.
    The periodic columns' periods scale with `N // 32768`, so a longer row
    keeps the same split-bin population. Columns at or past a row's bound
    are poisoned to `1e30`.
    """
    var sc = max(1, N // 32768)
    var out = List[Float32](capacity=B * N)
    for b in range(B):
        var L = bounds[b]
        for c in range(N):
            var v: Float32
            var u = _cand_noise(b, c)
            if c >= L:
                v = 1.0e30
            elif kind == 0:
                if c % 3 == 0:
                    v = 2.0
                elif c % 97 == 1:
                    v = 2.0 + Float32(c % 1000) * 1.0e-5
                else:
                    v = u * 1.8 - 0.9
            elif kind == 1 or kind == 2 or kind == 3:
                var n_c = 1500 if kind == 1 else 1000
                var c0 = 0 if kind == 3 else L - n_c
                var j = c - c0
                if j >= 0 and j < n_c:
                    if kind == 1:
                        v = 1.0 + Float32(j) * Float32(1.0 / 32768.0)
                    elif j < 200:
                        v = 1.05 - Float32(j) * Float32(1.0 / 65536.0)
                    elif j < 500:
                        v = 1.02
                    else:
                        v = 1.0 + Float32(j - 500) * Float32(1.0 / 65536.0)
                elif c % (19 * sc) == 0:
                    v = 50.0 + u
                else:
                    v = u * 0.5
            elif kind == 4:
                if c % (24 * sc) == 0:
                    v = 7.0e4 * Float32(1.0 + Float32(c // (24 * sc)) * 2.5e-3)
                else:
                    v = u * 2.0 - 1.0
            else:
                if c % (80 * sc) == 0:
                    v = 1.0 + Float32(c // sc) * 1.0e-5
                elif c % (21 * sc) == 1:
                    v = -(1.0e-7 + Float32(c % 997) * 1.0e-10)
                else:
                    v = -50.0 - Float32(c % 100)
            out.append(v)
    return out^


def test_topk_cluster_candidates(ctx: DeviceContext) raises:
    """The candidate path and both of its fallbacks, at every cluster size.

    `_cand_rows` steers round 0's split bin into each branch: the dense
    bin that skips the candidate scatter, a single peer whose list overflows
    after it, an exact plateau cut on a peer's and on rank 0's candidates,
    and the two window-edge bins whose refinement runs wide. Each case runs
    at both dtypes and every cluster size, on one full row and one bounded
    row; at CS 10 the last rank's slice (about 3480 columns) still holds
    kind 1's 1500 candidates past a peer's cap and kind 2's 1000 under it,
    and nine peers share rank 0's list. The candidate-path kinds also run on
    `test_topk_cluster_band`'s streaming shapes (and at CS 10), over both
    streaming load paths, where the candidate scatter re-reads the slice
    from global rather than from the staging.
    """
    comptime N = 34816
    comptime B = 2
    var bounds: List[Int] = [N, N - 1003]
    comptime for dt in [DType.float32, DType.bfloat16]:
        for kind in range(6):
            var K = 1024 if kind >= 4 else 2048
            var scores = _cand_rows(kind, B, N, bounds)
            for cs in [2, 3, 4, 5, 6, 8, 10, 16]:
                _run_and_check_cluster[dt](
                    ctx,
                    scores,
                    N,
                    K,
                    B,
                    bounds,
                    cs,
                    String("clu_cand", kind, "_cs", cs),
                )
        var s_shapes: List[Tuple[Int, Int]]
        comptime if dt == DType.float32:
            s_shapes = [(163840, 2), (550000, 10)]
        else:
            s_shapes = [(900000, 8), (1100000, 10)]
        for shape in s_shapes:
            _cand_streaming[dt](ctx, shape[0], shape[1])
    print("PASS test_topk_cluster_candidates")


def _cand_streaming[dt: DType](ctx: DeviceContext, sN: Int, scs: Int) raises:
    """`test_topk_cluster_candidates`' kinds 1-5 at a streaming shape."""
    comptime B = 2
    var sbounds: List[Int] = [sN, sN - 5003]
    for kind in range(1, 6):
        var K = 1024 if kind >= 4 else 2048
        var scores = _cand_rows(kind, B, sN, sbounds)
        _run_and_check_cluster[dt](
            ctx,
            scores,
            sN,
            K,
            B,
            sbounds,
            scs,
            String("clu_cand", kind, "_streaming_cs", scs),
        )
        _run_and_check_cluster[dt, use_tma=True](
            ctx,
            scores,
            sN,
            K,
            B,
            sbounds,
            scs,
            String("clu_cand", kind, "_streaming_tma_cs", scs),
        )


def test_topk_cluster_nan_pad(ctx: DeviceContext) raises:
    """Rows of negative NaNs beside fewer than `K` finite scores.

    At this length (2596 live columns) rank 0 selects the row alone. Its
    fp16 conversion returns every NaN as the positive canonical NaN, so the
    coarse bin takes its sign from the score: an all-ones NaN must bin
    lowest, as its `phi` is, or the finite columns fall below the NaNs and
    are dropped. bf16's 8-column groups leave the row's last group partial,
    so its dead columns (`0x7FFF`) sit beside the NaNs' bin 0 and must stay
    out of every class. This pins what such a row is still owed
    (`_check_nan_row`): no column past the bound, no column twice, every
    non-NaN column kept. The NaNs keep their all-ones bits at bf16 (see
    `_run_and_check_cluster`).
    """
    comptime K = 2048
    var neg_nan = bitcast[.float32, 1](UInt32(0xFFFFFFFF))[0]
    comptime N = 2600
    comptime bound = 2596
    var row = List[Float32](capacity=N)
    for c in range(N):
        if c >= bound:
            row.append(Float32(1.0e30))
        elif c < 652 and not (328 <= c and c < 338):
            row.append(Float32(1 + c % 13))
        else:
            row.append(neg_nan)
    comptime B = 2
    var scores = List[Float32](capacity=B * N)
    for _ in range(B):
        scores.extend(row.copy())
    var bounds = List[Int](length=B, fill=bound)
    _run_and_check_cluster[.float32, nan_rows=True](
        ctx, scores, N, K, B, bounds, 8, "clu_nan_f32_cs8"
    )
    for cs in [2, 4, 8]:
        _run_and_check_cluster[.bfloat16, nan_rows=True](
            ctx, scores, N, K, B, bounds, cs, String("clu_nan_bf16_cs", cs)
        )
    print("PASS test_topk_cluster_nan_pad")


def test_topk_cluster_stress(ctx: DeviceContext) raises:
    """Many rows, every cluster size, both dtypes, repeated launches.

    The cluster protocol runs one cluster barrier in round 0, whose arrive
    releases only shared::cta writes, then gathers round 0's own histogram
    from every peer; the candidate path's counts travel by `st.async` on
    each rank's mbarrier and the peers' candidate pairs by `st.async` on
    rank 0's, with no closing cluster barrier, and peers exit as soon as
    their columns are out. A race there shows up as a miscounted bin or a
    missing candidate on some row of some launch, not on every one. So each
    single-pass arm is 64 rows with varied bounds (every 16th row a trivial
    `count <= K` row beside full ones), launched three times: twice over
    distinct LCG scores and once over 17 coarse levels, whose plateaus put
    the split on a heavily tied bin cut first-come. A streaming arm repeats
    the two stimuli at CS 2 and 8 over both load paths.
    """
    comptime K = 2048
    comptime N = 49139
    comptime B = 64
    var bounds = List[Int](capacity=B)
    for b in range(B):
        bounds.append(K if b % 16 == 15 else N - (b * 7919) % 16000)
    comptime for dt in [DType.float32, DType.bfloat16]:
        for cs in [2, 3, 4, 5, 6, 8, 10, 16]:
            for rep in range(3):
                var seed = UInt32(0x57E50000 + cs * 16 + rep)
                var scores = _poisoned_bounded_scores(B, N, bounds, seed)
                if rep == 1:
                    for b in range(B):
                        for c in range(bounds[b]):
                            scores[b * N + c] = (
                                Float32((c * 37 + b * 11) % 17) / 16.0
                            )
                _run_and_check_cluster[dt](
                    ctx,
                    scores,
                    N,
                    K,
                    B,
                    bounds,
                    cs,
                    String("clu_stress_cs", cs, "_rep", rep),
                )
        # The streaming flavor under the same stimuli: every slice of a
        # full-length row exceeds the single-pass staging (51872 f32, 103744
        # bf16 elements) at CS 2 and 8, so each round's barrier is followed by
        # a slot refill (TMA) or a fresh global pass, and the scatter streams
        # too.
        comptime NS = 420000 if dt == DType.float32 else 840000
        comptime BS = 12
        var s_bounds = List[Int](capacity=BS)
        for b in range(BS):
            s_bounds.append(K if b % 6 == 5 else NS - (b * 7919) % 4096)
        for cs in [2, 8]:
            for rep in range(2):
                var scores = _poisoned_bounded_scores(
                    BS, NS, s_bounds, UInt32(0x57E5A000 + cs * 16 + rep)
                )
                if rep == 1:
                    for b in range(BS):
                        for c in range(s_bounds[b]):
                            scores[b * NS + c] = (
                                Float32((c * 37 + b * 11) % 17) / 16.0
                            )
                _run_and_check_cluster[dt](
                    ctx,
                    scores,
                    NS,
                    K,
                    BS,
                    s_bounds,
                    cs,
                    String("clu_stress_stream_cs", cs, "_rep", rep),
                )
                _run_and_check_cluster[dt, use_tma=True](
                    ctx,
                    scores,
                    NS,
                    K,
                    BS,
                    s_bounds,
                    cs,
                    String("clu_stress_stream_tma_cs", cs, "_rep", rep),
                )
    print("PASS test_topk_cluster_stress")


def _solo_tie_rows(
    B: Int, N: Int, L: Int, n_above: Int, n_bin: Int, spread: Bool
) -> List[Float32]:
    """Rows of `L` live columns (the rest poisoned) whose threshold bin holds
    `n_bin` columns below `n_above` clear winners, scattered over the row:
    one value (`spread=False`) or distinct f32 values inside one fp16 bin
    (`spread=True`, which bf16 rounds back into one value)."""
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            var p = (c * 7 + b * 13) % L
            if c >= L:
                scores.append(Float32(1.0e30))
            elif p < n_above:
                scores.append(Float32(10.0))
            elif p < n_above + n_bin:
                var d = Float32(p - n_above) * Float32(1.0e-5)
                scores.append(Float32(5.0) + (d if spread else Float32(0.0)))
            else:
                scores.append(Float32(-1.0))
    return scores^


def _solo_saturated_rows(
    B: Int, N: Int, L: Int, n_top: Int, n_low: Int
) -> List[Float32]:
    """Rows of `L` live columns (the rest poisoned) with `n_top` scores of
    `1e20` and `n_low` of `1e10` scattered over the rest of `-1`: both round
    to fp16 infinity, one coarse bin, but differ in their exponent byte."""
    var scores = List[Float32](capacity=B * N)
    for b in range(B):
        for c in range(N):
            var p = (c * 7 + b * 13) % L
            if c >= L:
                scores.append(Float32(1.0e30))
            elif p < n_top:
                scores.append(Float32(1.0e20))
            elif p < n_top + n_low:
                scores.append(Float32(1.0e10))
            else:
                scores.append(Float32(-1.0))
    return scores^


def test_topk_cluster_solo(ctx: DeviceContext) raises:
    """Rows rank 0 selects alone (live length at most 32768 at cluster sizes
    up to 4, 24576 from 5 to 8) and the rows just past it, at every cluster
    size and both dtypes.

    Lengths straddle each payload width (4096-column groups at f32, 8192 at
    bf16; 1, 2, 3, 4, 6 or 8 of them) and the cap; `K` runs from 1 to the
    selection width. The tie arms reach each way the threshold bin
    resolves: an exact fill, the rank on exact keys (bins up to 128 columns,
    equal and distinct keys), and the radix rounds (wider bins, a plateau the
    last round cuts, and a bracket that fills the row in the first round).
    Odd strides keep the row bases unaligned. Strides rank 0 covers take the
    launch without a cluster, through both entry points (bounded and
    unbounded), and the mixed batch puts both paths in one launch. A `K` of
    0 must launch nothing.
    """
    comptime N = 32800
    var lengths: List[Int] = [
        2049,
        4095,
        4096,
        4097,
        8191,
        8192,
        8193,
        12288,
        12289,
        16383,
        16384,
        16385,
        24576,
        24577,
        32767,
        32768,
        32769,
        32800,
    ]
    comptime for dt in [DType.float32, DType.bfloat16]:
        for cs in [2, 4, 8]:
            for k in [1, 100, 2047, 2048]:
                var B = len(lengths)
                var scores = _poisoned_bounded_scores(
                    B, N, lengths, 0x5010_0000 + UInt32(cs * 4099 + k)
                )
                _run_and_check_cluster[dt](
                    ctx,
                    scores,
                    N,
                    k,
                    B,
                    lengths,
                    cs,
                    String("solo_k", k, "_cs", cs),
                )
        comptime K = 2048
        # Every cluster size: each one instantiates its own payload widths,
        # and each needs every arm.
        for cs in [2, 4, 8]:
            for L in [4096, 16384, 32768]:
                var lens = List[Int](length=4, fill=L)
                for shape in [
                    (K - 100, 100),
                    (K - 10, 40),
                    (K - 40, 128),
                    (K - 60, 129),
                    (K - 200, 3000),
                ]:
                    for spread in [False, True]:
                        _run_and_check_cluster[dt](
                            ctx,
                            _solo_tie_rows(4, N, L, shape[0], shape[1], spread),
                            N,
                            K,
                            4,
                            lens,
                            cs,
                            String(
                                "solo_ties_cs",
                                cs,
                                "_",
                                L,
                                "_",
                                shape[0],
                                "_",
                                shape[1],
                                "_",
                                spread,
                            ),
                        )
                # Past fp16's range every score shares the top bin, so the
                # radix rounds see two exponents and the first round's
                # bracket fills the row exactly: the early exit.
                _run_and_check_cluster[dt](
                    ctx,
                    _solo_saturated_rows(4, N, L, K, 1000),
                    N,
                    K,
                    4,
                    lens,
                    cs,
                    String("solo_radix_exit_cs", cs, "_", L),
                )
        # An odd stride: row bases are not 16 B aligned, so the loads take
        # their scalar path on every row but the first.
        comptime NO = 16391
        var odd: List[Int] = [NO, 4097, 12289, 2049]
        _run_and_check_cluster[dt](
            ctx,
            _poisoned_bounded_scores(len(odd), NO, odd, 0x5010_0DD5),
            NO,
            K,
            len(odd),
            odd,
            4,
            "solo_odd_stride",
        )
        # A stride no longer than rank 0's reach launches without the
        # cluster, one CTA per row; the odd one keeps row bases unaligned.
        # At 8 ranks 24576 is the reach and 32768 launches the cluster, its
        # full rows split and its short ones on rank 0.
        # The unbounded arm is `persistent_topk_block_split(row_bounds=None)`'s
        # route there.
        for stride in [16384, 12291, 24576, 32768]:
            var short: List[Int] = [stride, stride - 1, 12289, 4097, 2049, 2048]
            var ss = _poisoned_bounded_scores(
                len(short), stride, short, 0x5010_5000 + UInt32(stride)
            )
            var full = List[Int](length=8, fill=stride)
            var fs = _lcg_scores(8, stride, 0x5010_F000 + UInt32(stride), 100.0)
            for cs in [2, 4, 8]:
                _run_and_check_cluster[dt](
                    ctx,
                    ss,
                    stride,
                    K,
                    len(short),
                    short,
                    cs,
                    String("solo_unclustered_n", stride, "_cs", cs),
                )
                _run_and_check_cluster[dt, use_bounds=False](
                    ctx,
                    fs,
                    stride,
                    K,
                    8,
                    full,
                    cs,
                    String("solo_unbounded_n", stride, "_cs", cs),
                )
        # Mixed: short rows rank 0 takes beside long rows the cluster splits.
        comptime NM = 107232
        var mixed: List[Int] = [
            NM,
            3000,
            32768,
            NM - 5,
            32769,
            9000,
            2049,
            2048,
        ]
        var ms = _poisoned_bounded_scores(len(mixed), NM, mixed, 0x5010_BEEF)
        for cs in [2, 4, 8]:
            _run_and_check_cluster[dt](
                ctx,
                ms,
                NM,
                K,
                len(mixed),
                mixed,
                cs,
                String("solo_mixed_cs", cs),
            )
        # `K = 0` selects nothing: the launcher returns before any launch, so
        # the output is never touched.
        var guard = ctx.enqueue_create_buffer[.int32](4)
        guard.enqueue_fill(Int32(-2))
        var zs = ctx.enqueue_create_buffer[dt](4 * 16384)
        zs.enqueue_fill(Scalar[dt](1))
        topk_cluster_split(
            ctx,
            rebind[ImmPointer[Scalar[dt], ImmutAnyOrigin]](zs.unsafe_ptr()),
            rebind[MutPointer[Int32, MutAnyOrigin]](guard.unsafe_ptr()),
            16384,
            0,
            total_seq_len=4,
            row_bounds=None,
            cluster_size=4,
        )
        with guard.map_to_host() as g:
            for i in range(4):
                assert_true(
                    g[i] == -2, String("K=0 wrote slot ", i, " = ", g[i])
                )
    print("PASS test_topk_cluster_solo")


def test_topk_cluster_sm90_assembles() raises:
    """The cluster kernel assembles for H100 (sm_90a), whose PTX is ISA 8.5.

    The dispatch compiles the cluster tier on SM90, but the GPU tests run on
    SM100, so an SM100-only instruction (such as the PTX 8.6
    `fence...sync_restrict`) would otherwise break every H100 build of
    `mla_index_fp8` unseen. The `object` emission runs ptxas, so this fails
    at test build time. CS 2 and 8 take both arms of the round-0 barrier.
    """
    comptime for dt in [DType.bfloat16, DType.float32]:
        comptime for cs in [2, 8]:
            var obj = _compile_code[
                _topk_cluster_kernel[dt, cs, False],
                emission_kind="object",
                target=get_gpu_target["sm_90a"](),
            ]().asm
            assert_true(obj.byte_length() > 0)
    print("PASS test_topk_cluster_sm90_assembles")


def test_topk_cluster_dispatch(ctx: DeviceContext) raises:
    """The launcher's cluster gate, at the shapes it takes and the one it must
    refuse.

    24 rows at `N=107232, bf16` is the GLM decode shape; 8 rows at `f32` the
    small batch; 74 rows is the gate's edge on B200; 75 must fall through to
    the histogram selects, which serve the same contract and so validate the
    same way -- the point of that arm is that the gate moves nothing else.
    The other shapes reach CS 16, 10, 8, 6, 5 and 3 through the dispatch.

    The row cap and the cluster-size table are pinned first, on B200's 148
    SMs and on other SM counts, so a retune shows up here as a deliberate
    edit.
    """
    assert_equal(_ctopk_max_rows[148](), 74, "row cap at sms=148")
    assert_equal(_ctopk_max_rows[160](), 80, "row cap at sms=160")
    assert_equal(_ctopk_max_rows[132](), 66, "row cap at sms=132")
    # (rows, N, expected cluster size), per SM count.
    var b200: List[Tuple[Int, Int, Int]] = [
        (1, 107232, 16),
        (7, 107232, 16),
        (8, 107232, 8),
        (7, 65536, 8),
        (11, 163840, 10),
        (12, 163840, 8),
        (1, 81920, 10),
        (7, 81920, 10),
        (11, 81920, 10),
        (1, 98304, 16),
        (1, 65536, 8),
        (1, 40960, 8),
        (15, 107232, 8),
        (16, 107232, 6),
        (22, 107232, 6),
        (23, 107232, 5),
        (26, 107232, 5),
        (27, 107232, 4),
        (33, 107232, 4),
        (34, 107232, 3),
        (45, 107232, 3),
        (46, 107232, 2),
        (64, 107232, 2),
        (74, 107232, 2),
    ]
    for t in b200:
        assert_equal(
            _choose_topk_cluster_size[148](t[0], t[1]),
            t[2],
            String("cluster size at rows=", t[0], " N=", t[1], " sms=148"),
        )
    # H100's GPC layout is not modeled: power-of-two sizes, flat estimate.
    var h100: List[Tuple[Int, Int, Int]] = [
        (1, 163840, 8),
        (13, 163840, 8),
        (14, 107232, 4),
        (29, 107232, 4),
        (30, 107232, 2),
    ]
    for t in h100:
        assert_equal(
            _choose_topk_cluster_size[132](t[0], t[1]),
            t[2],
            String("cluster size at rows=", t[0], " N=", t[1], " sms=132"),
        )
    var b300: List[Tuple[Int, Int, Int]] = [
        (1, 163840, 16),
        (16, 107232, 8),
        (17, 107232, 6),
    ]
    for t in b300:
        assert_equal(
            _choose_topk_cluster_size[160](t[0], t[1]),
            t[2],
            String("cluster size at rows=", t[0], " N=", t[1], " sms=160"),
        )

    comptime K = 2048
    comptime for dt in [DType.float32, DType.bfloat16]:
        for shape in [
            (24, 107232),
            (4, 163840),
            (10, 163840),
            (8, 107232),
            (16, 107232),
            (40, 65536),
            (8, 32769),
            (74, 12288),
            (75, 12288),
            (71, 32771),
        ]:
            var B = shape[0]
            var N = shape[1]
            var bounds = List[Int](length=B, fill=N)
            var scores = _poisoned_bounded_scores(
                B, N, bounds, 0xD15A0000 + UInt32(B * 7919)
            )
            _run_and_check_cluster[dt, via_dispatch=True](
                ctx,
                scores,
                N,
                K,
                B,
                bounds,
                4,
                String("dispatch_b", B),
            )
        # The bounded arm through the dispatch: bounds vary per row, so the
        # cluster's slice split must clamp at each row's live length.
        var B = 24
        var N = 107232
        var bounds = _cluster_bounds(N, K)
        while len(bounds) < B:
            bounds.append(N)
        var scores = _poisoned_bounded_scores(B, N, bounds, 0xD15AB0DE)
        _run_and_check_cluster[dt, via_dispatch=True](
            ctx, scores, N, K, B, bounds, 4, "dispatch_bounded"
        )
    print("PASS test_topk_cluster_dispatch")


def test_topk_cluster_graph_capture(ctx: DeviceContext) raises:
    """A captured device graph replays the cluster launch with its attributes.

    Serving runs the indexer under device-graph capture, so the cluster path
    is only usable if the launch attributes (the cluster dimension, the
    shared-memory opt-in) survive capture and replay. One graph per
    streaming load path -- the pipeline's capture is the arm that caught the
    env-reading launcher, so it is pinned rather than assumed. Two replays
    of each graph, each validated on its own: the nd contract does not
    promise the same tie cut twice, so the replays are checked against the
    predicate rather than against each other.
    """
    comptime K = 2048
    comptime B = 8
    # Two graphs, one per streaming load path. The plain arm's odd `N` keeps
    # its stride unpadded, the TMA arm's padded one reaches the pipeline --
    # and each must survive capture, so both variants' launch (attributes,
    # mbarriers' shared-memory request and all) are held to replay.
    comptime for N in [32769, 900000]:
        var bounds = _cluster_bounds(N, K)
        var scores = _poisoned_bounded_scores(
            B, N, bounds, UInt32(0x6AFC0000 + N)
        )

        var scores_dev = ctx.enqueue_create_buffer[.bfloat16](B * N)
        var idxs_dev = ctx.enqueue_create_buffer[.int32](B * K)
        var bounds_dev = ctx.enqueue_create_buffer[.int32](B)
        with scores_dev.map_to_host() as buf:
            for i in range(B * N):
                buf[i] = scores[i].cast[.bfloat16]()
        with bounds_dev.map_to_host() as buf:
            for b in range(B):
                buf[b] = Int32(bounds[b])

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            topk_cluster_split[RB=RowBoundsPtr](
                builder.recording_context(),
                rebind[ImmPointer[BFloat16, ImmutAnyOrigin]](
                    scores_dev.unsafe_ptr()
                ),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_dev.unsafe_ptr()),
                N,
                K,
                total_seq_len=B,
                row_bounds=Optional(
                    RowBoundsPtr(
                        rebind[ImmPointer[Int32, ImmutAnyOrigin]](
                            bounds_dev.unsafe_ptr()
                        )
                    )
                ),
                cluster_size=8,
                use_tma=N % 8 == 0,
            )

        var graph = DeviceGraph.create(ctx, build)
        var seen_scores = _as_read[.bfloat16](scores)
        for replay in range(2):
            idxs_dev.enqueue_fill(Int32(-2))
            graph.replay()
            ctx.synchronize()
            var idxs_host = ctx.enqueue_create_host_buffer[.int32](B * K)
            ctx.enqueue_copy(dst_buf=idxs_host, src_buf=idxs_dev)
            ctx.synchronize()
            for b in range(B):
                var bound = bounds[b]
                var live = min(K, bound)
                for k in range(live, K):
                    assert_true(
                        Int(idxs_host[b * K + k]) == -1,
                        String(
                            "N ",
                            N,
                            " graph replay ",
                            replay,
                            " row ",
                            b,
                            " tail slot ",
                            k,
                            " is not -1",
                        ),
                    )
                if live == 0:
                    continue
                var row_scores = List[Float32](capacity=bound)
                for i in range(bound):
                    row_scores.append(seen_scores[b * N + i])
                var row_idxs = List[Int](capacity=live)
                for k in range(live):
                    row_idxs.append(Int(idxs_host[b * K + k]))
                _check_score_valid_row(
                    row_scores,
                    row_idxs,
                    bound,
                    live,
                    String("graph_replay", replay),
                    b,
                )
            _ = idxs_host
    print("PASS test_topk_cluster_graph_capture")


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
        test_unordered_resident_band(ctx)
        test_phi16_identity()
        test_unord_nd_predicate_bites()
        test_unordered_wide_payload_band(ctx)
        test_unordered_prime_lengths(ctx)
        test_unordered_k_edges(ctx)
        test_unordered_row_counts(ctx)
        test_unordered_all_equal(ctx)
        test_unordered_plateau_straddles_k(ctx)
        test_unordered_plateau_edges(ctx)
        test_unordered_k1_all_equal(ctx)
        test_unordered_two_valued(ctx)
        test_unordered_masked_tail(ctx)
        test_unordered_negative_and_extremes(ctx)
        test_unordered_sorted_inputs(ctx)
        test_unordered_beyond_resident(ctx)
        test_unordered_small_n(ctx)
        test_bounded_2048_kernel(ctx)
        test_bounded_resident(ctx)
        test_bounded_resident_bin_digit(ctx)
        test_bounded_histsel_prefetch(ctx)
        test_bounded_histsel_fills_gpu(ctx)
        test_bounded_histsel_select_all_tail(ctx)
        test_bf16_block_multirow(ctx)
        test_bf16_resident(ctx)
        test_bf16_histsel(ctx)
        test_bf16_bounded(ctx)
        test_bf16_rounding_plateau(ctx)
        test_bf16_unord_nd_streaming(ctx)
        test_bf16_negative_plateau_streaming(ctx)
        test_bf16_negative_plateau_resident(ctx)
        test_bf16_negative_levels_masked(ctx)
        test_short_row_beside_long_row_bf16(ctx)
        comptime if (
            ctx.target._is_sm_9x_or_newer()
            and not ctx.target._is_sm_120x_or_newer()
        ):
            print("\n--- cluster-cooperative top-k (SM90+, non-SM12x) ---")
            test_topk_cluster_band(ctx)
            test_topk_cluster_ties(ctx)
            test_topk_cluster_candidates(ctx)
            test_topk_cluster_nan_pad(ctx)
            test_topk_cluster_stress(ctx)
            test_topk_cluster_solo(ctx)
            test_topk_cluster_dispatch(ctx)
            test_topk_cluster_graph_capture(ctx)
            test_topk_cluster_sm90_assembles()
    print("ALL TESTS PASSED")
