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

"""Host-side pins for the MLA decode split-K partition choice and the runtime
head-count binding that reaches it.

Split boundaries change online softmax and probability rounding, so bit-exact
oracles must match the split count. These host-side pins also catch performance
policy drift and runtime head-count binding errors that direct comptime kernel
tests cannot exercise.
"""

from std.math import ceildiv
from std.sys import get_defined_int

from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    _LAYOUT_G_DIRECT_HALF_ROWS_COST,
    _sparse_layout_e_cost_at_np,
    _select_sparse_layout_e_launch,
    _SparseLayoutEModels,
    _SplitCost,
    _NO_LAYOUT_E_MODELS,
    _sparse_layout_e_models,
    _sparse_layout_e_incumbent_variant,
    _sparse_pair_eligible_form,
    _SPARSE_BF16_Q_FP8_KV_COST,
    _SPARSE_E_FP8_COST,
    _SPARSE_PAIR_MAX_DIRECT_WALK,
    MLA_SPARSE_KV_FORM_NVFP4,
    MLA_SPARSE_KV_FORM_PLAIN,
    MLA_SPARSE_KV_FORM_SCALED_FP8,
    _NUM_PARTITION_BUCKETS,
    _bucket_num_partitions,
    _config_cost_partition_bucket,
    _compact_partition_bucket,
    _compute_num_partitions,
    _get_partition_bucket,
    _relax_sparse_split_floor,
    _layout_g_partition_bucket,
    _mla_decode_split,
    compute_mla_dispatch_key_runtime,
    compute_mla_dispatch_scalars,
    compute_mla_dispatch_scalars_runtime,
)

comptime SM_COUNT = 148
comptime HALF_SMS = SM_COUNT // 2


def _expect(what: StringLiteral, got: Int, want: Int) raises:
    if got != want:
        raise Error(what, ": got ", got, ", want ", want)


def _is_bucket(num_partitions: Int) -> Bool:
    comptime for i in range(_NUM_PARTITION_BUCKETS):
        if _get_partition_bucket[HALF_SMS, i]() == num_partitions:
            return True
    return False


def _expect_bucket(
    what: StringLiteral, num_partitions: Int, batch: Int, cache: Int, q: Int
) raises:
    if not _is_bucket(num_partitions):
        raise Error(
            what,
            " chose num_partitions=",
            num_partitions,
            " which has no compiled combine kernel (batch=",
            batch,
            " cache=",
            cache,
            " q=",
            q,
            ")",
        )


def test_bucket_tables_agree() raises:
    """The bucket table is written out three times and nothing keeps the copies
    in sync: as a literal in `_get_partition_bucket`, again in
    `_bucket_num_partitions`, and as the count `_NUM_PARTITION_BUCKETS`. A
    divergence reproduces the unreduced-output bug the dispatch now raises on.

    These properties pin the three to each other without restating the table,
    so the test cannot drift along with it.
    """
    comptime for i in range(_NUM_PARTITION_BUCKETS):
        var b = _get_partition_bucket[HALF_SMS, i]()
        # A bucket must map to itself. If `_bucket_num_partitions` is missing
        # this entry it rounds up to the next one instead.
        _expect(
            "bucket does not map to itself",
            _bucket_num_partitions[HALF_SMS](b),
            b,
        )

    comptime for i in range(1, _NUM_PARTITION_BUCKETS):
        var prev = _get_partition_bucket[HALF_SMS, i - 1]()
        var b = _get_partition_bucket[HALF_SMS, i]()
        if prev >= b:
            raise Error(
                "bucket table is not strictly ascending at index ",
                i,
                ": ",
                prev,
                " >= ",
                b,
            )
        # One past the previous bucket must land on this one. Catches the other
        # direction: an entry present only in `_bucket_num_partitions`.
        _expect(
            "wrong bucket for prev+1",
            _bucket_num_partitions[HALF_SMS](prev + 1),
            b,
        )

    # A stale count leaves the last checked entry short of the SM-adaptive one.
    _expect(
        "last bucket is not half_sms",
        _get_partition_bucket[HALF_SMS, _NUM_PARTITION_BUCKETS - 1](),
        HALF_SMS,
    )

    _expect("floor", _bucket_num_partitions[HALF_SMS](0), 1)
    _expect("floor", _bucket_num_partitions[HALF_SMS](1), 1)
    _expect(
        "clamp above the top bucket",
        _bucket_num_partitions[HALF_SMS](HALF_SMS + 1),
        HALF_SMS,
    )


def _check_every_path(batch: Int, cache: Int, q: Int, page: Int) raises:
    """Every head count, both split-K policies, both floor policies."""
    for r in range(2):
        var relax = r == 1
        for c in range(2):
            var cost = c == 1
            _expect_bucket(
                "h8",
                _compute_num_partitions[8, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h8 folded",
                _compute_num_partitions[8, True, HALF_SMS, True](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h8 bf16 kv",
                _compute_num_partitions[8, False, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h16",
                _compute_num_partitions[16, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h64",
                _compute_num_partitions[64, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h128",
                _compute_num_partitions[128, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )


def test_every_split_has_a_combine_kernel() raises:
    """Host mirror of the dispatch's runtime guard.

    A split outside the bucket set launches the decode and never reduces its
    partial outputs, which is indistinguishable from a correct result. The
    guard catches that at runtime on the shapes something happens to run; this
    catches it over a sweep, including prime lengths no production config uses.
    """
    var batches = [1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 128]
    var caches = [251, 512, 1021, 1536, 2048, 3072, 4099]
    var qs = [1, 2, 3, 5, 6, 7]
    var pages = [64, 128]
    for bi in range(len(batches)):
        for ci in range(len(caches)):
            for qi in range(len(qs)):
                for pi in range(len(pages)):
                    _check_every_path(
                        batches[bi], caches[ci], qs[qi], pages[pi]
                    )


def _np_h8(
    batch: Int, cache: Int, q: Int, page: Int, cost_optimal: Bool
) -> Int:
    return _compute_num_partitions[8, True, HALF_SMS, False](
        batch, cache, q, page, SM_COUNT, cost_optimal_split=cost_optimal
    )


def test_cost_model_split_is_pinned() raises:
    """Pin the shapes the cost model was fitted on.

    The dispatch reaches this path only for an unfolded sparse spec-decode
    verify, so `q > 1` throughout; `q = 1` is gated off at the call site and is
    deliberately not pinned here. Values are measured, not derived.
    """
    # The production shape. `cost_optimal_split` is what changes the answer.
    _expect("production shape, cost model on", _np_h8(8, 2048, 6, 64, True), 3)
    _expect(
        "production shape, cost model off", _np_h8(8, 2048, 6, 64, False), 8
    )

    # Batch curve at the production cache and q. Splitting stops paying as the
    # batch alone fills the machine.
    _expect("batch 1", _np_h8(1, 2048, 6, 64, True), 20)
    _expect("batch 2", _np_h8(2, 2048, 6, 64, True), 9)
    _expect("batch 4", _np_h8(4, 2048, 6, 64, True), 4)
    _expect("batch 16", _np_h8(16, 2048, 6, 64, True), 3)
    _expect("batch 32", _np_h8(32, 2048, 6, 64, True), 3)
    _expect("batch 128", _np_h8(128, 2048, 6, 64, True), 1)

    # q curve at the production batch: more query positions means more CTAs per
    # split, so the affordable split shrinks.
    _expect("q 2", _np_h8(8, 2048, 2, 64, True), 9)
    _expect("q 3", _np_h8(8, 2048, 3, 64, True), 4)
    _expect("q 5", _np_h8(8, 2048, 5, 64, True), 3)
    _expect("q 7", _np_h8(8, 2048, 7, 64, True), 2)
    _expect("q 8", _np_h8(8, 2048, 8, 64, True), 2)

    # The model minimises over gathered tokens, so page size does not enter it.
    _expect("page 128, cache 512", _np_h8(8, 512, 6, 128, True), 3)
    _expect("page 128, cache 2048", _np_h8(8, 2048, 6, 128, True), 3)
    _expect("page 128, cache 4096", _np_h8(8, 4096, 6, 128, True), 3)


def _layout_g_pick(batch: Int, split_len: Int, page: Int, q: Int = 6) -> Int:
    return _config_cost_partition_bucket[
        HALF_SMS, _LAYOUT_G_DIRECT_HALF_ROWS_COST
    ](q * batch, split_len, page, SM_COUNT, HALF_SMS)


def test_layout_g_cost_split_is_pinned() raises:
    """The routed half-row Layout G kernel's split, from its own cost record.

    The picks match the P7b forced-np sweep's best `np` at every measured
    cell; re-pin them when the record is refitted. Every pick must be a
    bucket.
    """
    var batches = [1, 2, 4, 8, 16, 32, 64]
    var want_2048 = [16, 8, 4, 3, 1, 1, 1]
    var want_4096 = [16, 8, 4, 3, 3, 2, 1]
    for page in [128, 64]:
        for i in range(len(batches)):
            var b = batches[i]
            var np2 = _layout_g_pick(b, 2048, page)
            _expect_bucket("layout G cost (top_k 2048)", np2, b, 2048, 6)
            _expect("layout G cost, top_k 2048", np2, want_2048[i])
            var np4 = _layout_g_pick(b, 4096, page)
            _expect_bucket("layout G cost (top_k 4096)", np4, b, 4096, 6)
            _expect("layout G cost, top_k 4096", np4, want_4096[i])
    _expect("layout G cost, empty split", _layout_g_pick(8, 0, 128), 1)
    _expect(
        "layout G cost, max_partitions 1",
        _config_cost_partition_bucket[
            HALF_SMS, _LAYOUT_G_DIRECT_HALF_ROWS_COST
        ](48, 2048, 128, SM_COUNT, 1),
        1,
    )
    # A cache shorter than top_k: 1006 keys are 8 pages, not 16.
    _expect("layout G cost, short cache", _layout_g_pick(8, 1006, 128), 3)
    # Where a second wave of the non-split kernel competes with np = 2: the
    # fitted record keeps np = 1 at B 28-37 (top_k 2048) and np = 2 at
    # B 25-27 and B 50-61 (top_k 4096).
    # Columns: batch, top_k, expected split count.
    var band: List[Tuple[Int, Int, Int]] = [
        (24, 2048, 1),
        (27, 2048, 2),
        (28, 2048, 1),
        (37, 2048, 1),
        (38, 2048, 1),
        (49, 4096, 1),
        (50, 4096, 2),
        (61, 4096, 2),
        (62, 4096, 1),
    ]
    for i in range(len(band)):
        var b = band[i][0]
        var k = band[i][1]
        _expect(
            "layout G cost, np = 1 band", _layout_g_pick(b, k, 128), band[i][2]
        )


def test_page_equivalent_split_buckets() raises:
    _expect(
        "top-k 2048, nine splits leave one empty",
        _compact_partition_bucket[HALF_SMS](9, 2048, 128),
        8,
    )
    _expect(
        "17 pages need nine productive splits",
        _compact_partition_bucket[HALF_SMS](16, 17 * 128, 128),
        9,
    )
    for page in [64, 128]:
        for pages in range(1, 65):
            comptime for i in range(_NUM_PARTITION_BUCKETS):
                var np = _get_partition_bucket[HALF_SMS, i]()
                var compact = _compact_partition_bucket[HALF_SMS](
                    np, pages * page, page
                )
                _expect_bucket("page-equivalent split", compact, 1, pages, 1)
                _expect(
                    "partition boundaries preserved",
                    ceildiv(pages, compact),
                    ceildiv(pages, np),
                )
                if (np == 1) != (compact == 1) or compact > np:
                    raise Error("page compaction changed the kernel class")


def test_topk2048_split_policy() raises:
    """The top-k 2048 record keeps one split from B = 24 through 28 at q = 6,
    where the general record splits B = 25 to 27 in two; other shapes keep the
    general record.
    """
    var batches = [8, 16, 25, 26, 27, 28, 32, 64]
    var expected = [3, 1, 1, 1, 1, 1, 1, 1]
    var general = [3, 1, 2, 2, 2, 1, 1, 1]
    for i in range(len(batches)):
        var b = batches[i]
        _expect(
            "top-k 2048",
            _layout_g_partition_bucket[HALF_SMS](
                6 * b,
                2048,
                128,
                SM_COUNT,
                HALF_SMS,
                top_k=2048,
                cache_len=32768,
            ),
            expected[i],
        )
        _expect(
            "general record at top-k 2048",
            _layout_g_pick(b, 2048, 128),
            general[i],
        )
        for length in [256, 512, 1006, 2047, 4096, 8192]:
            _expect(
                "other lengths and cache clamping keep the general record",
                _layout_g_partition_bucket[HALF_SMS](
                    6 * b,
                    length,
                    128,
                    SM_COUNT,
                    HALF_SMS,
                    top_k=2048,
                    cache_len=32768,
                ),
                _layout_g_pick(b, length, 128),
            )
        _expect(
            "other top-k values keep the general record",
            _layout_g_partition_bucket[HALF_SMS](
                6 * b,
                2048,
                128,
                SM_COUNT,
                HALF_SMS,
                top_k=4096,
                cache_len=32768,
            ),
            _layout_g_pick(b, 2048, 128),
        )
        for cache in [0, 1024, 2048, 4096, 6143]:
            _expect(
                "shorter and unknown caches keep the general record",
                _layout_g_partition_bucket[HALF_SMS](
                    6 * b,
                    2048,
                    128,
                    SM_COUNT,
                    HALF_SMS,
                    top_k=2048,
                    cache_len=cache,
                ),
                _layout_g_pick(b, 2048, 128),
            )
        for cache in [6144, 8192, 16384]:
            _expect(
                "caches from 6144 take the top-k 2048 record",
                _layout_g_partition_bucket[HALF_SMS](
                    6 * b,
                    2048,
                    128,
                    SM_COUNT,
                    HALF_SMS,
                    top_k=2048,
                    cache_len=cache,
                ),
                expected[i],
            )


def test_multi_head_group_ignores_the_cost_model() raises:
    """The 128-head path documents that it ignores `cost_optimal_split`. That
    claim is otherwise only prose, and the flag is threaded through the shared
    routing function.
    """
    var batches = [1, 2, 8, 16, 32, 128]
    var caches = [251, 512, 2048, 4099]
    for bi in range(len(batches)):
        for ci in range(len(caches)):
            var off = _compute_num_partitions[128, True, HALF_SMS, False](
                batches[bi],
                caches[ci],
                1,
                64,
                SM_COUNT,
                cost_optimal_split=False,
            )
            var on = _compute_num_partitions[128, True, HALF_SMS, False](
                batches[bi],
                caches[ci],
                1,
                64,
                SM_COUNT,
                cost_optimal_split=True,
            )
            if off != on:
                raise Error(
                    (
                        "multi-head-group split changed with the cost model at"
                        " batch="
                    ),
                    batches[bi],
                    " cache=",
                    caches[ci],
                    ": ",
                    off,
                    " vs ",
                    on,
                )


# Every head count `compute_mla_dispatch_scalars_runtime` enumerates.
comptime _RUNTIME_HEAD_COUNTS = [8, 12, 16, 24, 32, 48, 64, 128]


def _probe_shapes() -> List[Tuple[Int, Int, Int]]:
    """`(batch, cache, q)` triples that spread the head counts apart.

    Within one head group `num_heads` reaches the split count only through the
    spec-decode fold predicate `num_heads * q <= 64`, so every count collapses
    onto the same answer at `q = 1` and a decode-shaped probe would pin
    nothing. The last shape is the one where `is_fp8_kv` moves the np=1 clamp.
    """
    return [
        (4, 8192, 4),
        (8, 8192, 2),
        (2, 8192, 8),
        (4, 8192, 5),
        (64, 256, 1),
    ]


def _comptime_np_table[is_fp8_kv: Bool]() -> List[Int]:
    """Split count per (head count, probe shape), head-count major."""
    var shapes = _probe_shapes()
    var table = List[Int]()
    comptime for num_heads in _RUNTIME_HEAD_COUNTS:
        for si in range(len(shapes)):
            table.append(
                compute_mla_dispatch_scalars[num_heads, is_fp8_kv=is_fp8_kv](
                    shapes[si][0], shapes[si][1], shapes[si][2], SM_COUNT
                )[2]
            )
    return table^


def _head_counts() -> List[Int]:
    """`_RUNTIME_HEAD_COUNTS` as a runtime list, for the pairwise sweep."""
    var out = List[Int]()
    for num_heads in materialize[_RUNTIME_HEAD_COUNTS]():
        out.append(num_heads)
    return out^


def _expect_at(
    what: StringLiteral,
    num_heads: Int,
    is_fp8_kv: Bool,
    shape: Tuple[Int, Int, Int],
    got: Int,
    want: Int,
) raises:
    if got != want:
        raise Error(
            what,
            " at num_heads=",
            num_heads,
            " is_fp8_kv=",
            is_fp8_kv,
            " (batch=",
            shape[0],
            " cache=",
            shape[1],
            " q=",
            shape[2],
            "): got ",
            got,
            ", want ",
            want,
        )


def test_runtime_binding_matches_its_own_specialization() raises:
    """The runtime entry point hand-writes one arm per head count, and nothing
    else checks that an arm forwards to the specialization it names.

    The GPU tests cannot: they build `MLADispatchScalarArgs[num_heads=...]`,
    which calls `compute_mla_dispatch_scalars` at comptime and never crosses
    the runtime enumeration. A copy-paste slip binding a count to a
    neighbour's instantiation would leave them all green.
    """
    var shapes = _probe_shapes()
    comptime for is_fp8_kv in [False, True]:
        var want = _comptime_np_table[is_fp8_kv]()
        var i = 0
        comptime for num_heads in _RUNTIME_HEAD_COUNTS:
            for si in range(len(shapes)):
                var shape = shapes[si]
                var got = compute_mla_dispatch_scalars_runtime(
                    shape[0], shape[1], shape[2], num_heads, is_fp8_kv, SM_COUNT
                )
                _expect_at(
                    "batch_size", num_heads, is_fp8_kv, shape, got[0], shape[0]
                )
                _expect_at(
                    "q_max_seq_len",
                    num_heads,
                    is_fp8_kv,
                    shape,
                    got[1],
                    shape[2],
                )
                _expect_at(
                    "num_partitions",
                    num_heads,
                    is_fp8_kv,
                    shape,
                    got[2],
                    want[i],
                )
                i += 1


def _separable_in_principle(a: Int, b: Int) -> Bool:
    """Whether any input at all can tell two head counts apart.

    Across head groups `ceildiv(num_heads, 64)` scales the CTA count. Within
    one, `num_heads` reaches the split count only through the fold predicate
    `num_heads * q <= 64`, so two counts on the same side of it for every `q`
    return identical metadata for every input, forever. Beyond `q = 64` both
    products exceed 64 for any positive count, so the sweep is exhaustive.
    """
    if ceildiv(a, 64) != ceildiv(b, 64):
        return True
    for q in range(1, 65):
        if (a * q <= 64) != (b * q <= 64):
            return True
    return False


def test_the_binding_pin_is_not_vacuous() raises:
    """Shapes that map every head count onto one answer would pass the pin
    above whatever the arms are bound to.

    So hold the probe set to the separation the heuristic actually admits:
    every pair it can tell apart, these shapes must tell apart. The pairs it
    cannot -- 24/32 and 48/64, which sit on the same side of the fold predicate
    for every `q` -- must stay indistinguishable, or the reasoning in
    `_separable_in_principle` has gone stale. A mis-binding within such a pair
    returns identical metadata for every input and is unobservable here.
    """
    var shapes = _probe_shapes()
    var n = len(shapes)
    var bf16 = _comptime_np_table[False]()
    var fp8 = _comptime_np_table[True]()

    var fp8_matters = False
    for i in range(len(bf16)):
        if bf16[i] != fp8[i]:
            fp8_matters = True
    if not fp8_matters:
        raise Error(
            "no probe shape separates is_fp8_kv, so the fp8 half of the"
            " binding pin proves nothing"
        )

    var counts = _head_counts()
    for a in range(len(counts)):
        for b in range(a + 1, len(counts)):
            var separated = False
            for si in range(n):
                if (
                    bf16[a * n + si] != bf16[b * n + si]
                    or fp8[a * n + si] != fp8[b * n + si]
                ):
                    separated = True
            var expected = _separable_in_principle(counts[a], counts[b])
            if separated != expected:
                raise Error(
                    "num_heads=",
                    counts[a],
                    " vs ",
                    counts[b],
                    ": probe shapes ",
                    "separate" if separated else "do not separate",
                    " them, but the heuristic ",
                    "can" if expected else "cannot",
                )


def test_unenumerated_head_count_raises() raises:
    """The enumeration is deliberately partial, so the fall-through is a
    supported outcome and has to stay an error rather than a silent answer.
    """
    var raised = False
    try:
        _ = compute_mla_dispatch_scalars_runtime(1, 1024, 1, 7, False, SM_COUNT)
    except:
        raised = True
    if not raised:
        raise Error("unenumerated num_heads=7 returned instead of raising")


def _np_sparse_2048[num_heads: Int](batch: Int, cache: Int, q: Int) -> Int:
    """The dispatch's split for a plain fixed top-k 2048 sparse FP8 decode."""
    var relax = _relax_sparse_split_floor[num_heads](batch, q, 2048, cache)
    var split_len = min(2048, cache + q) if relax else 2048
    return _compute_num_partitions[num_heads, True, HALF_SMS, False](
        batch, split_len, q, 128, SM_COUNT, relax_split_floor=relax
    )


def test_layout_e_relaxed_split_is_pinned() raises:
    """Layout E at 64 heads drops the page floor at q 1 and 2 for any cache of
    at least 1024 tokens; Layout G keeps the KERN-3217 scope.
    """
    _expect("h64 q1 B1 cache 32768", _np_sparse_2048[64](1, 32768, 1), 16)
    _expect("h64 q1 B4 cache 32768", _np_sparse_2048[64](4, 32768, 1), 16)
    _expect("h64 q1 B8 cache 32768", _np_sparse_2048[64](8, 32768, 1), 16)
    _expect("h64 q2 B4 cache 32768", _np_sparse_2048[64](4, 32768, 2), 16)
    _expect("h64 q2 B8 cache 32768", _np_sparse_2048[64](8, 32768, 2), 9)
    _expect("h64 q1 B8 cache 2048", _np_sparse_2048[64](8, 2048, 1), 16)
    # Out of scope: the page floor of 4 holds.
    _expect("h64 q1 B16 cache 32768", _np_sparse_2048[64](16, 32768, 1), 4)
    _expect("h64 q3 B8 cache 32768", _np_sparse_2048[64](8, 32768, 3), 4)
    _expect("h64 q1 B8 cache 512", _np_sparse_2048[64](8, 512, 1), 4)
    _expect("h8 q1 B8 cache 32768", _np_sparse_2048[8](8, 32768, 1), 4)
    _expect("h8 q1 B8 cache 2048", _np_sparse_2048[8](8, 2048, 1), 16)
    _expect(
        "h128 q1 B8 cache 32768 relax",
        Int(_relax_sparse_split_floor[128](8, 1, 2048, 32768)),
        0,
    )


def _np_h128_sparse[q_type: DType](batch: Int, cache: Int) -> Int:
    """The 128-head dispatch's split for a plain fixed top-k 2048 sparse
    decode over a unit-scale FP8 cache at q 1.
    """
    return _np_sparse_page[q_type, DType.float8_e4m3fn, 128](
        batch, 1, cache, 128
    )


def _np_sparse_page[
    q_type: DType, kv_type: DType, heads: Int
](batch: Int, q: Int, cache: Int, page: Int) -> Int:
    """The dispatch's split for a plain fixed top-k 2048 sparse decode over
    an unquantized `kv_type` cache, at a KV `page` size.
    """
    return _mla_decode_split[
        q_type,
        kv_type,
        False,
        False,
        heads,
        HALF_SMS,
        sparse=True,
        _is_cache_length_accurate=False,
        per_token_scale_rope_aware=False,
        fold_shared_index=False,
        has_extra_k=False,
        fixed_topk=True,
        skip_masked_tiles=False,
    ](batch, q, cache, 2048, 0, page, SM_COUNT)[0]


def test_h128_relaxed_clamp_follows_the_cost_record() raises:
    """At 128 heads the FP8 Q record sizes its split from the top-k, across
    the whole 9-page corner of caches 1024 to 1151. The BF16 Q record over
    an FP8 cache keeps the relaxed clamp.
    """
    for batch in range(1, 9):
        for cache in [1024, 1088, 1151, 1152, 2048]:
            _expect(
                "h128 fp8 q",
                _np_h128_sparse[DType.float8_e4m3fn](batch, cache),
                _config_cost_partition_bucket[HALF_SMS, _SPARSE_E_FP8_COST](
                    2 * batch, 2048, 128, SM_COUNT, HALF_SMS
                ),
            )
            _expect(
                "h128 bf16 q over fp8",
                _np_h128_sparse[DType.bfloat16](batch, cache),
                _config_cost_partition_bucket[
                    HALF_SMS, _SPARSE_BF16_Q_FP8_KV_COST
                ](2 * batch, min(2048, cache + 1), 128, SM_COUNT, HALF_SMS),
            )
    # Not vacuous: the clamp moves both records at cache 1024 and B 1.
    _expect("fp8 q B1 1024", _np_h128_sparse[DType.float8_e4m3fn](1, 1024), 8)
    _expect(
        "fp8 q relaxed B1 1024",
        _config_cost_partition_bucket[HALF_SMS, _SPARSE_E_FP8_COST](
            2, 1025, 128, SM_COUNT, HALF_SMS
        ),
        9,
    )
    if _config_cost_partition_bucket[HALF_SMS, _SPARSE_BF16_Q_FP8_KV_COST](
        2, 2048, 128, SM_COUNT, HALF_SMS
    ) == _np_h128_sparse[DType.bfloat16](1, 1024):
        raise Error("bf16 q over fp8 B1 1024: the clamp moved nothing")


def _expect_key(
    what: StringLiteral,
    is_fp8_kv: Bool,
    batch: Int,
    q: Int,
    cache: Int,
    want_np: Int,
    want_pair: Int,
    *,
    kv_form: Int = MLA_SPARSE_KV_FORM_PLAIN,
    stride: Int = 2048,
    page_size: Int = 128,
    fixed_topk: Bool = True,
) raises:
    var key = compute_mla_dispatch_key_runtime(
        batch,
        cache,
        q,
        128,
        is_fp8_kv,
        stride,
        SM_COUNT,
        kv_form=kv_form,
        page_size=page_size,
        fixed_topk=fixed_topk,
        depth=0,
    )
    _expect(what, key[2], want_np)
    _expect(what, key[3], want_pair)


def test_sparse_graph_key_follows_the_launch() raises:
    """Pins the legacy key policy when the native caller omits depth eligibility.

    The 128-head sparse capture key follows the launch the dispatch makes,
    not the dense split of the cache length, which the sparse graph ops never
    launch: one split at most 512 keys, the cost-model split of the top-k
    above, and the pair only when its critical CTA walks past the form's route
    threshold (three 128-key tiles for FP8, half a tile for BF16). Only the
    scaled FP8 and NVFP4 forms take the relaxed split, and the BF16 pair's
    plain decode takes the one-wave split instead of the record's.
    """
    _expect_key("fp8 B1 q1 32768", True, 1, 1, 32768, 8, 0)
    _expect_key("fp8 B1 q1 256", True, 1, 1, 256, 1, 0)
    _expect_key("fp8 B1 q1 448", True, 1, 1, 448, 1, 1)
    _expect_key("fp8 B1 q1 1024", True, 1, 1, 1024, 8, 0)
    _expect_key("fp8 B8 q1 1536", True, 8, 1, 1536, 8, 0)
    _expect_key("fp8 B10 q1 1024", True, 10, 1, 1024, 4, 0)
    _expect_key("fp8 B10 q1 32768", True, 10, 1, 32768, 4, 1)
    _expect_key("fp8 B64 q1 32768", True, 64, 1, 32768, 1, 1)
    _expect_key("fp8 B64 q1 300", True, 64, 1, 300, 1, 0)
    _expect_key("fp8 B4 q2 32768", True, 4, 2, 32768, 8, 0)
    _expect_key("fp8 B16 q4 32768", True, 16, 4, 32768, 1, 1)
    # BF16 takes the pair at the shipped default (`MLA_SPARSE_CG2_BF16=1`):
    # its route threshold is 64 keys per CTA, so every BF16 cell here keeps
    # the pair. `-D MLA_SPARSE_CG2_BF16=0` is the BF16 off switch.
    comptime bf16_pair = (
        get_defined_int["MLA_SPARSE_CG2_BF16", 1]() != 0
        and get_defined_int["MLA_SPARSE_CG2", 1]() != 0
        and get_defined_int["MLA_SPARSE_BN_QK", 0]() != 64
    )
    comptime want_bf16_pair = Int(bf16_pair)
    _expect_key("bf16 B1 q1 32768", False, 1, 1, 32768, 16, want_bf16_pair)
    _expect_key("bf16 B1 q1 256", False, 1, 1, 256, 1, want_bf16_pair)
    _expect_key("bf16 B10 q1 32768", False, 10, 1, 32768, 4, want_bf16_pair)
    # The BF16 pair's plain decode splits only as far as one wave holds, where
    # its fitted record would buy a second half-wave and a combine launch: 96
    # and 192 CTAs pick one split where the record picks three and two. FP8
    # keeps the record (its pair is 4.9% faster at the record's split).
    comptime pair_one_wave = Int(
        bf16_pair and get_defined_int["MLA_SPARSE_CG2_ROUTE", 1]() != 0
    )
    if pair_one_wave != 0:
        _expect_key("bf16 B8 q6 32768", False, 8, 6, 32768, 1, want_bf16_pair)
        _expect_key("bf16 B16 q6 32768", False, 16, 6, 32768, 1, want_bf16_pair)
    else:
        _expect_key("bf16 B8 q6 32768", False, 8, 6, 32768, 3, want_bf16_pair)
        _expect_key("bf16 B16 q6 32768", False, 16, 6, 32768, 2, want_bf16_pair)
    # A 64-token page moves no 128-head key pick: the FP8 record sizes its
    # split from the top-k, and the BF16 record sweeps to the same picks at
    # both pages (pinned at the cell below), while the scaled forms' dense
    # heuristic never reads the KV page. Where the page still moves a split
    # pick, and a break in the plumbing would fail a test, is pinned right
    # after this test.
    _expect_key("fp8 B1 q1 1536 page 64", True, 1, 1, 1536, 8, 0, page_size=64)
    _expect_key("fp8 B64 q1 448 page 64", True, 64, 1, 448, 1, 1, page_size=64)
    _expect_key("bf16 B1 q1 1536", False, 1, 1, 1536, 16, want_bf16_pair)
    _expect_key(
        "bf16 B1 q1 1536 page 64",
        False,
        1,
        1,
        1536,
        16,
        want_bf16_pair,
        page_size=64,
    )
    # Scaled FP8 and NVFP4 caches take the dense heuristic of the top-k, with
    # the relaxed split unless per-token top-k lengths or a sink keep the page
    # floor; NVFP4's index row is twice its top-k.
    _expect_key(
        "scaled fp8 B1 q1 1024",
        True,
        1,
        1,
        1024,
        9,
        0,
        kv_form=MLA_SPARSE_KV_FORM_SCALED_FP8,
    )
    _expect_key(
        "scaled fp8 B1 q1 32768",
        True,
        1,
        1,
        32768,
        16,
        0,
        kv_form=MLA_SPARSE_KV_FORM_SCALED_FP8,
    )
    _expect_key(
        "nvfp4 B1 q1 1024",
        False,
        1,
        1,
        1024,
        9,
        0,
        kv_form=MLA_SPARSE_KV_FORM_NVFP4,
        stride=4096,
    )
    _expect_key(
        "nvfp4 B1 q1 32768",
        False,
        1,
        1,
        32768,
        16,
        0,
        kv_form=MLA_SPARSE_KV_FORM_NVFP4,
        stride=4096,
    )
    _expect_key(
        "nvfp4 B1 q1 1024 variable top-k",
        False,
        1,
        1,
        1024,
        16,
        0,
        kv_form=MLA_SPARSE_KV_FORM_NVFP4,
        stride=4096,
        fixed_topk=False,
    )
    # The key no longer moves with the cache past the top-k.
    for cache in [4096, 32768, 163840]:
        var key = compute_mla_dispatch_key_runtime(
            1, cache, 1, 128, True, 2048, SM_COUNT
        )
        _expect("fp8 B1 q1 np past the top-k", key[2], 8)
    # Dense dispatch keeps its existing key; sparse keys follow their own split.
    var dense = compute_mla_dispatch_scalars_runtime(
        1, 32768, 1, 128, True, SM_COUNT
    )
    var key = compute_mla_dispatch_key_runtime(
        1, 32768, 1, 128, True, 0, SM_COUNT
    )
    _expect("dense h128 np", key[2], dense[2])
    _expect("dense h128 variant", key[3], 0)
    var key64 = compute_mla_dispatch_key_runtime(
        1, 32768, 1, 64, True, 2048, SM_COUNT
    )
    _expect("sparse h64 np", key64[2], 8)
    _expect("sparse h64 variant", key64[3], 0)


def _np_prod[
    heads: Int, q_type: DType, kv_type: DType, quant: Bool, nvfp4: Bool
](batch: Int, q: Int, cache: Int) -> Int:
    """The dispatch's split for a plain top-k 2048 decode in one production
    cache form.
    """
    return _mla_decode_split[
        q_type,
        kv_type,
        quant,
        nvfp4,
        heads,
        HALF_SMS,
        sparse=True,
        _is_cache_length_accurate=False,
        per_token_scale_rope_aware=False,
        fold_shared_index=False,
        has_extra_k=False,
        fixed_topk=True,
        skip_masked_tiles=False,
    ](batch, q, cache, 4096 if nvfp4 else 2048, 0, 128, SM_COUNT)[0]


def _expect_key_matches_launch_form[
    heads: Int,
    qt: DType,
    kt: DType,
    quant: Bool,
    nvfp4: Bool,
    kv_form: Int,
](batch: Int, q: Int, cache: Int, stride: Int, is_fp8: Bool) raises:
    var key = compute_mla_dispatch_key_runtime(
        batch,
        cache,
        q,
        heads,
        is_fp8,
        stride,
        SM_COUNT,
        kv_form=kv_form,
        physical_indices=True,
        depth=576,
    )

    var launch = _mla_decode_split[
        qt,
        kt,
        quant,
        nvfp4,
        heads,
        HALF_SMS,
        sparse=True,
        _is_cache_length_accurate=False,
        per_token_scale_rope_aware=False,
        fold_shared_index=False,
        has_extra_k=False,
        fixed_topk=True,
        skip_masked_tiles=False,
    ](batch, q, cache, stride, 0, 128, SM_COUNT)
    if key[2] != launch[0]:
        raise Error(
            "capture key disagrees with launch: form=",
            kv_form,
            " h=",
            heads,
            " B=",
            batch,
            " q=",
            q,
            " cache=",
            cache,
            " key=",
            key[2],
            " launch=",
            launch[0],
        )


def _expect_key_matches_launch[
    heads: Int, kv_form: Int
](batch: Int, q: Int, cache: Int) raises:
    """The graph capture key's split is the split the dispatch launches.

    The key is captured at the longest cache that maps to it and replayed for
    shorter caches, so a key that disagrees with the launch would replay a
    different split than the policy picked. Cover both sides of the top-k
    boundary, where the launch may start reading the cache instead of the
    top-k, and every form in the production scope.
    """
    comptime if kv_form == MLA_SPARSE_KV_FORM_NVFP4:
        _expect_key_matches_launch_form[
            heads,
            DType.float8_e4m3fn,
            DType.uint8,
            True,
            True,
            MLA_SPARSE_KV_FORM_NVFP4,
        ](batch, q, cache, 4096, False)
    elif kv_form == MLA_SPARSE_KV_FORM_SCALED_FP8:
        _expect_key_matches_launch_form[
            heads,
            DType.bfloat16,
            DType.float8_e4m3fn,
            True,
            False,
            MLA_SPARSE_KV_FORM_SCALED_FP8,
        ](batch, q, cache, 2048, True)
    else:
        _expect_key_matches_launch_form[
            heads,
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            False,
            False,
            MLA_SPARSE_KV_FORM_PLAIN,
        ](batch, q, cache, 2048, True)


def test_capture_key_matches_launch_across_the_topk_boundary() raises:
    """Sweep the key against the launch for every production form and the
    caches around the 2047/2048 top-k boundary, including shorter caches.
    """
    comptime for form in [
        MLA_SPARSE_KV_FORM_PLAIN,
        MLA_SPARSE_KV_FORM_SCALED_FP8,
        MLA_SPARSE_KV_FORM_NVFP4,
    ]:
        comptime for heads in [64, 128]:
            for batch in [1, 2, 3, 4, 5, 8, 16]:
                for q in [1, 2]:
                    for cache in [
                        0,
                        512,
                        1024,
                        1535,
                        2047,
                        2048,
                        2049,
                        4096,
                        32768,
                    ]:
                        _expect_key_matches_launch[heads, form](batch, q, cache)


def test_topk2048_compaction_holds_for_every_cache_form() raises:
    """The top-k 2048 compaction keys on the index row the launch receives,
    so a packed NVFP4 row (twice as wide) and a direct call compact as the
    FP8 graph op does: 9 splits become 8 at h=64 q=2 B=8, 6% faster on B200
    for NVFP4 at caches 1152 to 2040.
    """
    for cache in [1152, 1536, 2040]:
        _expect(
            "plain FP8 top-k 2048 compacts",
            _np_prod[
                64, DType.float8_e4m3fn, DType.float8_e4m3fn, False, False
            ](8, 2, cache),
            8,
        )
        _expect(
            "NVFP4 top-k 2048 compacts",
            _np_prod[64, DType.float8_e4m3fn, DType.uint8, True, True](
                8, 2, cache
            ),
            8,
        )
        var key = compute_mla_dispatch_key_runtime(
            8,
            cache,
            2,
            64,
            True,
            4096,
            SM_COUNT,
            kv_form=MLA_SPARSE_KV_FORM_NVFP4,
            physical_indices=True,
            depth=576,
        )
        _expect("NVFP4 top-k 2048 key compacts", key[2], 8)


def test_production_forms_follow_the_wave_rule() raises:
    """The blockwise-FP8 and NVFP4 forms take the one-wave bucket; the FP8
    unit records only cap at it.

    The forced-split sweep behind this (graph replay, caches 4096 and 32768,
    B 1-64, q 1-6) found the blockwise heuristic under-splitting h=64 B=16
    q=1 by 33% and B=4 q=4 by 31%, and over-splitting h=128 B=16 q=1 by 17%
    and B=8 q=4 by 25%; NVFP4's joint model over-splits h=128 B=4 q=2 and
    B=8 q=1 by 12%. The one-wave bucket is within noise of the best measured
    arm in every cell of both forms. The FP8 unit records already sit on the
    one-wave choice, and forcing their low-batch fill costs up to 8%, so they
    keep their np when the bucket is larger.
    """
    # Blockwise BF16 Q over FP8, block 32.
    _expect(
        "blockwise h64 B4 q4",
        _np_prod[64, DType.bfloat16, DType.float8_e4m3fn, True, False](
            4, 4, 32768
        ),
        8,
    )
    _expect(
        "blockwise h64 B8 q2",
        _np_prod[64, DType.bfloat16, DType.float8_e4m3fn, True, False](
            8, 2, 32768
        ),
        8,
    )
    _expect(
        "blockwise h64 B16 q1",
        _np_prod[64, DType.bfloat16, DType.float8_e4m3fn, True, False](
            16, 1, 32768
        ),
        8,
    )
    _expect(
        "blockwise h128 B4 q4",
        _np_prod[128, DType.bfloat16, DType.float8_e4m3fn, True, False](
            4, 4, 32768
        ),
        4,
    )
    _expect(
        "blockwise h128 B8 q4",
        _np_prod[128, DType.bfloat16, DType.float8_e4m3fn, True, False](
            8, 4, 32768
        ),
        2,
    )
    _expect(
        "blockwise h128 B8 q6",
        _np_prod[128, DType.bfloat16, DType.float8_e4m3fn, True, False](
            8, 6, 32768
        ),
        1,
    )
    _expect(
        "blockwise h128 B16 q1",
        _np_prod[128, DType.bfloat16, DType.float8_e4m3fn, True, False](
            16, 1, 32768
        ),
        4,
    )
    # NVFP4, including the two cells the joint model's 9-way split loses.
    _expect(
        "nvfp4 h64 B8 q2",
        _np_prod[64, DType.float8_e4m3fn, DType.uint8, True, True](8, 2, 32768),
        8,
    )
    _expect(
        "nvfp4 h128 B4 q2",
        _np_prod[128, DType.float8_e4m3fn, DType.uint8, True, True](
            4, 2, 32768
        ),
        8,
    )
    _expect(
        "nvfp4 h128 B8 q1",
        _np_prod[128, DType.float8_e4m3fn, DType.uint8, True, True](
            8, 1, 32768
        ),
        8,
    )
    # FP8 unit: the records keep a smaller split than the bucket.
    _expect(
        "fp8 unit h64 B4 q2 keeps its record",
        _np_prod[64, DType.float8_e4m3fn, DType.float8_e4m3fn, False, False](
            4, 2, 32768
        ),
        8,
    )
    _expect(
        "fp8 unit h128 B4 q1 keeps its record",
        _np_prod[128, DType.float8_e4m3fn, DType.float8_e4m3fn, False, False](
            4, 1, 32768
        ),
        8,
    )
    # Below the top-k the cache bounds the walk and the old policy stands.
    _expect(
        "nvfp4 short cache stays at one split",
        _np_prod[128, DType.float8_e4m3fn, DType.uint8, True, True](1, 1, 128),
        1,
    )


def test_sparse_split_follows_the_kv_page() raises:
    """The KV page still reaches the records that size their split from the
    cache, so a 64-token page moves their pick; the page-invariant 128-head
    picks above document the complement.

    At 128 heads only the BF16-Q-over-FP8 record still clamps to the cache
    (the FP8 record sizes from the top-k, and BF16-over-BF16 sweeps to the
    same picks at both pages). Below 128 heads the full-row Layout G record
    and the relaxed-floor compact bucket read the page too. Each pair of
    picks differs, so these fail if `kv_page_size` stops reaching the cost
    page.
    """
    # 128 heads, BF16 Q over an FP8 cache.
    _expect(
        "bf16 q fp8 kv B1 1152 page 128",
        _np_sparse_page[DType.bfloat16, DType.float8_e4m3fn, 128](
            1, 1, 1152, 128
        ),
        8,
    )
    _expect(
        "bf16 q fp8 kv B1 1152 page 64",
        _np_sparse_page[DType.bfloat16, DType.float8_e4m3fn, 128](
            1, 1, 1152, 64
        ),
        16,
    )
    _expect(
        "bf16 q fp8 kv B8 1536 page 128",
        _np_sparse_page[DType.bfloat16, DType.float8_e4m3fn, 128](
            8, 1, 1536, 128
        ),
        8,
    )
    _expect(
        "bf16 q fp8 kv B8 1536 page 64",
        _np_sparse_page[DType.bfloat16, DType.float8_e4m3fn, 128](
            8, 1, 1536, 64
        ),
        9,
    )
    # 32 heads: the full-row Layout G record on the relaxed floor.
    _expect(
        "fp8 h32 B1 1536 page 128",
        _np_sparse_page[DType.float8_e4m3fn, DType.float8_e4m3fn, 32](
            1, 1, 1536, 128
        ),
        8,
    )
    _expect(
        "fp8 h32 B1 1536 page 64",
        _np_sparse_page[DType.float8_e4m3fn, DType.float8_e4m3fn, 32](
            1, 1, 1536, 64
        ),
        9,
    )
    # 64 heads: the record the relaxed floor feeds (the compact bucket step
    # is a no-op at these shapes; the pick moves in the record itself).
    _expect(
        "fp8 h64 B1 1536 page 128",
        _np_sparse_page[DType.float8_e4m3fn, DType.float8_e4m3fn, 64](
            1, 1, 1536, 128
        ),
        8,
    )
    _expect(
        "fp8 h64 B1 1536 page 64",
        _np_sparse_page[DType.float8_e4m3fn, DType.float8_e4m3fn, 64](
            1, 1, 1536, 64
        ),
        9,
    )


def test_sparse_key_head_counts_and_variants() raises:
    for heads in [8, 16, 24, 32, 40, 48, 56, 64, 128]:
        for nvfp4 in [False, True]:
            var key = compute_mla_dispatch_key_runtime(
                1,
                32768,
                1,
                heads,
                not nvfp4,
                256 if nvfp4 else 128,
                SM_COUNT,
                kv_form=MLA_SPARSE_KV_FORM_NVFP4 if nvfp4 else MLA_SPARSE_KV_FORM_PLAIN,
            )
            _expect("one-page sparse key split", key[2], 1)
            var variant = 0
            comptime if get_defined_int["MLA_SPARSE_BN_QK", 0]() == 128:
                if heads >= 64:
                    variant = 2
            _expect("sparse key tile variant", key[3], variant)
    comptime if get_defined_int["MLA_SPARSE_BN_QK", 0]() == 128:
        var key = compute_mla_dispatch_key_runtime(
            1,
            32768,
            1,
            128,
            False,
            768,
            SM_COUNT,
            kv_form=MLA_SPARSE_KV_FORM_NVFP4,
        )
        _expect("NVFP4 key counts value slots once", key[3], 2)


def _expect_forced_joint_variant[heads: Int](variant: Int) raises:
    comptime tile = get_defined_int["MLA_SPARSE_BN_QK", 0]()
    comptime pair = get_defined_int["MLA_SPARSE_CG2", 1]()
    comptime pair_route = get_defined_int["MLA_SPARSE_CG2_ROUTE", 1]()
    comptime if tile == 64:
        _expect("forced single64 variant", variant, 0)
    elif tile == 128:
        comptime if heads == 64 or pair == 0:
            _expect("forced single128 variant", variant, 2)
        elif pair_route == 0:
            _expect("forced pair128 variant", variant, 1)
        else:
            if variant != 1 and variant != 2:
                raise Error("forced128 selected64 keys")


def _check_joint_cell[
    heads: Int,
    nvfp4: Bool,
    skip: Bool,
    models: _SparseLayoutEModels = _sparse_layout_e_models[
        nvfp4, heads, skip
    ](),
    verify_capture: Bool = True,
](batch: Int, q: Int, cache: Int, topk: Int) raises:
    comptime kv_type = DType.uint8 if nvfp4 else DType.float8_e4m3fn
    var effective_cache = cache + q
    var keys = min(topk, effective_cache)
    # Multiple incumbent seeds expose tie-order bugs that disappear when a
    # test begins at the already-selected optimum.
    for seed in [1, 2, 4, 16, HALF_SMS]:
        var joint = _select_sparse_layout_e_launch[
            DType.float8_e4m3fn,
            kv_type,
            nvfp4,
            nvfp4,
            heads,
            HALF_SMS,
            skip_masked_tiles=skip,
            models=models,
        ](batch, q, keys, 128, SM_COUNT, seed, 0, effective_cache)
        var fixed = _select_sparse_layout_e_launch[
            DType.float8_e4m3fn,
            kv_type,
            nvfp4,
            nvfp4,
            heads,
            HALF_SMS,
            skip_masked_tiles=skip,
            models=models,
        ](batch, q, keys, 128, SM_COUNT, seed, joint[0], effective_cache)
        _expect("joint split equals fixed split", fixed[0], joint[0])
        _expect("joint variant equals fixed variant", fixed[1], joint[1])
        _expect_bucket("joint split", joint[0], batch, cache, q)
        _expect_forced_joint_variant[heads](joint[1])

    comptime for b in range(_NUM_PARTITION_BUCKETS):
        var np = _get_partition_bucket[HALF_SMS, b]()
        var fixed = _select_sparse_layout_e_launch[
            DType.float8_e4m3fn,
            kv_type,
            nvfp4,
            nvfp4,
            heads,
            HALF_SMS,
            skip_masked_tiles=skip,
            models=models,
        ](batch, q, keys, 128, SM_COUNT, 1, np, effective_cache)
        _expect("explicit split retained", fixed[0], np)
        _expect_forced_joint_variant[heads](fixed[1])

    comptime if not skip and verify_capture:
        var split = _mla_decode_split[
            DType.float8_e4m3fn,
            kv_type,
            nvfp4,
            nvfp4,
            heads,
            HALF_SMS,
            sparse=True,
            _is_cache_length_accurate=False,
            per_token_scale_rope_aware=False,
            fold_shared_index=False,
            has_extra_k=False,
            fixed_topk=True,
            skip_masked_tiles=False,
        ](batch, q, cache, topk * (2 if nvfp4 else 1), 0, 128, SM_COUNT)
        var fixed = _select_sparse_layout_e_launch[
            DType.float8_e4m3fn,
            kv_type,
            nvfp4,
            nvfp4,
            heads,
            HALF_SMS,
        ](
            batch,
            q,
            keys,
            split[1],
            SM_COUNT,
            split[0],
            split[0],
            effective_cache,
        )
        var key = compute_mla_dispatch_key_runtime(
            batch,
            cache,
            q,
            heads,
            not nvfp4,
            topk * (2 if nvfp4 else 1),
            SM_COUNT,
            kv_form=MLA_SPARSE_KV_FORM_NVFP4 if nvfp4 else MLA_SPARSE_KV_FORM_PLAIN,
            physical_indices=True,
            depth=576,
        )
        _expect("native key split follows final selection", key[2], fixed[0])
        _expect("native key variant follows final selection", key[3], fixed[1])


def _test_joint_form[
    heads: Int,
    nvfp4: Bool,
    skip: Bool,
    models: _SparseLayoutEModels = _sparse_layout_e_models[
        nvfp4, heads, skip
    ](),
    verify_capture: Bool = True,
]() raises:
    for batch in [1, 4, 8, 16, 32, 64]:
        for q in [1, 2, 4, 6, 8]:
            for cache in [0, 128, 384, 512, 2048, 32768]:
                _check_joint_cell[heads, nvfp4, skip, models, verify_capture](
                    batch, q, cache, 2048
                )
    for batch in [1, 4]:
        for q in [1, 2]:
            for topk in [
                1,
                63,
                64,
                65,
                127,
                128,
                129,
                191,
                192,
                255,
                256,
                257,
                384,
                2048,
            ]:
                _check_joint_cell[heads, nvfp4, skip, models, verify_capture](
                    batch, q, 4096, topk
                )
    for cache in [0, 32768]:
        _check_joint_cell[heads, nvfp4, skip, models, verify_capture](
            1, 8192, cache, 2048
        )
        _check_joint_cell[heads, nvfp4, skip, models, verify_capture](
            4, 2048, cache, 2048
        )


def test_joint_default_and_fixed_selection_agree() raises:
    _test_joint_form[64, False, False]()
    _test_joint_form[64, True, False]()
    _test_joint_form[128, False, False]()
    _test_joint_form[128, True, False]()
    _test_joint_form[64, False, True]()
    _test_joint_form[64, True, True]()
    _test_joint_form[128, False, True]()
    _test_joint_form[128, True, True]()


# Synthetic records exercise selection even before performance records ship.
comptime _SYNTHETIC_LAYOUT_E_MODELS = _SparseLayoutEModels(
    _SplitCost(10000, 1000, 10, 200, 300, 0, 10000, 200, 300),
    _SplitCost(100, 100, 1, 1, 2, 0, 100, 1, 2),
    _SplitCost(50, 50, 1, 1, 2, 0, 50, 1, 2),
    7,
    7,
)


def _check_unmeasured_fallback[
    heads: Int,
    nvfp4: Bool,
    window: Int,
    models: _SparseLayoutEModels,
]() raises:
    comptime kv_type = DType.uint8 if nvfp4 else DType.float8_e4m3fn
    comptime legacy_pair = _sparse_pair_eligible_form[
        DType.float8_e4m3fn,
        kv_type,
        nvfp4,
        nvfp4,
        heads,
        False,
        False,
    ]()
    comptime wide_forced = get_defined_int["MLA_SPARSE_BN_QK", 0]() == 128
    for np in [1, 2, 4, 16, HALF_SMS]:
        var expected = _sparse_layout_e_incumbent_variant[
            legacy_pair, wide_forced, _SPARSE_PAIR_MAX_DIRECT_WALK
        ](
            257,
            np,
            128,
        )
        for fixed in [0, np]:
            var selected = _select_sparse_layout_e_launch[
                DType.float8_e4m3fn,
                kv_type,
                nvfp4,
                nvfp4,
                heads,
                HALF_SMS,
                sliding_window_size=window,
                models=models,
            ](1, 1, 257, 128, SM_COUNT, np, fixed, 4096)
            # Nonnegative windows always preserve the incumbent. A negative
            # window only tests the unmeasured explicit empty-split case.
            comptime if window >= 0:
                _expect("window keeps split", selected[0], np)
                _expect("window keeps variant", selected[1], expected)
            else:
                if fixed > 3:
                    _expect(
                        "empty splits keep requested split", selected[0], np
                    )
                    _expect("empty splits keep variant", selected[1], expected)


def test_joint_synthetic_models_and_unmeasured_fallback() raises:
    comptime for heads in [64, 128]:
        comptime for nvfp4 in [False, True]:
            _test_joint_form[
                heads, nvfp4, False, _SYNTHETIC_LAYOUT_E_MODELS, False
            ]()
            _test_joint_form[
                heads, nvfp4, True, _SYNTHETIC_LAYOUT_E_MODELS, False
            ]()
            comptime for window in [-1, 0, 32, 128]:
                _check_unmeasured_fallback[
                    heads, nvfp4, window, _NO_LAYOUT_E_MODELS
                ]()
                _check_unmeasured_fallback[
                    heads, nvfp4, window, _SYNTHETIC_LAYOUT_E_MODELS
                ]()
    comptime if get_defined_int["MLA_SPARSE_BN_QK", 0]() == 0:
        var selected = _select_sparse_layout_e_launch[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            False,
            False,
            64,
            HALF_SMS,
            models=_SYNTHETIC_LAYOUT_E_MODELS,
        ](1, 1, 2048, 128, SM_COUNT, 1, 1, 4096)
        _expect("synthetic records change the selected variant", selected[1], 2)


def test_joint_logical_rows_keep_incumbent() raises:
    comptime for heads in [64, 128]:
        comptime legacy_pair = _sparse_pair_eligible_form[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            False,
            False,
            heads,
            False,
            False,
        ]()
        comptime wide = get_defined_int["MLA_SPARSE_BN_QK", 0]() == 128
        for np in [1, 2, 4, 16]:
            var actual = _select_sparse_layout_e_launch[
                DType.float8_e4m3fn,
                DType.float8_e4m3fn,
                False,
                False,
                heads,
                HALF_SMS,
                physical_indices=False,
                models=_SYNTHETIC_LAYOUT_E_MODELS,
            ](1, 1, 2048, 128, SM_COUNT, np, 0, 4096)
            _expect("logical rows keep incumbent split", actual[0], np)
            _expect(
                "logical rows keep incumbent variant",
                actual[1],
                _sparse_layout_e_incumbent_variant[
                    legacy_pair, wide, _SPARSE_PAIR_MAX_DIRECT_WALK
                ](2048, np, 128),
            )


def test_joint_unknown_and_nope_depth_keep_incumbent() raises:
    comptime for heads in [64, 128]:
        comptime for nvfp4 in [False, True]:
            comptime kv_type = DType.uint8 if nvfp4 else DType.float8_e4m3fn
            comptime pair = _sparse_pair_eligible_form[
                DType.float8_e4m3fn,
                kv_type,
                nvfp4,
                nvfp4,
                heads,
                False,
                False,
            ]()
            comptime wide = get_defined_int["MLA_SPARSE_BN_QK", 0]() == 128
            comptime for depth in [0, 512]:
                for np in [1, 2, 4, 16]:
                    var choice = _select_sparse_layout_e_launch[
                        DType.float8_e4m3fn,
                        kv_type,
                        nvfp4,
                        nvfp4,
                        heads,
                        HALF_SMS,
                        depth=depth,
                        models=_SYNTHETIC_LAYOUT_E_MODELS,
                    ](1, 1, 2048, 128, SM_COUNT, np, 0, 4096)
                    _expect("unmeasured depth keeps split", choice[0], np)
                    _expect(
                        "unmeasured depth keeps variant",
                        choice[1],
                        _sparse_layout_e_incumbent_variant[
                            pair, wide, _SPARSE_PAIR_MAX_DIRECT_WALK
                        ](2048, np, 128),
                    )


def _synthetic_constant_cost[split_ns: Int, single_ns: Int]() -> _SplitCost:
    return _SplitCost(split_ns, 0, 0, 0, 0, 0, single_ns, 0, 0)


def _check_joint_np_margin[split_ns: Int, margin: Int, expected: Int]() raises:
    comptime cost = _synthetic_constant_cost[split_ns, 1000]()
    comptime models = _SparseLayoutEModels(cost, cost, cost, 7, 7)
    var result = _select_sparse_layout_e_launch[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        False,
        False,
        64,
        HALF_SMS,
        np_margin_percent=margin,
        models=models,
    ](1, 1, 2048, 128, SM_COUNT, 1)
    _expect("independent NP margin", result[0], expected)
    var fixed = _select_sparse_layout_e_launch[
        DType.float8_e4m3fn,
        DType.float8_e4m3fn,
        False,
        False,
        64,
        HALF_SMS,
        np_margin_percent=margin,
        models=models,
    ](1, 1, 2048, 128, SM_COUNT, 1, result[0])
    _expect("NP margin fixed-selection consistency", fixed[1], result[1])


def test_joint_np_margin_is_independent() raises:
    _check_joint_np_margin[950, 2, 2]()
    _check_joint_np_margin[950, 10, 1]()
    _check_joint_np_margin[900, 10, 1]()
    _check_joint_np_margin[899, 10, 2]()
    comptime for wide_ns in [980, 979]:
        comptime base = _synthetic_constant_cost[1000, 1000]()
        comptime wide = _synthetic_constant_cost[wide_ns, wide_ns]()
        comptime models = _SparseLayoutEModels(base, base, wide, 5, 5)
        var choice = _sparse_layout_e_cost_at_np[
            models, False, False, False, _SPARSE_PAIR_MAX_DIRECT_WALK
        ](
            1,
            2048,
            128,
            SM_COUNT,
            1,
            5,
        )
        comptime if get_defined_int["MLA_SPARSE_BN_QK", 0]() == 0:
            _expect(
                "variant margin remains strict two percent",
                choice[0],
                2 if wide_ns == 979 else 0,
            )


def test_confirmed_nvfp4_partition_pins() raises:
    comptime if get_defined_int[
        "MLA_SPARSE_BN_QK", 0
    ]() != 0 or get_defined_int["MLA_SPARSE_NUM_PARTITIONS", 0]() != 0:
        return
    # Pins originate from the three-round offline policy replay, not this
    # helper's model. Changed choices require fixed-choice confirmation.
    # The one-wave rule moves three NVFP4 cells: h=128 B=4 q=2 takes 8, not
    # the record's 9 (13.6 us vs 15.4 us at ctx 32768), because 9 splits leave
    # two of the 16 pages unowned while still launching their CTAs, and h=128
    # B=8 q=1 takes 8, not 9. NVFP4 h=64 B=1 q=6 stays at 8: the fill would
    # raise it to 16 and measured 1.4% slower, while blockwise B=1 q=6 still
    # fills (24.8 -> 14.1 us).
    comptime cells = (
        (64, 1, 1, 0, 2048, 1),
        (64, 1, 1, 2048, 2048, 16),
        (64, 1, 1, 32768, 129, 1),
        (64, 1, 6, 2048, 2048, 8),
        (64, 1, 6, 32768, 2048, 8),
        (64, 4, 2, 2048, 2048, 16),
        (64, 8, 4, 2048, 2048, 4),
        (64, 16, 6, 32768, 2048, 1),
        (64, 64, 8, 32768, 2048, 1),
        (128, 1, 1, 0, 2048, 1),
        (128, 1, 1, 2048, 2048, 16),
        (128, 1, 1, 32768, 129, 1),
        (128, 1, 6, 2048, 2048, 8),
        (128, 1, 6, 32768, 2048, 8),
        (128, 4, 2, 2048, 2048, 8),
        (128, 8, 4, 2048, 2048, 2),
        (128, 16, 6, 32768, 2048, 1),
        (128, 64, 8, 32768, 2048, 1),
    )
    comptime for i in range(len(cells)):
        comptime cell = cells[i]
        var key = compute_mla_dispatch_key_runtime(
            cell[1],
            cell[3],
            cell[2],
            cell[0],
            False,
            cell[4] * 2,
            SM_COUNT,
            kv_form=MLA_SPARSE_KV_FORM_NVFP4,
            physical_indices=True,
            depth=576,
        )
        _expect("confirmed NVFP4 split", key[2], cell[5])
        _expect("confirmed NVFP4 single64", key[3], 0)


def test_joint_default_reseeds_empty_ctas() raises:
    comptime if get_defined_int[
        "MLA_SPARSE_BN_QK", 0
    ]() != 0 or get_defined_int["MLA_SPARSE_NUM_PARTITIONS", 0]() != 0:
        return
    comptime for heads in [64, 128]:
        for seed in [2, 4, 16, HALF_SMS]:
            var automatic = _select_sparse_layout_e_launch[
                DType.float8_e4m3fn,
                DType.uint8,
                True,
                True,
                heads,
                HALF_SMS,
            ](1, 1, 1, 128, SM_COUNT, seed)
            _expect("cold default uses a useful split", automatic[0], 1)
            _expect("cold default keeps single64", automatic[1], 0)
            var fixed = _select_sparse_layout_e_launch[
                DType.float8_e4m3fn,
                DType.uint8,
                True,
                True,
                heads,
                HALF_SMS,
            ](1, 1, 1, 128, SM_COUNT, seed, seed)
            _expect("explicit empty splits remain exact", fixed[0], seed)
            _expect("explicit empty split variant", fixed[1], 0)
        for q in [1, 6]:
            for cache in [0, 128]:
                var key = compute_mla_dispatch_key_runtime(
                    1,
                    cache,
                    q,
                    heads,
                    False,
                    4096,
                    SM_COUNT,
                    kv_form=MLA_SPARSE_KV_FORM_NVFP4,
                    physical_indices=True,
                    depth=576,
                )
                _expect("short cache useful default split", key[2], 1)
                _expect("short cache keeps single64", key[3], 0)
        for seed in [4, 16, HALF_SMS]:
            var longer = _select_sparse_layout_e_launch[
                DType.float8_e4m3fn,
                DType.uint8,
                True,
                True,
                heads,
                HALF_SMS,
            ](1, 1, 257, 128, SM_COUNT, seed)
            _expect("larger empty-CTA region keeps incumbent", longer[0], seed)


def main() raises:
    test_joint_default_reseeds_empty_ctas()
    test_confirmed_nvfp4_partition_pins()
    test_joint_np_margin_is_independent()
    test_joint_default_and_fixed_selection_agree()
    test_joint_synthetic_models_and_unmeasured_fallback()
    test_joint_logical_rows_keep_incumbent()
    test_joint_unknown_and_nope_depth_keep_incumbent()
    test_sparse_key_head_counts_and_variants()
    comptime if get_defined_int["MLA_SPARSE_TEST_KEY_ONLY", 0]() != 0:
        return
    test_bucket_tables_agree()
    test_every_split_has_a_combine_kernel()
    test_cost_model_split_is_pinned()
    test_layout_g_cost_split_is_pinned()
    test_page_equivalent_split_buckets()
    test_topk2048_split_policy()
    test_layout_e_relaxed_split_is_pinned()
    test_h128_relaxed_clamp_follows_the_cost_record()
    test_multi_head_group_ignores_the_cost_model()
    test_runtime_binding_matches_its_own_specialization()
    test_the_binding_pin_is_not_vacuous()
    test_unenumerated_head_count_raises()
    test_sparse_graph_key_follows_the_launch()
    test_sparse_split_follows_the_kv_page()
    test_capture_key_matches_launch_across_the_topk_boundary()
    test_production_forms_follow_the_wave_rule()
    test_topk2048_compaction_holds_for_every_cache_form()
    print("split policy pins OK")
