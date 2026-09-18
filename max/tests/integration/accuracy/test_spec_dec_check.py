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

"""Unit tests for SpecDecCheck's prefix tree and per-node statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from spec_dec_check.samples import Sample
from spec_dec_check.stats import aggregate, node_stats, permutation_null
from spec_dec_check.tree import (
    Node,
    _survives_bucketing,
    bucketize,
    build_tree,
)

_FLOOR = 10
"""The bucket floor the bucketing cases are written against."""


def _samples(
    server: str, sequences: Sequence[Sequence[int]], prompt_index: int = 0
) -> list[Sample]:
    """Builds one sample per sequence, all for one server and prompt."""
    return [
        Sample(
            server=server,
            prompt_index=prompt_index,
            request_index=index,
            seed=index + 1,
            prompt_token_ids=(1, 2),
            token_ids=tuple(sequence),
            finish_reason="length",
            completion_tokens=len(sequence),
            completed_at=float(index),
            text="",
        )
        for index, sequence in enumerate(sequences)
    ]


def _node(baseline: Mapping[int, int], test: Mapping[int, int]) -> Node:
    """Builds a depth-0 node straight from two next-token count maps."""
    return Node(
        prompt_index=0, prefix=(), counts=(Counter(baseline), Counter(test))
    )


# One token; two tokens both clearing the floor; a sub-floor tail that earns
# its own bucket; a sub-floor tail that folds; and a tail that takes the node
# below two buckets.
_BUCKET_CASES = [
    pytest.param({5: 20}, {5: 20}, (5,), (), None, id="single_token"),
    pytest.param(
        {1: 20, 2: 20}, {1: 20, 2: 20}, (1, 2), (), None, id="two_kept"
    ),
    pytest.param(
        {1: 20, 2: 20, 3: 4, 4: 4},
        {1: 20, 2: 20, 3: 4, 4: 4},
        (1, 2, None),
        (3, 4),
        None,
        id="tail_own_bucket",
    ),
    pytest.param(
        {1: 20, 2: 20, 3: 3},
        {1: 20, 2: 20, 3: 3},
        (1, 2),
        (3,),
        2,
        id="tail_folded",
    ),
]


@pytest.mark.parametrize(
    "baseline,test,labels,tail_tokens,folded_into", _BUCKET_CASES
)
def test_bucketize_places_the_tail(
    baseline: Mapping[int, int],
    test: Mapping[int, int],
    labels: tuple[int | None, ...],
    tail_tokens: tuple[int, ...],
    folded_into: int | None,
) -> None:
    """Buckets come out in canonical order with the tail placed correctly."""
    bucketed = bucketize(_node(baseline, test), bucket_floor=_FLOOR)

    assert bucketed is not None
    assert bucketed.labels == labels
    assert bucketed.tail_tokens == tail_tokens
    assert bucketed.tail_folded_into == folded_into


@pytest.mark.parametrize(
    "baseline,test,labels,tail_tokens,folded_into", _BUCKET_CASES
)
def test_bucketize_conserves_each_group_total(
    baseline: Mapping[int, int],
    test: Mapping[int, int],
    labels: tuple[int | None, ...],
    tail_tokens: tuple[int, ...],
    folded_into: int | None,
) -> None:
    """No observation is lost or double counted by bucketing.

    The permutation null conditions on each group's total, so a tail that
    folded into the wrong place would still be drawn against the right
    margins and the error would not surface as a failure anywhere else.
    """
    node = _node(baseline, test)
    bucketed = bucketize(node, bucket_floor=_FLOOR)

    assert bucketed is not None
    assert bucketed.n_per_group == node.group_counts


def test_bucketize_drops_a_node_that_cannot_reach_two_buckets() -> None:
    """One dominant token plus a sub-floor tail leaves nothing to compare.

    A node whose every observation is the same token is kept instead, as a
    single bucket with no degrees of freedom, so more variety in the data can
    mean fewer testable nodes rather than more.
    """
    dominant = _node({1: 20, 2: 2}, {1: 20, 2: 1})
    assert bucketize(dominant, bucket_floor=_FLOOR) is None
    assert bucketize(_node({1: 20}, {1: 20}), bucket_floor=_FLOOR) is not None


@pytest.mark.parametrize(
    "pooled",
    [
        pytest.param({5: 40}, id="single_token"),
        pytest.param({1: 40, 2: 40}, id="two_kept"),
        pytest.param({1: 40, 2: 40, 3: 8, 4: 8}, id="tail_own_bucket"),
        pytest.param({1: 40, 2: 40, 3: 6}, id="tail_folded"),
        pytest.param({1: 40, 2: 3}, id="dropped"),
        pytest.param({1: 3, 2: 3, 3: 3}, id="all_below_floor"),
    ],
)
def test_survives_bucketing_agrees_with_bucketize(
    pooled: Mapping[int, int],
) -> None:
    """The cheap predicate and the real bucketing must never disagree.

    ``relabeling_null`` walks prefixes without bucketing them, relying on
    this predicate to decide which would be tested. If the two drift apart
    the null is drawn for a different node set than the one under test, and
    every p-value taken from it is quietly wrong.
    """
    half = {token: count // 2 for token, count in pooled.items()}
    rest = {token: count - half[token] for token, count in pooled.items()}

    predicted = _survives_bucketing(pooled, bucket_floor=_FLOOR)
    actual = bucketize(_node(half, rest), bucket_floor=_FLOOR) is not None

    assert predicted == actual


def test_node_set_is_invariant_under_swapping_the_groups() -> None:
    """Relabeling the two groups leaves the same nodes with swapped counts.

    The per-group minimum is what makes this hold, and the permutation null
    assumes it: a node set that depended on which side was which would make
    the relabeled draws describe a different tree.
    """
    baseline = _samples("baseline", [(10, 20)] * 30 + [(11, 21)] * 8)
    test = _samples("test", [(10, 20)] * 25 + [(11, 21)] * 12)

    forward = build_tree(baseline + test, min_per_group=5, max_depth=3)
    swapped = build_tree(
        _samples("test", [s.token_ids for s in baseline])
        + _samples("baseline", [s.token_ids for s in test]),
        min_per_group=5,
        max_depth=3,
    )

    assert {(n.prompt_index, n.prefix) for n in forward} == {
        (n.prompt_index, n.prefix) for n in swapped
    }
    by_prefix = {n.prefix: n for n in swapped}
    for node in forward:
        assert node.counts == by_prefix[node.prefix].counts[::-1]


def test_build_tree_minimum_is_per_group_not_pooled() -> None:
    """A prefix one group barely reaches is not extended, however busy it is.

    Here 9 baseline and 2 test sequences continue past ``(10, 20)``, so the
    pooled count clears twice the minimum while the smaller group does not
    clear it once. Power at a node is set by its smaller side, so the prefix
    must be dropped.
    """
    samples = _samples("baseline", [(10, 20, 30)] * 9) + _samples(
        "test", [(10, 20)] * 6 + [(10, 20, 30)] * 2
    )

    prefixes = {n.prefix for n in build_tree(samples, min_per_group=5)}

    assert (10,) in prefixes
    assert (10, 20) not in prefixes


def test_node_stats_matches_a_hand_computed_chi_square() -> None:
    """The two-sample statistic agrees with the closed form on a 2x2 table.

    With 30/10 against 20/20 the Pearson statistic is
    ``N(ad - bc)^2 / (row1 row2 col1 col2)``, which is 16/3 here.
    """
    node = _node({1: 30, 2: 10}, {1: 20, 2: 20})
    bucketed = bucketize(node, bucket_floor=_FLOOR)
    assert bucketed is not None

    stats = node_stats(bucketed)

    assert stats is not None
    assert stats.dof == 1
    assert stats.chi_square == pytest.approx(16.0 / 3.0)
    assert stats.tvd == pytest.approx(0.25)


def test_node_stats_is_none_when_one_group_never_reached_the_prefix() -> None:
    """A node only one group reached has no two-sample test of its own."""
    bucketed = bucketize(_node({1: 20, 2: 20}, {}), bucket_floor=_FLOOR)
    assert bucketed is not None

    assert node_stats(bucketed) is None


def _null_p_value(
    rng: np.random.Generator, *, n: int, probabilities: Sequence[float]
) -> float:
    """Draws both groups from one distribution and returns the sum p-value."""
    baseline = rng.multinomial(n, probabilities)
    test = rng.multinomial(n, probabilities)
    node = _node(
        dict(enumerate(baseline.tolist())), dict(enumerate(test.tolist()))
    )
    bucketed = bucketize(node, bucket_floor=_FLOOR)
    assert bucketed is not None
    stats = node_stats(bucketed)
    assert stats is not None
    null = permutation_null(
        [stats], num_permutations=199, rng=rng, effect_min_per_group=0
    )
    return null.p_value("sum", stats.chi_square)


def test_permutation_p_values_are_calibrated_under_the_null() -> None:
    """Two groups from one distribution produce p-values that are not small.

    This exercises the whole chain -- bucketing, the chi-square, and the
    conditional draw -- against the property the tool rests on: when nothing
    differs, a p-value is uniform, so rejecting at 0.2 happens about a fifth
    of the time rather than most of the time.
    """
    rng = np.random.default_rng(0)
    probabilities = [0.5, 0.25, 0.15, 0.10]
    p_values = [
        _null_p_value(rng, n=200, probabilities=probabilities)
        for _ in range(200)
    ]

    below = sum(1 for p in p_values if p < 0.2) / len(p_values)
    assert 0.10 <= below <= 0.32
    assert min(p_values) > 0.0


def test_permutation_p_value_detects_a_real_difference() -> None:
    """A distribution that genuinely differs lands at the resolution floor."""
    rng = np.random.default_rng(0)
    baseline = rng.multinomial(2000, [0.50, 0.25, 0.15, 0.10])
    test = rng.multinomial(2000, [0.25, 0.50, 0.15, 0.10])
    bucketed = bucketize(
        _node(
            dict(enumerate(baseline.tolist())), dict(enumerate(test.tolist()))
        ),
        bucket_floor=_FLOOR,
    )
    assert bucketed is not None
    stats = node_stats(bucketed)
    assert stats is not None

    null = permutation_null(
        [stats], num_permutations=199, rng=rng, effect_min_per_group=0
    )

    assert null.p_value("sum", stats.chi_square) == pytest.approx(1.0 / 200.0)


_REFERENCE = [0.5, 0.3, 0.2]
"""The distribution the test group is always drawn from."""

_SHIFTED = [0.4, 0.4, 0.2]
"""A baseline distribution a fixed distance from :data:`_REFERENCE`."""

_SHIFTED_DIVERGENCE = 0.2254
"""What ``divergence`` estimates for that pair, ``sqrt(D)`` with
``D = sum((p - q)^2 / ((p + q) / 2))`` over the two distributions."""


@pytest.mark.parametrize("n", [100, 10000])
def test_divergence_does_not_grow_with_sample_size(n: int) -> None:
    """The debiased effect size estimates a distance, not a sample count.

    A raw chi-square grows with ``n`` even when the gap between the groups is
    fixed, so ``divergence`` subtracts the null expectation and the
    sample-size scaling back out. The same pair of distributions must
    therefore read the same at either ``n``, a hundredfold apart, while the
    null reads near zero at both.

    The aggregate is taken over many nodes because that is the only place the
    noise cancels: a single node's excess chi-square is signed and clipping
    happens after averaging, so one node under the null at ``n = 100`` ranges
    well above zero on its own.
    """
    rng = np.random.default_rng(0)

    def divergence(baseline_probabilities: Sequence[float]) -> float:
        stats = []
        for _ in range(40):
            baseline = rng.multinomial(n, baseline_probabilities)
            test = rng.multinomial(n, _REFERENCE)
            bucketed = bucketize(
                _node(
                    dict(enumerate(baseline.tolist())),
                    dict(enumerate(test.tolist())),
                ),
                bucket_floor=_FLOOR,
            )
            assert bucketed is not None
            node = node_stats(bucketed)
            assert node is not None
            stats.append(node)
        return aggregate(stats, effect_min_per_group=0).divergence

    assert divergence(_REFERENCE) < 0.20
    assert divergence(_SHIFTED) == pytest.approx(_SHIFTED_DIVERGENCE, abs=0.10)
