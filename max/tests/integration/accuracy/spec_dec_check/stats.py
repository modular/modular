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
"""Per-node statistics, aggregates and the permutation null for SpecDecCheck.

A bucketed node holds both groups' next-token counts over the same buckets, so
the question at a node is whether two multinomial samples came from one
distribution, and the statistic is the two-sample chi-square on those counts.
The names ``baseline`` and ``test`` stand only for the first and the second
group, which are the two servers in the main run and one server's two
temporal halves in the stability check.

Two aggregates gate a run: ``sum`` over the nodes catches a bug that shifts
many positions a little, and ``max`` catches one that wrecks a single position
among thousands of correct ones. The rest never decide pass or fail, because
their nulls depend on the per-node sample sizes and one threshold would mean a
different thing at every node. ``mean_tvd`` and ``max_tvd`` are total
variation distances, which sit on a noise floor that shrinks with sample
count; ``divergence`` subtracts that noise and the sample-size scaling out of
the chi-squares to estimate a fixed distance between the groups; ``pruned``,
``max_depth`` and ``mean_depth`` describe the tree itself, and take their null
from relabeling whole sequences rather than redealing within nodes, since all
three are functions of the very group sizes the node tests condition on.

The asymptotic chi-square distribution does not apply here -- nodes are
small, buckets are sparse, the nodes of one prompt are nested, and the same
samples both build the tree and get tested -- so p-values come from a
permutation null that conditions on each node's margins. Holding a node's
pooled bucket counts and both group totals fixed makes the first group's
bucket vector a multivariate hypergeometric draw, the second group's vector
its complement, and the expected counts per-node constants, which turns each
statistic into one vectorized expression over every permutation at once.
Nodes are accumulated per permutation, which treats them as independent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .tree import BucketedNode, Node

AGGREGATE_NAMES = (
    "sum",
    "max",
    "mean_tvd",
    "max_tvd",
    "divergence",
    "pruned",
    "max_depth",
    "mean_depth",
)
"""The aggregate names :class:`PermutationNull` accepts."""

_LOWER_TAILED = frozenset({"max_depth", "mean_depth"})
"""The aggregates for which a small value, not a large one, is the extreme."""


@dataclass(frozen=True, eq=False)
class NodeStats:
    """The two-sample statistics of one bucketed node.

    Args:
        bucketed: The node and the bucket counts these statistics describe.
        chi_square: The two-sample chi-square over the node's buckets.
        dof: The degrees of freedom, one less than the bucket count.
        tvd: The total variation distance between the groups, in ``[0, 1]``.
        contributions: Each bucket's share of ``chi_square``, so the largest
            entries name the buckets driving the node.
        test_excess: Each bucket's test-minus-baseline proportion, so a
            positive entry is a bucket the second group over-produced.
    """

    bucketed: BucketedNode
    chi_square: float
    dof: int
    tvd: float
    contributions: npt.NDArray[np.float64]
    test_excess: npt.NDArray[np.float64]

    @property
    def node(self) -> Node:
        """The prefix-tree node under test."""
        return self.bucketed.node

    @property
    def chi_square_per_dof(self) -> float:
        """The chi-square over its degrees of freedom, ``0.0`` without any."""
        return self.chi_square / self.dof if self.dof else 0.0

    @property
    def depth(self) -> int:
        """The node's depth, which is the generated position under test."""
        return self.node.depth

    @property
    def prefix(self) -> tuple[int, ...]:
        """The generated ids the tested samples share."""
        return self.node.prefix

    @property
    def prompt_index(self) -> int:
        """Index of the prompt the tested samples answer."""
        return self.node.prompt_index

    @property
    def pooled_count(self) -> int:
        """The number of observations over both groups."""
        return int(self.bucketed.pooled.sum())

    @property
    def n_per_group(self) -> tuple[int, int]:
        """The per-group observation totals, first group first."""
        return self.bucketed.n_per_group


@dataclass(frozen=True)
class AggregateStats:
    """The run-level statistics computed over the tested nodes.

    Args:
        sum_chi_square: The chi-square summed over every tested node.
        sum_dof: The degrees of freedom summed over every tested node.
        sum_chi_square_per_dof: Their ratio, which sits near ``1.0`` under
            the null whatever the node count.
        max_chi_square_per_dof: The largest per-node chi-square over its dof.
        max_node_index: Index of the node that attains it, or ``-1``.
        max_tvd: The largest eligible total variation distance.
        max_tvd_node_index: Index of the node that attains it, or ``-1``.
        mean_tvd: The total variation distance over every tested node,
            weighted by pooled observations, so agreeing positions pull it
            down.
        divergence: An effect size: the square root of the same weighted mean
            of each node's excess chi-square over its null expectation, scaled
            by ``N / (n_baseline * n_test)``, and clipped at zero after
            averaging. Clipping only the average lets node noise cancel rather
            than accumulate, so it sits at or near zero under the null at any
            sample size.
        pruned: The distinct prefixes the per-group minimum pruned, or
            ``None`` when the caller drew no null for it.
        max_depth: The deepest tested depth, ``0`` when there were no nodes.
        mean_depth: The mean tested depth, each node counting once.
        forced_nodes: How many tested nodes were a single token, agreeing
            completely and contributing nothing to the gates.
        num_nodes: How many nodes went into these aggregates.
    """

    sum_chi_square: float
    sum_dof: int
    sum_chi_square_per_dof: float
    max_chi_square_per_dof: float
    max_node_index: int
    max_tvd: float
    max_tvd_node_index: int
    mean_tvd: float
    divergence: float
    pruned: int | None
    max_depth: int
    mean_depth: float
    forced_nodes: int
    num_nodes: int


@dataclass(frozen=True, eq=False)
class PermutationNull:
    """The null distribution of each aggregate over the permutations.

    Every vector below holds one value per permutation. The last three come
    from relabeling whole sequences, and are ``None`` when none was drawn.

    Args:
        sum_chi_square: The summed chi-square.
        max_chi_square_per_dof: The largest chi-square per degree of freedom.
        mean_tvd: The observation-weighted mean total variation distance.
        max_tvd: The largest eligible total variation distance.
        divergence: The debiased divergence.
        pruned: The pruned-prefix count.
        max_depth: The deepest tested depth.
        mean_depth: The mean tested depth.
        num_permutations: The length of each of those vectors.
    """

    sum_chi_square: npt.NDArray[np.float64]
    max_chi_square_per_dof: npt.NDArray[np.float64]
    mean_tvd: npt.NDArray[np.float64]
    max_tvd: npt.NDArray[np.float64]
    divergence: npt.NDArray[np.float64]
    pruned: npt.NDArray[np.int64] | None
    max_depth: npt.NDArray[np.int64] | None
    mean_depth: npt.NDArray[np.float64] | None
    num_permutations: int

    def p_value(self, name: str, observed: float) -> float:
        """Computes the permutation p-value of an observed aggregate.

        Args:
            name: Which aggregate, one of :data:`AGGREGATE_NAMES`.
            observed: The value the real data gave for that aggregate.

        Returns:
            The share of null draws at least as extreme as ``observed``,
            where extreme means large for every aggregate but the depths,
            for which it means shallow. The observed value itself is
            counted, so the result is never ``0.0`` and stays a valid p-value
            however few permutations were drawn.

        Raises:
            ValueError: If ``name`` is not an aggregate name, or names a null
                this analysis did not draw.
        """
        values = self._values(name)
        if name in _LOWER_TAILED:
            at_least = int(np.count_nonzero(values <= observed))
        else:
            at_least = int(np.count_nonzero(values >= observed))
        return (1.0 + at_least) / (self.num_permutations + 1.0)

    def quantiles(
        self, name: str, qs: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Computes quantiles of one aggregate's null distribution.

        Args:
            name: Which aggregate, one of :data:`AGGREGATE_NAMES`.
            qs: The quantiles to take, each in ``[0, 1]``.

        Returns:
            The requested quantiles, in the order asked for.

        Raises:
            ValueError: If ``name`` is not an aggregate name.
        """
        # ``np.quantile`` is typed as returning an unspecified float width.
        return np.asarray(
            np.quantile(self._values(name), np.asarray(qs, dtype=np.float64)),
            dtype=np.float64,
        )

    def _values(self, name: str) -> npt.NDArray[np.float64]:
        """Returns one aggregate's value under each permutation."""
        if name == "sum":
            return self.sum_chi_square
        if name == "max":
            return self.max_chi_square_per_dof
        if name == "mean_tvd":
            return self.mean_tvd
        if name == "max_tvd":
            return self.max_tvd
        if name == "divergence":
            return self.divergence
        if name == "pruned":
            if self.pruned is None:
                raise ValueError("no pruned-prefix null was drawn")
            return self.pruned.astype(np.float64)
        if name == "max_depth":
            if self.max_depth is None:
                raise ValueError("no max-depth null was drawn")
            return self.max_depth.astype(np.float64)
        if name == "mean_depth":
            if self.mean_depth is None:
                raise ValueError("no mean-depth null was drawn")
            return self.mean_depth
        raise ValueError(
            f"unknown aggregate {name!r}, expected one of {AGGREGATE_NAMES}"
        )


@dataclass(frozen=True, eq=False)
class RelabelingNull:
    """What relabeling whole sequences says about the tree the run built.

    Args:
        pruned: The number of distinct prefixes the real labels pruned.
        pruned_null: That count under each relabeling, which is the fixed
            distinct-prefix count minus the candidates
            :func:`.tree.relabeling_null` draws.
        max_depth_null: The deepest tested depth under each relabeling, as
            the same function draws it; the observed depth comes from the
            tested nodes themselves.
        mean_depth_null: The mean tested depth under each relabeling,
            likewise.
    """

    pruned: int
    pruned_null: npt.NDArray[np.int64]
    max_depth_null: npt.NDArray[np.int64]
    mean_depth_null: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class ControlResult:
    """The outcome of the depth-0 control on server equivalence.

    Args:
        stats: The control node's statistics, or ``None`` when the control
            could not be computed.
        null: The one-node permutation null, ``None`` as for ``stats``.
        p_value: Its permutation p-value, ``None`` as for ``stats``.
        degenerate: Whether the control carried no information at all.
        passed: Whether the servers agreed at depth 0, false when degenerate.
        message: A one-line account of the outcome, without numbers.
    """

    stats: NodeStats | None
    null: PermutationNull | None
    p_value: float | None
    degenerate: bool
    passed: bool
    message: str


@dataclass(frozen=True)
class Verdict:
    """The pass or fail decision over the gating aggregates.

    Args:
        passed: Whether both gating p-values cleared ``alpha``.
        p_sum: The permutation p-value of the summed chi-square.
        p_max: The permutation p-value of the largest chi-square per dof.
        observed: The aggregates the real data gave.
        null: The null distribution the p-values came from.
        reasons: One entry per failing gate, empty when the run passed.
    """

    passed: bool
    p_sum: float
    p_max: float
    observed: AggregateStats
    null: PermutationNull
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    """Everything :func:`analyze` produced for one run.

    Args:
        node_stats: The statistics of every node that could be tested, in the
            order the indices in ``observed`` refer to.
        observed: The aggregates of the real data.
        null: The permutation null of those aggregates.
        verdict: The decision and its p-values.
    """

    node_stats: tuple[NodeStats, ...]
    observed: AggregateStats
    null: PermutationNull
    verdict: Verdict


def node_stats(bucketed: BucketedNode) -> NodeStats | None:
    """Computes one node's two-sample statistics.

    Args:
        bucketed: The node to test.

    Returns:
        The node's statistics, or ``None`` when one group never reached this
        prefix. Such a node needs no test of its own: the groups disagreeing
        about whether the prefix is reached at all is a difference in the
        parent node's next-token counts, where it is caught. A single-bucket
        node, where both groups always emitted the same token, has zero
        chi-square over zero degrees of freedom.
    """
    num_buckets = bucketed.num_buckets
    n_baseline, n_test = bucketed.n_per_group
    if n_baseline == 0 or n_test == 0:
        return None

    pooled = bucketed.pooled
    rows = bucketed.counts[0][np.newaxis, :]
    contributions = _chi_square_contributions(
        rows, pooled=pooled, n_baseline=n_baseline, n_test=n_test
    )[0]
    tvd = _total_variation(
        rows, pooled=pooled, n_baseline=n_baseline, n_test=n_test
    )[0]
    baseline_share = bucketed.counts[0].astype(np.float64) / n_baseline
    test_share = bucketed.counts[1].astype(np.float64) / n_test
    return NodeStats(
        bucketed=bucketed,
        chi_square=float(contributions.sum()),
        dof=num_buckets - 1,
        tvd=float(tvd),
        contributions=contributions,
        test_excess=test_share - baseline_share,
    )


def compute_node_stats(
    bucketed_nodes: Sequence[BucketedNode],
) -> list[NodeStats]:
    """Computes the statistics of every node given.

    Args:
        bucketed_nodes: The bucketed nodes to test, each reached by both
            groups, which the per-group prefix minimum guarantees.

    Returns:
        Their statistics, in the order given.

    Raises:
        ValueError: If only one group reached a node.
    """
    stats: list[NodeStats] = []
    for bucketed in bucketed_nodes:
        result = node_stats(bucketed)
        if result is None:
            node = bucketed.node
            raise ValueError(
                f"only one group reached prompt {node.prompt_index} prefix"
                f" {node.prefix}; node selection should have excluded it"
            )
        stats.append(result)
    return stats


def aggregate(
    node_stats: Sequence[NodeStats],
    *,
    effect_min_per_group: int,
    pruned: int | None = None,
) -> AggregateStats:
    """Reduces per-node statistics to the run-level aggregates.

    Args:
        node_stats: The per-node statistics.
        pruned: The observed pruned-prefix count, or ``None`` to leave it
            out.
        effect_min_per_group: The count each group needs at a node before its
            total variation distance is eligible for ``max_tvd``.

    Returns:
        The aggregates, with ``-1`` in place of an index no node attained.
    """
    sum_chi_square = 0.0
    sum_dof = 0
    max_ratio = 0.0
    max_node_index = -1
    max_tvd = 0.0
    max_tvd_node_index = -1
    weighted_tvd = 0.0
    weighted_divergence = 0.0
    total_pooled = 0
    forced_nodes = 0
    for index, stats in enumerate(node_stats):
        sum_chi_square += stats.chi_square
        sum_dof += stats.dof
        weighted_tvd += stats.pooled_count * stats.tvd
        total_pooled += stats.pooled_count
        if stats.dof == 0:
            forced_nodes += 1
            continue
        n_baseline, n_test = stats.n_per_group
        weighted_divergence += stats.pooled_count * float(
            _excess_divergence(
                np.asarray(stats.chi_square),
                dof=stats.dof,
                n_baseline=n_baseline,
                n_test=n_test,
            )
        )
        ratio = stats.chi_square_per_dof
        if max_node_index < 0 or ratio > max_ratio:
            max_ratio = ratio
            max_node_index = index
        if min(stats.n_per_group) >= effect_min_per_group and (
            max_tvd_node_index < 0 or stats.tvd > max_tvd
        ):
            max_tvd = stats.tvd
            max_tvd_node_index = index
    return AggregateStats(
        sum_chi_square=sum_chi_square,
        sum_dof=sum_dof,
        sum_chi_square_per_dof=(
            sum_chi_square / sum_dof if sum_dof > 0 else 0.0
        ),
        max_chi_square_per_dof=max_ratio,
        max_node_index=max_node_index,
        max_tvd=max_tvd,
        max_tvd_node_index=max_tvd_node_index,
        mean_tvd=weighted_tvd / total_pooled if total_pooled else 0.0,
        divergence=(
            math.sqrt(max(weighted_divergence / total_pooled, 0.0))
            if total_pooled
            else 0.0
        ),
        pruned=pruned,
        max_depth=max((stats.depth for stats in node_stats), default=0),
        mean_depth=(
            sum(stats.depth for stats in node_stats) / len(node_stats)
            if node_stats
            else 0.0
        ),
        forced_nodes=forced_nodes,
        num_nodes=len(node_stats),
    )


def permutation_null(
    node_stats: Sequence[NodeStats],
    *,
    num_permutations: int,
    rng: np.random.Generator,
    effect_min_per_group: int,
    pruned: npt.NDArray[np.int64] | None = None,
    max_depth: npt.NDArray[np.int64] | None = None,
    mean_depth: npt.NDArray[np.float64] | None = None,
) -> PermutationNull:
    """Draws the null distribution of the node aggregates.

    Args:
        node_stats: The per-node statistics.
        num_permutations: How many permutations to draw, which bounds every
            p-value's resolution.
        rng: The generator to draw with. Seeding belongs to the caller.
        effect_min_per_group: The count each group needs at a node before it
            contributes to the ``max_tvd`` null.
        pruned: The pruned-prefix count's null, one value per permutation;
            ``None`` leaves it out.
        max_depth: The deepest tested depth's null, drawn the same way;
            ``None`` leaves it out.
        mean_depth: The mean tested depth's null, likewise.

    Returns:
        The aggregates' null distributions.

    Raises:
        ValueError: If ``num_permutations`` is below 1, or if one of the
            relabeling nulls has a different length.
    """
    if num_permutations < 1:
        raise ValueError(
            f"num_permutations must be at least 1, got {num_permutations}"
        )
    for name, values in (
        ("pruned", pruned),
        ("max_depth", max_depth),
        ("mean_depth", mean_depth),
    ):
        if values is not None and len(values) != num_permutations:
            raise ValueError(
                f"{name} has {len(values)} values for {num_permutations}"
                " permutations"
            )
    sum_chi_square = np.zeros(num_permutations, dtype=np.float64)
    max_ratio = np.zeros(num_permutations, dtype=np.float64)
    max_tvd = np.zeros(num_permutations, dtype=np.float64)
    weighted_tvd = np.zeros(num_permutations, dtype=np.float64)
    weighted_divergence = np.zeros(num_permutations, dtype=np.float64)
    total_pooled = 0
    for stats in node_stats:
        total_pooled += stats.pooled_count
        if stats.dof == 0:
            # A single bucket permutes to itself: zero chi-square and TVD.
            continue
        pooled = stats.bucketed.pooled
        n_baseline, n_test = stats.n_per_group
        draws = rng.multivariate_hypergeometric(
            pooled, n_baseline, size=num_permutations
        )
        chi_square = _chi_square_contributions(
            draws, pooled=pooled, n_baseline=n_baseline, n_test=n_test
        ).sum(axis=1)
        sum_chi_square += chi_square
        np.maximum(max_ratio, chi_square / stats.dof, out=max_ratio)
        weighted_divergence += stats.pooled_count * _excess_divergence(
            chi_square, dof=stats.dof, n_baseline=n_baseline, n_test=n_test
        )
        tvd = _total_variation(
            draws, pooled=pooled, n_baseline=n_baseline, n_test=n_test
        )
        weighted_tvd += stats.pooled_count * tvd
        if min(stats.n_per_group) >= effect_min_per_group:
            np.maximum(max_tvd, tvd, out=max_tvd)
    return PermutationNull(
        sum_chi_square=sum_chi_square,
        max_chi_square_per_dof=max_ratio,
        mean_tvd=weighted_tvd / total_pooled if total_pooled else weighted_tvd,
        max_tvd=max_tvd,
        divergence=(
            np.sqrt(np.maximum(weighted_divergence / total_pooled, 0.0))
            if total_pooled
            else weighted_divergence
        ),
        pruned=pruned,
        max_depth=max_depth,
        mean_depth=mean_depth,
        num_permutations=num_permutations,
    )


def control_check(
    bucketed: BucketedNode | None,
    *,
    num_permutations: int,
    rng: np.random.Generator,
    alpha: float,
) -> ControlResult:
    """Tests one depth-0 node for agreement between the servers.

    Depth 0 is a control, not a test: both servers emit that token straight
    out of prefill with no drafting involved, so a difference there means they
    are not serving the same model the same way and nothing deeper is
    interpretable.

    Args:
        bucketed: The bucketed depth-0 node, or ``None`` when it had fewer
            than two buckets.
        num_permutations: How many permutations to draw for this one node.
        rng: The generator to draw with.
        alpha: The significance level the control must clear.

    Returns:
        The control's statistics, its p-value and whether it passed, with a
        message the caller can print before aborting.

    Raises:
        ValueError: If ``num_permutations`` is below 1.
    """
    if bucketed is None:
        return ControlResult(
            stats=None,
            null=None,
            p_value=None,
            degenerate=True,
            passed=False,
            message=(
                "degenerate: the control node has fewer than two token"
                " buckets, so there is nothing to compare."
            ),
        )

    stats = node_stats(bucketed)
    if stats is None:
        return ControlResult(
            stats=None,
            null=None,
            p_value=None,
            degenerate=True,
            passed=False,
            message=(
                "degenerate: only one server returned anything for this"
                " prompt, so there is no comparison to make."
            ),
        )

    null = permutation_null(
        [stats],
        num_permutations=num_permutations,
        rng=rng,
        effect_min_per_group=0,
    )
    p_value = null.p_value("sum", stats.chi_square)
    passed = p_value >= alpha
    if passed:
        message = "passed: the servers agree on the first token."
    else:
        message = (
            "FAILED: the servers disagree on the first token. Drafting"
            " cannot cause this, since both servers produce that token from"
            " prefill; they differ in weights, sampling defaults or numerics."
        )
    return ControlResult(
        stats=stats,
        null=null,
        p_value=p_value,
        degenerate=False,
        passed=passed,
        message=message,
    )


def verdict(
    observed: AggregateStats, null: PermutationNull, *, alpha: float
) -> Verdict:
    """Decides pass or fail from the gating aggregates.

    Only ``sum`` and ``max`` gate; the TVD, divergence, pruned-prefix and
    depth aggregates are reported alongside them and never fail a run on
    their own.

    Args:
        observed: The aggregates of the real data.
        null: The permutation null of those aggregates.
        alpha: The significance level each gating p-value must clear. The two
            gates share the run's false-alarm budget, so each gets half of it.

    Returns:
        The decision, both gating p-values and a reason per failing gate. A
        run with no tested nodes passes vacuously, which the caller should
        treat as no evidence rather than as a pass.
    """
    p_sum = null.p_value("sum", observed.sum_chi_square)
    p_max = null.p_value("max", observed.max_chi_square_per_dof)
    reasons: list[str] = []
    if p_sum < alpha:
        reasons.append(
            f"summed chi-square {observed.sum_chi_square:.1f} on"
            f" {observed.sum_dof} dof has permutation p = {p_sum:.3g} <"
            f" alpha = {alpha:.3g}"
        )
    if p_max < alpha:
        reasons.append(
            f"largest per-node chi-square/dof"
            f" {observed.max_chi_square_per_dof:.2f} at node"
            f" {observed.max_node_index} has permutation p = {p_max:.3g} <"
            f" alpha = {alpha:.3g}"
        )
    return Verdict(
        passed=not reasons,
        p_sum=p_sum,
        p_max=p_max,
        observed=observed,
        null=null,
        reasons=tuple(reasons),
    )


def analyze(
    bucketed_nodes: Sequence[BucketedNode],
    *,
    num_permutations: int,
    alpha: float,
    effect_min_per_group: int,
    rng: np.random.Generator,
    relabeling: RelabelingNull | None = None,
) -> AnalysisResult:
    """Runs the whole per-node-to-verdict analysis over a run's nodes.

    Args:
        bucketed_nodes: The bucketed nodes to test. Control nodes belong to
            :func:`control_check`, not here.
        num_permutations: How many permutations to draw for the null.
        alpha: The significance level the gating p-values must clear.
        effect_min_per_group: The count each group needs at a node before its
            total variation distance is eligible for ``max_tvd``.
        rng: The generator to draw with. Seeding belongs to the caller.
        relabeling: The pruned-prefix count and the depth nulls, when the
            caller drew them. These never gate.

    Returns:
        The per-node statistics, the aggregates, the null and the verdict.

    Raises:
        ValueError: If ``num_permutations`` is below 1, or if only one group
            reached a node.
    """
    stats = compute_node_stats(bucketed_nodes)
    observed = aggregate(
        stats,
        effect_min_per_group=effect_min_per_group,
        pruned=None if relabeling is None else relabeling.pruned,
    )
    null = permutation_null(
        stats,
        num_permutations=num_permutations,
        rng=rng,
        effect_min_per_group=effect_min_per_group,
        pruned=None if relabeling is None else relabeling.pruned_null,
        max_depth=None if relabeling is None else relabeling.max_depth_null,
        mean_depth=(None if relabeling is None else relabeling.mean_depth_null),
    )
    return AnalysisResult(
        node_stats=tuple(stats),
        observed=observed,
        null=null,
        verdict=verdict(observed, null, alpha=alpha),
    )


def _excess_divergence(
    chi_square: npt.NDArray[np.float64],
    *,
    dof: int,
    n_baseline: int,
    n_test: int,
) -> npt.NDArray[np.float64]:
    """Turns a node's chi-square into an estimate of the groups' divergence.

    Conditional on the margins, the chi-square's expectation under the null
    is ``dof * N / (N - 1)``, and under a real difference the excess over it
    is the divergence between the two next-token distributions scaled by the
    effective sample size ``n_baseline * n_test / N``. Undoing both leaves an
    estimate that neither grows with sample count nor sits on a noise floor.
    The result is signed, so the caller's average cancels node noise.

    Args:
        chi_square: One chi-square per draw, or a single observed one.
        dof: The node's degrees of freedom, at least 1.
        n_baseline: The first group's observations at the node.
        n_test: The second group's observations at the node.

    Returns:
        The estimated divergence for each element of ``chi_square``.
    """
    total = n_baseline + n_test
    expected = dof * total / (total - 1)
    return (chi_square - expected) * (total / (n_baseline * n_test))


def _chi_square_contributions(
    baseline_counts: npt.NDArray[np.int64],
    *,
    pooled: npt.NDArray[np.int64],
    n_baseline: int,
    n_test: int,
) -> npt.NDArray[np.float64]:
    """Computes the per-bucket chi-square terms of baseline count vectors.

    The margins fix the second group's counts at ``pooled - baseline`` and
    both groups' expected counts, so one first-group vector per row is all
    the input this needs, and a row of the result sums to that vector's
    chi-square statistic.
    """
    baseline = baseline_counts.astype(np.float64)
    pooled_float = pooled.astype(np.float64)
    test = pooled_float - baseline
    proportions = pooled_float / float(n_baseline + n_test)
    expected_baseline = n_baseline * proportions
    expected_test = n_test * proportions
    return _square_over(
        baseline - expected_baseline, expected_baseline
    ) + _square_over(test - expected_test, expected_test)


def _square_over(
    deviation: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes ``deviation ** 2 / expected``, zero where nothing is expected.

    A bucket nothing was expected in has no observations either, so treating
    it as a zero term keeps it from turning the statistic into a NaN.
    """
    return np.divide(
        deviation * deviation,
        expected,
        out=np.zeros_like(deviation),
        where=expected > 0.0,
    )


def _total_variation(
    baseline_counts: npt.NDArray[np.int64],
    *,
    pooled: npt.NDArray[np.int64],
    n_baseline: int,
    n_test: int,
) -> npt.NDArray[np.float64]:
    """Computes the total variation distance of baseline count vectors.

    Takes its input the same way ``_chi_square_contributions`` does and
    returns one distance per row.
    """
    baseline = baseline_counts.astype(np.float64)
    test = pooled.astype(np.float64) - baseline
    difference = baseline / n_baseline - test / n_test
    return 0.5 * np.abs(difference).sum(axis=1)
