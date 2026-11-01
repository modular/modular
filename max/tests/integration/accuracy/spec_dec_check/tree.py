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
"""Prefix tree, node selection and bucketing for SpecDecCheck.

A node is one prefix of generated ids for one prompt, carrying the
distribution of the next token among the sequences that share that prefix,
counted per group, so the test at a node is a conditional-distribution test.
The two groups are the two servers in the main run and one server's two
temporal halves in the stability check, so nothing here names a server.
Bucketing then reduces a node's token counts to the cells the chi-square and
permutation tests consume, in a canonical order, since the permutation test
draws conditional on these exact buckets.

The tree is built breadth first and pruned: a prefix that fewer than
``min_per_group`` sequences of either group continue past cannot have a child
that clears the same bar, so it is neither reported nor extended, which bounds
the node count instead of letting it grow with every unique prefix. The bar is
per group rather than pooled because a node's power is set by its smaller
side; being symmetric, it still leaves the node set invariant under relabeling
the groups.

How many prefixes clear that bar is itself a statistic: two groups that
diverge send their sequences down different branches, so prefixes fail the
per-group minimum on imbalance rather than on traffic, and both the candidate
count and the deepest tested depth drop. :func:`relabeling_null` draws their
null by relabeling whole sequences.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .samples import Sample


@dataclass
class Node:
    """One prefix of generated ids for one prompt and its next-token counts.

    Args:
        prompt_index: Index of the prompt the samples under this node answer.
        prefix: The generated ids shared by those samples, empty at depth 0.
        counts: Per-group next-token counts, ordered like the ``groups``
            argument that built the tree.
    """

    prompt_index: int
    prefix: tuple[int, ...]
    counts: tuple[Counter[int], Counter[int]]

    @property
    def depth(self) -> int:
        """The prefix length, which is also the position under test."""
        return len(self.prefix)

    @property
    def pooled_count(self) -> int:
        """The number of next-token observations across both groups."""
        return sum(sum(counter.values()) for counter in self.counts)

    @property
    def group_counts(self) -> tuple[int, int]:
        """The number of next-token observations in each group."""
        return (
            sum(self.counts[0].values()),
            sum(self.counts[1].values()),
        )

    def pooled_counts(self) -> Counter[int]:
        """Merges the per-group next-token counts into one counter.

        Returns:
            Each next token's count summed over both groups.
        """
        merged: Counter[int] = Counter()
        for counter in self.counts:
            merged.update(counter)
        return merged

    def top_token_probability(self) -> float:
        """Computes the pooled share of the most frequent next token.

        Returns:
            That share, or ``0.0`` when the node has no observations.
        """
        merged = self.pooled_counts()
        total = sum(merged.values())
        if total == 0:
            return 0.0
        return max(merged.values()) / total

    def entropy_bits(self) -> float:
        """Computes the entropy of the pooled next-token distribution.

        Returns:
            The Shannon entropy in bits, ``0.0`` without observations.
        """
        merged = self.pooled_counts()
        total = sum(merged.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in merged.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy


@dataclass(frozen=True, eq=False)
class BucketedNode:
    """A node's next-token counts collapsed onto test buckets.

    Args:
        node: The node these buckets summarize.
        labels: The token id behind each bucket, ``None`` for the tail bucket.
        counts: One bucket-count vector per group, in ``groups`` order.
        tail_tokens: The token ids pooled into the tail, ascending, empty
            when every token cleared the bucket floor.
        tail_folded_into: The label of the bucket the tail was merged into,
            or ``None`` when the tail kept its own bucket or did not exist.
    """

    node: Node
    labels: tuple[int | None, ...]
    counts: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
    tail_tokens: tuple[int, ...]
    tail_folded_into: int | None

    @property
    def num_buckets(self) -> int:
        """The bucket count, which is the length of each count vector."""
        return len(self.labels)

    @property
    def n_per_group(self) -> tuple[int, int]:
        """The per-group observation totals, in ``groups`` order."""
        return (int(self.counts[0].sum()), int(self.counts[1].sum()))

    @property
    def pooled(self) -> npt.NDArray[np.int64]:
        """The bucket counts summed over both groups."""
        return self.counts[0] + self.counts[1]


def _server_group(sample: Sample) -> str:
    """Returns the server a sample came from, the default grouping."""
    return sample.server


def build_tree(
    samples: Sequence[Sample],
    *,
    min_per_group: int,
    groups: tuple[str, str] = ("baseline", "test"),
    group_of: Callable[[Sample], str] | None = None,
    max_depth: int | None = None,
) -> list[Node]:
    """Builds the pruned prefix tree over a run's samples.

    Args:
        samples: Every sample to compare, both groups and all prompts.
        min_per_group: The number of next-token observations each group
            needs at a prefix before it is reported or extended. A prompt's
            depth-0 node is reported whatever its counts.
        groups: The two group names, which also fix the order of every
            per-group pair this module produces.
        group_of: Which group a sample belongs to, by server when ``None``.
        max_depth: An exclusive bound on the depths built, for groups drawn
            under different token budgets. A budget only truncates the end of
            a sequence, so every depth below the smaller one stays a valid
            comparison while a deeper node would carry observations from one
            group alone. ``None`` builds every depth.

    Returns:
        The nodes, ordered by prompt index, then depth, then prefix.

    Raises:
        ValueError: This includes when ``min_per_group`` or ``max_depth`` is
            below 1, when the two group names are equal, and when a sample
            falls in a group outside ``groups``.
    """
    if min_per_group < 1:
        raise ValueError(
            f"min_per_group must be at least 1, got {min_per_group}"
        )
    if max_depth is not None and max_depth < 1:
        raise ValueError(
            f"max_depth must be at least 1 to keep the depth-0 control, got"
            f" {max_depth}"
        )
    by_prompt = _sequences_by_prompt(samples, groups=groups, group_of=group_of)
    nodes: list[Node] = []
    for prompt_index in sorted(by_prompt):
        sequences, sequence_groups = by_prompt[prompt_index]
        nodes.extend(
            _prompt_nodes(
                prompt_index,
                sequences,
                sequence_groups,
                min_per_group,
                max_depth,
            )
        )
    return nodes


def _sequences_by_prompt(
    samples: Sequence[Sample],
    *,
    groups: tuple[str, str],
    group_of: Callable[[Sample], str] | None,
) -> dict[int, tuple[list[tuple[int, ...]], list[int]]]:
    """Splits the samples by prompt into parallel sequence and group lists.

    The tree refers to sequences by position into these lists, so nothing
    per-sample is copied per node.

    Raises:
        ValueError: If the two group names are equal, or if a sample falls in
            a group outside ``groups``.
    """
    if groups[0] == groups[1]:
        raise ValueError(f"the two groups must differ, got {groups!r}")
    group_indices = {groups[0]: 0, groups[1]: 1}
    group_key = _server_group if group_of is None else group_of
    by_prompt: dict[int, tuple[list[tuple[int, ...]], list[int]]] = {}
    for sample in samples:
        group = group_key(sample)
        group_index = group_indices.get(group)
        if group_index is None:
            raise ValueError(f"sample group {group!r} is not one of {groups!r}")
        sequences, sequence_groups = by_prompt.setdefault(
            sample.prompt_index, ([], [])
        )
        sequences.append(sample.token_ids)
        sequence_groups.append(group_index)
    return by_prompt


def _prompt_nodes(
    prompt_index: int,
    sequences: Sequence[tuple[int, ...]],
    sequence_groups: Sequence[int],
    min_per_group: int,
    max_depth: int | None,
) -> list[Node]:
    """Builds one prompt's nodes breadth first, pruning short prefixes.

    ``sequence_groups`` holds the group index of each sequence, positionally
    aligned with ``sequences``.
    """
    nodes: list[Node] = []
    level: list[tuple[tuple[int, ...], list[int]]] = [
        ((), list(range(len(sequences))))
    ]
    depth = 0
    while level and (max_depth is None or depth < max_depth):
        next_level: list[tuple[tuple[int, ...], list[int]]] = []
        for prefix, members in level:
            counts: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
            children: dict[int, list[int]] = {}
            for member in members:
                sequence = sequences[member]
                if len(sequence) <= depth:
                    continue
                token = sequence[depth]
                counts[sequence_groups[member]][token] += 1
                children.setdefault(token, []).append(member)
            nodes.append(Node(prompt_index, prefix, counts))
            for token in sorted(children):
                child_members = children[token]
                # A child's observations are the members that go on past it,
                # and no grandchild can have more in either group, so this
                # bounds the tree.
                observations = [0, 0]
                for m in child_members:
                    if len(sequences[m]) > depth + 1:
                        observations[sequence_groups[m]] += 1
                if min(observations) >= min_per_group:
                    next_level.append((prefix + (token,), child_members))
        level = next_level
        depth += 1
    return nodes


@dataclass(frozen=True, eq=False)
class RelabelingDraws:
    """Three tree statistics under each whole-sequence relabeling.

    Args:
        candidates: How many prefixes at depth 1 and deeper cleared the
            per-group minimum, per relabeling.
        max_depth: The deepest depth at which a candidate also survives
            bucketing and so would be tested, per relabeling; ``0`` when none
            would.
        mean_depth: The mean depth of those tested prefixes, each counting
            once, per relabeling; ``0.0`` when none would be tested.
    """

    candidates: npt.NDArray[np.int64]
    max_depth: npt.NDArray[np.int64]
    mean_depth: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class _RelabelingSums:
    """The running per-relabeling totals the walk accumulates in place."""

    candidates: npt.NDArray[np.int64]
    max_depth: npt.NDArray[np.int64]
    tested: npt.NDArray[np.int64]
    tested_depth: npt.NDArray[np.int64]


def relabeling_null(
    samples: Sequence[Sample],
    *,
    min_per_group: int,
    bucket_floor: int,
    num_permutations: int,
    rng: np.random.Generator,
    groups: tuple[str, str] = ("baseline", "test"),
    group_of: Callable[[Sample], str] | None = None,
    max_depth: int | None = None,
) -> RelabelingDraws:
    """Draws the null of the candidate count and the deepest tested depth.

    The candidates are the prefixes at depth 1 and deeper that
    :func:`build_tree` keeps and :func:`select_nodes` returns: those that at
    least ``min_per_group`` sequences of each group continue past. Under the
    null the group labels are exchangeable, so this relabels whole sequences
    at random within each prompt, keeping each group's total, and counts the
    candidates that result. Which prefixes exist and how many sequences reach
    each one is fixed by the pooled samples, so only each prefix's split
    between the groups is random, and whether a candidate survives
    :func:`bucketize` is a fixed flag per prefix.

    The split is sampled exactly and top down rather than by shuffling
    labels: given how many first-group sequences a prefix has, the number
    that continue with each next token is a multivariate hypergeometric draw
    over its children, independent of every other branch, so one vectorized
    draw per child covers every permutation at once. A prefix that fewer than
    ``2 * min_per_group`` sequences continue past can never qualify, nor can
    anything below it, so the walk stops there.

    Args:
        samples: Every sample, both groups and all prompts.
        min_per_group: The per-group minimum given to :func:`build_tree`.
        bucket_floor: The floor given to :func:`bucketize`.
        num_permutations: How many relabelings to draw.
        rng: The generator to draw with. Seeding belongs to the caller.
        groups: The two group names, as for :func:`build_tree`.
        group_of: Which group a sample belongs to, by server when ``None``.
        max_depth: The exclusive depth bound given to :func:`build_tree`.

    Returns:
        Both statistics under each relabeling, over all prompts.

    Raises:
        ValueError: This includes when ``min_per_group``, ``bucket_floor`` or
            ``num_permutations`` is below 1, when the two group names are
            equal, and when a sample falls in a group outside ``groups``.
    """
    if min_per_group < 1:
        raise ValueError(
            f"min_per_group must be at least 1, got {min_per_group}"
        )
    if bucket_floor < 1:
        raise ValueError(f"bucket_floor must be at least 1, got {bucket_floor}")
    if num_permutations < 1:
        raise ValueError(
            f"num_permutations must be at least 1, got {num_permutations}"
        )
    by_prompt = _sequences_by_prompt(samples, groups=groups, group_of=group_of)
    sums = _RelabelingSums(
        candidates=np.zeros(num_permutations, dtype=np.int64),
        max_depth=np.zeros(num_permutations, dtype=np.int64),
        tested=np.zeros(num_permutations, dtype=np.int64),
        tested_depth=np.zeros(num_permutations, dtype=np.int64),
    )
    for sequences, sequence_groups in by_prompt.values():
        first_group = sum(1 for group in sequence_groups if group == 0)
        _walk_candidates(
            sequences,
            list(range(len(sequences))),
            np.full(num_permutations, first_group, dtype=np.int64),
            depth=0,
            min_per_group=min_per_group,
            bucket_floor=bucket_floor,
            max_depth=max_depth,
            rng=rng,
            sums=sums,
        )
    return RelabelingDraws(
        candidates=sums.candidates,
        max_depth=sums.max_depth,
        mean_depth=np.divide(
            sums.tested_depth,
            sums.tested,
            out=np.zeros(num_permutations, dtype=np.float64),
            where=sums.tested > 0,
        ),
    )


def _walk_candidates(
    sequences: Sequence[tuple[int, ...]],
    members: list[int],
    first: npt.NDArray[np.int64],
    *,
    depth: int,
    min_per_group: int,
    bucket_floor: int,
    max_depth: int | None,
    rng: np.random.Generator,
    sums: _RelabelingSums,
) -> None:
    """Accumulates the statistics at and below one prefix into ``sums``.

    ``members`` are the sequences carrying this prefix and ``first`` how many
    of them each relabeling put in the first group. Sequences that end here
    do not count as observations, so the first-group count among those that
    continue is one more hypergeometric draw.
    """
    continuing = [m for m in members if len(sequences[m]) > depth]
    ending = len(members) - len(continuing)
    if ending:
        first = rng.hypergeometric(len(continuing), ending, first)
    total = len(continuing)
    children: dict[int, list[int]] = {}
    for m in continuing:
        children.setdefault(sequences[m][depth], []).append(m)
    if depth >= 1:
        candidate = (first >= min_per_group) & (total - first >= min_per_group)
        np.add(sums.candidates, candidate, out=sums.candidates)
        sizes = {token: len(child) for token, child in children.items()}
        if _survives_bucketing(sizes, bucket_floor=bucket_floor):
            tested = np.where(candidate, depth, 0)
            np.maximum(sums.max_depth, tested, out=sums.max_depth)
            np.add(sums.tested, candidate, out=sums.tested)
            np.add(sums.tested_depth, tested, out=sums.tested_depth)
    if max_depth is not None and depth + 1 >= max_depth:
        return
    # Children are dealt from the remaining pool one at a time, which draws
    # the multivariate hypergeometric split marginal by marginal. Children
    # that cannot qualify stay in the pool without a draw of their own.
    remaining_total = total
    remaining_first = first
    for token in sorted(children):
        child = children[token]
        observations = sum(1 for m in child if len(sequences[m]) > depth + 1)
        if observations < 2 * min_per_group:
            continue
        drawn = rng.hypergeometric(
            len(child), remaining_total - len(child), remaining_first
        )
        remaining_total -= len(child)
        remaining_first = remaining_first - drawn
        _walk_candidates(
            sequences,
            child,
            drawn,
            depth=depth + 1,
            min_per_group=min_per_group,
            bucket_floor=bucket_floor,
            max_depth=max_depth,
            rng=rng,
            sums=sums,
        )


def count_prefixes(
    samples: Sequence[Sample], *, max_depth: int | None = None
) -> int:
    """Counts the distinct prefixes that could have become nodes.

    A prefix counts once per prompt when at least one sequence continues past
    it, at every depth from 1 up to but excluding ``max_depth``. Together with
    the pruned tree this says how many prefixes pruning removed.

    Args:
        samples: Every sample, all prompts.
        max_depth: The same exclusive depth bound given to :func:`build_tree`,
            or ``None`` for every depth the samples reach.

    Returns:
        The number of distinct ``(prompt, prefix)`` pairs.
    """
    longest = max((len(sample.token_ids) for sample in samples), default=0)
    bound = longest if max_depth is None else min(longest, max_depth)
    total = 0
    for depth in range(1, bound):
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for sample in samples:
            if len(sample.token_ids) > depth:
                seen.add((sample.prompt_index, sample.token_ids[:depth]))
        total += len(seen)
    return total


def control_nodes(nodes: Sequence[Node]) -> list[Node]:
    """Picks the depth-0 control node of each prompt.

    Args:
        nodes: Nodes from :func:`build_tree`.

    Returns:
        The depth-0 nodes, in prompt-index order.
    """
    return sorted(
        (node for node in nodes if node.depth == 0),
        key=lambda node: node.prompt_index,
    )


def select_nodes(
    nodes: Sequence[Node], *, min_per_group: int, min_depth: int = 1
) -> list[Node]:
    """Selects the conditional nodes with enough observations to test.

    Args:
        nodes: Nodes from :func:`build_tree`.
        min_per_group: The next-token count each group needs at a node.
        min_depth: The shallowest depth to keep. The default leaves depth 0
            to the equivalence control.

    Returns:
        The qualifying nodes, in the order given.
    """
    return [
        node
        for node in nodes
        if node.depth >= min_depth and min(node.group_counts) >= min_per_group
    ]


def bucketize(node: Node, *, bucket_floor: int) -> BucketedNode | None:
    """Collapses one node's next-token counts onto test buckets.

    Rare tokens pool into a tail bucket, which itself folds into the smallest
    kept bucket when the tail is under the floor too.

    Args:
        node: The node to bucket.
        bucket_floor: The pooled count a token, or the tail, needs to keep a
            bucket of its own.

    Returns:
        The bucketed node. A node whose every observation is one token is
        returned as a single bucket, since both groups agreeing completely is
        a result. ``None`` when several tokens were seen but fewer than two
        buckets survive the floor, which leaves nothing to compare.

    Raises:
        ValueError: If ``bucket_floor`` is below 1.
    """
    if bucket_floor < 1:
        raise ValueError(f"bucket_floor must be at least 1, got {bucket_floor}")
    pooled = node.pooled_counts()
    if len(pooled) == 1:
        (token,) = pooled
        return BucketedNode(
            node=node,
            labels=(token,),
            counts=(
                np.array([node.counts[0][token]], dtype=np.int64),
                np.array([node.counts[1][token]], dtype=np.int64),
            ),
            tail_tokens=(),
            tail_folded_into=None,
        )
    kept, tail_tokens, labels, fold_index = _bucket_plan(
        pooled, bucket_floor=bucket_floor
    )
    if len(labels) < 2:
        return None

    vectors: list[npt.NDArray[np.int64]] = []
    for counter in node.counts:
        vector = np.zeros(len(labels), dtype=np.int64)
        tail_count = sum(counter[token] for token in tail_tokens)
        for index, label in enumerate(labels):
            vector[index] = tail_count if label is None else counter[label]
        if fold_index is not None:
            vector[fold_index] += tail_count
        vectors.append(vector)

    return BucketedNode(
        node=node,
        labels=tuple(labels),
        counts=(vectors[0], vectors[1]),
        tail_tokens=tail_tokens,
        tail_folded_into=None if fold_index is None else kept[fold_index],
    )


def _bucket_plan(
    pooled: Mapping[int, int], *, bucket_floor: int
) -> tuple[list[int], tuple[int, ...], list[int | None], int | None]:
    """Decides a node's buckets from its pooled next-token counts.

    Returns:
        The kept tokens in canonical order, the tail tokens, the bucket
        labels with ``None`` standing for a tail bucket of its own, and the
        index of the kept bucket the tail folds into, if it does.
    """
    kept = sorted(
        (token for token, count in pooled.items() if count >= bucket_floor),
        key=lambda token: (-pooled[token], token),
    )
    tail_tokens = tuple(
        sorted(token for token, count in pooled.items() if count < bucket_floor)
    )
    tail_pooled = sum(pooled[token] for token in tail_tokens)

    labels: list[int | None] = list(kept)
    fold_index: int | None = None
    if tail_tokens:
        if tail_pooled >= bucket_floor or not kept:
            labels.append(None)
        else:
            # Under the canonical order the last kept bucket is the smallest,
            # so the tail lands there without disturbing the other positions.
            fold_index = len(kept) - 1
    return kept, tail_tokens, labels, fold_index


def _survives_bucketing(
    pooled: Mapping[int, int], *, bucket_floor: int
) -> bool:
    """Whether ``bucketize`` would keep a node with these pooled counts."""
    if len(pooled) == 1:
        return True
    _, _, labels, _ = _bucket_plan(pooled, bucket_floor=bucket_floor)
    return len(labels) >= 2


def bucketize_all(
    nodes: Sequence[Node], *, bucket_floor: int
) -> list[BucketedNode]:
    """Buckets every node, dropping the ones with nothing to compare.

    Args:
        nodes: The nodes to bucket.
        bucket_floor: The pooled count a token needs to keep its own bucket.

    Returns:
        The bucketed nodes, in the order given.

    Raises:
        ValueError: If ``bucket_floor`` is below 1.
    """
    bucketed: list[BucketedNode] = []
    for node in nodes:
        result = bucketize(node, bucket_floor=bucket_floor)
        if result is not None:
            bucketed.append(result)
    return bucketed
