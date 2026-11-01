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
"""Per-expert token counts and dispatch layouts for MoE kernel benchmarks.

A MoE FFN kernel sees the rows the router sent to its local experts. The
counts below derive those rows from the batch instead of picking them by hand,
so a benchmark covers the balanced case, random routing, and hot experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Rows per scale-factor block in the NVFP4 / MXFP8 scale tile.
SF_MN_GROUP_SIZE = 128


class Routing(str, Enum):
    """How tokens pick their top-k experts."""

    BALANCED = "balanced"
    """Every expert gets the floor or ceil of the average count."""
    UNIFORM = "uniform"
    """Each token picks top-k distinct experts uniformly at random."""
    SKEWED = "skewed"
    """Expert popularity follows a power law, so a few experts run hot."""


def routed_topk(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    routing: Routing,
    *,
    ep_size: int = 1,
    seed: int = 0,
    skew: float = 1.2,
) -> np.ndarray:
    """Returns each token's `top_k` distinct experts.

    Args:
        num_tokens: Tokens entering the MoE layer across all ranks.
        top_k: Experts each token is routed to.
        num_experts: Routed experts in the model.
        routing: How tokens pick their experts.
        ep_size: Expert-parallel degree, so `Routing.BALANCED` also balances
            the ranks.
        seed: Seed for the random routings.
        skew: Power-law exponent for `Routing.SKEWED`.

    Returns:
        An int32 array of shape `[num_tokens, top_k]`.
    """
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} exceeds num_experts {num_experts}")
    if routing is Routing.BALANCED:
        # Round-robin over experts interleaved across ranks: a token's picks
        # are distinct, every expert gets the floor or ceil of the average,
        # and the leftover picks spread evenly over the ranks.
        order = np.arange(num_experts).reshape(ep_size, -1).T.reshape(-1)
        slots = np.arange(num_tokens * top_k).reshape(num_tokens, top_k)
        return order[slots % num_experts].astype(np.int32)

    rng = np.random.default_rng(seed)
    if routing is Routing.UNIFORM:
        probs = None
    else:
        weights = 1.0 / np.arange(1, num_experts + 1) ** skew
        probs = rng.permutation(weights / weights.sum())
    return np.stack(
        [
            rng.choice(num_experts, size=top_k, replace=False, p=probs)
            for _ in range(num_tokens)
        ]
    ).astype(np.int32)


def routed_counts(topk_ids: np.ndarray, num_experts: int) -> np.ndarray:
    """Returns the per-expert row counts of a routing, as int64."""
    return np.bincount(topk_ids.reshape(-1), minlength=num_experts).astype(
        np.int64
    )


def rank_counts(counts: np.ndarray, ep_size: int) -> np.ndarray:
    """Splits global per-expert counts into `[ep_size, num_local_experts]`."""
    num_experts = counts.shape[0]
    if num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not split over {ep_size}")
    return counts.reshape(ep_size, num_experts // ep_size)


def pick_rank(counts: np.ndarray, ep_size: int, how: str = "busiest") -> int:
    """Returns the EP rank to benchmark.

    Args:
        counts: Global per-expert row counts.
        ep_size: Expert-parallel degree.
        how: `busiest` for the rank with the most rows, which sets an EP
            step's latency because the step waits for its slowest rank;
            `median` for a typical rank; or a rank index.

    Returns:
        The rank index.
    """
    totals = rank_counts(counts, ep_size).sum(axis=1)
    if how == "busiest":
        return int(np.argmax(totals))
    if how == "median":
        return int(np.argsort(totals, kind="stable")[(ep_size - 1) // 2])
    return int(how)


@dataclass(frozen=True)
class DispatchLayout:
    """The routing tensors an EP dispatch hands to the local expert FFN."""

    expert_start: np.ndarray
    """`[E + 1]` uint32 row prefix sums over the local experts."""
    a_scale_offsets: np.ndarray
    """`[E]` uint32; expert e's first scale block is
    `expert_start[e] // SF_MN_GROUP_SIZE + a_scale_offsets[e]`."""
    expert_ids: np.ndarray
    """`[E]` int32 expert id of each slot."""
    num_scale_blocks: int
    """Scale blocks the routed rows occupy."""

    @property
    def num_rows(self) -> int:
        return int(self.expert_start[-1])


def dispatch_layout(local_counts: np.ndarray) -> DispatchLayout:
    """Builds the dispatch outputs for one rank's per-expert counts.

    Each expert's scales start on a fresh `SF_MN_GROUP_SIZE`-row block, which
    is what the EP dispatch's `pad_expert_offsets` produces.
    """
    counts = local_counts.astype(np.int64)
    expert_start = np.concatenate([[0], np.cumsum(counts)])
    blocks = -(-counts // SF_MN_GROUP_SIZE)
    block_start = np.concatenate([[0], np.cumsum(blocks)])
    offsets = block_start[:-1] - expert_start[:-1] // SF_MN_GROUP_SIZE
    return DispatchLayout(
        expert_start=expert_start.astype(np.uint32),
        a_scale_offsets=offsets.astype(np.uint32),
        expert_ids=np.arange(counts.shape[0], dtype=np.int32),
        num_scale_blocks=int(block_start[-1]),
    )


def scale_rows_capacity(max_rows: int, num_local_experts: int) -> int:
    """Scale-tile rows for a `max_rows` token buffer, one pad block per expert."""
    return max_rows // SF_MN_GROUP_SIZE + num_local_experts
