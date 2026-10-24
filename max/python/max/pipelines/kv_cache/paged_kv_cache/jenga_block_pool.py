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
"""The fungible page pool that every flat KV cache draws from.

The pool itself, :class:`JengaBlockPool`, lives in Rust (``kv-core``) and is
re-exported here; this module plans the geometry it is built with.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import lcm

from max._kv_core import JengaBlockPool as JengaBlockPool
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

# A padded page may spend at most this share of itself on bytes the kernel
# never reads. A round allowance, not tuned to any model: enough for an awkward
# page to tile a far smaller huge block, while adding at most a tenth to it.
_MAX_PADDING_FRACTION = float(
    os.getenv("MODULAR_KV_JENGA_MAX_PADDING_FRACTION") or 0.1
)

# The search will not consider a huge block larger than this. Past it the
# allocation quantum is coarse enough that stranding beats any padding it saves.
_MAX_HUGE_PAGE_BYTES = int(
    os.getenv("MODULAR_KV_JENGA_MAX_HUGE_PAGE_BYTES") or 128 * 1024 * 1024
)

# How many little pages of one cache a huge block is considered to hold. Bounds
# the candidate search; ratios above this imply a page so much smaller than the
# block that padding it tightly is free anyway.
_MAX_TILING_RATIO = int(os.getenv("MODULAR_KV_JENGA_MAX_TILING_RATIO") or 256)

# The share of the budget whole huge blocks may strand before planning warns.
_STRANDING_WARN_FRACTION = 0.05


def compute_jenga_ratios(
    available_bytes: int,
    cache_sizes: Mapping[str, int],
    include_null_block: bool = True,
) -> tuple[int, int, dict[str, int]]:
    """Fits a byte budget to a huge block geometry every cache tiles exactly.

    A huge block is the least common multiple of the caches' page sizes, so it
    holds a whole number of pages of each: ``ratios[cache_id]`` of them. The
    budget therefore backs ``num_huge_blocks`` huge blocks -- that is,
    ``num_huge_blocks * ratios[cache_id]`` pages of each cache -- the first of
    which is the null block every cache shares.

    Args:
        available_bytes: The per-device KV budget the pool may occupy.
        cache_sizes: Each cache's page size in bytes.
        include_null_block: Whether to include the null block.

    Returns:
        How many huge blocks the budget holds, size of each huge block in bytes,
        and how many pages of each cache it holds.

    Raises:
        ValueError: If the arguments are not positive, or if the budget is too
            small to hold a null block and one allocatable block.
    """
    if len(cache_sizes) == 0:
        raise ValueError(f"cache_sizes must be non-empty, found {cache_sizes}")
    if any(size <= 0 for size in cache_sizes.values()):
        raise ValueError(f"cache_sizes must be positive, found {cache_sizes}")
    if available_bytes <= 0:
        raise ValueError(
            f"available_bytes must be positive, found {available_bytes}"
        )

    huge_page_bytes = lcm(*cache_sizes.values())
    num_huge_blocks = available_bytes // huge_page_bytes

    pages = ", ".join(
        f"{cache_id}={to_human_readable_bytes(size)}"
        for cache_id, size in cache_sizes.items()
    )
    if include_null_block and num_huge_blocks < 2:
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to "
            f"build a pool. A huge block is the least common multiple of the "
            f"page sizes ({pages}), so it takes "
            f"{to_human_readable_bytes(huge_page_bytes)}, and the pool needs "
            f"at least two of them -- "
            f"{to_human_readable_bytes(2 * huge_page_bytes)} -- because huge "
            f"block 0 is the null page every cache shares."
        )
    if num_huge_blocks < 1:
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to "
            f"build a pool. A huge block is the least common multiple of the "
            f"page sizes ({pages}), so it takes "
            f"{to_human_readable_bytes(huge_page_bytes)}, and the pool needs "
            f"at least one of them."
        )

    return (
        num_huge_blocks,
        huge_page_bytes,
        {
            cache_id: huge_page_bytes // size
            for cache_id, size in cache_sizes.items()
        },
    )


@dataclass(frozen=True)
class JengaGeometry:
    """A huge-block geometry every cache tiles exactly, once padded.

    :func:`compute_jenga_ratios` demands exact tiling, which makes the huge
    block the least common multiple of the page sizes -- cheap when they divide
    each other, ruinous when a coprime factor survives into one of them (an odd
    layer count, a head width that is not a power of two). Padding each page up
    to a divisor of a *searched* huge block buys tractability back: a leaf
    spends a few percent of its page on bytes the kernel never reads, and the
    huge block stays small enough to allocate against a real budget.

    """

    num_huge_blocks: int
    """How many huge blocks the budget holds, counting the null block when one
    was requested."""

    huge_page_bytes: int
    """Size of one huge block."""

    ratios: dict[str, int]
    """How many little pages of each cache one huge block holds."""

    padded_sizes: dict[str, int]
    """Each cache's page stride after padding. Every value divides
    ``huge_page_bytes``, and is either the cache's own page size or a multiple
    of its stride alignment."""

    def padding_fraction(self, cache_id: str, data_bytes: int) -> float:
        """Returns the share of ``cache_id``'s page that is padding."""
        return self.padded_sizes[cache_id] / data_bytes - 1


def _smallest_tiling_page(
    huge_page_bytes: int, data_bytes: int, stride_alignment: int | None
) -> int | None:
    """Returns the smallest page holding ``data_bytes`` that tiles the block.

    A page must divide ``huge_page_bytes``, so a huge block holds a whole
    number of them. One larger than the data must also be a multiple of
    ``stride_alignment``, the strides the cache's kernels can address; a cache
    without one gets only its data.

    Searching the ratio rather than the page keeps this cheap: the ratio cannot
    exceed ``huge_page_bytes // data_bytes``, a few hundred at the sizes we
    allocate, and the largest valid ratio yields the smallest page.

    Returns:
        The page size, or ``None`` when no divisor of ``huge_page_bytes``
        qualifies.
    """
    if huge_page_bytes % data_bytes == 0:
        return data_bytes
    if stride_alignment is None:
        return None
    for ratio in range(huge_page_bytes // data_bytes, 0, -1):
        if huge_page_bytes % ratio:
            continue
        page = huge_page_bytes // ratio
        if page % stride_alignment == 0:
            return page
    return None


def _fit_huge_page(
    huge_page_bytes: int,
    cache_sizes: Mapping[str, int],
    stride_alignments: Mapping[str, int | None],
    max_padding_fraction: float,
) -> dict[str, int] | None:
    """Pads every cache onto ``huge_page_bytes``, or gives up on it.

    Returns ``None`` if any cache cannot tile the block, or would have to spend
    more than ``max_padding_fraction`` of its page on padding to do so.
    """
    padded: dict[str, int] = {}
    for cache_id, data_bytes in cache_sizes.items():
        page = _smallest_tiling_page(
            huge_page_bytes, data_bytes, stride_alignments[cache_id]
        )
        if page is None or page / data_bytes - 1 > max_padding_fraction:
            return None
        padded[cache_id] = page
    return padded


def _padding_bytes(
    cache_sizes: Mapping[str, int], padded_sizes: Mapping[str, int]
) -> int:
    """Returns the padding bytes in one page of every cache."""
    return sum(padded_sizes.values()) - sum(cache_sizes.values())


def _huge_page_candidates(
    cache_sizes: Mapping[str, int],
    stride_alignments: Mapping[str, int | None],
    ceiling: int,
) -> list[int]:
    """Returns the huge-block sizes worth pricing, smallest first.

    A packed page divides the huge block and a padded one is a multiple of its
    alignment, so the block is a multiple of the lcm of both -- which is the
    grid the search runs on. On that grid the sizes that matter are the ones
    where some cache tiles *tightly*: for each cache and each small ratio, the
    smallest grid point at or above ``ratio * page``. That keeps the candidate
    set in the low thousands while still containing the optimum, since a
    geometry is only as good as its worst-padded leaf.
    """
    grid = lcm(
        *(
            data_bytes
            if (alignment := stride_alignments[cache_id]) is None
            else alignment
            for cache_id, data_bytes in cache_sizes.items()
        )
    )
    floor = max(cache_sizes.values())
    candidates = set()
    for data_bytes in cache_sizes.values():
        for ratio in range(1, _MAX_TILING_RATIO + 1):
            size = ceildiv(ratio * data_bytes, grid) * grid
            if size < floor:
                continue
            if size > ceiling:
                break
            candidates.add(size)
    return sorted(candidates)


def plan_jenga_geometry(
    available_bytes: int,
    cache_sizes: Mapping[str, int],
    stride_alignments: Mapping[str, int | None],
    *,
    include_null_block: bool = True,
    max_padding_fraction: float = _MAX_PADDING_FRACTION,
) -> JengaGeometry:
    """Fits a byte budget to a huge-block geometry, padding pages to get there.

    Only a cache with a stride alignment is padded. The search takes the huge
    block with the fewest padding bytes, then the smaller block, so the exact
    geometry wins whenever it fits under the ceiling. The budget only rules
    out blocks it cannot hold two of, so pools with different budgets, such
    as prefill and decode or dKV peers, agree on every page stride.

    Args:
        available_bytes: The per-device KV budget the pool may occupy.
        cache_sizes: Each cache's page size in bytes, before padding.
        stride_alignments: Each cache's stride alignment, or ``None`` for a
            cache that must stay packed; see
            :attr:`~max.nn.kv_cache.KVLeafRegion.stride_alignment`.
        include_null_block: Whether to include the null block.
        max_padding_fraction: The most of a page any one cache may spend on
            padding. Caps the search rather than the result: a geometry needing
            more is simply not considered. Defaults to
            ``MODULAR_KV_JENGA_MAX_PADDING_FRACTION``, itself 10%.

    Returns:
        The geometry to allocate.

    Raises:
        ValueError: If the arguments are not positive or disagree on cache ids,
            or if no geometry fits the budget.
    """
    if len(cache_sizes) == 0:
        raise ValueError(f"cache_sizes must be non-empty, found {cache_sizes}")
    if any(size <= 0 for size in cache_sizes.values()):
        raise ValueError(f"cache_sizes must be positive, found {cache_sizes}")
    if any(
        alignment is not None and alignment <= 0
        for alignment in stride_alignments.values()
    ):
        raise ValueError(
            f"stride_alignments must be positive or None, found "
            f"{stride_alignments}"
        )
    if cache_sizes.keys() != stride_alignments.keys():
        raise ValueError(
            f"cache_sizes and stride_alignments must cover the same caches, "
            f"found {sorted(cache_sizes)} and {sorted(stride_alignments)}"
        )
    if available_bytes <= 0:
        raise ValueError(
            f"available_bytes must be positive, found {available_bytes}"
        )

    min_huge_blocks = 2 if include_null_block else 1
    exact = lcm(*cache_sizes.values())
    candidates = _huge_page_candidates(
        cache_sizes, stride_alignments, _MAX_HUGE_PAGE_BYTES
    )
    if exact <= _MAX_HUGE_PAGE_BYTES:
        candidates.append(exact)

    best: tuple[int, int, dict[str, int]] | None = None
    for huge_page_bytes in candidates:
        if available_bytes // huge_page_bytes < min_huge_blocks:
            continue
        padded = _fit_huge_page(
            huge_page_bytes,
            cache_sizes,
            stride_alignments,
            max_padding_fraction,
        )
        if padded is None:
            continue
        cost = _padding_bytes(cache_sizes, padded)
        if best is None or (cost, huge_page_bytes) < best[:2]:
            best = (cost, huge_page_bytes, padded)

    if best is None and available_bytes // exact >= min_huge_blocks:
        best = (0, exact, dict(cache_sizes))

    if best is None:
        pages = ", ".join(
            f"{cache_id}={to_human_readable_bytes(size)}"
            for cache_id, size in cache_sizes.items()
        )
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to build "
            f"a pool from the page sizes ({pages}), even allowing each page "
            f"with a stride alignment to pad by {max_padding_fraction:.0%} to "
            f"tile a smaller huge block. Exact tiling would take "
            f"{to_human_readable_bytes(exact)} per huge block."
        )

    _cost, huge_page_bytes, padded_sizes = best
    return JengaGeometry(
        num_huge_blocks=available_bytes // huge_page_bytes,
        huge_page_bytes=huge_page_bytes,
        ratios={
            cache_id: huge_page_bytes // size
            for cache_id, size in padded_sizes.items()
        },
        padded_sizes=padded_sizes,
    )


def _stranding_warning(
    geometry: JengaGeometry, num_leaves: int, available_bytes: int
) -> str | None:
    """Returns a warning when the huge block can strand more than a threshold of the budget.

    In the worst case each leaf holds one nearly empty huge block, plus the
    null block.

    Args:
        geometry: The planned geometry.
        num_leaves: Leaves sharing the pool.
        available_bytes: The per-device budget the geometry was planned for.

    Returns:
        The warning, or ``None``.
    """
    stranded = (num_leaves + 1) * geometry.huge_page_bytes
    if stranded <= _STRANDING_WARN_FRACTION * available_bytes:
        return None
    return (
        f"Jenga KV geometry can strand {stranded / available_bytes:.1%} of "
        f"the per-device KV budget: {num_leaves} leaves plus the null block "
        f"each own whole {to_human_readable_bytes(geometry.huge_page_bytes)} "
        f"huge blocks, out of {to_human_readable_bytes(available_bytes)}. "
        f"Setting a stride alignment on the leaf that inflates the huge block "
        f"lets the planner pad it."
    )


def _pristine_pool_can_satisfy(
    allocatable_huge_blocks: int,
    cache_ratios: Mapping[str, int],
    demand: Mapping[str, int],
) -> bool:
    """Returns whether an empty pool can carve ``demand`` blocks per cache.

    A huge block is carved for exactly one cache, so every cache's demand,
    converted at its own ratio, is charged against the same huge blocks.
    Depends on the geometry alone, not on a pool instance.
    """
    claimable = allocatable_huge_blocks
    for cache_id, num_blocks in demand.items():
        if num_blocks > 0:
            claimable -= ceildiv(num_blocks, cache_ratios[cache_id])
    return claimable >= 0
