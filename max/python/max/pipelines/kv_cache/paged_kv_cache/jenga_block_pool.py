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
# never reads. Zero by default.
_MAX_PADDING_FRACTION = float(
    os.getenv("MODULAR_KV_JENGA_MAX_PADDING_FRACTION") or 0.0
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
    """Each cache's page size after padding. Every value divides
    ``huge_page_bytes`` and is a whole number of that cache's rows."""

    def padding_fraction(self, cache_id: str, data_bytes: int) -> float:
        """Returns the share of ``cache_id``'s page that is padding."""
        return self.padded_sizes[cache_id] / data_bytes - 1


def _smallest_tiling_page(
    huge_page_bytes: int, data_bytes: int, row_bytes: int
) -> int | None:
    """Returns the smallest page holding ``data_bytes`` that tiles the block.

    A page must divide ``huge_page_bytes``, so a huge block holds a whole
    number of them, and must be a whole number of rows, because
    ``PagedKVCache`` expresses paging as row arithmetic -- ``_stride()`` divides
    the page stride by ``num_heads * head_size`` and that division has to be
    exact, or there is no row index for the start of a page.

    Searching the ratio rather than the page keeps this cheap: the ratio cannot
    exceed ``huge_page_bytes // data_bytes``, a few hundred at the sizes we
    allocate, and the largest valid ratio yields the smallest page.

    Returns:
        The page size, or ``None`` when no divisor of ``huge_page_bytes``
        qualifies.
    """
    for ratio in range(huge_page_bytes // data_bytes, 0, -1):
        if huge_page_bytes % ratio:
            continue
        page = huge_page_bytes // ratio
        if page % row_bytes == 0:
            return page
    return None


def _fit_huge_page(
    huge_page_bytes: int,
    cache_sizes: Mapping[str, int],
    row_sizes: Mapping[str, int],
    max_padding_fraction: float,
) -> dict[str, int] | None:
    """Pads every cache onto ``huge_page_bytes``, or gives up on it.

    Returns ``None`` if any cache cannot tile the block, or would have to spend
    more than ``max_padding_fraction`` of its page on padding to do so.
    """
    padded: dict[str, int] = {}
    for cache_id, data_bytes in cache_sizes.items():
        page = _smallest_tiling_page(
            huge_page_bytes, data_bytes, row_sizes[cache_id]
        )
        if page is None or page / data_bytes - 1 > max_padding_fraction:
            return None
        padded[cache_id] = page
    return padded


def _geometry_cost(
    available_bytes: int,
    huge_page_bytes: int,
    cache_sizes: Mapping[str, int],
    padded_sizes: Mapping[str, int],
    include_null_block: bool,
) -> float | None:
    """Returns the share of the budget a geometry cannot put cache data in.

    Two terms, and they pull against each other -- which is why the huge block
    is chosen on their sum rather than on padding alone. A larger block lets
    every page land closer above its data (less padding) but quantises the
    budget more coarsely, so it strands a bigger remainder and spends more on
    the null block. Minimising padding alone walks straight into the coarse end.

    Returns ``None`` when the budget cannot hold the blocks the pool needs.
    """
    num_huge_blocks = available_bytes // huge_page_bytes
    if num_huge_blocks < (2 if include_null_block else 1):
        return None

    padding = sum(
        padded_sizes[cache_id] / data_bytes - 1
        for cache_id, data_bytes in cache_sizes.items()
    ) / len(cache_sizes)
    unusable = available_bytes - num_huge_blocks * huge_page_bytes
    if include_null_block:
        unusable += huge_page_bytes
    return padding + unusable / available_bytes


def _huge_page_candidates(
    cache_sizes: Mapping[str, int],
    row_sizes: Mapping[str, int],
    ceiling: int,
) -> list[int]:
    """Returns the huge-block sizes worth pricing, smallest first.

    Every page is a whole number of its own rows and divides the huge block, so
    the block is necessarily a multiple of ``lcm(row_sizes)`` -- which is the
    grid the search runs on. On that grid the sizes that matter are the ones
    where some cache tiles *tightly*: for each cache and each small ratio, the
    smallest grid point at or above ``ratio * page``. That keeps the candidate
    set in the low thousands while still containing the optimum, since a
    geometry is only as good as its worst-padded leaf.
    """
    grid = lcm(*row_sizes.values())
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
    row_sizes: Mapping[str, int],
    *,
    include_null_block: bool = True,
    max_padding_fraction: float = _MAX_PADDING_FRACTION,
) -> JengaGeometry:
    """Fits a byte budget to a huge-block geometry, padding pages to get there.

    The exact-tiling geometry :func:`compute_jenga_ratios` computes is priced
    first and wins whenever it is competitive, so a model whose page sizes
    already divide each other is allocated exactly as before. When a coprime
    factor makes that least common multiple too coarse to allocate -- the case
    that takes a vision tower pooled with a text cache to a 23.7 GiB huge block
    -- the search pads each page up to a divisor of a smaller block instead.

    Args:
        available_bytes: The per-device KV budget the pool may occupy.
        cache_sizes: Each cache's page size in bytes, before padding.
        row_sizes: The granularity each cache's padded page must be a whole
            number of -- a row, ``num_heads * head_size * dtype_size``, widened
            where a kernel needs more alignment than a row gives.
        include_null_block: Whether to include the null block.
        max_padding_fraction: The most of a page any one cache may spend on
            padding. Caps the search rather than the result: a geometry needing
            more is simply not considered. Defaults to
            ``MODULAR_KV_JENGA_MAX_PADDING_FRACTION``, itself zero.

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
    if any(size <= 0 for size in row_sizes.values()):
        raise ValueError(f"row_sizes must be positive, found {row_sizes}")
    if cache_sizes.keys() != row_sizes.keys():
        raise ValueError(
            f"cache_sizes and row_sizes must cover the same caches, found "
            f"{sorted(cache_sizes)} and {sorted(row_sizes)}"
        )
    for cache_id, data_bytes in cache_sizes.items():
        if data_bytes % row_sizes[cache_id]:
            raise ValueError(
                f"cache {cache_id!r} holds {data_bytes} bytes per page, which "
                f"is not a whole number of its {row_sizes[cache_id]}-byte "
                f"rows. A page the kernel addresses by row cannot be a "
                f"fraction of one."
            )
    if available_bytes <= 0:
        raise ValueError(
            f"available_bytes must be positive, found {available_bytes}"
        )

    best: tuple[float, int, dict[str, int]] | None = None

    exact = lcm(*cache_sizes.values())
    exact_cost = _geometry_cost(
        available_bytes, exact, cache_sizes, cache_sizes, include_null_block
    )
    if exact_cost is not None:
        best = (exact_cost, exact, dict(cache_sizes))

    ceiling = min(_MAX_HUGE_PAGE_BYTES, available_bytes // 2)
    for huge_page_bytes in _huge_page_candidates(
        cache_sizes, row_sizes, ceiling
    ):
        padded = _fit_huge_page(
            huge_page_bytes, cache_sizes, row_sizes, max_padding_fraction
        )
        if padded is None:
            continue
        cost = _geometry_cost(
            available_bytes,
            huge_page_bytes,
            cache_sizes,
            padded,
            include_null_block,
        )
        # Ties go to the smaller block: same cost, finer allocation quantum.
        if cost is not None and (best is None or cost < best[0]):
            best = (cost, huge_page_bytes, padded)

    if best is None:
        pages = ", ".join(
            f"{cache_id}={to_human_readable_bytes(size)}"
            for cache_id, size in cache_sizes.items()
        )
        raise ValueError(
            f"{to_human_readable_bytes(available_bytes)} is too small to build "
            f"a pool from the page sizes ({pages}), even allowing each page to "
            f"pad by {max_padding_fraction:.0%} to tile a smaller huge block. "
            f"Exact tiling would take "
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
