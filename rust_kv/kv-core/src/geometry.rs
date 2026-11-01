//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//! The huge-block geometry a Jenga pool is carved from.
//!
//! A port of the geometry half of MAX's
//! `max/python/max/pipelines/kv_cache/paged_kv_cache/jenga_block_pool.py`
//! (`compute_jenga_ratios`, `JengaGeometry`, `plan_jenga_geometry` and its
//! helpers, `_pristine_pool_can_satisfy`) and of the capacity search in
//! `jenga_block_manager.py` (`_max_seq_len_fitting_in_geometry`). The Python
//! is still what MAX runs.
//!
//! It differs from the Python in four ways:
//!
//! - A cache is its position in the input slice rather than a string id, and
//!   every per-cache output is in input order. Python's check that
//!   `cache_sizes` and `stride_alignments` name the same caches has no
//!   counterpart: both live in one [`CachePage`].
//! - The search bounds Python reads from `MODULAR_KV_JENGA_*` at import are
//!   a [`SearchBounds`] argument. This crate reads no environment.
//! - Byte counts are `u64` and block counts `usize`. A least common multiple
//!   that would grow past `u64` is larger than any budget, which is all
//!   Python concludes from it too.
//! - The padding cap is checked in plain `f64` arithmetic, so a page right at
//!   the cap can differ from Python's in the last bit. The cost a geometry is
//!   chosen on is whole bytes, as in Python.

use std::{collections::BTreeSet, fmt};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;
const TIB: u64 = 1024 * GIB;

/// Bounds on the padded geometry search.
///
/// Python reads each from an environment variable at import; the defaults
/// here are the ones it falls back to when the variable is unset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchBounds {
    /// The most of a page any one cache with a stride alignment may spend on
    /// bytes the kernel never reads. It caps the search rather than the
    /// result: a geometry needing more is not considered.
    /// `MODULAR_KV_JENGA_MAX_PADDING_FRACTION`, 10% by default.
    pub max_padding_fraction: f64,
    /// The largest huge block the search considers. Past it the allocation
    /// quantum is coarse enough that stranding beats any padding it saves.
    /// `MODULAR_KV_JENGA_MAX_HUGE_PAGE_BYTES`, 128 MiB by default.
    pub max_huge_page_bytes: u64,
    /// How many little pages of one cache a huge block is considered to hold.
    /// Bounds the candidate search; a ratio above it implies a page so much
    /// smaller than the block that padding it tightly is free anyway.
    /// `MODULAR_KV_JENGA_MAX_TILING_RATIO`, 256 by default.
    pub max_tiling_ratio: u64,
}

impl Default for SearchBounds {
    fn default() -> Self {
        SearchBounds {
            max_padding_fraction: 0.1,
            max_huge_page_bytes: 128 * MIB,
            max_tiling_ratio: 256,
        }
    }
}

/// One cache's page, as the planner sizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachePage {
    /// Bytes one page holds, before padding.
    pub data_bytes: u64,
    /// The bytes a padded page stride must be a multiple of, or `None` for a
    /// cache that must stay packed. Python's `KVLeafRegion.stride_alignment`.
    pub stride_alignment: Option<u64>,
}

/// A huge-block geometry every cache tiles exactly, once padded.
///
/// [`compute_jenga_ratios`] demands exact tiling, which makes the huge block
/// the least common multiple of the page sizes: cheap when they divide each
/// other, ruinous when a coprime factor survives into one of them (an odd
/// layer count, a head width that is not a power of two).
/// [`plan_jenga_geometry`] pads each page up to a divisor of a searched,
/// smaller huge block instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JengaGeometry {
    /// How many huge blocks the budget holds, counting the null block when
    /// one was requested.
    pub num_huge_blocks: usize,
    /// Size of one huge block.
    pub huge_page_bytes: u64,
    /// How many little pages of each cache one huge block holds.
    pub ratios: Vec<usize>,
    /// Each cache's page stride after padding. Every value divides
    /// `huge_page_bytes`, and is either the cache's own page size or a
    /// multiple of its stride alignment.
    pub padded_sizes: Vec<u64>,
}

impl JengaGeometry {
    /// Returns the share of `cache`'s padded page that is padding.
    ///
    /// # Panics
    /// If the geometry has no cache `cache`.
    #[must_use]
    pub fn padding_fraction(&self, cache: usize, data_bytes: u64) -> f64 {
        self.padded_sizes[cache] as f64 / data_bytes as f64 - 1.0
    }
}

/// A budget or a set of pages no pool can be built from. Python raises
/// `ValueError` for each.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    /// There are no caches to tile.
    NoCaches,
    /// A cache's page holds no bytes.
    ZeroPageSize {
        /// The cache, by its position in the input.
        cache: usize,
    },
    /// A cache's stride alignment is zero.
    ZeroStrideAlignment {
        /// The cache, by its position in the input.
        cache: usize,
    },
    /// The budget is zero.
    ZeroBudget,
    /// [`compute_jenga_ratios`]: the budget holds too few exactly tiled huge
    /// blocks -- two with a null block, one without.
    BudgetBelowExactTiling {
        /// The budget.
        available_bytes: u64,
        /// Every cache's page size.
        cache_sizes: Vec<u64>,
        /// The least common multiple of `cache_sizes`, or `None` past
        /// `u64::MAX`.
        huge_page_bytes: Option<u64>,
        /// Whether the pool reserves huge block 0 as its null block.
        include_null_block: bool,
    },
    /// [`plan_jenga_geometry`]: no geometry, exact or padded, fits the budget.
    NoGeometryFits {
        /// The budget.
        available_bytes: u64,
        /// Every cache's page size, before padding.
        cache_sizes: Vec<u64>,
        /// The padding each page was allowed.
        max_padding_fraction: f64,
        /// The exactly tiled huge block, or `None` past `u64::MAX`.
        exact_huge_page_bytes: Option<u64>,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometryError::NoCaches => write!(f, "cache_sizes must be non-empty"),
            GeometryError::ZeroPageSize { cache } => {
                write!(f, "cache_sizes must be positive, found 0 for cache {cache}")
            }
            GeometryError::ZeroStrideAlignment { cache } => write!(
                f,
                "stride_alignments must be positive or None, found 0 for cache {cache}"
            ),
            GeometryError::ZeroBudget => write!(f, "available_bytes must be positive, found 0"),
            GeometryError::BudgetBelowExactTiling {
                available_bytes,
                cache_sizes,
                huge_page_bytes,
                include_null_block,
            } => {
                write!(
                    f,
                    "{} is too small to build a pool. A huge block is the least common \
                     multiple of the page sizes ({}), so it takes {}, and the pool needs at \
                     least ",
                    human_bytes(*available_bytes),
                    list_pages(cache_sizes),
                    human_lcm(*huge_page_bytes),
                )?;
                if *include_null_block {
                    let two = huge_page_bytes.and_then(|bytes| bytes.checked_mul(2));
                    write!(
                        f,
                        "two of them -- {} -- because huge block 0 is the null page every \
                         cache shares.",
                        human_lcm(two),
                    )
                } else {
                    write!(f, "one of them.")
                }
            }
            GeometryError::NoGeometryFits {
                available_bytes,
                cache_sizes,
                max_padding_fraction,
                exact_huge_page_bytes,
            } => write!(
                f,
                "{} is too small to build a pool from the page sizes ({}), even allowing \
                 each page with a stride alignment to pad by {:.0}% to tile a smaller huge \
                 block. Exact tiling would take {} per huge block.",
                human_bytes(*available_bytes),
                list_pages(cache_sizes),
                max_padding_fraction * 100.0,
                human_lcm(*exact_huge_page_bytes),
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

/// Fits a byte budget to a huge-block geometry every cache tiles exactly.
///
/// A huge block is the least common multiple of the caches' page sizes, so it
/// holds a whole number of pages of each: `ratios[cache]` of them. The budget
/// therefore backs `num_huge_blocks` huge blocks -- that is,
/// `num_huge_blocks * ratios[cache]` pages of each cache -- the first of which
/// is the null block every cache shares when `include_null_block` is set.
/// Nothing is padded, so `padded_sizes` is `cache_sizes`.
///
/// # Errors
/// If there is no cache, a page or the budget is zero, or the budget is too
/// small to hold a null block (if asked for) and one allocatable block.
pub fn compute_jenga_ratios(
    available_bytes: u64,
    cache_sizes: &[u64],
    include_null_block: bool,
) -> Result<JengaGeometry, GeometryError> {
    if cache_sizes.is_empty() {
        return Err(GeometryError::NoCaches);
    }
    if let Some(cache) = cache_sizes.iter().position(|&size| size == 0) {
        return Err(GeometryError::ZeroPageSize { cache });
    }
    if available_bytes == 0 {
        return Err(GeometryError::ZeroBudget);
    }

    let huge_page_bytes = lcm_of(cache_sizes.iter().copied());
    match huge_page_bytes {
        Some(huge) if available_bytes / huge >= min_huge_blocks(include_null_block) => {
            Ok(JengaGeometry {
                num_huge_blocks: count(available_bytes / huge),
                huge_page_bytes: huge,
                ratios: cache_sizes.iter().map(|&size| count(huge / size)).collect(),
                padded_sizes: cache_sizes.to_vec(),
            })
        }
        _ => Err(GeometryError::BudgetBelowExactTiling {
            available_bytes,
            cache_sizes: cache_sizes.to_vec(),
            huge_page_bytes,
            include_null_block,
        }),
    }
}

/// Fits a byte budget to a huge-block geometry, padding pages to get there.
///
/// Only a cache with a stride alignment is padded. The search takes the huge
/// block with the fewest padding bytes, then the smaller block, so the exact
/// geometry wins whenever it fits under the ceiling. The budget only rules
/// out blocks it cannot hold two of, so pools with different budgets agree
/// on every page stride.
///
/// # Errors
/// If there is no cache, a page, an alignment or the budget is zero, or no
/// geometry fits the budget.
pub fn plan_jenga_geometry(
    available_bytes: u64,
    pages: &[CachePage],
    include_null_block: bool,
    bounds: &SearchBounds,
) -> Result<JengaGeometry, GeometryError> {
    if pages.is_empty() {
        return Err(GeometryError::NoCaches);
    }
    if let Some(cache) = pages.iter().position(|page| page.data_bytes == 0) {
        return Err(GeometryError::ZeroPageSize { cache });
    }
    if let Some(cache) = pages
        .iter()
        .position(|page| page.stride_alignment == Some(0))
    {
        return Err(GeometryError::ZeroStrideAlignment { cache });
    }
    if available_bytes == 0 {
        return Err(GeometryError::ZeroBudget);
    }

    let cache_sizes: Vec<u64> = pages.iter().map(|page| page.data_bytes).collect();
    let min_blocks = min_huge_blocks(include_null_block);
    let exact = lcm_of(cache_sizes.iter().copied());
    let mut candidates =
        huge_page_candidates(pages, bounds.max_huge_page_bytes, bounds.max_tiling_ratio);
    if let Some(exact) = exact
        && exact <= bounds.max_huge_page_bytes
    {
        candidates.push(exact);
    }

    let data_total: u64 = cache_sizes.iter().sum();
    let mut best: Option<(u64, u64, Vec<u64>)> = None;
    for huge_page_bytes in candidates {
        if available_bytes / huge_page_bytes < min_blocks {
            continue;
        }
        let Some(padded) = fit_huge_page(huge_page_bytes, pages, bounds.max_padding_fraction)
        else {
            continue;
        };
        let cost = padded.iter().sum::<u64>() - data_total;
        if best.as_ref().is_none_or(|(best_cost, best_huge, _)| {
            (cost, huge_page_bytes) < (*best_cost, *best_huge)
        }) {
            best = Some((cost, huge_page_bytes, padded));
        }
    }
    if best.is_none()
        && let Some(exact) = exact
        && available_bytes / exact >= min_blocks
    {
        best = Some((0, exact, cache_sizes.clone()));
    }

    let Some((_cost, huge_page_bytes, padded_sizes)) = best else {
        return Err(GeometryError::NoGeometryFits {
            available_bytes,
            cache_sizes,
            max_padding_fraction: bounds.max_padding_fraction,
            exact_huge_page_bytes: exact,
        });
    };
    Ok(JengaGeometry {
        num_huge_blocks: count(available_bytes / huge_page_bytes),
        huge_page_bytes,
        ratios: padded_sizes
            .iter()
            .map(|&size| count(huge_page_bytes / size))
            .collect(),
        padded_sizes,
    })
}

/// Returns the smallest page holding `data_bytes` that tiles the block.
///
/// A page must divide `huge_page_bytes`, so a huge block holds a whole number
/// of them. One larger than the data must also be a multiple of
/// `stride_alignment`, the strides the cache's kernels can address; a cache
/// without one gets only its data.
///
/// Searching the ratio rather than the page keeps this cheap: the ratio
/// cannot exceed `huge_page_bytes / data_bytes`, a few hundred at the sizes
/// pools allocate, and the largest valid ratio yields the smallest page.
fn smallest_tiling_page(
    huge_page_bytes: u64,
    data_bytes: u64,
    stride_alignment: Option<u64>,
) -> Option<u64> {
    if huge_page_bytes.is_multiple_of(data_bytes) {
        return Some(data_bytes);
    }
    let alignment = stride_alignment?;
    (1..=huge_page_bytes / data_bytes).rev().find_map(|ratio| {
        let page = huge_page_bytes / ratio;
        (huge_page_bytes.is_multiple_of(ratio) && page.is_multiple_of(alignment)).then_some(page)
    })
}

/// Pads every cache onto `huge_page_bytes`, or gives up on it.
///
/// `None` if any cache cannot tile the block, or would have to spend more than
/// `max_padding_fraction` of its page on padding to do so.
fn fit_huge_page(
    huge_page_bytes: u64,
    pages: &[CachePage],
    max_padding_fraction: f64,
) -> Option<Vec<u64>> {
    pages
        .iter()
        .map(|page| {
            let padded =
                smallest_tiling_page(huge_page_bytes, page.data_bytes, page.stride_alignment)?;
            // Not `<=`: a NaN bound rejects nothing in Python.
            if padded as f64 / page.data_bytes as f64 - 1.0 > max_padding_fraction {
                None
            } else {
                Some(padded)
            }
        })
        .collect()
}

/// Returns the huge-block sizes worth pricing, smallest first.
///
/// A packed page divides the huge block and a padded one is a multiple of its
/// alignment, so the block is a multiple of the lcm of both -- which is the
/// grid the search runs on. On that grid the sizes that matter
/// are the ones where some cache tiles tightly: for each cache and each small
/// ratio, the smallest grid point at or above `ratio * page`. That keeps the
/// candidate set in the low thousands while still containing the optimum,
/// since a geometry is only as good as its worst-padded leaf.
fn huge_page_candidates(pages: &[CachePage], ceiling: u64, max_tiling_ratio: u64) -> Vec<u64> {
    // A grid past `u64::MAX` has no point under the ceiling.
    let Some(grid) = lcm_of(
        pages
            .iter()
            .map(|page| page.stride_alignment.unwrap_or(page.data_bytes)),
    ) else {
        return Vec::new();
    };
    let floor = pages.iter().map(|page| page.data_bytes).max().unwrap_or(0);
    let mut candidates = BTreeSet::new();
    for page in pages {
        for ratio in 1..=max_tiling_ratio {
            // Sizes only grow with the ratio, so one past `u64::MAX` is past
            // the ceiling for this cache and every larger ratio.
            let Some(size) = ratio
                .checked_mul(page.data_bytes)
                .and_then(|bytes| bytes.div_ceil(grid).checked_mul(grid))
            else {
                break;
            };
            if size < floor {
                continue;
            }
            if size > ceiling {
                break;
            }
            candidates.insert(size);
        }
    }
    candidates.into_iter().collect()
}

/// Returns whether an empty pool can carve `demand[cache]` blocks per cache.
///
/// A huge block is carved for exactly one cache, so every cache's demand,
/// converted at its own ratio, is charged against the same huge blocks.
/// Depends on the geometry alone, not on a pool instance.
///
/// # Panics
/// If `cache_ratios` and `demand` differ in length, or a cache with demand
/// has a ratio of zero.
#[must_use]
pub fn pristine_pool_can_satisfy(
    allocatable_huge_blocks: usize,
    cache_ratios: &[usize],
    demand: &[usize],
) -> bool {
    assert_eq!(
        cache_ratios.len(),
        demand.len(),
        "demand must name every cache of the geometry, and only those",
    );
    // Subtract rather than total the claims: a total past `usize::MAX` would
    // have to saturate, and a saturated total fits a pool of `usize::MAX`.
    demand
        .iter()
        .zip(cache_ratios)
        .filter(|&(&num_blocks, _)| num_blocks > 0)
        .try_fold(
            allocatable_huge_blocks,
            |remaining, (&num_blocks, &ratio)| remaining.checked_sub(num_blocks.div_ceil(ratio)),
        )
        .is_some()
}

/// How many blocks a leaf draws for one request, by the shape of the leaf.
///
/// The four kinds of `KVLeafRegion` MAX pools, reduced to their
/// `blocks_to_reserve`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafDemand {
    /// A full-attention leaf keeps every block of the request.
    Full,
    /// A sliding-window leaf keeps only the blocks its window spans.
    SlidingWindow {
        /// Tokens the window spans, the query token included.
        window_size: usize,
        /// Tokens per page of this leaf.
        page_size: usize,
    },
    /// A recurrent state keeps its live block, plus a checkpoint behind it
    /// when prefix caching is on.
    Recurrent,
    /// Per-request scratch keeps one block for the request's whole life.
    Scratch,
}

impl LeafDemand {
    /// Returns how many blocks one request draws to fill `num_blocks` slots.
    ///
    /// Fewer than `num_blocks` when some slots hold the null block: a sliding
    /// window keeps only its span, and a state keeps only its live block and
    /// one checkpoint.
    ///
    /// # Panics
    /// If a sliding window's `page_size` is zero.
    #[must_use]
    pub fn blocks_to_reserve(&self, num_blocks: usize, enable_prefix_caching: bool) -> usize {
        match *self {
            LeafDemand::Full => num_blocks,
            LeafDemand::SlidingWindow {
                window_size,
                page_size,
            } => num_blocks.min(window_size.div_ceil(page_size)),
            LeafDemand::Recurrent => {
                if enable_prefix_caching {
                    2
                } else {
                    1
                }
            }
            LeafDemand::Scratch => 1,
        }
    }
}

/// The longest single request an empty pool serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxSeqLen {
    /// No length exhausts the pool: every leaf's demand plateaus (sliding
    /// windows, states). Python returns `None`.
    Unbounded,
    /// The longest request, in tokens.
    Longest(usize),
    /// Not even an empty request fits: the blocks the state leaves draw
    /// up front already overflow the pool. Python returns -1.
    NothingFits,
}

/// Returns the longest single request an empty pool of this geometry serves.
///
/// Binary search over the fit, which is monotonic because no leaf's demand
/// decreases with length. `leaves[cache]` and `cache_ratios[cache]` describe
/// the same cache. `allocatable_huge_blocks` excludes the null block.
///
/// Where Python's search bound would grow past `usize::MAX` this one stops
/// there, so a request longer than any `usize` reads as
/// [`MaxSeqLen::Unbounded`].
///
/// # Panics
/// If there are no leaves, `leaves` and `cache_ratios` differ in length,
/// `block_size` is zero, or a leaf with demand has a ratio of zero.
#[must_use]
pub fn max_seq_len_fitting_in_geometry(
    leaves: &[LeafDemand],
    block_size: usize,
    allocatable_huge_blocks: usize,
    cache_ratios: &[usize],
    enable_prefix_caching: bool,
) -> MaxSeqLen {
    assert_eq!(
        leaves.len(),
        cache_ratios.len(),
        "every leaf needs a ratio, and every ratio a leaf",
    );
    let max_ratio = cache_ratios
        .iter()
        .copied()
        .max()
        .expect("a geometry has at least one leaf");

    let mut demand = vec![0; leaves.len()];
    let mut fits = |seq_len: usize| {
        let num_blocks = seq_len.div_ceil(block_size);
        for (blocks, leaf) in demand.iter_mut().zip(leaves) {
            *blocks = leaf.blocks_to_reserve(num_blocks, enable_prefix_caching);
        }
        pristine_pool_can_satisfy(allocatable_huge_blocks, cache_ratios, &demand)
    };

    // No request outruns one leaf given the whole budget at the largest ratio,
    // so that bounds the search; fitting at the bound means no finite bound.
    let upper_bound = allocatable_huge_blocks
        .saturating_mul(block_size)
        .saturating_mul(max_ratio);
    let search_space = upper_bound.saturating_add(2);

    // Python's `bisect_left`, midpoints and all: the first length that does
    // not fit.
    let (mut lo, mut hi) = (0, search_space);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == search_space {
        MaxSeqLen::Unbounded
    } else if lo == 0 {
        MaxSeqLen::NothingFits
    } else {
        MaxSeqLen::Longest(lo - 1)
    }
}

/// A block count computed in bytes, as the index type the pool counts in.
fn count(blocks: u64) -> usize {
    usize::try_from(blocks).expect("kv-core runs on 64-bit hosts")
}

/// How many huge blocks the smallest pool holds.
fn min_huge_blocks(include_null_block: bool) -> u64 {
    if include_null_block { 2 } else { 1 }
}

/// The least common multiple of `values`, which must all be positive, or
/// `None` past `u64::MAX`.
fn lcm_of(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values.into_iter().try_fold(1u64, |acc, value| {
        (acc / gcd(acc, value)).checked_mul(value)
    })
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Python's `to_human_readable_bytes`.
fn human_bytes(bytes: u64) -> String {
    let (unit, name) = if bytes >= TIB {
        (TIB, "TiB")
    } else if bytes >= GIB {
        (GIB, "GiB")
    } else if bytes >= MIB {
        (MIB, "MiB")
    } else {
        (KIB, "KiB")
    };
    format!("{:.2} {name}", bytes as f64 / unit as f64)
}

/// A least common multiple for an error message; `None` is past `u64::MAX`.
fn human_lcm(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => human_bytes(bytes),
        None => format!("more than {}", human_bytes(u64::MAX)),
    }
}

/// Python's `", ".join(f"{cache_id}={size}")`, with positions for ids.
fn list_pages(cache_sizes: &[u64]) -> String {
    cache_sizes
        .iter()
        .enumerate()
        .map(|(cache, &size)| format!("{cache}={}", human_bytes(size)))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests;
