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
//! The pool itself.
//!
//! Ported from MAX's Python `jenga_block_pool.py`, which it now replaces. See
//! [`crate::pool`] for how it differs.

use std::{collections::HashMap, fmt, hash::Hash, ops::Range};

use crate::{
    geometry::pristine_pool_can_satisfy,
    pool::{
        block::{HugeBlock, LittleBlock},
        free_list::{FreeList, FreeQueue, Linked, Stamps},
    },
};

/// The null block's `bid` in every cache, inside huge block 0.
const NULL_BID: usize = 0;

/// A pool that cannot be built, or a cache the pool can no longer serve.
#[derive(Debug, Clone)]
pub enum JengaError {
    /// The requested geometry is not a pool. Python raises `ValueError`.
    InvalidGeometry(String),
    /// No bytes are left to serve this cache. Python raises
    /// `InsufficientBlocksError`, which the block manager catches to preempt.
    InsufficientBlocks(usize),
}

impl fmt::Display for JengaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JengaError::InvalidGeometry(reason) => {
                write!(f, "invalid jenga pool geometry: {reason}")
            }
            JengaError::InsufficientBlocks(cache) => {
                write!(f, "no free blocks available for cache {cache}")
            }
        }
    }
}

impl std::error::Error for JengaError {}

/// How a [`JengaBlockPool`] behaves beyond its geometry. Everything is off by
/// default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolOptions {
    /// Reserves huge block 0 for the null block, as MAX's device pool does.
    ///
    /// Dummy and padding requests point their lookup tables at block 0 of
    /// every cache, so that block is never handed out, freed, evicted or
    /// committed, and the rest of huge block 0 (ids `1..ratio` of each cache)
    /// never exists. The pool then needs at least two huge blocks.
    pub null_block: bool,
    /// Drops every commit, so the pool stages blocks without ever serving a
    /// prefix hit. The tiered connector sets it when the host tier only stages
    /// blocks on their way to disk.
    pub skip_prefix_cache: bool,
}

#[derive(Debug, Clone, Copy)]
enum AllocSource {
    Claimed,
    Parked,
    Huge,
}

/// What one cache holds of a [`JengaBlockPool`], for observability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheOccupancy {
    /// Little blocks a request references.
    pub referenced: usize,
    /// Unreferenced little blocks still committed, which allocation evicts.
    pub cached: usize,
    /// The rest of the cache's free blocks, allocatable without evicting
    /// one of its own commits.
    pub free: usize,
    /// Huge blocks typed to the cache that a request references.
    pub claimed_huge_blocks: usize,
    /// Huge blocks parked while still typed to the cache.
    pub parked_huge_blocks: usize,
}

/// A pool of huge blocks, each subdividable into one cache's little blocks.
///
/// Every cache tiles the same bytes at its own page size, so a huge block is
/// `cache_ratios[cache]` blocks of that cache:
///
/// ```text
///   huge block     |     0     |     1     |     2     |     3     |
///   global  (x4)   | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
///   sliding (x2)   |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
/// ```
///
/// With [`PoolOptions::null_block`], huge block 0 is spent on the null block
/// (`N`) that dummy and padding requests share, which leaves the rest of it
/// (`.`) unusable, and starts real ids at 1 and at `ratio`:
///
/// ```text
///   huge block     |     0     |     1     |     2     |     3     |
///   global  (x4)   | N| .| .| .| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
///   sliding (x2)   |  N  |  .  |  2  |  3  |  4  |  5  |  6  |  7  |
/// ```
///
/// Those views alias, so a huge block serves one cache at a time -- its
/// `little_block_type` -- and at most one row of each column exists. Here the
/// global cache holds huge block 1, the sliding cache holds 2, and 3 is free
/// for either of them to claim:
///
/// ```text
///   huge block     |     0     |     1     |     2     |     3     |
///   global  (x4)   | 0| 1| 2| 3| 4| 5| 6| 7| - - - - - | - - - - - |
///   sliding (x2)   | - - - - - | - - - - - |  4  |  5  | - - - - - |
/// ```
///
/// Bytes change hands only while nothing references them, so the split between
/// caches follows live demand rather than a knob. A huge block other than the
/// null block's is therefore always in one of two states:
///
/// ```text
///   parked                                      claimed by cache c
///   +-------------------------------+           +--------------------------+
///   | num_referenced == 0           |   claim   | num_referenced >= 1      |
///   | in free_huge_blocks           |  ------>  | little_block_type == c   |
///   | any cache may claim it        |  <------  | only c allocates from it |
///   | commits still in prefix cache |   park    |                          |
///   +-------------------------------+           +--------------------------+
/// ```
///
/// Parked means no request holds any of its little blocks, so the bytes are up
/// for grabs. Claimed means one cache owns them: each of that cache's little
/// blocks in the huge block is either referenced by a request or queued in
/// that cache's free list as an eviction candidate, never both and never
/// neither.
///
/// [`Self::alloc_block`] claims a parked block when its cache has no free
/// little block left, and [`Self::touch`] claims one back when a prefix hit
/// takes its reference count up from 0. [`Self::free_block`] parks the block
/// again as soon as its last reference goes away, which is what lets another
/// cache reuse the bytes.
///
/// Parking keeps commits: a parked block's little blocks stay in their cache's
/// prefix cache, so that cache can reclaim it and still hit them. Only another
/// cache claiming the bytes evicts them.
///
/// `H` is the hash a block is committed under.
#[derive(Debug)]
pub struct JengaBlockPool<H> {
    /// How many of each cache's little blocks fit in a huge block.
    pub cache_ratios: Vec<usize>,
    /// Indexed by huge `bid`, including the null block's huge block 0.
    pub huge_blocks: Vec<HugeBlock>,
    /// `little_blocks[cache][bid]`, for every cache, including the ones in the
    /// null block's huge block 0.
    pub little_blocks: Vec<Vec<LittleBlock<H>>>,
    /// Each cache's eviction candidates in huge blocks it has claimed: blocks
    /// no request references, beside ones a request does.
    pub free_little_blocks: Vec<FreeQueue>,
    /// Each cache's eviction candidates in huge blocks parked while typed to
    /// it. Parking does not evict them -- only another cache claiming the
    /// bytes does -- so a cache can still allocate or hit them.
    ///
    /// Kept apart from `free_little_blocks` so allocation fills claimed huge
    /// blocks first and lets parked ones stay parked for any cache to claim.
    pub parked_little_blocks: Vec<FreeQueue>,
    /// Orders both, so a page keeps its turn when it moves between them.
    stamps: Stamps,
    /// The huge blocks no cache holds, least recently used first.
    pub free_huge_blocks: FreeList,
    /// Each cache's committed blocks, by hash.
    pub prefix_caches: Vec<HashMap<H, usize>>,
    /// How many of `free_huge_blocks` are parked while still typed to each
    /// cache -- their little blocks are already counted in
    /// `parked_little_blocks`, so `num_free_blocks` must not also count the
    /// huge block itself for that same cache.
    parked_and_typed: Vec<usize>,
    /// How many of each cache's committed blocks no request references, kept
    /// up to date so [`Self::occupancy`] need not scan the prefix cache.
    num_cached: Vec<usize>,
    options: PoolOptions,
}

impl<H: Copy + Eq + Hash> JengaBlockPool<H> {
    /// Builds a pool of `num_huge_blocks` huge blocks, all parked but the null
    /// block's.
    ///
    /// `cache_ratios[cache]` is how many of that cache's pages fit in a huge
    /// block, as Python's `compute_jenga_ratios` returns them (by leaf order).
    /// `num_huge_blocks` counts the null block's huge block, as Python's does.
    ///
    /// # Errors
    /// [`JengaError::InvalidGeometry`] if no huge block is left to allocate,
    /// or if `cache_ratios` is empty or gives a cache a ratio of zero.
    pub fn new(
        num_huge_blocks: usize,
        cache_ratios: Vec<usize>,
        options: PoolOptions,
    ) -> Result<Self, JengaError> {
        if options.null_block && num_huge_blocks < 2 {
            return Err(JengaError::InvalidGeometry(format!(
                "num_huge_blocks must be at least 2, since huge block 0 is the null block, \
                 found {num_huge_blocks}"
            )));
        }
        if num_huge_blocks == 0 {
            return Err(JengaError::InvalidGeometry(
                "num_huge_blocks must be at least 1, found 0".to_owned(),
            ));
        }
        if cache_ratios.is_empty() {
            return Err(JengaError::InvalidGeometry(
                "cache_ratios must be non-empty".to_owned(),
            ));
        }
        if let Some(idx) = cache_ratios.iter().position(|&ratio| ratio == 0) {
            return Err(JengaError::InvalidGeometry(format!(
                "cache_ratios must be positive, but cache {idx}'s ratio is 0"
            )));
        }

        // Every allocatable huge block starts parked, in ascending id order,
        // so no cache has any little block in circulation yet.
        let mut huge_blocks: Vec<HugeBlock> =
            (0..num_huge_blocks).map(|_| HugeBlock::default()).collect();
        let first_huge_bid = usize::from(options.null_block);
        let free_huge_blocks =
            FreeList::with_ids(&mut huge_blocks, first_huge_bid..num_huge_blocks);

        let num_caches = cache_ratios.len();

        Ok(JengaBlockPool {
            little_blocks: cache_ratios
                .iter()
                .map(|ratio| {
                    (0..ratio * num_huge_blocks)
                        .map(|_| LittleBlock::default())
                        .collect()
                })
                .collect(),
            free_little_blocks: (0..num_caches).map(|_| FreeQueue::default()).collect(),
            parked_little_blocks: (0..num_caches).map(|_| FreeQueue::default()).collect(),
            stamps: Stamps::default(),
            prefix_caches: (0..num_caches).map(|_| HashMap::new()).collect(),
            parked_and_typed: vec![0; num_caches],
            num_cached: vec![0; num_caches],
            cache_ratios,
            huge_blocks,
            free_huge_blocks,
            options,
        })
    }

    /// Returns the null block's `bid`, the same in every cache, if the pool
    /// reserves one.
    #[must_use]
    pub fn null_bid(&self) -> Option<usize> {
        self.options.null_block.then_some(NULL_BID)
    }

    /// Returns the huge blocks allocation can hand out: all of them, or all
    /// but huge block 0 when it holds the null block.
    #[must_use]
    pub fn allocatable_huge_bids(&self) -> Range<usize> {
        usize::from(self.options.null_block)..self.huge_blocks.len()
    }

    fn is_null(&self, bid: usize) -> bool {
        self.options.null_block && bid == NULL_BID
    }

    /// Returns where the next block of `cache` comes from: its claimed huge
    /// blocks, its parked ones, or a huge block off the parked list.
    ///
    /// Filling claimed huge blocks first keeps a cache's live pages packed
    /// into few huge blocks, so the rest can park for any cache to claim. A
    /// page whose loss evicts no commit comes first, and a pristine huge
    /// block before evicting a prefix. Eviction takes the least recently used
    /// commit, parked or not: the commits beside live pages are the newest.
    fn alloc_source(&self, cache: usize) -> Option<AllocSource> {
        let lists = [
            (AllocSource::Claimed, &self.free_little_blocks[cache]),
            (AllocSource::Parked, &self.parked_little_blocks[cache]),
        ];
        for (source, list) in lists {
            if list
                .peek_front()
                .is_some_and(|bid| self.little_blocks[cache][bid].block_hash.is_none())
            {
                return Some(source);
            }
        }
        if self
            .free_huge_blocks
            .peek_front()
            .is_some_and(|huge_bid| self.huge_blocks[huge_bid].little_block_type.is_none())
        {
            return Some(AllocSource::Huge);
        }
        // Both queues share one set of stamps, so the smaller front stamp is
        // the cache's least recently used commit.
        lists
            .into_iter()
            .filter_map(|(source, list)| {
                list.peek_front()
                    .map(|bid| (source, self.little_blocks[cache][bid].free))
            })
            .min_by_key(|&(_, stamp)| stamp)
            .map(|(source, _)| source)
            .or_else(|| (self.free_huge_blocks.len > 0).then_some(AllocSource::Huge))
    }

    /// Returns a fresh block of `cache`, claiming huge blocks as needed. It is
    /// never the null block.
    ///
    /// # Errors
    /// [`JengaError::InsufficientBlocks`] if the pool has no bytes left to
    /// serve this cache. Fragmentation can cause this while other caches still
    /// hold little blocks, so it is a signal to preempt, not a bug.
    ///
    /// # Panics
    /// If the pool does not serve `cache`.
    pub fn alloc_block(&mut self, cache: usize) -> Result<usize, JengaError> {
        let ratio = self.cache_ratios[cache];
        let bid = match self.alloc_source(cache) {
            None => return Err(JengaError::InsufficientBlocks(cache)),
            Some(AllocSource::Claimed) => self.free_little_blocks[cache]
                .pop_front(&mut self.little_blocks[cache])
                .expect("alloc_source checked the list"),
            Some(AllocSource::Parked) => {
                let bid = self.parked_little_blocks[cache]
                    .peek_front()
                    .expect("alloc_source checked the list");
                self.claim_huge_block(bid / ratio, cache);
                self.free_little_blocks[cache].remove(&mut self.little_blocks[cache], bid);
                bid
            }
            Some(AllocSource::Huge) => {
                let huge_bid = self
                    .free_huge_blocks
                    .peek_front()
                    .expect("alloc_source checked the list");
                // A huge block parked under `cache` keeps its little blocks on
                // `cache`'s parked list, so we would have taken one of those
                // rather than reaching for a huge block at all.
                assert_ne!(self.huge_blocks[huge_bid].little_block_type, Some(cache));
                self.claim_huge_block(huge_bid, cache);
                self.free_little_blocks[cache]
                    .pop_front(&mut self.little_blocks[cache])
                    .expect("a claimed huge block puts its little blocks in circulation")
            }
        };

        // Handing out a committed block evicts it: its bytes are about to be
        // overwritten, so it can no longer serve its hash.
        self.uncommit_block(cache, bid);

        self.little_blocks[cache][bid].ref_cnt += 1;
        self.huge_blocks[bid / ratio].num_referenced += 1;
        Ok(bid)
    }

    /// Drops a block from its cache's prefix cache, if it is committed.
    ///
    /// # Panics
    /// If the pool does not serve `cache`.
    pub fn uncommit_block(&mut self, cache: usize, bid: usize) {
        let blocks = &mut self.little_blocks[cache];
        if let Some(hash) = blocks[bid].block_hash.take() {
            self.prefix_caches[cache].remove(&hash);
            if blocks[bid].ref_cnt == 0 {
                self.num_cached[cache] -= 1;
            }
        }
    }

    /// Makes a filled block reusable by anyone hashing the same tokens.
    ///
    /// # Panics
    /// If the pool does not serve `cache`, if `bid` is the null block or
    /// already carries a hash, or if `hash` is already committed in this
    /// cache.
    pub fn commit_into_prefix_cache(&mut self, cache: usize, hash: H, bid: usize) {
        assert!(!self.is_null(bid), "the null block cannot be committed");
        if self.options.skip_prefix_cache {
            return;
        }

        let blocks = self.little_blocks.get_mut(cache).unwrap();
        assert!(
            blocks[bid].block_hash.is_none(),
            "block {bid} of {cache} is already committed"
        );
        blocks[bid].block_hash = Some(hash);
        if blocks[bid].ref_cnt == 0 {
            self.num_cached[cache] += 1;
        }

        let prefix_cache = self.prefix_caches.get_mut(cache).unwrap();
        let previous = prefix_cache.insert(hash, bid);
        assert!(previous.is_none(), "hash is already committed in {cache}");
    }

    /// Drops one reference, parking the huge block once it holds none.
    ///
    /// Freeing the null block does nothing: dummy requests share it, so it is
    /// never counted.
    ///
    /// # Panics
    /// If the pool does not serve `cache`, or if the reference count would go
    /// negative -- a double free.
    pub fn free_block(&mut self, cache: usize, bid: usize) {
        let ratio = self.cache_ratios[cache];
        if self.is_null(bid) {
            return;
        }
        let JengaBlockPool {
            huge_blocks,
            little_blocks,
            free_little_blocks,
            free_huge_blocks,
            parked_and_typed,
            num_cached,
            stamps,
            ..
        } = self;
        let blocks = &mut little_blocks[cache];
        let free = &mut free_little_blocks[cache];

        blocks[bid].ref_cnt -= 1;
        assert!(
            blocks[bid].ref_cnt >= 0,
            "block {bid} of {cache} double-freed (ref_cnt < 0)"
        );
        if blocks[bid].ref_cnt > 0 {
            return;
        }
        // Only a 1 -> 0 transition drops a little block out of the huge
        // block's referenced count; a still-referenced twin (e.g. a dedup hit
        // besides this one) must not be double-decremented.
        let huge_bid = bid / ratio;
        huge_blocks[huge_bid].num_referenced -= 1;
        if blocks[bid].block_hash.is_none() {
            free.append_front(stamps, blocks, bid);
        } else {
            free.append_back(stamps, blocks, bid);
            num_cached[cache] += 1;
        }

        if huge_blocks[huge_bid].num_referenced > 0 {
            return;
        }

        // The whole huge block went idle, so park it where any cache can claim
        // it. Its little blocks move to `parked_little_blocks[cache]` (still
        // committed, still directly allocable by `cache` without a reclaim) and
        // the block stays typed: whoever claims it next needs to know whose
        // commits its bytes still back, to evict them only if the bytes change
        // hands. `parked_and_typed` tracks that those little blocks are already
        // counted, so `num_free_blocks` doesn't double-count them.
        assert_eq!(huge_blocks[huge_bid].little_block_type, Some(cache));
        parked_and_typed[cache] += 1;
        free_huge_blocks.append_back(huge_blocks, huge_bid);
        self.move_free_pages(cache, huge_bid, true);
    }

    /// Moves every page of `huge_bid` between `cache`'s claimed and parked
    /// free lists, which holds them all while no request references any. Each
    /// keeps its turn, so eviction order survives the move.
    fn move_free_pages(&mut self, cache: usize, huge_bid: usize, park: bool) {
        let ratio = self.cache_ratios[cache];
        let blocks = &self.little_blocks[cache];
        let (from, to) = if park {
            (
                &mut self.free_little_blocks[cache],
                &mut self.parked_little_blocks[cache],
            )
        } else {
            (
                &mut self.parked_little_blocks[cache],
                &mut self.free_little_blocks[cache],
            )
        };
        for bid in huge_bid * ratio..(huge_bid + 1) * ratio {
            from.move_to(to, blocks, bid);
        }
    }

    /// Commits a block, or returns the twin already holding its bytes.
    ///
    /// Returns the committed block to use instead of `bid`, which has been
    /// freed, or `None` if `bid` itself now serves the hash.
    ///
    /// # Panics
    /// If the pool does not serve `cache`, or if `bid` is the null block.
    pub fn get_or_commit_into_prefix_cache(
        &mut self,
        cache: usize,
        hash: H,
        bid: usize,
    ) -> Option<usize> {
        assert!(!self.is_null(bid), "the null block cannot be committed");
        // Check if a block with the same hash is already committed. If so, we
        // reuse the already committed block.
        if let Some(committed) = self.prefix_caches[cache].get(&hash) {
            if *committed == bid {
                return None;
            }
            self.touch(cache, *committed);

            // Free the block we currently have. It was never committed, or
            // `committed` would be it.
            assert!(self.little_blocks[cache][bid].block_hash.is_none());
            self.free_block(cache, bid);
        } else {
            self.commit_into_prefix_cache(cache, hash, bid);
            return None;
        }

        let committed = self.prefix_caches[cache][&hash];

        Some(committed)
    }

    /// Takes a reference on a block, reviving it if it was out of use.
    ///
    /// Touching the null block does nothing, as freeing it does.
    ///
    /// # Panics
    /// If the pool does not serve `cache`, or if the block's bytes have been
    /// retyped to another cache -- reviving it would hand back that cache's
    /// tensor, so only a cache's own commits are touchable.
    pub fn touch(&mut self, cache: usize, bid: usize) {
        let ratio = self.cache_ratios[cache];
        if self.is_null(bid) {
            return;
        }
        let huge_bid = bid / ratio;
        assert_eq!(
            self.huge_blocks[huge_bid].little_block_type,
            Some(cache),
            "block {bid} does not belong to {cache}"
        );

        // ref_cnt == 0 means this block is out of circulation: either an
        // eviction candidate in its cache's free queue, or parked with the rest
        // of an idle huge block, which referencing it takes back for this cache.
        if self.little_blocks[cache][bid].ref_cnt == 0 {
            if self.huge_blocks[huge_bid].is_free() {
                self.claim_huge_block(huge_bid, cache);
            }
            let free = self.free_little_blocks.get_mut(cache).unwrap();
            let blocks = self.little_blocks.get_mut(cache).unwrap();
            free.remove(blocks, bid);
            if blocks[bid].block_hash.is_some() {
                self.num_cached[cache] -= 1;
            }
            // Only a 0 -> 1 transition adds a newly-referenced little block to
            // the huge block's count; a second reference to one already held
            // must not double-count it.
            self.huge_blocks[huge_bid].num_referenced += 1;
        }
        self.little_blocks[cache][bid].ref_cnt += 1;
    }

    /// Returns how many more blocks of `cache` the pool can still serve.
    ///
    /// # Panics
    /// If the pool does not serve `cache`.
    #[must_use]
    pub fn num_free_blocks(&self, cache: usize) -> usize {
        // Parked huge blocks already typed to `cache` contribute nothing extra
        // here: their little blocks are already counted in
        // `parked_little_blocks[cache]`, so only the other parked huge blocks
        // (pristine, or typed to a different cache) would yield new ones if claimed.
        let claimable_huge_blocks = self.free_huge_blocks.len - self.parked_and_typed[cache];
        claimable_huge_blocks * self.cache_ratios[cache]
            + self.free_little_blocks[cache].len()
            + self.parked_little_blocks[cache].len()
    }

    /// Returns how many blocks of `cache` the pool is currently serving. The
    /// null block does not count.
    ///
    /// # Panics
    /// If the pool does not serve `cache`.
    #[must_use]
    pub fn num_used_blocks(&self, cache: usize) -> usize {
        self.cache_ratios[cache] * self.allocatable_huge_bids().len() - self.num_free_blocks(cache)
    }

    /// Returns what `cache` holds, scanning every huge block.
    ///
    /// # Panics
    /// If the pool does not serve `cache`.
    #[must_use]
    pub fn occupancy(&self, cache: usize) -> CacheOccupancy {
        let mut referenced = 0;
        let mut claimed_huge_blocks = 0;
        for huge in &self.huge_blocks {
            if huge.little_block_type == Some(cache) && !huge.is_free() {
                claimed_huge_blocks += 1;
                referenced += huge.num_referenced;
            }
        }
        // Retyping evicts the outgoing cache's commits, so every commit sits
        // in a huge block typed to `cache`, and an unreferenced one is on its
        // free list.
        let cached = self.num_cached[cache];
        CacheOccupancy {
            referenced,
            cached,
            free: self.num_free_blocks(cache) - cached,
            claimed_huge_blocks,
            parked_huge_blocks: self.parked_and_typed[cache],
        }
    }

    /// Returns how many parked huge blocks no cache has ever claimed.
    #[must_use]
    pub fn num_pristine_huge_blocks(&self) -> usize {
        self.free_huge_blocks.len - self.parked_and_typed.iter().sum::<usize>()
    }

    /// Returns whether the pool can allocate `demand[cache]` more little
    /// blocks of every cache at once.
    ///
    /// Unlike summing [`Self::num_free_blocks`], this charges every cache to
    /// the same huge blocks: each cache's count alone assumes it may retype
    /// every parked huge block, which only one cache can actually do.
    ///
    /// `at_capacity` asks the same of a pool that has handed nothing out yet,
    /// making the answer a property of the pool's geometry rather than of what
    /// it currently holds.
    ///
    /// # Panics
    /// If `demand` does not name every cache the pool serves, by index.
    #[must_use]
    pub fn can_satisfy_demand(&self, demand: &[usize], at_capacity: bool) -> bool {
        if at_capacity {
            return pristine_pool_can_satisfy(
                self.allocatable_huge_bids().len(),
                &self.cache_ratios,
                demand,
            );
        }
        assert_eq!(
            demand.len(),
            self.cache_ratios.len(),
            "demand must name every cache of the pool, and only those"
        );
        // A parked huge block stays typed only while its cache needs it to
        // satisfy demand. The remaining idle pages may be retyped.
        let mut claimable = self.free_huge_blocks.len;
        let mut shortfall = Vec::with_capacity(demand.len());
        for (cache, &num_blocks) in demand.iter().enumerate() {
            let ratio = self.cache_ratios[cache];
            let fixed_free_blocks = self.free_little_blocks[cache].len();
            let carved = fixed_free_blocks + self.parked_little_blocks[cache].len();
            let parked = self.parked_and_typed[cache];
            let parked_to_keep =
                parked.min(num_blocks.saturating_sub(fixed_free_blocks).div_ceil(ratio));
            claimable -= parked_to_keep;
            shortfall.push(num_blocks.saturating_sub(carved));
        }
        pristine_pool_can_satisfy(claimable, &self.cache_ratios, &shortfall)
    }

    /// Returns the largest `m` for which [`Self::can_satisfy_demand`] accepts
    /// `m` times `unit`, or 0 if `unit` asks for nothing.
    ///
    /// # Panics
    /// If `unit` does not name every cache the pool serves, by index.
    #[must_use]
    pub fn max_satisfiable_multiple(&self, unit: &[usize], at_capacity: bool) -> usize {
        let num_huge_blocks = self.allocatable_huge_bids().len();
        let Some(mut hi) = unit
            .iter()
            .zip(&self.cache_ratios)
            .filter(|&(&blocks, _)| blocks > 0)
            .map(|(&blocks, &ratio)| num_huge_blocks * ratio / blocks)
            .min()
        else {
            return 0;
        };
        let mut lo = 0;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            let demand: Vec<usize> = unit.iter().map(|&blocks| blocks * mid).collect();
            if self.can_satisfy_demand(&demand, at_capacity) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// Drops every commit no request is holding, in every cache.
    ///
    /// A commit a request still references survives, because its block cannot
    /// be handed out while it is in use.
    ///
    /// Returns how many blocks were purged from each cache's prefix cache.
    pub fn reset_prefix_cache(&mut self) -> Vec<usize> {
        let JengaBlockPool {
            little_blocks,
            prefix_caches,
            num_cached,
            ..
        } = self;
        let mut purged = Vec::with_capacity(prefix_caches.len());
        for (cache, prefix_cache) in prefix_caches.iter_mut().enumerate() {
            let blocks = little_blocks.get_mut(cache).unwrap();
            let before = prefix_cache.len();
            prefix_cache.retain(|_, &mut bid| {
                let referenced = blocks[bid].ref_cnt > 0;
                if !referenced {
                    blocks[bid].block_hash = None;
                }
                referenced
            });
            // Every purged commit was unreferenced, so all of them were cached.
            let num_purged = before - prefix_cache.len();
            num_cached[cache] -= num_purged;
            purged.push(num_purged);
        }
        purged
    }

    /// Puts an unreferenced huge block's little blocks in circulation.
    ///
    /// Retyping to a different cache hands the bytes over, so whatever the
    /// outgoing cache had committed in them is evicted first. Reclaiming for
    /// the same cache keeps those commits, so a prefix hit can still serve
    /// them.
    fn claim_huge_block(&mut self, huge_bid: usize, cache: usize) {
        assert_eq!(
            self.huge_blocks[huge_bid].num_referenced, 0,
            "huge block {huge_bid} is still referenced"
        );
        let previous = self.huge_blocks[huge_bid].little_block_type;
        if self.huge_blocks[huge_bid].is_free() {
            self.free_huge_blocks
                .remove(&mut self.huge_blocks, huge_bid);
            if let Some(previous) = previous {
                self.parked_and_typed[previous] -= 1;
            }
        }

        if previous == Some(cache) {
            // Reclaiming our own parked huge block: its little blocks are
            // still free-listed (see `free_block`), on the parked list.
            self.move_free_pages(cache, huge_bid, false);
            return;
        }

        // The outgoing cache's little blocks are all unreferenced, waiting on
        // its parked list.
        if let Some(previous) = previous {
            let ratio = self.cache_ratios[previous];
            for bid in huge_bid * ratio..(huge_bid + 1) * ratio {
                self.parked_little_blocks[previous].remove(&mut self.little_blocks[previous], bid);
                self.uncommit_block(previous, bid);
            }
        }

        self.huge_blocks[huge_bid].little_block_type = Some(cache);
        let ratio = self.cache_ratios[cache];
        let free = self.free_little_blocks.get_mut(cache).unwrap();
        let blocks = self.little_blocks.get_mut(cache).unwrap();
        // Reversed so the front-appended pristine blocks come off in
        // ascending bid order, not descending.
        for bid in (huge_bid * ratio..(huge_bid + 1) * ratio).rev() {
            if blocks[bid].block_hash.is_none() {
                free.append_front(&mut self.stamps, blocks, bid);
            } else {
                free.append_back(&mut self.stamps, blocks, bid);
            }
        }
    }

    /// Returns the block `cache` has committed under `hash`, if any.
    #[must_use]
    pub fn lookup_hash(&self, cache: usize, hash: H) -> Option<usize> {
        if self.options.skip_prefix_cache {
            None
        } else {
            self.prefix_caches[cache].get(&hash).copied()
        }
    }
}

// Ports MAX's deleted `jenga/test_jenga_block_pool.py`: a
// test named after a Python case is that case. The Python pool always has a
// null block, so each runs with one, where it reproduces the Python ids, and
// without, the host tier's pool, where every id sits one huge block lower.
#[cfg(test)]
mod tests {
    use super::*;

    /// Python commits `bytes`; any `Copy + Eq + Hash` serves.
    type Pool = JengaBlockPool<&'static str>;

    const WITH_NULL: PoolOptions = PoolOptions {
        null_block: true,
        skip_prefix_cache: false,
    };

    /// Builds a pool with `allocatable` huge blocks to hand out, plus huge
    /// block 0 when the null block takes it, so a Python test's
    /// `num_huge_blocks=n` is `pool(n - 1, .., true)`.
    fn pool(allocatable: usize, ratios: &[usize], null_block: bool) -> Pool {
        let options = PoolOptions {
            null_block,
            ..PoolOptions::default()
        };
        Pool::new(
            allocatable + usize::from(null_block),
            ratios.to_vec(),
            options,
        )
        .expect("valid geometry")
    }

    fn alloc_n(p: &mut Pool, cache: usize, n: usize) -> Vec<usize> {
        (0..n)
            .map(|_| p.alloc_block(cache).expect("a free block"))
            .collect()
    }

    fn expect_exhausted(p: &mut Pool, cache: usize) {
        let err = p.alloc_block(cache).expect_err("the pool to be exhausted");
        assert_eq!(
            err.to_string(),
            format!("no free blocks available for cache {cache}")
        );
    }

    /// The huge blocks a run of little blocks lives in.
    fn huge_bids(p: &Pool, cache: usize, bids: &[usize]) -> Vec<usize> {
        let ratio = p.cache_ratios[cache];
        let mut ids: Vec<usize> = bids.iter().map(|bid| bid / ratio).collect();
        ids.dedup();
        ids
    }

    /// Asserts the pool's structural invariants, mirroring `check_invariants`
    /// in `test_jenga_block_pool.py`.
    ///
    /// A huge block is either parked in `free_huge_blocks` with none of its
    /// little blocks in circulation, or claimed by one cache, in which case
    /// each of that cache's little blocks in it is referenced or free-listed,
    /// never both and never neither. The null block's huge block is neither:
    /// it is never parked or typed, so the walk below also checks that none of
    /// its little blocks -- the null blocks included -- is ever referenced,
    /// free-listed or committed.
    fn check_invariants<H: Copy + Eq + Hash + fmt::Debug>(p: &JengaBlockPool<H>) {
        if p.null_bid().is_some() {
            let null_huge = &p.huge_blocks[0];
            assert!(!null_huge.is_free(), "the null huge block is parked");
            assert_eq!(null_huge.little_block_type, None);
        }

        let mut num_free = vec![0; p.cache_ratios.len()];
        for huge_bid in 0..p.huge_blocks.len() {
            let huge = &p.huge_blocks[huge_bid];
            let parked = huge.is_free();
            if parked {
                assert_eq!(
                    huge.num_referenced, 0,
                    "parked huge block {huge_bid} is referenced"
                );
            }
            for (cache, ratio) in p.cache_ratios.iter().enumerate() {
                // A huge block still typed to `cache` -- parked or actively
                // claimed -- lets `cache` freely allocate its little blocks;
                // only a cache it ISN'T typed to sees them as out of circulation.
                let typed_to_cache = huge.little_block_type == Some(cache);
                for bid in huge_bid * ratio..(huge_bid + 1) * ratio {
                    let block = &p.little_blocks[cache][bid];
                    if typed_to_cache {
                        assert_ne!(
                            block.ref_cnt > 0,
                            block.is_free(),
                            "{cache}/{bid} is both or neither referenced and free"
                        );
                        if block.is_free() {
                            num_free[cache] += 1;
                            let queue = if parked {
                                &p.parked_little_blocks[cache]
                            } else {
                                &p.free_little_blocks[cache]
                            };
                            assert!(
                                queue.contains(&p.little_blocks[cache], bid),
                                "{cache}/{bid} is free on the wrong queue"
                            );
                        }
                    } else {
                        assert_eq!(block.ref_cnt, 0, "{cache}/{bid} is referenced");
                        assert!(!block.is_free(), "{cache}/{bid} is in circulation");
                    }
                }
            }
        }

        for (cache, ratio) in p.cache_ratios.iter().enumerate() {
            assert_eq!(
                p.parked_little_blocks[cache].len(),
                p.parked_and_typed[cache] * ratio,
                "{cache}'s parked list holds pages of claimed huge blocks"
            );
            assert_eq!(
                p.free_little_blocks[cache].len() + p.parked_little_blocks[cache].len(),
                num_free[cache],
                "{cache}'s free queues hold blocks that are not free"
            );
        }

        for (cache, prefix_cache) in p.prefix_caches.iter().enumerate() {
            for (hash, &bid) in prefix_cache {
                assert_eq!(p.little_blocks[cache][bid].block_hash.as_ref(), Some(hash));
                // Retyping the bytes out from under a committed block must drop it.
                assert_eq!(
                    p.huge_blocks[bid / p.cache_ratios[cache]].little_block_type,
                    Some(cache)
                );
            }
            let cached = prefix_cache
                .values()
                .filter(|&&bid| p.little_blocks[cache][bid].ref_cnt == 0)
                .count();
            assert_eq!(
                p.num_cached[cache], cached,
                "cache {cache}'s cached count drifted"
            );
        }

        // The running per-huge-block counts must match a fresh walk, and must be
        // zero exactly when Python's sum-of-reference-counts is.
        for huge_bid in 0..p.huge_blocks.len() {
            let huge = &p.huge_blocks[huge_bid];
            let mut counted = 0usize;
            let mut summed = 0i64;
            if let Some(cache) = huge.little_block_type {
                let ratio = p.cache_ratios[cache];
                for bid in huge_bid * ratio..(huge_bid + 1) * ratio {
                    let ref_cnt = p.little_blocks[cache][bid].ref_cnt;
                    counted += usize::from(ref_cnt > 0);
                    summed += ref_cnt;
                }
            }
            assert_eq!(
                huge.num_referenced, counted,
                "huge block {huge_bid} count drifted"
            );
            assert_eq!(huge.num_referenced == 0, summed == 0);
        }
    }

    #[test]
    fn constructor() {
        let ratios = [5, 2];
        for null_block in [false, true] {
            let p = pool(42, &ratios, null_block);
            let null_huge_blocks = usize::from(null_block);

            // Python's `huge_blocks` and `little_blocks` leave out huge block
            // 0 when it holds the null block. Here it stays, so a bid indexes
            // the arrays directly and `bid / ratio` is still its huge block.
            assert_eq!(p.huge_blocks.len(), 42 + null_huge_blocks);
            assert_eq!(
                p.allocatable_huge_bids(),
                null_huge_blocks..42 + null_huge_blocks
            );
            assert_eq!(p.free_huge_blocks.len, 42);
            for (cache, &ratio) in ratios.iter().enumerate() {
                assert_eq!(
                    p.little_blocks[cache].len(),
                    ratio * (42 + null_huge_blocks)
                );
                assert_eq!(p.cache_ratios[cache], ratio);
                // Nothing is in circulation yet, but every parked huge block is
                // still available to whichever cache asks first.
                assert_eq!(p.free_little_blocks[cache].len(), 0);
                assert_eq!(p.num_free_blocks(cache), 42 * ratio);
                assert_eq!(p.num_used_blocks(cache), 0);
                assert!(p.prefix_caches[cache].is_empty());
            }
            check_invariants(&p);
        }
    }

    #[test]
    fn invalid_constructor() {
        // A pool of one huge block is all null block and nothing to hand out.
        for num_huge_blocks in [0, 1] {
            let err = Pool::new(num_huge_blocks, vec![1], WITH_NULL).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "invalid jenga pool geometry: num_huge_blocks must be at least 2, since huge \
                     block 0 is the null block, found {num_huge_blocks}"
                )
            );
        }
        // Without a null block one huge block is a usable pool, so only zero
        // is rejected. Python has no such pool.
        let err = Pool::new(0, vec![1], PoolOptions::default()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid jenga pool geometry: num_huge_blocks must be at least 1, found 0"
        );
        assert!(Pool::new(1, vec![1], PoolOptions::default()).is_ok());

        for options in [PoolOptions::default(), WITH_NULL] {
            let err = Pool::new(42, vec![], options).unwrap_err();
            assert_eq!(
                err.to_string(),
                "invalid jenga pool geometry: cache_ratios must be non-empty"
            );

            let err = Pool::new(42, vec![1, 0], options).unwrap_err();
            assert_eq!(
                err.to_string(),
                "invalid jenga pool geometry: cache_ratios must be positive, but cache 1's ratio \
                 is 0"
            );
        }
    }

    /// `test_alloc`.
    #[test]
    fn alloc_claims_huge_blocks_as_needed() {
        for null_block in [false, true] {
            let mut p = pool(2, &[42, 43, 3], null_block);

            let bids = alloc_n(&mut p, 2, 6);
            expect_exhausted(&mut p, 2);

            let (expected_bids, expected_huge_bids) = if null_block {
                (3..9, [1, 2])
            } else {
                (0..6, [0, 1])
            };
            assert_eq!(bids, expected_bids.collect::<Vec<_>>());
            assert!(bids.iter().all(|&bid| p.little_blocks[2][bid].ref_cnt == 1));
            assert_eq!(huge_bids(&p, 2, &bids), expected_huge_bids);
            check_invariants(&p);
        }
    }

    #[test]
    fn caches_split_the_pool_by_live_demand() {
        for null_block in [false, true] {
            let mut p = pool(5, &[4, 1], null_block);

            // No knob decides the split: whichever cache asks takes the next
            // huge block.
            let global_bids = alloc_n(&mut p, 0, 8);
            let sliding_bids = alloc_n(&mut p, 1, 2);

            let first = usize::from(null_block);
            assert_eq!(p.free_huge_blocks.len, 1);
            assert_eq!(huge_bids(&p, 0, &global_bids), [first, first + 1]);
            assert_eq!(huge_bids(&p, 1, &sliding_bids), [first + 2, first + 3]);
            check_invariants(&p);

            for &bid in &global_bids {
                p.free_block(0, bid);
            }
            for &bid in &sliding_bids {
                p.free_block(1, bid);
            }

            // With the pool idle, either cache can have all of it.
            let global_bids = alloc_n(&mut p, 0, 20);
            expect_exhausted(&mut p, 0);
            for &bid in &global_bids {
                p.free_block(0, bid);
            }

            alloc_n(&mut p, 1, 5);
            expect_exhausted(&mut p, 1);
            check_invariants(&p);
        }
    }

    #[test]
    fn fragmentation_can_cause_oom() {
        for null_block in [false, true] {
            let mut p = pool(4, &[1, 2, 3, 4, 5], null_block);
            for cache in [0, 1, 2, 3] {
                p.alloc_block(cache).expect("a free block");
            }

            // Cache 0 or 4 needs a whole new huge block, which is not available.
            for cache in [0, 4] {
                expect_exhausted(&mut p, cache);
            }

            // Can allocate little blocks for caches 1, 2 and 3.
            for cache in [1, 2, 3] {
                p.alloc_block(cache).expect("a free block");
            }
            check_invariants(&p);
        }
    }

    #[test]
    fn partly_freed_huge_block_stays_with_its_cache() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2, 2], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.free_block(0, bids[0]);

            // One live reference keeps the bytes typed, so the other cache
            // cannot have them however much it wants them.
            assert_eq!(p.free_huge_blocks.len, 1);
            assert_eq!(p.free_little_blocks[0].len(), 1);
            assert_eq!(p.num_free_blocks(1), 2);
            check_invariants(&p);

            p.free_block(0, bids[1]);
            assert_eq!(p.free_huge_blocks.len, 2);
            // The huge block went idle, but its little blocks stay on cache
            // 0's parked list -- still directly allocable by cache 0 without
            // a reclaim -- even though the huge block itself is now also up
            // for grabs.
            assert_eq!(p.parked_little_blocks[0].len(), 2);
            assert_eq!(p.num_free_blocks(1), 4);
            check_invariants(&p);
        }
    }

    #[test]
    fn freeing_a_pristine_block_is_reused_before_a_committed_one() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "prefix", bids[0]);
            // Free the committed block first, then the pristine one -- despite
            // that order, the pristine one is reused first.
            p.free_block(0, bids[0]);
            p.free_block(0, bids[1]);

            let next = p.alloc_block(0).expect("a free block");

            assert_eq!(next, bids[1]);
            assert_eq!(p.little_blocks[0][bids[0]].block_hash, Some("prefix"));
            check_invariants(&p);
        }
    }

    #[test]
    fn idle_huge_block_can_be_retyped() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2, 1], null_block);
            // Python's `huge_blocks[0]`, which leaves out the null block's.
            let only_huge_bid = usize::from(null_block);

            let bid = p.alloc_block(0).expect("a free block");
            p.free_block(0, bid);
            let retyped = p.alloc_block(1).expect("a free block");

            assert_eq!(p.huge_blocks[only_huge_bid].little_block_type, Some(1));
            assert_eq!(retyped / p.cache_ratios[1], only_huge_bid);
            assert_eq!(p.little_blocks[0][bid].ref_cnt, 0);
            assert_eq!(p.num_free_blocks(1), 0);
            // The stolen-from cache must not still think its parked-and-typed
            // huge block counts as headroom now that it's gone.
            assert_eq!(p.num_free_blocks(0), 0);
            check_invariants(&p);
        }
    }

    #[test]
    fn prefer_claim_huge_avoids_evicting_a_commit() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "apple", bids[0]);
            p.free_block(0, bids[0]);
            p.commit_into_prefix_cache(0, "banana", bids[1]);
            p.free_block(0, bids[1]);

            // The second huge block is still virgin, so the next alloc claims
            // it instead of evicting either commit.
            let new = p.alloc_block(0).expect("a free block");

            let second_huge_bid = if null_block { 2 } else { 1 };
            assert_eq!(new / p.cache_ratios[0], second_huge_bid);
            assert_eq!(
                p.prefix_caches[0],
                HashMap::from([("apple", bids[0]), ("banana", bids[1])])
            );
            check_invariants(&p);
        }
    }

    #[test]
    fn parking_keeps_the_head_of_a_prefix_evicted_last() {
        for null_block in [false, true] {
            let mut p = pool(1, &[3], null_block);
            let bids = alloc_n(&mut p, 0, 3);
            // Reuse hands the prefix its blocks out of bid order.
            p.free_block(0, bids[0]);
            p.free_block(0, bids[1]);
            let head = p.alloc_block(0).expect("a free block");
            let tail = p.alloc_block(0).expect("a free block");
            assert!(head > tail);
            p.commit_into_prefix_cache(0, "head", head);
            p.commit_into_prefix_cache(0, "tail", tail);
            p.free_block(0, tail);
            p.free_block(0, head);

            // Park the huge block, then claim it back.
            p.free_block(0, bids[2]);
            assert_eq!(p.free_huge_blocks.len, 1);
            assert_eq!(p.alloc_block(0).expect("a free block"), bids[2]);

            assert_eq!(p.alloc_block(0).expect("a free block"), tail);
            assert_eq!(p.prefix_caches[0], HashMap::from([("head", head)]));
            check_invariants(&p);
        }
    }

    #[test]
    fn alloc_fills_claimed_huge_blocks_before_parked_ones() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);
            let bids = alloc_n(&mut p, 0, 4);
            // The first huge block frees a page while a request still holds
            // its other one, then the second huge block parks.
            p.free_block(0, bids[0]);
            p.free_block(0, bids[2]);
            p.free_block(0, bids[3]);

            // Taking a parked page would unpark bytes any cache could claim.
            assert_eq!(p.alloc_block(0).expect("a free block"), bids[0]);
            assert_eq!(p.free_huge_blocks.len, 1);
            check_invariants(&p);
        }
    }

    #[test]
    fn alloc_evicts_the_least_recently_used_commit_even_when_parked() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);
            let bids = alloc_n(&mut p, 0, 4);
            for (bid, hash) in bids.iter().zip(["a", "b", "c", "d"]) {
                p.commit_into_prefix_cache(0, hash, *bid);
            }
            // The second huge block parks, then the first frees a page while a
            // request still holds its other one.
            p.free_block(0, bids[2]);
            p.free_block(0, bids[3]);
            p.free_block(0, bids[0]);

            // The page beside a live one is the newest commit, so it stays.
            assert_eq!(p.alloc_block(0).expect("a free block"), bids[2]);
            assert_eq!(
                p.prefix_caches[0],
                HashMap::from([("a", bids[0]), ("b", bids[1]), ("d", bids[3])])
            );
            check_invariants(&p);
        }
    }

    /// Ports the design simulation's churn: requests grow a full cache and
    /// slide a two-page window. A window cache spread over many huge blocks
    /// pins each with a page or two. Nothing is committed, as with prefix
    /// caching off: commits are evicted in LRU order, not packed.
    #[test]
    fn sliding_windows_pin_few_huge_blocks() {
        const FULL: usize = 0;
        const WINDOW: usize = 1;
        let mut p =
            JengaBlockPool::<u64>::new(257, vec![16, 8], WITH_NULL).expect("valid geometry");
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut requests: Vec<(Vec<usize>, Vec<usize>, u64)> = Vec::new();
        for step in 0..2000 {
            while requests.len() < 48 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                requests.push((Vec::new(), Vec::new(), 20 + rng % 40));
            }
            for (full, window, steps_left) in &mut requests {
                if *steps_left % 4 == 0 {
                    full.push(p.alloc_block(FULL).expect("a free block"));
                }
                window.push(p.alloc_block(WINDOW).expect("a free block"));
                if window.len() > 2 {
                    p.free_block(WINDOW, window.remove(0));
                }
                *steps_left -= 1;
            }
            for (full, window, _) in requests.extract_if(.., |request| request.2 == 0) {
                for bid in full.into_iter().rev() {
                    p.free_block(FULL, bid);
                }
                for bid in window.into_iter().rev() {
                    p.free_block(WINDOW, bid);
                }
            }
            if step >= 100 {
                let live: usize = requests.iter().map(|request| request.1.len()).sum();
                let pinned = p
                    .huge_blocks
                    .iter()
                    .filter(|huge| {
                        huge.little_block_type == Some(WINDOW) && huge.num_referenced > 0
                    })
                    .count();
                assert!(
                    pinned <= 2 * live.div_ceil(8),
                    "step {step}: {pinned} huge blocks pinned by {live} window pages"
                );
            }
        }
        check_invariants(&p);
    }

    #[test]
    fn retyping_evicts_what_the_outgoing_cache_committed() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2, 1], null_block);

            let bid = p.alloc_block(0).expect("a free block");
            p.commit_into_prefix_cache(0, "prefix", bid);
            p.free_block(0, bid);
            // Parked, but still serving its hash: the global cache gets it back
            // for free if it asks before anyone else claims the bytes.
            assert_eq!(p.prefix_caches[0], HashMap::from([("prefix", bid)]));

            p.alloc_block(1).expect("a free block");

            assert!(p.prefix_caches[0].is_empty());
            assert_eq!(p.little_blocks[0][bid].block_hash, None);
            check_invariants(&p);
        }
    }

    #[test]
    fn prefix_hit_reclaims_a_parked_huge_block() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);

            let bid = p.alloc_block(0).expect("a free block");
            assert_eq!(bid, if null_block { 2 } else { 0 });
            p.commit_into_prefix_cache(0, "prefix", bid);
            p.free_block(0, bid);
            assert_eq!(p.free_huge_blocks.len, 2);

            let hit = p.lookup_hash(0, "prefix").expect("a committed block");
            p.touch(0, hit);

            assert_eq!(p.little_blocks[0][bid].ref_cnt, 1);
            assert_eq!(p.little_blocks[0][bid].block_hash, Some("prefix"));
            assert_eq!(p.free_huge_blocks.len, 1);
            // Its sibling was never evicted from circulation by the park -- it
            // stayed on cache 0's free list the whole time -- so it is still
            // free.
            assert!(p.little_blocks[0][bid + 1].is_free());
            check_invariants(&p);
        }
    }

    #[test]
    fn allocating_over_a_committed_block_evicts_it() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "apple", bids[0]);
            p.free_block(0, bids[0]);
            p.commit_into_prefix_cache(0, "banana", bids[1]);
            p.free_block(0, bids[1]);
            assert_eq!(p.prefix_caches[0].len(), 2);

            let new = p.alloc_block(0).expect("a free block");

            // Allocating a new global block evicts a single oldest entry from
            // the global prefix cache.
            assert_eq!(new, bids[0]);
            assert_eq!(p.little_blocks[0][new].block_hash, None);
            assert_eq!(p.prefix_caches[0], HashMap::from([("banana", bids[1])]));
            check_invariants(&p);
        }
    }

    #[test]
    fn allocating_from_different_cache_over_a_committed_block_evicts_it() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2, 1], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "apple", bids[0]);
            p.free_block(0, bids[0]);
            p.commit_into_prefix_cache(0, "banana", bids[1]);
            p.free_block(0, bids[1]);
            assert_eq!(p.prefix_caches[0].len(), 2);

            p.alloc_block(1).expect("a free block");

            // Allocating a single sliding block evicts the whole global prefix
            // cache, because it takes the only huge block backing it.
            assert_eq!(p.little_blocks[0][bids[0]].block_hash, None);
            assert_eq!(p.little_blocks[0][bids[1]].block_hash, None);
            assert!(p.prefix_caches[0].is_empty());
            check_invariants(&p);
        }
    }

    #[test]
    fn get_or_commit_dedupes_onto_the_committed_block() {
        for null_block in [false, true] {
            let mut p = pool(1, &[2], null_block);

            let first = p.alloc_block(0).expect("a free block");
            assert_eq!(p.get_or_commit_into_prefix_cache(0, "prefix", first), None);

            let duplicate = p.alloc_block(0).expect("a free block");
            assert_eq!(
                p.get_or_commit_into_prefix_cache(0, "prefix", duplicate),
                Some(first)
            );

            assert_eq!(p.little_blocks[0][first].ref_cnt, 2);
            // The duplicate's bytes are redundant, so they go back to the pool.
            assert_eq!(p.little_blocks[0][duplicate].ref_cnt, 0);
            assert!(p.little_blocks[0][duplicate].is_free());
            check_invariants(&p);
        }
    }

    #[test]
    fn get_or_commit_is_a_noop_for_an_already_committed_block() {
        for null_block in [false, true] {
            let mut p = pool(1, &[1], null_block);

            let bid = p.alloc_block(0).expect("a free block");
            p.commit_into_prefix_cache(0, "prefix", bid);

            assert_eq!(p.get_or_commit_into_prefix_cache(0, "prefix", bid), None);
            assert_eq!(p.little_blocks[0][bid].ref_cnt, 1);
            check_invariants(&p);
        }
    }

    #[test]
    fn caches_do_not_share_a_prefix_cache() {
        for null_block in [false, true] {
            let mut p = pool(2, &[1, 1], null_block);

            // The same tokens hash the same way for every cache, but each cache
            // stores different bytes for them, so a hash names one block per
            // cache.
            let global_bid = p.alloc_block(0).expect("a free block");
            let sliding_bid = p.alloc_block(1).expect("a free block");
            p.commit_into_prefix_cache(0, "prefix", global_bid);
            p.commit_into_prefix_cache(1, "prefix", sliding_bid);

            assert_eq!(p.lookup_hash(0, "prefix"), Some(global_bid));
            assert_eq!(p.lookup_hash(1, "prefix"), Some(sliding_bid));
            check_invariants(&p);
        }
    }

    #[test]
    fn every_cache_gets_a_null_block_at_id_zero() {
        let mut p = pool(2, &[2, 1], true);

        // Dummy and padding requests point at block 0, which is why real
        // blocks start at id `ratio` and real huge blocks at id 1.
        assert_eq!(p.null_bid(), Some(0));
        assert_eq!(p.allocatable_huge_bids(), 1..3);
        // The null block costs no capacity, since it is outside the pool.
        assert_eq!(p.num_free_blocks(0), 4);
        check_invariants(&p);

        // Given the whole pool, a cache gets every id but huge block 0's: its
        // null block, and the ids past it that never exist.
        for (cache, ratio) in [(0, 2), (1, 1)] {
            let bids = alloc_n(&mut p, cache, 2 * ratio);
            expect_exhausted(&mut p, cache);
            assert_eq!(bids, (ratio..3 * ratio).collect::<Vec<_>>());
            assert_eq!(p.num_used_blocks(cache), 2 * ratio);
            check_invariants(&p);
            for bid in bids {
                p.free_block(cache, bid);
            }
        }
        check_invariants(&p);

        assert_eq!(pool(2, &[2, 1], false).null_bid(), None);
    }

    #[test]
    fn the_null_block_is_never_allocated_or_freed() {
        let mut p = pool(1, &[1], true);
        let null = p.null_bid().expect("a null block");

        // Exhausting the pool hands out its one real block, not the null one.
        assert_ne!(p.alloc_block(0).expect("a free block"), null);
        expect_exhausted(&mut p, 0);

        // Requests share the null block, so releasing or reviving it must not
        // put it in circulation -- however many dummy requests do. Python pins
        // its reference count instead; here it is never counted at all.
        for _ in 0..3 {
            p.free_block(0, null);
            p.touch(0, null);
        }

        assert_eq!(p.little_blocks[0][null].ref_cnt, 0);
        assert!(!p.little_blocks[0][null].is_free());
        expect_exhausted(&mut p, 0);
        check_invariants(&p);
    }

    #[test]
    #[should_panic(expected = "the null block cannot be committed")]
    fn the_null_block_cannot_be_committed() {
        let mut p = pool(1, &[1], true);
        // Its bytes are shared by every dummy request, so no hash describes
        // them.
        p.commit_into_prefix_cache(0, "prefix", 0);
    }

    #[test]
    #[should_panic(expected = "the null block cannot be committed")]
    fn the_null_block_cannot_be_deduplicated() {
        let mut p = pool(1, &[1], true);
        let bid = p.alloc_block(0).expect("a free block");
        p.commit_into_prefix_cache(0, "prefix", bid);
        p.get_or_commit_into_prefix_cache(0, "prefix", 0);
    }

    #[test]
    #[should_panic(expected = "does not belong to")]
    fn ids_past_the_null_block_do_not_exist() {
        let mut p = pool(1, &[2], true);
        p.touch(0, 1);
    }

    #[test]
    fn retyping_never_reaches_the_null_block() {
        let mut p = pool(1, &[2, 1], true);

        let bid = p.alloc_block(0).expect("a free block");
        p.free_block(0, bid);
        // The only bytes to retype are huge block 1's: huge block 0 holds every
        // cache's null block at once, so it is never claimed.
        let retyped = p.alloc_block(1).expect("a free block");
        assert_eq!(retyped, 1);
        expect_exhausted(&mut p, 0);
        expect_exhausted(&mut p, 1);

        p.free_block(1, retyped);
        assert_eq!(p.alloc_block(0).expect("a free block"), 2);
        check_invariants(&p);
    }

    #[test]
    fn uncommitting_drops_the_hash_and_keeps_the_block() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);

            let bid = p.alloc_block(0).expect("a free block");
            p.commit_into_prefix_cache(0, "prefix", bid);

            p.uncommit_block(0, bid);

            // The block keeps its reference; only the hash goes, so nobody can
            // hit it again.
            assert_eq!(p.little_blocks[0][bid].ref_cnt, 1);
            assert_eq!(p.little_blocks[0][bid].block_hash, None);
            assert!(p.prefix_caches[0].is_empty());

            // Uncommitting an uncommitted block is a no-op.
            p.uncommit_block(0, bid);
            assert!(p.prefix_caches[0].is_empty());
            check_invariants(&p);
        }
    }

    #[test]
    fn reset_prefix_cache_purges_only_unreferenced_commits() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2, 1], null_block);

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "apple", bids[0]);
            p.commit_into_prefix_cache(0, "banana", bids[1]);
            p.free_block(0, bids[1]);
            let sliding_bid = p.alloc_block(1).expect("a free block");
            p.commit_into_prefix_cache(1, "apple", sliding_bid);

            assert_eq!(p.reset_prefix_cache(), vec![1, 0]);

            // A commit a request still references survives.
            assert_eq!(p.prefix_caches[0], HashMap::from([("apple", bids[0])]));
            assert_eq!(p.prefix_caches[1], HashMap::from([("apple", sliding_bid)]));
            check_invariants(&p);
        }
    }

    #[test]
    fn occupancy_counts_unreferenced_commits_as_cached() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2], null_block);
            let cached = |p: &Pool| p.occupancy(0).cached;

            let bids = alloc_n(&mut p, 0, 2);
            p.commit_into_prefix_cache(0, "apple", bids[0]);
            assert_eq!(cached(&p), 0);
            p.free_block(0, bids[0]);
            assert_eq!(cached(&p), 1);
            p.touch(0, bids[0]);
            assert_eq!(cached(&p), 0);
            p.free_block(0, bids[0]);
            p.free_block(0, bids[1]);
            // Nothing stops a caller committing a block no request holds.
            p.commit_into_prefix_cache(0, "banana", bids[1]);
            assert_eq!(cached(&p), 2);
            check_invariants(&p);

            p.reset_prefix_cache();
            assert_eq!(cached(&p), 0);
            check_invariants(&p);
        }
    }

    #[test]
    fn skip_prefix_cache_drops_every_commit() {
        let options = PoolOptions {
            skip_prefix_cache: true,
            ..PoolOptions::default()
        };
        let mut p = Pool::new(1, vec![2], options).expect("valid geometry");

        let bids = alloc_n(&mut p, 0, 2);
        p.commit_into_prefix_cache(0, "prefix", bids[0]);
        assert_eq!(
            p.get_or_commit_into_prefix_cache(0, "prefix", bids[1]),
            None
        );

        assert_eq!(p.lookup_hash(0, "prefix"), None);
        assert!(p.prefix_caches[0].is_empty());
        assert_eq!(p.little_blocks[0][bids[0]].block_hash, None);
        check_invariants(&p);
    }

    /// `test_can_satisfy_demand_charges_every_cache_to_one_budget`.
    #[test]
    fn can_satisfy_demand_charges_every_cache_to_one_budget() {
        for null_block in [false, true] {
            // 4 allocatable huge blocks: 8 GLOBAL pages or 16 SCALES pages on
            // their own, but only 3 huge blocks' worth once both are asked for
            // at once.
            let p = pool(4, &[2, 4], null_block);
            assert_eq!(p.num_free_blocks(0), 8);
            assert_eq!(p.num_free_blocks(1), 16);

            assert!(p.can_satisfy_demand(&[8, 0], false));
            assert!(p.can_satisfy_demand(&[0, 16], false));
            // 8 GLOBAL pages take all 4 huge blocks, leaving none for the scales.
            assert!(!p.can_satisfy_demand(&[8, 1], false));
            assert!(p.can_satisfy_demand(&[6, 4], false));
        }
    }

    /// `test_can_satisfy_demand_credits_pages_already_carved`.
    #[test]
    fn can_satisfy_demand_credits_pages_already_carved() {
        for null_block in [false, true] {
            let mut p = pool(2, &[2, 4], null_block);
            // Carve one huge block for each cache and hand a page back, so both
            // have free pages while no huge block is left to claim.
            for cache in 0..2 {
                let bid = p.alloc_block(cache).expect("a free block");
                p.free_block(cache, bid);
            }

            // Each cache alone would retype the other's parked huge block, so
            // on its own it counts far more than the pages both can hold at
            // once.
            assert_eq!(p.num_free_blocks(0), 4);
            assert_eq!(p.num_free_blocks(1), 8);
            assert!(p.can_satisfy_demand(&[2, 4], false));
            assert!(!p.can_satisfy_demand(&[3, 4], false));
            assert!(!p.can_satisfy_demand(&[4, 8], false));
            check_invariants(&p);
        }
    }

    /// `test_can_satisfy_demand_retypes_surplus_parked_pages`.
    #[test]
    fn can_satisfy_demand_retypes_surplus_parked_pages() {
        for null_block in [false, true] {
            let mut p = pool(4, &[2, 4], null_block);
            let bids = alloc_n(&mut p, 1, 8);
            for bid in bids {
                p.free_block(1, bid);
            }

            // One parked SCALES huge block satisfies its demand. The other can
            // be retyped with the two pristine blocks to satisfy GLOBAL's.
            assert!(p.can_satisfy_demand(&[6, 4], false));
            check_invariants(&p);
        }
    }

    /// `test_can_satisfy_demand_at_capacity_ignores_what_is_out`.
    #[test]
    fn can_satisfy_demand_at_capacity_ignores_what_is_out() {
        for null_block in [false, true] {
            let mut p = pool(4, &[2, 4], null_block);
            let demand = [6, 4];
            assert!(p.can_satisfy_demand(&demand, false));

            // Drain the pool: the live answer flips, the geometric one does not.
            alloc_n(&mut p, 0, 8);
            assert!(!p.can_satisfy_demand(&demand, false));
            assert!(p.can_satisfy_demand(&demand, true));
            // Still bounded by the geometry -- 4 huge blocks is all there ever is.
            assert!(!p.can_satisfy_demand(&[8, 1], true));
        }
    }

    #[test]
    #[should_panic(expected = "double-freed")]
    fn double_free_panics() {
        let mut p = pool(1, &[1], false);
        let bid = p.alloc_block(0).expect("a free block");
        p.free_block(0, bid);
        p.free_block(0, bid);
    }

    #[test]
    #[should_panic(expected = "does not belong to")]
    fn touching_a_retyped_block_panics() {
        let mut p = pool(1, &[2, 1], false);

        let bid = p.alloc_block(0).expect("a free block");
        p.free_block(0, bid);
        p.alloc_block(1).expect("a free block");

        // The bytes now hold the sliding cache's tensor, so reviving the global
        // block would hand back the wrong pages.
        p.touch(0, bid);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn naming_a_cache_the_pool_lacks_panics() {
        let mut p = pool(1, &[1], false);
        drop(p.alloc_block(99));
    }
}
