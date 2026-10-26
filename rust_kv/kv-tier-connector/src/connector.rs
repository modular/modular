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
//! The host/disk tiered KV connector — the only implementation behind the
//! `tiered` and `rust_tiered` (host + disk, write-through) connector types,
//! whose Python versions it replaced.
//!
//! `load`/`offload` run on the scheduler thread
//! (via pyo3, GIL released), do only cheap bookkeeping against the host
//! `JengaBlockPool`, and hand the copies/disk I/O to lanes. A load returns a
//! [`Counter`] the scheduler polls; the request is cordoned until it completes
//! (host-observed), so the forward never reads KV that has not landed. Host
//! blocks are pinned at submit and freed by `reclaim` on the scheduler thread
//! once their transfer completes — the lanes only hand ids back through a
//! thread-safe queue, so the pool needs no lock.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use kv_core::pool::block_pool::{JengaBlockPool, PoolOptions};

use crate::{
    copy_engine::CopyEngine,
    disk_tier::DiskTier,
    lane::{AlreadyComplete, Pollable, TransferLane},
    trace,
    transfer::Counter,
    utils::{BlockHash, LeafHash},
};

/// Result of a `load`/`offload`: the device (G0) blocks the manager pins until
/// completion, and the completion the scheduler polls.
pub struct TransferResult {
    /// `g0_blocks_per_leaf[leaf_idx]` are the device blocks pinned for that
    /// leaf -- a plain `Vec<usize>` can't carry this (an `offload` can pin a
    /// different depth per leaf, e.g. a leaf already host-resident or out of
    /// free blocks is skipped for that hash, so the leaves aren't lockstep).
    pub g0_blocks_per_leaf: Vec<Vec<usize>>,
    pub counter: Counter,
    /// Blocks an `offload` skipped because the host tier already held them,
    /// counted per `(hash, leaf)`. The caller subtracts these from what it
    /// asked for, so only blocks the pool had no room for count as dropped.
    pub already_held: usize,
}

impl TransferResult {
    fn complete(num_leaves: usize) -> Self {
        TransferResult {
            counter: Counter::new(0),
            g0_blocks_per_leaf: vec![Vec::new(); num_leaves],
            already_held: 0,
        }
    }
}

/// A host block's lifecycle event, handed back by a lane to the scheduler thread.
///
/// One queue carries both kinds, drained in push order, so a `Release` can never
/// be applied without the `Commit` pushed before it: a block released with its
/// publish still owed could be reallocated to another hash and then receive the
/// stale publish, corrupting the host tier.
enum HostBlockEvent {
    /// The block holds the KV for this hash; publish it into the prefix cache.
    Commit(LeafHash, usize),
    /// The block's last reader is done; release the pin taken for the transfer.
    Release(usize, usize),
}

/// One block of one leaf for an onload to copy host -> device. On an `is_disk`
/// hit, `host_bid` is unset until `reserve_disk_promotions` allocates it.
struct Hit {
    host_bid: usize,
    device_bid: usize,
    hash: LeafHash,
    is_disk: bool,
}

/// Allocates each disk promotion in `hits` a host block to land in.
///
/// Runs after every host hit is pinned, so it cannot evict a hash this load
/// needs. On exhaustion, unpins the whole load and returns false.
fn reserve_disk_promotions(pool: &mut JengaBlockPool<BlockHash>, hits: &mut [Hit]) -> bool {
    for i in 0..hits.len() {
        if !hits[i].is_disk {
            continue;
        }
        let Ok(host_bid) = pool.alloc_block(hits[i].hash.leaf_idx) else {
            // `i` and later never got a block, so only earlier promotions and
            // the host hits pinned while planning hold a pin.
            for (j, hit) in hits.iter().enumerate() {
                if !hit.is_disk || j < i {
                    pool.free_block(hit.hash.leaf_idx, hit.host_bid);
                }
            }
            tracing::warn!(
                leaf_idx = hits[i].hash.leaf_idx,
                num_hits = hits.len(),
                "kv host pool saturated by in-flight transfers, abandoning onload"
            );
            return false;
        };
        hits[i].host_bid = host_bid;
    }
    true
}

/// Plans an onload: the device block each leaf-hash lands in, and the sources
/// to copy from.
///
/// Refuses before pinning anything, so a refusal has nothing to unwind. The
/// residency it re-checks can genuinely have changed: an offload's
/// write-through LRU-evicts from the disk index on the disk-write lane. A
/// vanished hash is [`ConnectorError::TransferFailed`]; wrong shapes are
/// [`ConnectorError::Config`].
fn plan_load(
    pool: &mut JengaBlockPool<BlockHash>,
    disk: &DiskTier,
    num_leaves: usize,
    block_ids: &[Vec<usize>],
    hashes_per_leaf: &[Vec<BlockHash>],
) -> crate::error::Result<(Vec<Vec<usize>>, Vec<Hit>)> {
    if hashes_per_leaf.len() != num_leaves || block_ids.len() != num_leaves {
        return Err(crate::error::ConnectorError::Config(format!(
            "load needs one hash list and one block list per leaf, got {} and {} for {num_leaves} leaves",
            hashes_per_leaf.len(),
            block_ids.len(),
        )));
    }
    for (leaf_idx, hashes) in hashes_per_leaf.iter().enumerate() {
        if block_ids[leaf_idx].len() < hashes.len() {
            return Err(crate::error::ConnectorError::Config(format!(
                "leaf {leaf_idx} was given {} destination blocks for {} hashes",
                block_ids[leaf_idx].len(),
                hashes.len()
            )));
        }
        for &block_hash in hashes {
            let leaf_hash = LeafHash {
                leaf_idx,
                block_hash,
            };
            if pool
                .lookup_hash(leaf_hash.leaf_idx, leaf_hash.block_hash)
                .is_none()
                && !disk.contains(&leaf_hash)
            {
                return Err(crate::error::ConnectorError::TransferFailed(format!(
                    "load asked for {} but neither tier holds it any more",
                    leaf_hash.to_hex()
                )));
            }
        }
    }

    let mut g0_blocks_per_leaf: Vec<Vec<usize>> = vec![Vec::new(); num_leaves];
    let mut hits: Vec<Hit> = Vec::new();
    for (leaf_idx, hashes) in hashes_per_leaf.iter().enumerate() {
        for (hash_idx, &block_hash) in hashes.iter().enumerate() {
            let hash = LeafHash {
                leaf_idx,
                block_hash,
            };
            let device_bid = block_ids[leaf_idx][hash_idx];
            g0_blocks_per_leaf[leaf_idx].push(device_bid);
            match pool.lookup_hash(hash.leaf_idx, hash.block_hash) {
                Some(host_bid) => {
                    pool.touch(leaf_idx, host_bid);
                    hits.push(Hit {
                        host_bid,
                        device_bid,
                        hash,
                        is_disk: false,
                    });
                }
                // Don't allocate a host block for a disk promotion here: doing
                // so could evict a hash this same load needs.
                None => hits.push(Hit {
                    host_bid: 0,
                    device_bid,
                    hash,
                    is_disk: true,
                }),
            }
        }
    }
    Ok((g0_blocks_per_leaf, hits))
}

/// One block of one leaf for an offload to copy device -> host.
struct ToOffload {
    host_bid: usize,
    device_bid: usize,
    hash: LeafHash,
}

/// Plans an offload: the host block each leaf-hash lands in.
///
/// Best effort per block, unlike a load: a leaf's block is skipped, not
/// refused, when the host tier already holds its hash or has no free block
/// to take it. Returns the plan and how many skips were already held. Wrong shapes are
/// [`ConnectorError::Config`](crate::error::ConnectorError::Config).
///
/// Leaves need not agree on a depth -- a recurrent leaf publishes a block
/// only at the boundary its checkpoint landed on -- but the plan stays
/// hash-major across them, so a lookup racing the copies sees a prefix of
/// the run in every leaf rather than one leaf's whole run and nothing of
/// the next.
fn plan_offload(
    pool: &mut JengaBlockPool<BlockHash>,
    num_leaves: usize,
    block_ids: &[Vec<usize>],
    hashes_per_leaf: &[Vec<BlockHash>],
) -> crate::error::Result<(Vec<ToOffload>, usize)> {
    if hashes_per_leaf.len() != num_leaves || block_ids.len() != num_leaves {
        return Err(crate::error::ConnectorError::Config(format!(
            "offload needs one hash list and one block list per leaf, got {} and {} for {num_leaves} leaves",
            hashes_per_leaf.len(),
            block_ids.len(),
        )));
    }
    for (leaf_idx, hashes) in hashes_per_leaf.iter().enumerate() {
        if block_ids[leaf_idx].len() < hashes.len() {
            return Err(crate::error::ConnectorError::Config(format!(
                "leaf {leaf_idx} was given {} source blocks for {} hashes",
                block_ids[leaf_idx].len(),
                hashes.len()
            )));
        }
    }

    let depth = hashes_per_leaf.iter().map(Vec::len).max().unwrap_or(0);
    let mut plan: Vec<ToOffload> = Vec::new();
    let mut already_held = 0;
    for hash_idx in 0..depth {
        for (leaf_idx, (hashes, ids)) in hashes_per_leaf.iter().zip(block_ids).enumerate() {
            let (Some(&block_hash), Some(&device_bid)) = (hashes.get(hash_idx), ids.get(hash_idx))
            else {
                continue;
            };
            if pool.prefix_caches[leaf_idx].contains_key(&block_hash) {
                already_held += 1;
                continue;
            }
            if pool.num_free_blocks(leaf_idx) == 0 {
                continue;
            }
            let Ok(host_bid) = pool.alloc_block(leaf_idx) else {
                continue;
            };
            plan.push(ToOffload {
                host_bid,
                device_bid,
                hash: LeafHash {
                    leaf_idx,
                    block_hash,
                },
            });
        }
    }
    Ok((plan, already_held))
}

/// Metrics counters, in bytes (updated from lane threads, so atomic).
///
/// Bytes rather than blocks: these tiers are byte budgets the operator sizes in
/// bytes, their block width need not match the device's, and bytes rate
/// directly against PCIe and disk bandwidth.
#[derive(Default)]
struct Metrics {
    h2d_bytes: AtomicU64,
    d2h_bytes: AtomicU64,
    disk_read_bytes: AtomicU64,
    disk_write_bytes: AtomicU64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafRecency {
    /// The whole sequence, last block first, so the shared prefix at its
    /// start is evicted last.
    FullSequence,
    /// The last `blocks` of the sequence, in order, so blocks the window
    /// slid past age toward eviction.
    SlidingWindow { blocks: usize },
    /// The whole sequence, in order, so the latest state is evicted last.
    Recurrent,
}

/// Returns `hashes` in the order a leaf with `recency` touches them.
fn touch_order(recency: LeafRecency, hashes: &[BlockHash]) -> Vec<BlockHash> {
    match recency {
        LeafRecency::FullSequence => hashes.iter().rev().copied().collect(),
        LeafRecency::SlidingWindow { blocks } => {
            hashes[hashes.len().saturating_sub(blocks)..].to_vec()
        }
        LeafRecency::Recurrent => hashes.to_vec(),
    }
}

/// Moves the host block holding `block_hash` to the most-recently-used end of
/// `leaf_idx`'s free queue, if the host tier holds one.
fn touch_host(pool: &mut JengaBlockPool<BlockHash>, leaf_idx: usize, block_hash: BlockHash) {
    if let Some(host_bid) = pool.lookup_hash(leaf_idx, block_hash) {
        pool.touch(leaf_idx, host_bid);
        pool.free_block(leaf_idx, host_bid);
    }
}

/// The tiered KV connector: host tier plus an optional disk last level.
pub struct TieredConnector {
    num_leaves: usize,
    pub pool: JengaBlockPool<BlockHash>,
    pub disk: Arc<DiskTier>,
    engine: Arc<CopyEngine>,
    bytes_per_leaf: Vec<usize>,
    recency: Vec<LeafRecency>,

    onload_lanes: Vec<Arc<TransferLane>>,
    onload_rr: AtomicUsize,
    offload_lane: Arc<TransferLane>,
    disk_read_lane: Arc<TransferLane>,
    disk_write_lane: Arc<TransferLane>,

    // Host block events the lanes hand back, applied by `reclaim` on the
    // scheduler thread -- so the pool itself stays lock-free.
    host_block_events: Arc<Mutex<Vec<HostBlockEvent>>>,

    metrics: Arc<Metrics>,
}

/// Round-robin index into `n` onload lanes.
fn next_onload_lane(rr: &AtomicUsize, n: usize) -> usize {
    debug_assert!(n > 0);
    rr.fetch_add(1, Ordering::Relaxed) % n
}

impl TieredConnector {
    /// Builds a connector.
    #[allow(
        clippy::too_many_arguments,
        reason = "connector construction threads through all tier and leaf config"
    )]
    pub fn new(
        total_num_host_blocks: usize,
        cache_ratios: Vec<usize>,
        disk: Arc<DiskTier>,
        engine: Arc<CopyEngine>,
        bytes_per_leaf: Vec<usize>,
        only_last_level: bool,
        num_h2d_onload_lanes: usize,
        recency: Vec<LeafRecency>,
    ) -> crate::error::Result<Self> {
        if num_h2d_onload_lanes == 0 {
            return Err(crate::error::ConnectorError::Config(
                "num_h2d_onload_lanes must be at least 1".to_owned(),
            ));
        }
        if recency.len() != cache_ratios.len() {
            return Err(crate::error::ConnectorError::Config(format!(
                "got a recency for {} leaves, expected {}",
                recency.len(),
                cache_ratios.len()
            )));
        }
        let skip_host_cache_tier = only_last_level && disk.enabled;
        Ok(TieredConnector {
            num_leaves: cache_ratios.len(),
            pool: JengaBlockPool::new(
                total_num_host_blocks,
                cache_ratios,
                PoolOptions {
                    skip_prefix_cache: skip_host_cache_tier,
                    ..PoolOptions::default()
                },
            )
            .unwrap(),
            disk,
            engine,
            bytes_per_leaf,
            recency,
            // Each onload lane does a blocking `memcpy; sync` per copy on its
            // own H2D stream. N>1 lets host-hit admits overlap. N=1 is the
            // historical FIFO that yields the copy engine after every block.
            onload_lanes: (0..num_h2d_onload_lanes)
                .map(|i| Arc::new(TransferLane::new(&format!("kv-onload-{i}"))))
                .collect(),
            onload_rr: AtomicUsize::new(0),
            offload_lane: Arc::new(TransferLane::new("kv-offload")),
            disk_read_lane: Arc::new(TransferLane::new("kv-disk-read")),
            disk_write_lane: Arc::new(TransferLane::new("kv-disk-write")),
            host_block_events: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Metrics::default()),
        })
    }

    /// Next onload lane in round-robin order, and its stream index.
    fn pick_onload_lane(&self) -> (usize, Arc<TransferLane>) {
        let i = next_onload_lane(&self.onload_rr, self.onload_lanes.len());
        (i, Arc::clone(&self.onload_lanes[i]))
    }

    /// Which of `hashes` each of `leaf_idxs` holds, one mask per leaf.
    ///
    /// Presence only: how deep a prefix these add up to depends on how far each
    /// leaf's attention reads back, and that is the cache manager's business,
    /// not this connector's. A `true` promises the block is readable from the
    /// host or disk tier of *that one leaf*; it says nothing about its siblings.
    ///
    /// Takes no pin, and the disk index is not ours alone: an offload's
    /// write-through LRU-evicts from it on the disk-write lane, so a
    /// disk-resident block reported here can be gone by the time
    /// [`Self::load`] asks for it. The host tier has no such race.
    #[must_use]
    pub fn lookup(&self, leaf_idxs: &[usize], hashes: &[BlockHash]) -> Vec<Vec<bool>> {
        leaf_idxs
            .iter()
            .map(|&leaf_idx| {
                hashes
                    .iter()
                    .map(|&block_hash| {
                        self.contains(&LeafHash {
                            leaf_idx,
                            block_hash,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Whether either tier can serve `leaf_hash`.
    fn contains(&self, leaf_hash: &LeafHash) -> bool {
        self.pool
            .lookup_hash(leaf_hash.leaf_idx, leaf_hash.block_hash)
            .is_some()
            || self.disk.contains(leaf_hash)
    }

    /// Refreshes each leaf's recency for the blocks of `hashes` it reads,
    /// ordered by the leaf's [`LeafRecency`], so a later-touched block is
    /// evicted after an earlier one.
    ///
    /// `hashes` is the whole sequence, in order. A host block goes to the
    /// most-recently-used end of its free queue; one a transfer still pins goes
    /// there when the pin drops. Hashes neither tier holds are skipped.
    pub fn touch(&mut self, hashes: &[BlockHash]) {
        for leaf_idx in 0..self.recency.len() {
            let ordered = touch_order(self.recency[leaf_idx], hashes);
            for &block_hash in &ordered {
                touch_host(&mut self.pool, leaf_idx, block_hash);
            }
            self.disk
                .touch(ordered.into_iter().map(|block_hash| LeafHash {
                    leaf_idx,
                    block_hash,
                }));
        }
    }

    /// Applies the host block events the lanes handed back: publishes filled
    /// blocks into the host prefix cache and releases the pins of blocks whose
    /// last reader is done (scheduler thread only). Called at the start of every
    /// `load`/`offload`.
    pub fn reclaim(&mut self) {
        let events: Vec<HostBlockEvent> =
            std::mem::take(&mut *self.host_block_events.lock().unwrap());
        for event in events {
            match event {
                HostBlockEvent::Commit(hash, host_bid) => {
                    if self
                        .pool
                        .lookup_hash(hash.leaf_idx, hash.block_hash)
                        .is_none()
                    {
                        self.pool.commit_into_prefix_cache(
                            hash.leaf_idx,
                            hash.block_hash,
                            host_bid,
                        );
                    }
                }
                HostBlockEvent::Release(leaf_idx, host_bid) => {
                    self.pool.free_block(leaf_idx, host_bid)
                }
            }
        }
    }

    /// Loads exactly the blocks it is told to, leaf by leaf.
    ///
    /// `hashes_per_leaf[leaf_idx]` names the hashes to read for that leaf, in
    /// the order its row holds them, and `block_ids[leaf_idx]` the device
    /// blocks to land them in, one per hash. Which hashes those are -- the
    /// whole prefix for a full-attention leaf, a window's worth for a
    /// sliding-window one -- is settled by the caller before it gets here.
    ///
    /// Returns the device blocks the copies will write (`block_ids` restated
    /// per leaf) and a completion the scheduler polls.
    ///
    /// # Errors
    ///
    /// Fails, without submitting anything, if a requested hash is resident in
    /// neither tier. A caller that looked the hashes up first will only see
    /// this if one was evicted in between; short-loading the run instead would
    /// hand back a row whose tail the caller then trusts as cached KV.
    pub fn load(
        &mut self,
        block_ids: &[Vec<usize>],
        hashes_per_leaf: &[Vec<BlockHash>],
        replica_idx: usize,
    ) -> crate::error::Result<TransferResult> {
        self.reclaim();
        let (g0_blocks_per_leaf, mut hits) = plan_load(
            &mut self.pool,
            &self.disk,
            self.num_leaves,
            block_ids,
            hashes_per_leaf,
        )?;

        if hits.is_empty() {
            return Ok(TransferResult::complete(self.num_leaves));
        }

        // The disk-write lane can have evicted a disk hit since `plan_load`
        // checked it. Pinning keeps its slot from being reused before the read
        // lands. It runs before the host allocations below so that a refusal
        // has evicted nothing from the host cache, and only the host hits
        // `plan_load` touched need unpinning.
        let disk_hashes: Vec<LeafHash> = hits
            .iter()
            .filter(|hit| hit.is_disk)
            .map(|hit| hit.hash)
            .collect();
        let Some(disk_pins) = self.disk.pin_reads(&disk_hashes) else {
            for hit in hits.iter().filter(|hit| !hit.is_disk) {
                self.pool.free_block(hit.hash.leaf_idx, hit.host_bid);
            }
            return Err(crate::error::ConnectorError::TransferFailed(
                "load asked for a disk block that was evicted while it was planned".to_owned(),
            ));
        };

        // Allocate before submitting anything: the pool runs full by design and
        // a pinned block can't be evicted, so running out here only means
        // in-flight transfers have it saturated. Nothing is on a lane yet, so
        // give the load up whole rather than truncate the hit prefix.
        if !reserve_disk_promotions(&mut self.pool, &mut hits) {
            self.disk.release_pins(&disk_pins);
            return Ok(TransferResult::complete(self.num_leaves));
        }
        let mut disk_pins = disk_pins.into_iter();

        let h2d_bytes: u64 = hits
            .iter()
            .map(|hit| self.bytes_per_leaf[hit.hash.leaf_idx] as u64)
            .sum();
        self.metrics
            .h2d_bytes
            .fetch_add(h2d_bytes, Ordering::AcqRel);
        let counter = Counter::new(hits.len());
        let gate = Arc::new(self.engine.snapshot_compute(replica_idx));

        for hit in &hits {
            let engine = Arc::clone(&self.engine);
            let gate = Arc::clone(&gate);
            let counter = counter.clone();
            let events = Arc::clone(&self.host_block_events);
            let device_bid = hit.device_bid;
            let host_bid = hit.host_bid;
            let hash = hit.hash;
            let leaf_idx = hash.leaf_idx;
            let host_ptr = self.engine.host_ptr(leaf_idx, host_bid);
            let metrics = Arc::clone(&self.metrics);
            let leaf_bytes = self.bytes_per_leaf[leaf_idx] as u64;
            let (lane_idx, onload_lane) = self.pick_onload_lane();

            // The H2D work item (onload lane): copy host -> device, mark done,
            // then release the host source. The H2D is that block's last reader,
            // so its pin (the `touch` on a host hit, the alloc on a disk
            // promotion) has to outlive the copy, not just its submission --
            // releasing earlier would let another transfer reallocate the block
            // and overwrite it before this copy read it.
            let make_h2d = move || {
                let engine = Arc::clone(&engine);
                let gate = Arc::clone(&gate);
                let counter = counter.clone();
                let events = Arc::clone(&events);
                (
                    Box::new(move || -> Box<dyn Pollable> {
                        let _r = trace::Range::new("kv-onload h2d", trace::color::ONLOAD);
                        // Blocks this onload lane until the H2D lands (one
                        // in-flight on this stream); already complete on
                        // return. A failed copy aborts inside `h2d_sync`.
                        engine.h2d_sync(
                            replica_idx,
                            leaf_idx,
                            device_bid,
                            host_bid,
                            lane_idx,
                            &gate,
                        );
                        Box::new(AlreadyComplete)
                    }) as Box<dyn FnOnce() -> Box<dyn Pollable> + Send>,
                    Box::new(move || {
                        counter.mark_one();
                        events
                            .lock()
                            .unwrap()
                            .push(HostBlockEvent::Release(leaf_idx, host_bid));
                    }) as Box<dyn FnOnce() + Send>,
                )
            };

            if hit.is_disk {
                // disk-read lane: read into the host block, then chain the H2D.
                let pin = disk_pins
                    .next()
                    .expect("one disk pin per disk hit, in hit order");
                let disk = Arc::clone(&self.disk);
                let events = Arc::clone(&self.host_block_events);
                self.disk_read_lane.submit(
                    Box::new(move || {
                        // The real disk-read span is emitted on the disk worker
                        // thread inside `read_block_async`; this just enqueues.
                        metrics
                            .disk_read_bytes
                            .fetch_add(leaf_bytes, Ordering::AcqRel);
                        Box::new(disk.read_block_async(pin, host_ptr)) as Box<dyn Pollable>
                    }),
                    Box::new(move || {
                        // The read filled the dest, so it is publishable now; the
                        // H2D releases it once it has copied it out.
                        events
                            .lock()
                            .unwrap()
                            .push(HostBlockEvent::Commit(hash, host_bid));
                        let (run, done) = make_h2d();
                        onload_lane.submit(run, done);
                    }),
                );
            } else {
                let (run, done) = make_h2d();
                onload_lane.submit(run, done);
            }
        }

        Ok(TransferResult {
            g0_blocks_per_leaf,
            counter,
            already_held: 0,
        })
    }

    /// Writes `hashes_per_leaf[leaf_idx]` out of `block_ids[leaf_idx]`, to the
    /// host and through to disk. Returns the device source blocks (pinned
    /// until the D2H lands) and the completion. See [`plan_offload`].
    pub fn offload(
        &mut self,
        block_ids: &[Vec<usize>],
        hashes_per_leaf: &[Vec<BlockHash>],
        replica_idx: usize,
    ) -> crate::error::Result<TransferResult> {
        self.reclaim();
        let (plan, already_held) =
            plan_offload(&mut self.pool, self.num_leaves, block_ids, hashes_per_leaf)?;

        if plan.is_empty() {
            return Ok(TransferResult {
                already_held,
                ..TransferResult::complete(self.num_leaves)
            });
        }

        let d2h_bytes: u64 = plan
            .iter()
            .map(|hit| self.bytes_per_leaf[hit.hash.leaf_idx] as u64)
            .sum();
        self.metrics
            .d2h_bytes
            .fetch_add(d2h_bytes, Ordering::AcqRel);
        let counter = Counter::new(plan.len());
        let mut g0_blocks_per_leaf: Vec<Vec<usize>> = vec![Vec::new(); self.num_leaves];
        for hit in &plan {
            g0_blocks_per_leaf[hit.hash.leaf_idx].push(hit.device_bid);
        }
        let gate = Arc::new(self.engine.snapshot_compute(replica_idx));

        for hit in &plan {
            let engine = Arc::clone(&self.engine);
            let gate = Arc::clone(&gate);
            let counter = counter.clone();
            let events = Arc::clone(&self.host_block_events);
            let device_bid = hit.device_bid;
            let host_bid = hit.host_bid;
            let hash = hit.hash;
            let leaf_idx = hash.leaf_idx;
            let host_ptr = self.engine.host_ptr(leaf_idx, host_bid);
            let metrics = Arc::clone(&self.metrics);
            let disk = Arc::clone(&self.disk);
            let disk_write_lane = Arc::clone(&self.disk_write_lane);
            let leaf_bytes = self.bytes_per_leaf[leaf_idx] as u64;

            self.offload_lane.submit(
                Box::new(move || -> Box<dyn Pollable> {
                    let _r = trace::Range::new("kv-offload d2h", trace::color::OFFLOAD);
                    // Blocks the offload lane thread until the D2H lands (one
                    // in-flight, no polling); already complete when it returns. A
                    // failed copy aborts the process inside `d2h_sync`.
                    engine.d2h_sync(replica_idx, leaf_idx, host_bid, device_bid, &gate);
                    Box::new(AlreadyComplete)
                }),
                Box::new(move || {
                    // Device source is free once the D2H lands, and the dest it
                    // filled is now publishable.
                    counter.mark_one();
                    events
                        .lock()
                        .unwrap()
                        .push(HostBlockEvent::Commit(hash, host_bid));
                    // Chain the write-through, which reads the dest, so the pin
                    // is released only once that write has landed. A declined
                    // write (already saved, pending, or no disk tier) completes
                    // immediately and releases the pin there.
                    disk_write_lane.submit(
                        Box::new(move || -> Box<dyn Pollable> {
                            // The real disk-write span is emitted on the disk
                            // worker thread inside `write_block_async`; this
                            // just enqueues. Counted only once queued, so a
                            // declined write is not reported as written.
                            match disk.write_block_async(hash, host_ptr) {
                                Some(op) => {
                                    metrics
                                        .disk_write_bytes
                                        .fetch_add(leaf_bytes, Ordering::AcqRel);
                                    Box::new(op)
                                }
                                None => Box::new(AlreadyComplete),
                            }
                        }),
                        Box::new(move || {
                            events
                                .lock()
                                .unwrap()
                                .push(HostBlockEvent::Release(leaf_idx, host_bid));
                        }),
                    );
                }),
            );
        }

        Ok(TransferResult {
            g0_blocks_per_leaf,
            counter,
            already_held,
        })
    }

    /// Total host tier capacity, in bytes.
    ///
    /// Every leaf tiles the same huge-block byte budget at its own ratio, so
    /// leaf 0 is as representative as any other.
    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        (self.pool.huge_blocks.len() * self.bytes_per_leaf[0] * self.pool.cache_ratios[0]) as u64
    }
    /// Host tier bytes not holding committed (cached) data, across every leaf.
    ///
    /// Not the pool's physical free list: a `ref_cnt == 0` block can still be
    /// committed as an eviction candidate (parking keeps commits), so this is
    /// `total - committed`, which is usually smaller than a raw free-block count.
    #[must_use]
    pub fn free_host_bytes(&self) -> u64 {
        self.host_bytes() - self.committed_host_bytes()
    }
    /// Host tier bytes currently holding committed (cached) data, summed
    /// across every leaf at its own byte width.
    #[must_use]
    fn committed_host_bytes(&self) -> u64 {
        (0..self.num_leaves)
            .map(|leaf_idx| {
                (self.pool.prefix_caches[leaf_idx].len() * self.bytes_per_leaf[leaf_idx]) as u64
            })
            .sum()
    }
    /// Total disk tier capacity, in bytes.
    #[must_use]
    pub fn disk_bytes(&self) -> u64 {
        self.disk.max_bytes() as u64
    }
    /// Disk tier bytes not currently indexed (cached).
    #[must_use]
    pub fn free_disk_bytes(&self) -> u64 {
        (self.disk.max_bytes() - self.disk.used_bytes()) as u64
    }

    /// `(h2d, d2h, disk_read, disk_write)` byte counts since the last reset.
    #[must_use]
    pub fn metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.metrics.h2d_bytes.load(Ordering::Acquire),
            self.metrics.d2h_bytes.load(Ordering::Acquire),
            self.metrics.disk_read_bytes.load(Ordering::Acquire),
            self.metrics.disk_write_bytes.load(Ordering::Acquire),
        )
    }

    /// Live count of disk read/write/delete ops queued or running on the disk
    /// pool (a gauge, not a cumulative counter -- unaffected by `take_metrics`).
    pub fn inflight_disk_ops(&self) -> u64 {
        self.disk.inflight_disk_ops() as u64
    }

    /// `(h2d, d2h, disk_read, disk_write)` byte counts read and cleared in one
    /// step, swapping each counter to zero as it reads it.
    pub fn take_metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.metrics.h2d_bytes.swap(0, Ordering::AcqRel),
            self.metrics.d2h_bytes.swap(0, Ordering::AcqRel),
            self.metrics.disk_read_bytes.swap(0, Ordering::AcqRel),
            self.metrics.disk_write_bytes.swap(0, Ordering::AcqRel),
        )
    }

    /// Purges the host prefix cache and the disk tier.
    pub fn reset_prefix_cache(&mut self) {
        // Quiesce all in-flight transfers before clearing the host and disk tiers.
        self.wait_for_writes();
        self.pool.reset_prefix_cache();
        // Clear the disk tier too, matching the Python tiered connector
        // (`reset_prefix_cache` resets both the host and disk caches).
        self.disk.reset();
    }

    /// Drains all in-flight transfers (D2H/H2D, disk read/write) without
    /// clearing any tier, then reclaims completed transfers so their pinned
    /// host/device blocks are released. Quiesces the connector so a following
    /// reset / residency check sees a stable state.
    pub fn wait_for_writes(&mut self) {
        // Drain in dependency order (matching shutdown): a disk read chains to
        // the onload lane, and an offload's D2H chains the write-through to the
        // disk-write lane, whose host source stays pinned until the write lands.
        self.disk_read_lane.wait_until_idle();
        for lane in &self.onload_lanes {
            lane.wait_until_idle();
        }
        self.offload_lane.wait_until_idle();
        self.disk_write_lane.wait_until_idle();
        self.disk.wait_for_writes();
        self.reclaim();
    }

    /// Returns the total number of bytes in the host tier.
    pub fn max_bytes(&self) -> usize {
        self.pool.huge_blocks.len() * self.bytes_per_leaf[0] * self.pool.cache_ratios[0]
    }

    /// Returns the number of bytes in the host tier that are currently committed.
    pub fn committed_bytes(&self) -> usize {
        (0..self.num_leaves)
            .map(|leaf_idx| self.pool.prefix_caches[leaf_idx].len() * self.bytes_per_leaf[leaf_idx])
            .sum()
    }

    /// Drains the lanes and disk, then the copy engine syncs on drop.
    ///
    /// Order matters: drain disk-read first (its `on_complete`s chain onto the
    /// onload lane), then onload, then the offload path.
    pub fn shutdown(&mut self) {
        self.disk_read_lane.shutdown();
        for lane in &self.onload_lanes {
            lane.shutdown();
        }
        self.offload_lane.shutdown();
        self.disk_write_lane.shutdown();
        self.disk.wait_for_writes();
        self.disk.shutdown();
        self.reclaim();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{
        BlockHash, DiskTier, Hit, JengaBlockPool, LeafHash, LeafRecency, PoolOptions,
        next_onload_lane, plan_load, plan_offload, reserve_disk_promotions, touch_host,
        touch_order,
    };

    fn hashes(count: u128) -> Vec<BlockHash> {
        (0..count).map(|value| BlockHash(value, value)).collect()
    }

    #[test]
    fn onload_lanes_round_robin() {
        let rr = AtomicUsize::new(0);
        assert_eq!(next_onload_lane(&rr, 2), 0);
        assert_eq!(next_onload_lane(&rr, 2), 1);
        assert_eq!(next_onload_lane(&rr, 2), 0);
        assert_eq!(rr.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_full_sequence_is_touched_last_block_first() {
        assert_eq!(
            touch_order(LeafRecency::FullSequence, &hashes(3)),
            [BlockHash(2, 2), BlockHash(1, 1), BlockHash(0, 0)]
        );
    }

    #[test]
    fn a_sliding_window_touches_only_its_window_in_order() {
        let window = LeafRecency::SlidingWindow { blocks: 2 };
        assert_eq!(
            touch_order(window, &hashes(4)),
            [BlockHash(2, 2), BlockHash(3, 3)]
        );
        // A window wider than the sequence is the whole sequence.
        assert_eq!(touch_order(window, &hashes(1)), [BlockHash(0, 0)]);
    }

    #[test]
    fn a_recurrent_leaf_is_touched_in_order() {
        assert_eq!(touch_order(LeafRecency::Recurrent, &hashes(2)), hashes(2));
    }

    #[test]
    fn a_touched_host_block_is_evicted_after_an_untouched_one() {
        let mut pool = JengaBlockPool::new(2, vec![1], PoolOptions::default()).expect("a pool");
        let [first, second] = [BlockHash(1, 1), BlockHash(2, 2)];
        for hash in [first, second] {
            let bid = pool.alloc_block(0).expect("a block");
            pool.commit_into_prefix_cache(0, hash, bid);
            pool.free_block(0, bid);
        }

        touch_host(&mut pool, 0, first);
        pool.alloc_block(0).expect("a block");

        assert!(pool.lookup_hash(0, first).is_some());
        assert!(pool.lookup_hash(0, second).is_none());
    }

    /// A disk promotion awaiting its host block.
    fn disk_hit(leaf_idx: usize, hash: u128) -> Hit {
        Hit {
            host_bid: 0,
            device_bid: 0,
            hash: LeafHash {
                leaf_idx,
                block_hash: BlockHash(hash, hash),
            },
            is_disk: true,
        }
    }

    #[test]
    fn reserving_disk_promotions_serves_every_hit_when_the_pool_has_room() {
        let mut pool = JengaBlockPool::new(4, vec![1], PoolOptions::default()).expect("a pool");
        let mut hits: Vec<Hit> = (0..4).map(|i| disk_hit(0, i)).collect();

        assert!(reserve_disk_promotions(&mut pool, &mut hits));

        let served: HashSet<usize> = hits.iter().map(|hit| hit.host_bid).collect();
        assert_eq!(served.len(), 4, "each promotion needs its own host block");
        assert_eq!(pool.num_free_blocks(0), 0);
    }

    #[test]
    fn an_exhausted_pool_refuses_the_load_and_keeps_no_blocks() {
        let mut pool = JengaBlockPool::new(2, vec![1], PoolOptions::default()).expect("a pool");
        // One more promotion than the pool can serve.
        let mut hits: Vec<Hit> = (0..3).map(|i| disk_hit(0, i)).collect();

        assert!(!reserve_disk_promotions(&mut pool, &mut hits));

        assert_eq!(
            pool.num_free_blocks(0),
            2,
            "a refused load must not strand the blocks it allocated"
        );
    }

    /// The reason the allocation is deferred: a host hit is pinned while
    /// planning, so no promotion can be handed the block it is about to read.
    #[test]
    fn a_promotion_cannot_be_handed_a_pinned_host_hit() {
        let hit_hash = BlockHash(9, 9);
        let mut pool = JengaBlockPool::new(3, vec![1], PoolOptions::default()).expect("a pool");

        // The host hit, committed first so it is the least recently used and
        // thus first in line to be evicted if nothing pins it.
        let hit_bid = pool.alloc_block(0).expect("a block");
        pool.commit_into_prefix_cache(0, hit_hash, hit_bid);
        pool.free_block(0, hit_bid);
        // A commit behind it, which is what the promotion should take instead.
        let spare_bid = pool.alloc_block(0).expect("a block");
        pool.commit_into_prefix_cache(0, BlockHash(8, 8), spare_bid);
        pool.free_block(0, spare_bid);
        // The last block held by an unrelated in-flight transfer, so the pool
        // has to evict a commit rather than claim a free huge block.
        pool.alloc_block(0).expect("a block");

        // Planning pins the host hit.
        pool.touch(0, hit_bid);

        let mut hits = vec![disk_hit(0, 1)];
        assert!(reserve_disk_promotions(&mut pool, &mut hits));

        assert_eq!(
            hits[0].host_bid, spare_bid,
            "the promotion evicted the host hit instead of the spare commit"
        );
        let hit_leaf_hash = LeafHash {
            leaf_idx: 0,
            block_hash: hit_hash,
        };
        assert_eq!(
            pool.lookup_hash(hit_leaf_hash.leaf_idx, hit_leaf_hash.block_hash),
            Some(hit_bid)
        );
    }

    /// Host hits are pinned while planning, before any allocation runs, so a
    /// refusal has to drop those pins too.
    #[test]
    fn a_refused_load_unpins_the_host_hits_it_did_not_allocate() {
        let mut pool = JengaBlockPool::new(2, vec![1], PoolOptions::default()).expect("a pool");

        // A committed, unreferenced block, as a host hit would find it.
        let host_bid = pool.alloc_block(0).expect("a block");
        pool.commit_into_prefix_cache(0, BlockHash(9, 9), host_bid);
        pool.free_block(0, host_bid);
        // Planning pins it, then asks for two promotions the pool cannot serve.
        pool.touch(0, host_bid);
        let mut hits = vec![
            Hit {
                host_bid,
                device_bid: 0,
                hash: LeafHash {
                    leaf_idx: 0,
                    block_hash: BlockHash(9, 9),
                },
                is_disk: false,
            },
            disk_hit(0, 1),
            disk_hit(0, 2),
        ];

        assert!(!reserve_disk_promotions(&mut pool, &mut hits));

        assert_eq!(
            pool.num_free_blocks(0),
            2,
            "the host hit's pin must be released along with the allocations"
        );
    }

    /// A committed, unreferenced host block, as a host hit would find it.
    fn commit_host_hit(pool: &mut JengaBlockPool<BlockHash>, hash: BlockHash) -> usize {
        let bid = pool.alloc_block(0).expect("a block");
        pool.commit_into_prefix_cache(0, hash, bid);
        pool.free_block(0, bid);
        bid
    }

    #[test]
    fn plan_load_pins_every_host_hit_it_found() {
        let mut pool = JengaBlockPool::new(4, vec![1], PoolOptions::default()).expect("a pool");
        let hashes = hashes(2);
        for &hash in &hashes {
            commit_host_hit(&mut pool, hash);
        }
        let free_before = pool.num_free_blocks(0);

        let (g0, hits) = plan_load(
            &mut pool,
            &DiskTier::disabled(),
            1,
            &[vec![7, 8]],
            std::slice::from_ref(&hashes),
        )
        .expect("both hashes are resident");

        assert_eq!(g0, vec![vec![7, 8]]);
        assert!(hits.iter().all(|hit| !hit.is_disk));
        assert_eq!(
            pool.num_free_blocks(0),
            free_before - 2,
            "each host hit is pinned for the copy that will read it"
        );
    }

    /// The disk index is the one tier a lane thread mutates, so a hash the
    /// caller looked up can be gone by now. Refusing before anything is pinned
    /// is what keeps that from leaking host blocks -- a leak here shrinks the
    /// host tier for the life of the process.
    #[test]
    fn plan_load_refuses_a_vanished_hash_and_pins_nothing() {
        let mut pool = JengaBlockPool::new(4, vec![1], PoolOptions::default()).expect("a pool");
        let hashes = hashes(2);
        commit_host_hit(&mut pool, hashes[0]);
        let free_before = pool.num_free_blocks(0);

        let refused = plan_load(
            &mut pool,
            &DiskTier::disabled(),
            1,
            &[vec![7, 8]],
            &[hashes],
        );

        assert!(refused.is_err());
        assert_eq!(
            pool.num_free_blocks(0),
            free_before,
            "the host hit scanned before the missing one must not stay pinned"
        );
    }

    #[test]
    fn plan_load_refuses_a_leaf_count_that_is_not_the_trees() {
        let mut pool = JengaBlockPool::new(2, vec![1], PoolOptions::default()).expect("a pool");
        assert!(
            plan_load(
                &mut pool,
                &DiskTier::disabled(),
                2,
                &[vec![7]],
                &[hashes(1)]
            )
            .is_err()
        );
    }

    #[test]
    fn plan_load_refuses_a_row_shorter_than_its_hashes() {
        let mut pool = JengaBlockPool::new(4, vec![1], PoolOptions::default()).expect("a pool");
        let hashes = hashes(2);
        for &hash in &hashes {
            commit_host_hit(&mut pool, hash);
        }
        assert!(
            plan_load(&mut pool, &DiskTier::disabled(), 1, &[vec![7]], &[hashes]).is_err(),
            "one destination block cannot hold two hashes"
        );
    }

    #[test]
    fn lookup_answers_each_leaf_from_its_own_keyspace() {
        let mut pool = JengaBlockPool::new(4, vec![1, 1], PoolOptions::default()).expect("a pool");
        let hashes = hashes(2);
        // Leaf 0 holds the first hash; leaf 1 holds neither.
        let bid = pool.alloc_block(0).expect("a block");
        pool.commit_into_prefix_cache(0, hashes[0], bid);
        pool.free_block(0, bid);

        let tiered = |leaf_hash: &LeafHash| {
            pool.lookup_hash(leaf_hash.leaf_idx, leaf_hash.block_hash)
                .is_some()
        };
        let masks: Vec<Vec<bool>> = [0, 1]
            .iter()
            .map(|&leaf_idx| {
                hashes
                    .iter()
                    .map(|&block_hash| {
                        tiered(&LeafHash {
                            leaf_idx,
                            block_hash,
                        })
                    })
                    .collect()
            })
            .collect();

        assert_eq!(masks, vec![vec![true, false], vec![false, false]]);
    }

    /// `(leaf_idx, device_bid)` of each planned copy, in submission order.
    fn planned(plan: &[super::ToOffload]) -> Vec<(usize, usize)> {
        plan.iter()
            .map(|copy| (copy.hash.leaf_idx, copy.device_bid))
            .collect()
    }

    #[test]
    fn plan_offload_takes_hashes_per_leaf_hash_major() {
        let mut pool = JengaBlockPool::new(8, vec![1, 1], PoolOptions::default()).expect("a pool");
        let run = hashes(3);
        // The second leaf publishes only at the run's last boundary, as a
        // recurrent state does.
        let (plan, already_held) = plan_offload(
            &mut pool,
            2,
            &[vec![10, 11, 12], vec![20]],
            &[run.clone(), vec![run[2]]],
        )
        .expect("a plan");
        assert_eq!(already_held, 0);

        assert_eq!(planned(&plan), [(0, 10), (1, 20), (0, 11), (0, 12)]);
        let hosts: HashSet<(usize, usize)> = plan
            .iter()
            .map(|copy| (copy.hash.leaf_idx, copy.host_bid))
            .collect();
        assert_eq!(hosts.len(), 4, "every copy lands in its own host block");
    }

    #[test]
    fn plan_offload_skips_a_hash_the_host_already_holds() {
        let mut pool = JengaBlockPool::new(8, vec![1], PoolOptions::default()).expect("a pool");
        let run = hashes(2);
        commit_host_hit(&mut pool, run[0]);

        let (plan, already_held) =
            plan_offload(&mut pool, 1, &[vec![10, 11]], &[run]).expect("a plan");

        assert_eq!(planned(&plan), [(0, 11)]);
        assert_eq!(already_held, 1);
    }

    #[test]
    fn plan_offload_refuses_a_leaf_count_that_is_not_the_trees() {
        let mut pool = JengaBlockPool::new(8, vec![1, 1], PoolOptions::default()).expect("a pool");
        let run = hashes(1);

        let error = plan_offload(&mut pool, 2, &[vec![10]], &[run])
            .err()
            .expect("a refusal");

        assert!(matches!(error, crate::error::ConnectorError::Config(_)));
        assert_eq!(pool.num_used_blocks(0), 0, "a refusal allocates nothing");
    }

    #[test]
    fn plan_offload_refuses_a_row_shorter_than_its_hashes() {
        let mut pool = JengaBlockPool::new(8, vec![1], PoolOptions::default()).expect("a pool");
        let run = hashes(2);

        let error = plan_offload(&mut pool, 1, &[vec![10]], &[run])
            .err()
            .expect("a refusal");

        assert!(matches!(error, crate::error::ConnectorError::Config(_)));
        assert_eq!(pool.num_used_blocks(0), 0, "a refusal allocates nothing");
    }
}
