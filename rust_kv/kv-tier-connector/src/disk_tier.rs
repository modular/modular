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
//! Slot-file disk tier: the last level of the KV prefix cache.
//!
//! Each leaf owns one file, `<cache_dir>/leaf-<leaf_idx>.bin`, cut into slots
//! of that leaf's block size, so the tier holds a handful of files however
//! many blocks it caches. A file per block would cost an inode per block per
//! leaf, millions per engine, and the kernel does per-inode work for each when
//! the process's cgroup exits. On Linux 6.17 that cgroup writeback inode switch
//! serializes on one lock and has stalled nodes for hours.
//!
//! A priority thread pool (reads preempt hole punches preempt writes) runs the
//! blocking file I/O off the connector's lanes; each op reads/writes the
//! *pinned host buffer* bytes directly (the caller passes the host address of
//! the block's row), so a disk read lands straight into the page an H2D will
//! copy. An in-memory LRU index bounds disk usage; it starts cold (see
//! [`DiskTier::shutdown`] for why there is no warm-start). Evicting a block
//! punches its slot out of the file, so the bytes on disk track the index
//! rather than each leaf's high-water mark.
//!
//! The pool holds a raw host pointer per op; the caller guarantees the block
//! stays pinned (host `JengaBlockPool` ref count) for the op's duration.
//!
//! Slots are reused, so a slot must not take a new block while an op on its
//! old one is outstanding, or a read would return the new block's bytes (or
//! the zeros of a hole) as the old one's. A slot's pending write and each
//! pinned read hold it, and eviction skips a held slot.
//!
//! [`DiskTier::new`] refuses a filesystem that cannot punch holes, and a
//! directory another disk tier already holds, since both tiers would write
//! the same slots.
#![allow(
    unsafe_code,
    reason = "raw host-pointer slices over caller-pinned blocks, and the fallocate call; each block is documented with SAFETY"
)]

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};

use crate::{
    error::{ConnectorError, Result, fatal},
    lane::Pollable,
    utils::LeafHash,
};

/// Priorities: lower runs first. Reads preempt hole punches preempt writes,
/// matching the Python `PriorityExecutor`'s reads, deletes and writes.
const PRIO_READ: u8 = 0;
const PRIO_PUNCH: u8 = 1;
const PRIO_WRITE: u8 = 2;

// Disk-tier I/O that bypasses the page cache. Blocks are only read back into
// the connector's own pinned host buffer, so page-cache copies are never
// reused. `use_disk_direct_io` opts each tier into O_DIRECT with no buffered
// fallback, so `probe_direct_io` rejects an unsupported filesystem or staging
// region up front.
//
// `O_DIRECT` and the alignment bound are declared here rather than taken from
// `libc`, to avoid a new dependency.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub mod page_cache {
    use std::{
        fs::{File, OpenOptions},
        os::unix::fs::OpenOptionsExt,
        path::Path,
    };

    /// `O_DIRECT` from `asm-generic/fcntl.h` (`__O_DIRECT`), identical on
    /// x86_64 and aarch64. Not to be confused with `0o200000`, which is
    /// `__O_DIRECTORY` in the generic header (but `O_DIRECT` on 32-bit ARM).
    pub(crate) const O_DIRECT: i32 = 0o40000;

    /// Opens `path` with `O_DIRECT` in `opts`, returning the errno text on
    /// failure: the caller decides whether that is fatal.
    pub(super) fn open_direct(opts: &mut OpenOptions, path: &Path) -> std::io::Result<File> {
        opts.custom_flags(O_DIRECT).open(path)
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
pub mod page_cache {
    use std::{
        fs::{File, OpenOptions},
        path::Path,
    };

    pub(crate) const O_DIRECT_UNAVAILABLE: bool = true;

    pub(super) fn open_direct(_opts: &mut OpenOptions, path: &Path) -> std::io::Result<File> {
        Err(std::io::Error::other(format!(
            "the disk tier's direct I/O is only supported on 64-bit Linux, and \
             this build targets another platform (wanted to open {})",
            path.display()
        )))
    }
}

/// O_DIRECT requires the buffer address, length and file offset to be aligned
/// to the device's logical block size; 4 KiB covers the devices MAX runs on.
/// Every chunked transfer stays inside this bound (see `transfer_direct`).
pub const DIRECT_IO_ALIGN: usize = 4096;

/// Whether a host address and byte count could take the O_DIRECT path.
pub fn is_direct_io_aligned(ptr: usize, len: usize) -> bool {
    ptr.is_multiple_of(DIRECT_IO_ALIGN) && len.is_multiple_of(DIRECT_IO_ALIGN)
}

/// Checks that every leaf's block size can take direct I/O: a positive
/// multiple of [`DIRECT_IO_ALIGN`], since a block's length is its leaf's row.
///
/// # Errors
///
/// Returns a config error naming the first offending leaf.
pub fn check_direct_io_leaf_sizes(bytes_per_leaf: &[usize]) -> Result<()> {
    if let Some((idx, bytes)) = bytes_per_leaf
        .iter()
        .enumerate()
        .find(|(_, bytes)| **bytes == 0 || **bytes % DIRECT_IO_ALIGN != 0)
    {
        return Err(ConnectorError::Config(format!(
            "use_disk_direct_io requires every leaf's block size to be a \
             positive multiple of {DIRECT_IO_ALIGN} bytes; leaf {idx} is \
             {bytes} bytes"
        )));
    }
    Ok(())
}

/// A pinned host block address + byte length. `Send` because the address is
/// stable pinned memory the caller keeps alive (ref-counted) for the op.
#[derive(Clone, Copy)]
struct HostBlock {
    ptr: usize,
    len: usize,
}

// SAFETY: the address refers to pinned host memory owned by MAX and kept alive
// for the op's duration; only one op touches a given block at a time.
unsafe impl Send for HostBlock {}

/// Completion handle for one disk op; a lane polls `is_ready`.
#[derive(Clone)]
pub struct DiskOp {
    done: Arc<AtomicBool>,
}

impl DiskOp {
    fn new() -> Self {
        DiskOp {
            done: Arc::new(AtomicBool::new(false)),
        }
    }
    fn complete(&self) {
        self.done.store(true, Ordering::Release);
    }
}

impl Pollable for DiskOp {
    fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

struct Job {
    task: Box<dyn FnOnce() + Send>,
}

struct PoolInner {
    // Jobs ordered by (priority, seq); pop-min runs the highest priority oldest.
    queue: BTreeMap<(u8, u64), Job>,
    shutting_down: bool,
}

/// A fixed-size priority thread pool.
struct PriorityPool {
    inner: Mutex<PoolInner>,
    cv: Condvar,
    seq: AtomicU64,
    inflight: AtomicUsize,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl PriorityPool {
    fn new(num_workers: usize) -> Arc<Self> {
        let pool = Arc::new(PriorityPool {
            inner: Mutex::new(PoolInner {
                queue: BTreeMap::new(),
                shutting_down: false,
            }),
            cv: Condvar::new(),
            seq: AtomicU64::new(0),
            inflight: AtomicUsize::new(0),
            workers: Mutex::new(Vec::new()),
        });
        let mut handles = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let p = Arc::clone(&pool);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("kv-disk-{i}"))
                    .spawn(move || p.worker())
                    .expect("spawn disk worker"),
            );
        }
        *pool.workers.lock().unwrap() = handles;
        pool
    }

    fn submit(&self, priority: u8, task: Box<dyn FnOnce() + Send>) {
        let seq = self.seq.fetch_add(1, Ordering::AcqRel);
        self.inflight.fetch_add(1, Ordering::AcqRel);
        self.inner
            .lock()
            .unwrap()
            .queue
            .insert((priority, seq), Job { task });
        self.cv.notify_one();
    }

    fn worker(self: Arc<Self>) {
        loop {
            let job = {
                let mut inner = self.inner.lock().unwrap();
                loop {
                    if let Some((&key, _)) = inner.queue.iter().next() {
                        break Some(inner.queue.remove(&key).unwrap());
                    }
                    if inner.shutting_down {
                        break None;
                    }
                    inner = self.cv.wait(inner).unwrap();
                }
            };
            match job {
                Some(job) => {
                    (job.task)();
                    self.inflight.fetch_sub(1, Ordering::AcqRel);
                    self.cv.notify_all(); // wake wait_until_idle
                }
                None => return,
            }
        }
    }

    fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    fn wait_until_idle(&self) {
        let mut inner = self.inner.lock().unwrap();
        while !inner.queue.is_empty() || self.inflight.load(Ordering::Acquire) > 0 {
            inner = self.cv.wait(inner).unwrap();
        }
    }

    fn shutdown(&self) {
        self.inner.lock().unwrap().shutting_down = true;
        self.cv.notify_all();
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.workers.lock().unwrap());
        for h in handles {
            drop(h.join());
        }
    }
}

impl Drop for PriorityPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Releases the extent behind `[offset, offset + len)` of `file`, keeping its
/// length. Declared here rather than taken from `libc`, like `O_DIRECT`.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn punch_hole(file: &File, offset: u64, len: u64) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    // From `linux/falloc.h`. A hole punch must keep the file size.
    const FALLOC_FL_KEEP_SIZE: i32 = 0x01;
    const FALLOC_FL_PUNCH_HOLE: i32 = 0x02;
    unsafe extern "C" {
        fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
    }
    let (Ok(offset), Ok(len)) = (i64::try_from(offset), i64::try_from(len)) else {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    };
    // SAFETY: `fallocate` only reads its integer arguments, and the descriptor
    // is borrowed from `file`, which outlives the call.
    let rc = unsafe {
        fallocate(
            file.as_raw_fd(),
            FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE,
            offset,
            len,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn punch_hole(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// A block's place on disk, a slot in its leaf's file.
///
/// [`DiskTier::pin_reads`] hands these out as read pins, and the slot stays
/// reserved for the block until [`DiskTier::read_block_async`] consumes the
/// pin, so every pin must be read exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskSlot {
    leaf_idx: usize,
    slot: usize,
}

/// One leaf's slot allocator.
#[derive(Default)]
struct LeafSlots {
    // Punched-out slots, ready for a write.
    free: Vec<usize>,
    // One past the highest slot ever handed out.
    high_water: usize,
    // Outstanding holds per slot, from its pending write and each pinned read.
    // Eviction skips a held slot.
    holds: HashMap<usize, usize>,
}

impl LeafSlots {
    fn alloc(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            self.high_water += 1;
            self.high_water - 1
        })
    }

    fn hold(&mut self, slot: usize) {
        *self.holds.entry(slot).or_insert(0) += 1;
    }

    fn release(&mut self, slot: usize) {
        let Some(holds) = self.holds.get_mut(&slot) else {
            fatal(format!("disk slot {slot} released more often than held"));
        };
        *holds -= 1;
        if *holds == 0 {
            self.holds.remove(&slot);
        }
    }

    fn is_held(&self, slot: usize) -> bool {
        self.holds.contains_key(&slot)
    }
}

#[derive(Clone, Copy)]
struct Entry {
    seq: u64,
    slot: usize,
}

#[derive(Default)]
struct DiskIndex {
    // LRU: seq -> hash and hash -> (seq, slot). Lowest seq is the eviction
    // victim.
    lru_by_seq: BTreeMap<u64, LeafHash>,
    entries: HashMap<LeafHash, Entry>,
    pending_writes: HashSet<LeafHash>,
    next_seq: u64,

    used_bytes: usize,
    leaves: Vec<LeafSlots>,
}

impl DiskIndex {
    fn new(num_leaves: usize) -> Self {
        DiskIndex {
            leaves: (0..num_leaves).map(|_| LeafSlots::default()).collect(),
            ..DiskIndex::default()
        }
    }
    fn touch(&mut self, hash: LeafHash) {
        if let Some(entry) = self.entries.get_mut(&hash) {
            self.lru_by_seq.remove(&entry.seq);
            entry.seq = self.next_seq;
            self.lru_by_seq.insert(self.next_seq, hash);
            self.next_seq += 1;
        }
    }
    fn insert(&mut self, hash: LeafHash, slot: usize) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.lru_by_seq.insert(seq, hash);
        self.entries.insert(hash, Entry { seq, slot });
    }
    fn contains(&self, hash: &LeafHash) -> bool {
        self.entries.contains_key(hash)
    }
    /// Removes the least recently used block whose slot nothing holds.
    fn pop_lru_unheld(&mut self) -> Option<(LeafHash, usize)> {
        let (&seq, &hash) = self
            .lru_by_seq
            .iter()
            .find(|(_, hash)| !self.leaves[hash.leaf_idx].is_held(self.entries[*hash].slot))?;
        self.lru_by_seq.remove(&seq);
        let entry = self.entries.remove(&hash).unwrap();
        Some((hash, entry.slot))
    }
}

struct LeafFile {
    file: File,
    path: PathBuf,
}

/// What the disk tier shares with its workers.
struct Store {
    /// Emptied by [`DiskTier::shutdown`]. Closing the files is what lets the
    /// caller's removal of the directory give their space back.
    files: RwLock<Vec<LeafFile>>,
    bytes_per_leaf: Vec<usize>,
    /// Whether reads and writes go through `O_DIRECT` (no page cache). See
    /// [`DiskTier::new`]; when false every op takes the buffered path.
    direct_io: bool,
    index: Mutex<DiskIndex>,
}

/// Set once a hole punch fails, so the warning is logged once per process.
static PUNCH_FAILED: AtomicBool = AtomicBool::new(false);

impl Store {
    fn offset(&self, slot: DiskSlot) -> u64 {
        (slot.slot * self.bytes_per_leaf[slot.leaf_idx]) as u64
    }

    fn describe(leaf: &LeafFile, slot: DiskSlot) -> String {
        format!("{} slot {}", leaf.path.display(), slot.slot)
    }

    fn unhold(&self, slot: DiskSlot) {
        self.index.lock().unwrap().leaves[slot.leaf_idx].release(slot.slot);
    }

    /// Punches out an evicted slot, then frees it. Punching before freeing
    /// means no write can land in the slot ahead of the punch and be zeroed
    /// by it.
    fn release_slot(&self, slot: DiskSlot) {
        let len = self.bytes_per_leaf[slot.leaf_idx];
        {
            let files = self.files.read().unwrap();
            let leaf = &files[slot.leaf_idx];
            if let Err(e) = punch_hole(&leaf.file, self.offset(slot), len as u64)
                && !PUNCH_FAILED.swap(true, Ordering::Relaxed)
            {
                // `DiskTier::new` proved punching works here, so this is
                // unexpected. Not fatal. The slot is still reused, so blocks
                // stay correct, but its leaf's file keeps the space.
                tracing::warn!(
                    "kv disk tier failed to punch a hole in {}: {e}",
                    leaf.path.display()
                );
            }
        }
        self.index.lock().unwrap().leaves[slot.leaf_idx]
            .free
            .push(slot.slot);
    }

    fn read_slot(&self, slot: DiskSlot, dst: HostBlock) {
        // Fail-fast: this only runs for blocks the index reports present, whose
        // slot was fully written (writes fail-fast too), so any error here is
        // genuine corruption. Silently returning would promote uninitialized
        // host bytes.
        // SAFETY: dst.ptr is a pinned host block of dst.len bytes, kept alive
        // by the caller's ref count for this op; we are the only writer.
        let buf = unsafe { std::slice::from_raw_parts_mut(dst.ptr as *mut u8, dst.len) };
        let files = self.files.read().unwrap();
        let leaf = &files[slot.leaf_idx];
        let offset = self.offset(slot);
        if self.direct_io {
            debug_assert!(
                is_direct_io_aligned(dst.ptr, dst.len),
                "the probe verified every block address and length is direct-I/O aligned"
            );
            transfer_direct(
                &format!("read {}", Self::describe(leaf, slot)),
                buf.len(),
                |off| leaf.file.read_at(&mut buf[off..], offset + off as u64),
            );
            return;
        }
        leaf.file
            .read_exact_at(buf, offset)
            .unwrap_or_else(|e| fatal(format!("disk read {}: {e}", Self::describe(leaf, slot))));
    }

    fn write_slot(&self, slot: DiskSlot, src: HostBlock) {
        // SAFETY: src.ptr is a pinned host block of src.len bytes, kept alive
        // by the caller's ref count for this op; read-only here.
        let buf = unsafe { std::slice::from_raw_parts(src.ptr as *const u8, src.len) };
        let files = self.files.read().unwrap();
        let leaf = &files[slot.leaf_idx];
        let offset = self.offset(slot);
        // Fail-fast on every step: a swallowed write leaves the index
        // advertising a block whose slot holds stale bytes or a hole, which a
        // later disk read would then promote as garbage. No reader sees a
        // partial slot, since the block is readable only once this returns.
        if self.direct_io {
            debug_assert!(
                is_direct_io_aligned(src.ptr, src.len),
                "the probe verified every block address and length is direct-I/O aligned"
            );
            transfer_direct(
                &format!("write {}", Self::describe(leaf, slot)),
                buf.len(),
                |off| leaf.file.write_at(&buf[off..], offset + off as u64),
            );
        } else {
            leaf.file.write_all_at(buf, offset).unwrap_or_else(|e| {
                fatal(format!("disk write {}: {e}", Self::describe(leaf, slot)))
            });
        }
        leaf.file.sync_data().unwrap_or_else(|e| {
            fatal(format!(
                "disk write sync {}: {e}",
                Self::describe(leaf, slot)
            ))
        });
    }
}

fn open_leaf_file(path: PathBuf, direct_io: bool) -> Result<LeafFile> {
    let mut opts = OpenOptions::new();
    // The index starts cold, so nothing a previous run left in the file is
    // reachable.
    opts.read(true).write(true).create(true).truncate(true);
    let file = if direct_io {
        page_cache::open_direct(&mut opts, &path)
    } else {
        opts.open(&path)
    }
    .map_err(|e| ConnectorError::Config(format!("open disk tier file {}: {e}", path.display())))?;
    Ok(LeafFile { file, path })
}

/// Takes the exclusive lock on `cache_dir` that keeps a second disk tier out.
///
/// Two tiers in one directory would truncate and overwrite each other's leaf
/// files, then serve each other's bytes as their own, so the second one is
/// refused before it opens anything.
fn lock_cache_dir(cache_dir: &Path) -> Result<File> {
    let path = cache_dir.join(".disk-tier.lock");
    let lock = File::create(&path)
        .map_err(|e| ConnectorError::Config(format!("create {}: {e}", path.display())))?;
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(ConnectorError::Config(format!(
            "another disk tier is already using {}; give each engine its own \
             disk_offload_dir",
            cache_dir.display()
        ))),
        Err(TryLockError::Error(e)) => Err(ConnectorError::Config(format!(
            "lock {}: {e}",
            path.display()
        ))),
    }
}

/// The disk tier. Optional: see [`DiskTier::disabled`].
pub struct DiskTier {
    /// False for [`DiskTier::disabled`], which declines every op.
    pub enabled: bool,
    max_bytes: usize,
    pool: Arc<PriorityPool>,
    store: Arc<Store>,
    /// Held from [`DiskTier::new`] until [`DiskTier::shutdown`]. See
    /// [`lock_cache_dir`].
    dir_lock: Mutex<Option<File>>,
}

impl DiskTier {
    /// Opens the disk tier at `cache_dir`, one file per leaf, with a cold
    /// (empty) index -- see [`DiskTier::shutdown`] for why a prior run's
    /// blocks are never reloaded.
    ///
    /// `direct_io` opts block reads and writes into `O_DIRECT`, keeping the
    /// tier's files out of the page cache; there is no buffered fallback, so
    /// the caller must first prove the combination works with
    /// [`probe_direct_io`] (the connector constructor does), or every op
    /// errors mid-serving.
    ///
    /// # Errors
    ///
    /// Returns a config error if another disk tier holds `cache_dir`, if its
    /// filesystem cannot punch holes, or if a leaf's size cannot take direct
    /// I/O.
    pub fn new(
        cache_dir: &str,
        bytes_per_leaf: &[usize],
        max_bytes: usize,
        num_workers: usize,
        direct_io: bool,
    ) -> Result<Self> {
        let cache_dir = Path::new(cache_dir);
        std::fs::create_dir_all(cache_dir)
            .map_err(|e| ConnectorError::Config(format!("create cache dir: {e}")))?;
        if direct_io {
            // O_DIRECT requires the buffer address, length and file offset to
            // be aligned to the device's logical block size. A block's length
            // is its leaf's row, and its offset a whole number of rows. The
            // staging region's base alignment is checked by the caller's
            // probe, which sees the actual addresses.
            check_direct_io_leaf_sizes(bytes_per_leaf)?;
        }
        let dir_lock = lock_cache_dir(cache_dir)?;
        let files = (0..bytes_per_leaf.len())
            .map(|leaf_idx| {
                open_leaf_file(cache_dir.join(format!("leaf-{leaf_idx}.bin")), direct_io)
            })
            .collect::<Result<Vec<_>>>()?;
        // Eviction gives a slot's space back with a hole punch. Without one,
        // every leaf's file keeps its peak size and the tier outgrows the cap.
        if let Some(leaf) = files.first() {
            punch_hole(&leaf.file, 0, DIRECT_IO_ALIGN as u64).map_err(|e| {
                ConnectorError::Config(format!(
                    "the disk tier needs hole punching (fallocate PUNCH_HOLE), \
                     which the filesystem under {} does not support: {e}",
                    cache_dir.display()
                ))
            })?;
        }

        Ok(DiskTier {
            enabled: true,
            max_bytes,
            pool: PriorityPool::new(num_workers.max(1)),
            store: Arc::new(Store {
                files: RwLock::new(files),
                bytes_per_leaf: bytes_per_leaf.to_vec(),
                direct_io,
                index: Mutex::new(DiskIndex::new(bytes_per_leaf.len())),
            }),
            dir_lock: Mutex::new(Some(dir_lock)),
        })
    }

    /// The disabled tier, for a connector whose last level is the host tier.
    ///
    /// Every query answers empty and every write is declined, so the tiered
    /// connector's disk call sites degrade to host-only unchanged. It owns no
    /// workers, so a submitted op would never complete -- hence nothing is ever
    /// submitted.
    #[must_use]
    pub fn disabled() -> Self {
        DiskTier {
            enabled: false,
            max_bytes: 0,
            pool: PriorityPool::new(0),
            store: Arc::new(Store {
                files: RwLock::new(Vec::new()),
                bytes_per_leaf: Vec::new(),
                direct_io: false,
                index: Mutex::new(DiskIndex::default()),
            }),
            dir_lock: Mutex::new(None),
        }
    }

    /// Total number of bytes the disk tier can hold.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Number of bytes currently indexed (saved or pending write).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.store.index.lock().unwrap().used_bytes
    }

    /// Number of disk ops queued or running.
    #[must_use]
    pub fn inflight_disk_ops(&self) -> usize {
        self.pool.inflight()
    }

    /// Whether `hash` is saved and readable (excludes in-flight writes).
    #[must_use]
    pub fn contains(&self, hash: &LeafHash) -> bool {
        if !self.enabled {
            return false;
        }
        let idx = self.store.index.lock().unwrap();
        idx.contains(hash) && !idx.pending_writes.contains(hash)
    }

    /// Moves each of `hashes` that is saved to the most-recently-used end of
    /// the disk LRU, in order.
    pub fn touch(&self, hashes: impl IntoIterator<Item = LeafHash>) {
        if !self.enabled {
            return;
        }
        let mut idx = self.store.index.lock().unwrap();
        for hash in hashes {
            idx.touch(hash);
        }
    }

    /// Clears every block from disk: empties the in-memory index and truncates
    /// every leaf's file, which releases all of its extents at once.
    ///
    /// The caller must have drained in-flight disk ops first (see
    /// `TieredConnector::reset_prefix_cache`) so no worker races the truncate
    /// and no read pin is outstanding.
    pub fn reset(&self) {
        let files = self.store.files.read().unwrap();
        let mut idx = self.store.index.lock().unwrap();
        *idx = DiskIndex::new(self.store.bytes_per_leaf.len());
        for leaf in files.iter() {
            if let Err(e) = leaf.file.set_len(0) {
                tracing::warn!("kv disk tier reset truncate {}: {e}", leaf.path.display());
            }
        }
    }

    /// Pins every one of `hashes` for reading and LRU-touches it, or pins
    /// none if any is no longer saved. Between a lookup and the load it plans, an
    /// offload's write-through can evict on the disk-write lane, and a pin is
    /// what keeps it from doing so again before the read lands.
    ///
    /// Each returned pin must be passed to [`Self::read_block_async`] or
    /// [`Self::release_pins`] exactly once, or its slot is never reused.
    #[must_use]
    pub fn pin_reads(&self, hashes: &[LeafHash]) -> Option<Vec<DiskSlot>> {
        if hashes.is_empty() {
            return Some(Vec::new());
        }
        if !self.enabled {
            return None;
        }
        let mut idx = self.store.index.lock().unwrap();
        if hashes
            .iter()
            .any(|hash| !idx.contains(hash) || idx.pending_writes.contains(hash))
        {
            return None;
        }
        Some(
            hashes
                .iter()
                .map(|&hash| {
                    idx.touch(hash);
                    let slot = idx.entries[&hash].slot;
                    idx.leaves[hash.leaf_idx].hold(slot);
                    DiskSlot {
                        leaf_idx: hash.leaf_idx,
                        slot,
                    }
                })
                .collect(),
        )
    }

    /// Drops pins a load took but will not read, for a load given up after
    /// pinning.
    pub fn release_pins(&self, pins: &[DiskSlot]) {
        for &pin in pins {
            self.store.unhold(pin);
        }
    }

    /// Reads the block `pin` names into the pinned host block at `dest_ptr`
    /// at READ priority, consuming the pin. Returns a pollable op.
    pub fn read_block_async(&self, pin: DiskSlot, dest_ptr: usize) -> DiskOp {
        if !self.enabled {
            // Unreachable: `pin_reads` pins nothing, so nothing is ever read.
            fatal("disk read on a disabled disk tier");
        }
        let op = DiskOp::new();
        let store = Arc::clone(&self.store);
        let dst = HostBlock {
            ptr: dest_ptr,
            len: store.bytes_per_leaf[pin.leaf_idx],
        };
        let op2 = op.clone();
        self.pool.submit(
            PRIO_READ,
            Box::new(move || {
                let _r =
                    crate::trace::Range::new("kv-disk read_block", crate::trace::color::DISK_READ);
                store.read_slot(pin, dst);
                op2.complete();
                store.unhold(pin);
            }),
        );
        op
    }

    /// Writes the pinned host block at `src_ptr` into a slot for `hash`,
    /// evicting the least recently used unheld blocks first if at capacity.
    /// Returns `None` if already saved, if a write is already pending, or if
    /// every block it could evict is held by a read or write in flight;
    /// otherwise a pollable op at WRITE priority.
    pub fn write_block_async(&self, hash: LeafHash, src_ptr: usize) -> Option<DiskOp> {
        if !self.enabled {
            return None;
        }
        let bytes_needed = self.store.bytes_per_leaf[hash.leaf_idx];

        let mut releases: Vec<DiskSlot> = Vec::new();
        let slot = {
            let mut index = self.store.index.lock().unwrap();
            if index.contains(&hash) || index.pending_writes.contains(&hash) {
                return None;
            }
            let mut room = true;
            while self.max_bytes - index.used_bytes < bytes_needed {
                let Some((victim, victim_slot)) = index.pop_lru_unheld() else {
                    if index.used_bytes == 0 {
                        fatal(format!(
                            "the disk tier's {} bytes cannot hold a {bytes_needed}-byte block",
                            self.max_bytes
                        ));
                    }
                    // Every saved block is in use by a read or write in
                    // flight. Dropping this write-through costs a future disk
                    // hit, not correctness.
                    room = false;
                    break;
                };
                index.used_bytes -= self.store.bytes_per_leaf[victim.leaf_idx];
                releases.push(DiskSlot {
                    leaf_idx: victim.leaf_idx,
                    slot: victim_slot,
                });
            }
            room.then(|| {
                let leaf = &mut index.leaves[hash.leaf_idx];
                let slot = leaf.alloc();
                leaf.hold(slot);
                index.pending_writes.insert(hash);
                index.insert(hash, slot);
                index.used_bytes += bytes_needed;
                DiskSlot {
                    leaf_idx: hash.leaf_idx,
                    slot,
                }
            })
        };
        for victim in releases {
            let store = Arc::clone(&self.store);
            self.pool
                .submit(PRIO_PUNCH, Box::new(move || store.release_slot(victim)));
        }
        let slot = slot?;

        let op = DiskOp::new();
        let src = HostBlock {
            ptr: src_ptr,
            len: bytes_needed,
        };
        let store = Arc::clone(&self.store);
        let op2 = op.clone();
        self.pool.submit(
            PRIO_WRITE,
            Box::new(move || {
                let _r = crate::trace::Range::new(
                    "kv-disk write_block",
                    crate::trace::color::DISK_WRITE,
                );
                store.write_slot(slot, src);
                store.index.lock().unwrap().pending_writes.remove(&hash);
                op2.complete();
                store.unhold(slot);
            }),
        );
        Some(op)
    }

    /// Blocks until every queued/running disk op completes.
    pub fn wait_for_writes(&self) {
        self.pool.wait_until_idle();
    }

    /// Drains and stops the disk workers.
    ///
    /// `cache_dir` is left in place: the disk tier is ephemeral scratch (there
    /// is no warm-start — blocks are never reloaded from a directory a previous
    /// process left), but its lifetime belongs to whoever created it. The
    /// caller removes it once this returns, which is the point after which no
    /// worker can still be writing into it. See the Python
    /// `connectors/_offload_dir.py`, which owns that directory end to end so a
    /// directory abandoned by a SIGKILL is still reclaimed on the next start.
    ///
    /// Closes the leaf files and releases the directory lock, so removing the
    /// directory frees its space even while this object lives on.
    pub fn shutdown(&self) {
        self.pool.wait_until_idle();
        self.pool.shutdown();
        self.store.files.write().unwrap().clear();
        self.dir_lock.lock().unwrap().take();
    }
}

/// Drives `op(offset)` until `len` bytes have moved, like `read_exact` /
/// `write_all`, in chunks that keep the file offset `DIRECT_IO_ALIGN`-aligned
/// so the next `op` is a legal O_DIRECT transfer. Any error or zero-length
/// transfer is fatal: there is no buffered fallback, by design.
fn transfer_direct(ctx: &str, len: usize, mut op: impl FnMut(usize) -> io::Result<usize>) {
    // Chunked like the buffered path's read_exact/write_all loops: Linux
    // truncates a single read(2)/write(2) past about 2 GiB.
    let mut off = 0;
    while off < len {
        match op(off) {
            Ok(0) => fatal(format!("disk {ctx} ended after {off} of its {len} bytes")),
            Ok(n) => off += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => fatal(format!("disk {ctx}: {e}")),
        }
        debug_assert!(
            off == len || off.is_multiple_of(DIRECT_IO_ALIGN),
            "chunking keeps the O_DIRECT offset aligned; a short transfer that \
             broke alignment would make the next op EINVAL"
        );
    }
}

/// Proves the disk tier's direct I/O works end to end before the connector
/// starts serving, returning `Err` naming what failed.
///
/// Runs one real O_DIRECT write + fsync + rename + read through `probe_buf`
/// (a `DIRECT_IO_ALIGN`-byte window into the *actual* staging memory, so a
/// staging region that cannot take direct I/O -- driver-allocated GTT memory
/// on AMD rejects it with EFAULT -- is caught here, at startup, rather than
/// one block into serving) into `cache_dir`. With no buffered fallback, this
/// check is what keeps `use_disk_direct_io` a safe thing to ask for.
pub fn probe_direct_io(
    cache_dir: &Path,
    probe_buf: &mut [u8],
) -> std::result::Result<(), ConnectorError> {
    if probe_buf.len() != DIRECT_IO_ALIGN {
        return Err(ConnectorError::Config(format!(
            "the direct-I/O probe buffer must be exactly {DIRECT_IO_ALIGN} bytes, got {}",
            probe_buf.len()
        )));
    }
    // SAFETY: `probe_buf` is a live, exclusively borrowed slice of exactly
    // `DIRECT_IO_ALIGN` bytes, which is all `probe_direct_io_at` touches.
    unsafe { probe_direct_io_at(probe_buf.as_mut_ptr() as usize, cache_dir) }
}

/// [`probe_direct_io`] over a raw host address, for bindings that carry the
/// staging region as addresses (the pyo3 connector).
///
/// # Safety
///
/// `host_addr` must name `DIRECT_IO_ALIGN` bytes that are live, readable and
/// writable, and that nothing else accesses until this returns. The caller
/// owns that memory; this only borrows it for the call. Alignment is checked
/// here and reported as a config error.
pub unsafe fn probe_direct_io_at(
    host_addr: usize,
    cache_dir: &Path,
) -> std::result::Result<(), ConnectorError> {
    // SAFETY: the caller upholds this function's `# Safety` contract: a live,
    // exclusively held span of `DIRECT_IO_ALIGN` bytes at `host_addr`.
    let probe_buf =
        unsafe { std::slice::from_raw_parts_mut(host_addr as *mut u8, DIRECT_IO_ALIGN) };
    if !is_direct_io_aligned(probe_buf.as_ptr() as usize, probe_buf.len()) {
        return Err(ConnectorError::Config(format!(
            "the staging region's probe address {:#x} is not {DIRECT_IO_ALIGN}-byte \
             aligned, so use_disk_direct_io cannot serve any block",
            probe_buf.as_ptr() as usize
        )));
    }
    let probe_path = cache_dir.join(".direct-io-probe");
    let pattern: u8 = 0xA5;
    probe_buf.fill(pattern);
    // A plain (buffered) write of a different pattern first, so a direct read
    // back of `pattern` proves the read came from the file and not a stale
    // buffer.
    std::fs::write(&probe_path, vec![!pattern; DIRECT_IO_ALIGN])
        .map_err(|e| ConnectorError::Config(format!("direct-I/O probe setup: {e}")))?;
    // Errors here -- not `fatal`: this runs at construction, where the
    // connector reports config errors to its caller instead of killing the
    // process.
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    let mut f = page_cache::open_direct(&mut opts, &probe_path).map_err(|e| {
        ConnectorError::Config(format!(
            "use_disk_direct_io: the disk cache directory {} does not support \
             direct I/O (open with O_DIRECT: {e}). tmpfs before Linux 6.6 and some \
             network/overlay filesystems reject it",
            cache_dir.display()
        ))
    })?;
    f.write_all(probe_buf).map_err(|e| {
        ConnectorError::Config(format!(
            "use_disk_direct_io: a direct write into the disk cache directory {} \
             failed: {e}. If the host staging region is driver-allocated \
             (host_staging_alloc='driver'), its memory rejects O_DIRECT with \
             EFAULT on AMD; use the registered staging region instead",
            cache_dir.display()
        ))
    })?;
    f.sync_all()
        .map_err(|e| ConnectorError::Config(format!("direct-I/O probe sync: {e}")))?;
    // Overwrite the buffer so a successful read must come from the file.
    probe_buf.fill(0);
    let mut opts = OpenOptions::new();
    opts.read(true);
    let mut f = page_cache::open_direct(&mut opts, &probe_path).map_err(|e| {
        ConnectorError::Config(format!(
            "use_disk_direct_io: opening a direct read from the disk cache \
             directory {} failed: {e}",
            cache_dir.display()
        ))
    })?;
    f.read_exact(probe_buf)
        .map_err(|e| ConnectorError::Config(format!("direct-I/O probe read: {e}")))?;
    if probe_buf.iter().any(|&b| b != pattern) {
        return Err(ConnectorError::Config(
            "use_disk_direct_io: a direct read returned bytes other than the \
             ones directly written"
                .into(),
        ));
    }
    drop(std::fs::remove_file(&probe_path));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::BlockHash;

    fn simple_hash(block_hash: u128) -> LeafHash {
        LeafHash {
            leaf_idx: 0,
            block_hash: BlockHash(block_hash, block_hash),
        }
    }

    fn tmpdir(tag: &str) -> String {
        let p = std::env::temp_dir().join(format!(
            "kvtier-disk-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = tmpdir("rt");
        let tier = DiskTier::new(&dir, &[16], 1 << 30, 4, false).unwrap();
        let hash = simple_hash(42);
        let src: Vec<u8> = (0..16u8).collect();
        assert!(!tier.contains(&hash));
        let op = tier.write_block_async(hash, src.as_ptr() as usize).unwrap();
        tier.wait_for_writes();
        assert!(op.is_ready());
        assert!(tier.contains(&hash));

        let mut dst = vec![0u8; 16];
        let rop = tier.read_block_async(pin(&tier, hash), dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        assert!(rop.is_ready());
        assert_eq!(dst, src);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    fn pin(tier: &DiskTier, hash: LeafHash) -> DiskSlot {
        tier.pin_reads(&[hash]).expect("the block is saved")[0]
    }

    fn read_back(tier: &DiskTier, hash: LeafHash, len: usize) -> Vec<u8> {
        let mut dst = vec![0u8; len];
        tier.read_block_async(pin(tier, hash), dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        dst
    }

    fn file_names(dir: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn every_block_of_a_leaf_shares_one_file() {
        let dir = tmpdir("one-file");
        let tier = DiskTier::new(&dir, &[8, 16], 1 << 30, 4, false).unwrap();
        let srcs: Vec<Vec<u8>> = (0..64u8).map(|i| vec![i; 16]).collect();
        for (i, src) in srcs.iter().enumerate() {
            let hash = LeafHash {
                leaf_idx: i % 2,
                block_hash: BlockHash(i as u128, 0),
            };
            tier.write_block_async(hash, src.as_ptr() as usize).unwrap();
        }
        tier.wait_for_writes();
        assert_eq!(
            file_names(&dir),
            [".disk-tier.lock", "leaf-0.bin", "leaf-1.bin"]
        );
        for (i, src) in srcs.iter().enumerate() {
            let leaf_idx = i % 2;
            let hash = LeafHash {
                leaf_idx,
                block_hash: BlockHash(i as u128, 0),
            };
            let len = [8, 16][leaf_idx];
            assert_eq!(read_back(&tier, hash, len), src[..len]);
        }
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn evicted_slots_are_reused() {
        let dir = tmpdir("reuse");
        // Capacity for 2 blocks of 8 bytes, one worker so punches and writes
        // run in submission-priority order.
        let tier = DiskTier::new(&dir, &[8], 16, 1, false).unwrap();
        let srcs: Vec<[u8; 8]> = (0..32u8).map(|i| [i; 8]).collect();
        for (i, src) in srcs.iter().enumerate() {
            tier.write_block_async(simple_hash(i as u128), src.as_ptr() as usize)
                .unwrap();
            tier.wait_for_writes();
        }
        // Two live blocks, plus the slot a victim's punch held back from the
        // write that evicted it.
        let len = std::fs::metadata(Path::new(&dir).join("leaf-0.bin"))
            .unwrap()
            .len();
        assert!(len <= 3 * 8, "the file grew to {len} bytes");
        assert!(!tier.contains(&simple_hash(29)));
        for i in [30, 31] {
            assert_eq!(read_back(&tier, simple_hash(i), 8), srcs[i as usize]);
        }
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn a_touched_block_is_evicted_after_an_untouched_one() {
        let dir = tmpdir("touch");
        // Capacity for 2 blocks of 8 bytes.
        let tier = DiskTier::new(&dir, &[8], 16, 1, false).unwrap();
        let srcs: Vec<[u8; 8]> = (0..3u8).map(|i| [i; 8]).collect();
        for i in 0..2 {
            tier.write_block_async(simple_hash(i), srcs[i as usize].as_ptr() as usize)
                .unwrap();
            tier.wait_for_writes();
        }

        tier.touch([simple_hash(0)]);
        tier.write_block_async(simple_hash(2), srcs[2].as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();

        assert!(tier.contains(&simple_hash(0)));
        assert!(!tier.contains(&simple_hash(1)));
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn a_pinned_block_is_never_evicted() {
        let dir = tmpdir("pinned");
        // Room for two blocks, so every write past the second evicts one.
        let tier = DiskTier::new(&dir, &[8], 16, 1, false).unwrap();
        let srcs: Vec<[u8; 8]> = (0..8u8).map(|i| [i + 1; 8]).collect();
        tier.write_block_async(simple_hash(0), srcs[0].as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();
        let held = pin(&tier, simple_hash(0));
        // Each later write evicts the unpinned block before it, never the
        // pinned one, so its slot is never reused under the read.
        for (i, src) in srcs.iter().enumerate().skip(1) {
            tier.write_block_async(simple_hash(i as u128), src.as_ptr() as usize)
                .unwrap();
            tier.wait_for_writes();
            assert!(tier.contains(&simple_hash(0)));
        }
        let mut dst = [0u8; 8];
        tier.read_block_async(held, dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        assert_eq!(dst, srcs[0]);
        assert_eq!(read_back(&tier, simple_hash(7), 8), srcs[7]);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn a_write_is_declined_while_every_block_is_held() {
        let dir = tmpdir("declined");
        let tier = DiskTier::new(&dir, &[8], 8, 1, false).unwrap();
        let src = [4u8; 8];
        tier.write_block_async(simple_hash(0), src.as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();
        let held = pin(&tier, simple_hash(0));
        assert!(
            tier.write_block_async(simple_hash(1), src.as_ptr() as usize)
                .is_none()
        );
        assert!(tier.contains(&simple_hash(0)));
        let mut dst = [0u8; 8];
        tier.read_block_async(held, dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        // The read released the only block, so the next write evicts it.
        tier.write_block_async(simple_hash(1), src.as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();
        assert!(!tier.contains(&simple_hash(0)));
        assert!(tier.contains(&simple_hash(1)));
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn release_pins_lets_a_pinned_block_be_evicted() {
        let dir = tmpdir("release-pins");
        let tier = DiskTier::new(&dir, &[8], 8, 1, false).unwrap();
        let src = [5u8; 8];
        tier.write_block_async(simple_hash(0), src.as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();
        let pins = tier.pin_reads(&[simple_hash(0)]).unwrap();
        tier.release_pins(&pins);
        tier.write_block_async(simple_hash(1), src.as_ptr() as usize)
            .unwrap();
        tier.wait_for_writes();
        assert!(!tier.contains(&simple_hash(0)));
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn a_second_tier_in_the_same_directory_is_refused() {
        let dir = tmpdir("shared");
        let first = DiskTier::new(&dir, &[8], 1 << 20, 1, false).unwrap();
        let err = DiskTier::new(&dir, &[8], 1 << 20, 1, false)
            .err()
            .expect("a second tier in a held directory must be refused");
        assert!(err.to_string().contains("another disk tier"), "{err}");
        // Shutting the first down releases the directory.
        first.shutdown();
        let second = DiskTier::new(&dir, &[8], 1 << 20, 1, false).unwrap();
        second.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn shutdown_closes_the_leaf_files() {
        let dir = tmpdir("closes");
        let open_under_dir = || {
            std::fs::read_dir("/proc/self/fd")
                .unwrap()
                .flatten()
                .filter_map(|fd| std::fs::read_link(fd.path()).ok())
                .filter(|target| target.starts_with(&dir))
                .count()
        };
        let tier = DiskTier::new(&dir, &[8, 16], 1 << 20, 1, false).unwrap();
        // Two leaf files and the directory lock.
        assert_eq!(open_under_dir(), 3);
        tier.shutdown();
        assert_eq!(open_under_dir(), 0);
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn pin_reads_is_all_or_nothing() {
        let dir = tmpdir("pin-all");
        let tier = DiskTier::new(&dir, &[8], 1 << 30, 1, false).unwrap();
        // Hold the single worker so the write stays pending.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        tier.pool.submit(
            PRIO_READ,
            Box::new(move || while release_rx.recv().is_ok() {}),
        );
        let src = [9u8; 8];
        tier.write_block_async(simple_hash(1), src.as_ptr() as usize)
            .unwrap();
        assert!(
            tier.pin_reads(&[simple_hash(1)]).is_none(),
            "a pending write is not readable yet"
        );
        drop(release_tx);
        tier.wait_for_writes();
        assert!(tier.pin_reads(&[simple_hash(1), simple_hash(2)]).is_none());
        assert_eq!(tier.pin_reads(&[]), Some(Vec::new()));
        assert_eq!(read_back(&tier, simple_hash(1), 8), src);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn eviction_releases_disk_space() {
        use std::os::unix::fs::MetadataExt;

        let dir = tmpdir("punch");
        let block = 16 * DIRECT_IO_ALIGN;
        let tier = DiskTier::new(&dir, &[block], block, 1, false).unwrap();
        let srcs: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i + 1; block]).collect();
        for (i, src) in srcs.iter().enumerate() {
            tier.write_block_async(simple_hash(i as u128), src.as_ptr() as usize)
                .unwrap();
            tier.wait_for_writes();
        }
        let meta = std::fs::metadata(Path::new(&dir).join("leaf-0.bin")).unwrap();
        // `blocks` counts 512-byte units actually allocated.
        let allocated = meta.blocks() * 512;
        assert!(
            meta.len() > block as u64,
            "the evictions spread across slots"
        );
        assert!(
            allocated <= block as u64,
            "{allocated} bytes allocated for one live {block}-byte block"
        );
        assert_eq!(read_back(&tier, simple_hash(3), block), srcs[3]);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn duplicate_write_returns_none() {
        let dir = tmpdir("dup");
        let tier = DiskTier::new(&dir, &[8], 1 << 30, 2, false).unwrap();
        let hash = simple_hash(42);
        let src = [7u8; 8];
        assert!(
            tier.write_block_async(hash, src.as_ptr() as usize)
                .is_some()
        );
        tier.wait_for_writes();
        // Already saved -> None.
        assert!(
            tier.write_block_async(hash, src.as_ptr() as usize)
                .is_none()
        );
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn inflight_disk_ops_tracks_queued_ops() {
        let dir = tmpdir("inflight");
        // One worker, so a blocked worker + a queued op are both observable.
        let tier = DiskTier::new(&dir, &[8], 1 << 30, 1, false).unwrap();
        assert_eq!(tier.inflight_disk_ops(), 0);

        // Occupy the single worker with a job that blocks until released.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        tier.pool.submit(
            PRIO_WRITE,
            // Blocks until the test drops `release_tx` (recv then errors).
            Box::new(move || while release_rx.recv().is_ok() {}),
        );
        // Queue a second op behind the blocked worker.
        tier.pool.submit(PRIO_WRITE, Box::new(|| {}));

        // Both ops are counted while the worker is blocked (submit accounts for
        // them synchronously; neither has completed).
        assert_eq!(tier.inflight_disk_ops(), 2);

        // Release the worker and drain; the gauge returns to zero.
        drop(release_tx);
        tier.wait_for_writes();
        assert_eq!(tier.inflight_disk_ops(), 0);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn reset_clears_disk() {
        let dir = tmpdir("reset");
        let tier = DiskTier::new(&dir, &[8], 1 << 30, 2, false).unwrap();
        let src = [5u8; 8];
        tier.write_block_async(simple_hash(1), src.as_ptr() as usize);
        tier.write_block_async(simple_hash(2), src.as_ptr() as usize);
        tier.wait_for_writes();
        assert_eq!(tier.used_bytes(), 16);

        let file = Path::new(&dir).join("leaf-0.bin");
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 16);

        tier.reset();

        // Index cleared and the leaf's file truncated, not unlinked.
        assert_eq!(tier.used_bytes(), 0);
        assert!(!tier.contains(&simple_hash(1)));
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 0);

        // Disk is still usable after reset, from slot 0 again.
        let src3 = [6u8; 8];
        tier.write_block_async(simple_hash(3), src3.as_ptr() as usize);
        tier.wait_for_writes();
        assert!(tier.contains(&simple_hash(3)));
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 8);
        assert_eq!(read_back(&tier, simple_hash(3), 8), src3);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn eviction_bounds_capacity() {
        let dir = tmpdir("evict");
        // Capacity for 2 blocks of 8 bytes.
        let tier = DiskTier::new(&dir, &[8], 16, 2, false).unwrap();
        let src = [3u8; 8];
        tier.write_block_async(simple_hash(10), src.as_ptr() as usize);
        tier.write_block_async(simple_hash(11), src.as_ptr() as usize);
        tier.wait_for_writes();
        // Third write evicts the LRU (block 10).
        tier.write_block_async(simple_hash(12), src.as_ptr() as usize);
        tier.wait_for_writes();
        assert!(tier.used_bytes() <= 16);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }
    #[test]
    fn disabled_tier_is_inert() {
        let tier = DiskTier::disabled();
        let src = [7u8; 8];
        // The declined write is what releases the offload's host pin; enqueueing
        // instead would strand it, since the disabled tier owns no workers.
        let declined = tier.write_block_async(simple_hash(1), src.as_ptr() as usize);
        assert!(declined.is_none());
        assert!(!tier.contains(&simple_hash(1)));
        assert_eq!((tier.max_bytes(), tier.used_bytes()), (0, 0));
        assert_eq!(tier.inflight_disk_ops(), 0);
        // Teardown must not touch the CWD or block.
        tier.reset();
        tier.wait_for_writes();
        tier.shutdown();
    }

    /// Whether this platform + the filesystem under `dir` accept O_DIRECT
    /// (tmpfs before Linux 6.6 does not; macOS has no O_DIRECT).
    fn supports_o_direct(dir: &Path) -> bool {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        page_cache::open_direct(&mut opts, &dir.join("probe")).is_ok()
    }

    /// A `DIRECT_IO_ALIGN`-byte window into `backing` at its first
    /// `DIRECT_IO_ALIGN` boundary: heap Vecs are not reliably 4 KiB-aligned,
    /// but the first boundary inside an over-allocated one is.
    fn aligned_window(backing: &mut [u8]) -> &mut [u8] {
        assert!(backing.len() >= 2 * DIRECT_IO_ALIGN);
        let base = backing.as_ptr() as usize;
        let start = base.div_ceil(DIRECT_IO_ALIGN) * DIRECT_IO_ALIGN - base;
        // SAFETY: the window lies fully inside `backing` by construction.
        let ptr = unsafe { backing.as_mut_ptr().add(start) };
        // SAFETY: `ptr` is in-bounds and the window fits, per the assert.
        unsafe { std::slice::from_raw_parts_mut(ptr, DIRECT_IO_ALIGN) }
    }

    #[test]
    fn direct_io_round_trips_when_supported() {
        let dir = tmpdir("direct-rt");
        std::fs::create_dir_all(&dir).unwrap();
        let dir_path = Path::new(&dir);
        if !supports_o_direct(dir_path) {
            // The no-fallback contract is enforced by the probe at
            // construction; on a filesystem without O_DIRECT there is
            // nothing to exercise here.
            return;
        }
        let tier = DiskTier::new(&dir, &[DIRECT_IO_ALIGN], 1 << 30, 2, true).unwrap();
        let hash = simple_hash(7);
        let mut src_backing = vec![0u8; 3 * DIRECT_IO_ALIGN];
        let src = aligned_window(&mut src_backing);
        src.fill(0x5A);
        let mut dst_backing = vec![0u8; 3 * DIRECT_IO_ALIGN];
        let dst = aligned_window(&mut dst_backing);
        tier.write_block_async(hash, src.as_ptr() as usize).unwrap();
        tier.wait_for_writes();
        tier.read_block_async(pin(&tier, hash), dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        assert_eq!(dst, src);
        tier.shutdown();
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn probe_rejects_a_filesystem_without_o_direct() {
        let dir = tmpdir("probe-unsupported");
        std::fs::create_dir_all(&dir).unwrap();
        let dir_path = Path::new(&dir);
        let mut backing = vec![0u8; 2 * DIRECT_IO_ALIGN];
        let probed = probe_direct_io(dir_path, aligned_window(&mut backing));
        if supports_o_direct(dir_path) {
            probed.expect("a filesystem that accepts O_DIRECT passes the probe");
            assert!(!dir_path.join(".direct-io-probe").exists());
        } else {
            probed.expect_err("a filesystem without O_DIRECT fails the probe");
        }
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn probe_rejects_an_unaligned_buffer() {
        let dir = tmpdir("probe-unaligned");
        std::fs::create_dir_all(&dir).unwrap();
        // Misalign by one byte: unaligned address must fail before any I/O.
        let backing = vec![0u8; 2 * DIRECT_IO_ALIGN];
        let start = backing.as_ptr() as usize + 1;
        let len = DIRECT_IO_ALIGN;
        // SAFETY: `backing` outlives the probe call; the probe only reads
        // and writes inside [start, start + len), which is inside the Vec.
        let probe = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) };
        probe_direct_io(Path::new(&dir), probe)
            .expect_err("an unaligned probe address must be rejected");
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn leaf_size_check_rejects_short_unaligned_and_empty_leaves() {
        check_direct_io_leaf_sizes(&[DIRECT_IO_ALIGN, 4 * DIRECT_IO_ALIGN])
            .expect("aligned leaves pass");
        for bad in [16, 0, DIRECT_IO_ALIGN + 16] {
            let err = check_direct_io_leaf_sizes(&[DIRECT_IO_ALIGN, bad])
                .expect_err("an unaligned leaf must be rejected");
            assert!(err.to_string().contains("leaf 1"), "{err}");
        }
    }

    #[test]
    fn probe_rejects_a_short_buffer() {
        let dir = tmpdir("probe-short");
        std::fs::create_dir_all(&dir).unwrap();
        let mut backing = vec![0u8; 2 * DIRECT_IO_ALIGN];
        let window = aligned_window(&mut backing);
        let err = probe_direct_io(Path::new(&dir), &mut window[..DIRECT_IO_ALIGN / 2])
            .expect_err("a probe buffer shorter than DIRECT_IO_ALIGN must be rejected");
        assert!(err.to_string().contains("exactly"));
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn direct_io_rejects_unaligned_leaf_sizes() {
        let dir = tmpdir("direct-unaligned-leaf");
        // 4096+16 is not a multiple of the 4 KiB direct-I/O alignment.
        let err = DiskTier::new(&dir, &[DIRECT_IO_ALIGN + 16], 1 << 30, 1, true)
            .err()
            .expect("an unaligned leaf must be rejected");
        assert!(err.to_string().contains("use_disk_direct_io"));
        drop(std::fs::remove_dir_all(&dir));
    }

    #[test]
    fn transfer_direct_moves_whole_aligned_chunks() {
        // The no-fallback contract makes any error or zero-length transfer
        // fatal, so the only observable behavior for aligned successes is the
        // chunking itself: three whole 4 KiB chunks complete without error
        // and the offset stays alignment-clean between them.
        let calls = std::cell::Cell::new(0);
        transfer_direct("probe", DIRECT_IO_ALIGN * 3, |off| {
            calls.set(calls.get() + 1);
            // Whole 4 KiB chunks: never short, never zero, never misaligned.
            Ok(DIRECT_IO_ALIGN.min(3 * DIRECT_IO_ALIGN - off))
        });
        assert_eq!(calls.get(), 3);
    }
}
