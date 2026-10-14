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
//! Flat-file disk tier: the last level of the KV prefix cache.
//!
//! Blocks are stored one-file-per-block at
//! `<cache_dir>/<first-byte-hex>/<hash-hex>.bin` (256 shard buckets). A priority
//! thread pool (reads preempt deletes preempt writes) runs the blocking file
//! I/O off the connector's lanes; each op reads/writes the *pinned host buffer*
//! bytes directly (the caller passes the host address of the block's row), so a
//! disk read lands straight into the page an H2D will copy. An in-memory LRU
//! index bounds disk usage; it starts cold (see [`DiskTier::shutdown`] for why
//! there is no warm-start).
//!
//! The pool holds a raw host pointer per op; the caller guarantees the block
//! stays pinned (host `JengaBlockPool` ref count) for the op's duration.
#![allow(
    unsafe_code,
    reason = "raw host-pointer slices over caller-pinned blocks; each block is documented with SAFETY"
)]

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::OpenOptions,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};

use crate::{
    error::{ConnectorError, Result, fatal},
    lane::Pollable,
    utils::LeafHash,
};

/// Priorities: lower runs first. Reads preempt deletes preempt writes, matching
/// the Python `PriorityExecutor`.
const PRIO_READ: u8 = 0;
const PRIO_DELETE: u8 = 1;
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

#[derive(Default)]
struct DiskIndex {
    // LRU: seq -> hash and hash -> seq. Lowest seq is the eviction victim.
    lru_by_seq: BTreeMap<u64, LeafHash>,
    seq_by_hash: HashMap<LeafHash, u64>,
    pending_writes: HashSet<LeafHash>,
    next_seq: u64,

    used_bytes: usize,
}

impl DiskIndex {
    fn touch(&mut self, hash: LeafHash) {
        if let Some(old) = self.seq_by_hash.get(&hash) {
            self.lru_by_seq.remove(old);
            let s = self.next_seq;
            self.next_seq += 1;
            self.lru_by_seq.insert(s, hash);
            self.seq_by_hash.insert(hash, s);
        }
    }
    fn insert(&mut self, hash: LeafHash) {
        let s = self.next_seq;
        self.next_seq += 1;
        self.lru_by_seq.insert(s, hash);
        self.seq_by_hash.insert(hash, s);
    }
    fn contains(&self, hash: &LeafHash) -> bool {
        self.seq_by_hash.contains_key(hash)
    }
    fn pop_lru(&mut self) -> Option<LeafHash> {
        let (&s, _) = self.lru_by_seq.iter().next()?;
        let hash = self.lru_by_seq.remove(&s).unwrap();
        self.seq_by_hash.remove(&hash);
        Some(hash)
    }
}

/// The disk tier. Optional: see [`DiskTier::disabled`].
pub struct DiskTier {
    cache_dir: PathBuf,
    /// False for [`DiskTier::disabled`], which declines every op.
    pub enabled: bool,
    bytes_per_leaf: Vec<usize>,
    max_bytes: usize,
    /// Whether reads and writes open with `O_DIRECT` (no page cache). See
    /// [`DiskTier::new`]; when false every op takes the buffered path, which
    /// is exactly the behavior before `use_disk_direct_io` existed.
    direct_io: bool,
    pool: Arc<PriorityPool>,
    index: Arc<Mutex<DiskIndex>>,
}

impl DiskTier {
    /// Opens the disk tier at `cache_dir` with a cold (empty) index -- see
    /// [`DiskTier::shutdown`] for why a prior run's blocks are never reloaded.
    ///
    /// `direct_io` opts block reads and writes into `O_DIRECT`, keeping the
    /// tier's files out of the page cache; there is no buffered fallback, so
    /// the caller must first prove the combination works with
    /// [`probe_direct_io`] (the connector constructor does), or every op
    /// errors mid-serving.
    pub fn new(
        cache_dir: &str,
        bytes_per_leaf: &[usize],
        max_bytes: usize,
        num_workers: usize,
        direct_io: bool,
    ) -> Result<Self> {
        let cache_dir = PathBuf::from(cache_dir);
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| ConnectorError::Config(format!("create cache dir: {e}")))?;
        for b in 0u16..256 {
            std::fs::create_dir_all(cache_dir.join(format!("{b:02x}")))
                .map_err(|e| ConnectorError::Config(format!("create bucket: {e}")))?;
        }
        if direct_io {
            // O_DIRECT requires the buffer address and length to be aligned to
            // the device's logical block size, and a block's length is its
            // leaf's row. The staging region's base alignment is checked by
            // the caller's probe, which sees the actual addresses.
            check_direct_io_leaf_sizes(bytes_per_leaf)?;
        }

        Ok(DiskTier {
            cache_dir,
            enabled: true,
            bytes_per_leaf: bytes_per_leaf.to_vec(),
            max_bytes,
            direct_io,
            pool: PriorityPool::new(num_workers.max(1)),
            index: Arc::new(Mutex::new(DiskIndex::default())),
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
            cache_dir: PathBuf::new(),
            enabled: false,
            bytes_per_leaf: Vec::new(),
            max_bytes: 0,
            direct_io: false,
            pool: PriorityPool::new(0),
            index: Arc::new(Mutex::new(DiskIndex::default())),
        }
    }

    fn block_path(&self, hash: LeafHash) -> PathBuf {
        // The shard is the first byte of the hash. Hashes with different shards
        // are partitioned into different subdirectories. This is to avoid having
        // millions of files in a single massive directory as the fileio perf
        // degrades with the number of files in a directory.
        let shard = (hash.block_hash.0 & 0xff) as u8;
        self.cache_dir
            .join(format!("{shard:02x}"))
            .join(format!("{}.bin", hash.to_hex()))
    }
    /// Total number of bytes the disk tier can hold.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Number of bytes currently indexed (saved or pending write).
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.index.lock().unwrap().used_bytes
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
        let idx = self.index.lock().unwrap();
        idx.contains(hash) && !idx.pending_writes.contains(hash)
    }

    /// Clears every block from disk: empties the in-memory index and unlinks all
    /// `.bin` files (keeping the shard subdirs so later writes still land).
    ///
    /// The caller must have drained in-flight disk writes first (see
    /// `TieredConnector::reset_prefix_cache`) so no worker races the unlink.
    pub fn reset(&self) {
        if !self.enabled {
            // `cache_dir` is empty, so the bucket joins below would resolve
            // relative to the process CWD.
            return;
        }
        {
            let mut idx = self.index.lock().unwrap();
            idx.lru_by_seq.clear();
            idx.seq_by_hash.clear();
            idx.pending_writes.clear();
            idx.next_seq = 0;
            idx.used_bytes = 0;
        }
        for bucket in 0u16..256 {
            let dir = self.cache_dir.join(format!("{bucket:02x}"));
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                    drop(std::fs::remove_file(&path));
                }
            }
        }
    }

    /// Reads `hash` into the pinned host block at `dest_ptr`. LRU-touches the
    /// hash and submits at READ priority. Returns a pollable op.
    pub fn read_block_async(&self, hash: LeafHash, dest_ptr: usize) -> DiskOp {
        if !self.enabled {
            // Unreachable: `contains` is false, so nothing is ever a disk hit.
            fatal("disk read on a disabled disk tier");
        }
        self.index.lock().unwrap().touch(hash);
        let op = DiskOp::new();
        let path = self.block_path(hash);
        let dst = HostBlock {
            ptr: dest_ptr,
            len: self.bytes_per_leaf[hash.leaf_idx],
        };
        let direct_io = self.direct_io;
        let op2 = op.clone();
        self.pool.submit(
            PRIO_READ,
            Box::new(move || {
                let _r =
                    crate::trace::Range::new("kv-disk read_block", crate::trace::color::DISK_READ);
                read_file_into(&path, dst, direct_io);
                op2.complete();
            }),
        );
        op
    }

    /// Writes the pinned host block at `src_ptr` to `hash`'s file, evicting LRU
    /// blocks first if at capacity. Returns `None` if already saved or a write
    /// is already pending; otherwise a pollable op at WRITE priority.
    pub fn write_block_async(&self, hash: LeafHash, src_ptr: usize) -> Option<DiskOp> {
        if !self.enabled {
            return None;
        }
        let bytes_needed = self.bytes_per_leaf[hash.leaf_idx];

        let mut evictions: Vec<LeafHash> = Vec::new();
        {
            let mut index = self.index.lock().unwrap();
            if index.contains(&hash) || index.pending_writes.contains(&hash) {
                return None;
            }
            while self.max_bytes - index.used_bytes < bytes_needed {
                match index.pop_lru() {
                    Some(victim) => {
                        index.used_bytes -= self.bytes_per_leaf[victim.leaf_idx];
                        evictions.push(victim);
                    }
                    _ => fatal(format!(
                        "not enough space in disk tier to offload {bytes_needed} bytes, \
                         even after evicting all evictable blocks"
                    )),
                }
            }
            index.pending_writes.insert(hash);
            index.insert(hash);
            index.used_bytes += bytes_needed;
        }
        for victim in evictions {
            let path = self.block_path(victim);
            self.pool.submit(
                PRIO_DELETE,
                Box::new(move || drop(std::fs::remove_file(&path))),
            );
        }

        let op = DiskOp::new();
        let path = self.block_path(hash);
        let src = HostBlock {
            ptr: src_ptr,
            len: bytes_needed,
        };
        let index = Arc::clone(&self.index);
        let direct_io = self.direct_io;
        let op2 = op.clone();
        self.pool.submit(
            PRIO_WRITE,
            Box::new(move || {
                let _r = crate::trace::Range::new(
                    "kv-disk write_block",
                    crate::trace::color::DISK_WRITE,
                );
                write_file_from(&path, src, direct_io);
                index.lock().unwrap().pending_writes.remove(&hash);
                op2.complete();
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
    pub fn shutdown(&self) {
        self.pool.wait_until_idle();
        self.pool.shutdown();
    }
}

fn read_file_into(path: &Path, dst: HostBlock, direct_io: bool) {
    // Fail-fast: this only runs for blocks the index reports present, whose file
    // was durably written (writes fail-fast too), so any error here is genuine
    // corruption. Silently returning would promote uninitialized host bytes.
    // SAFETY: dst.ptr is a pinned host block of dst.len bytes, kept alive by the
    // caller's ref count for this op; we are the only writer.
    let buf = unsafe { std::slice::from_raw_parts_mut(dst.ptr as *mut u8, dst.len) };
    if direct_io {
        debug_assert!(
            is_direct_io_aligned(dst.ptr, dst.len),
            "the probe verified every block address and length is direct-I/O aligned"
        );
        read_direct(path, buf);
        return;
    }
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| fatal(format!("disk read open {}: {e}", path.display())));
    f.read_exact(buf)
        .unwrap_or_else(|e| fatal(format!("disk read {}: {e}", path.display())));
}

fn write_file_from(path: &Path, src: HostBlock, direct_io: bool) {
    // SAFETY: src.ptr is a pinned host block of src.len bytes, kept alive by the
    // caller's ref count for this op; read-only here.
    let buf = unsafe { std::slice::from_raw_parts(src.ptr as *const u8, src.len) };
    let tmp = path.with_extension("bin.tmp");
    // Fail-fast on every step: a swallowed write leaves the index advertising a
    // block whose file is missing/truncated, which a later disk read would then
    // promote as garbage. Write to a temp file and atomically rename so a
    // reader never observes a partial file.
    if direct_io {
        debug_assert!(
            is_direct_io_aligned(src.ptr, src.len),
            "the probe verified every block address and length is direct-I/O aligned"
        );
        write_direct(&tmp, buf, path);
        return;
    }
    let mut f = std::fs::File::create(&tmp)
        .unwrap_or_else(|e| fatal(format!("disk write create {}: {e}", tmp.display())));
    f.write_all(buf)
        .unwrap_or_else(|e| fatal(format!("disk write {}: {e}", tmp.display())));
    f.sync_all()
        .unwrap_or_else(|e| fatal(format!("disk write sync {}: {e}", tmp.display())));
    std::fs::rename(&tmp, path).unwrap_or_else(|e| {
        fatal(format!(
            "disk write rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    });
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

fn read_direct(path: &Path, buf: &mut [u8]) {
    let mut opts = OpenOptions::new();
    opts.read(true);
    let mut f = page_cache::open_direct(&mut opts, path)
        .unwrap_or_else(|e| fatal(format!("disk read open {}: {e}", path.display())));
    transfer_direct(&format!("read {}", path.display()), buf.len(), |off| {
        f.read(&mut buf[off..])
    });
}

fn write_direct(tmp: &Path, buf: &[u8], final_path: &Path) {
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    let mut f = page_cache::open_direct(&mut opts, tmp)
        .unwrap_or_else(|e| fatal(format!("disk write create {}: {e}", tmp.display())));
    transfer_direct(&format!("write {}", tmp.display()), buf.len(), |off| {
        f.write(&buf[off..])
    });
    f.sync_all()
        .unwrap_or_else(|e| fatal(format!("disk write sync {}: {e}", tmp.display())));
    std::fs::rename(tmp, final_path).unwrap_or_else(|e| {
        fatal(format!(
            "disk write rename {} -> {}: {e}",
            tmp.display(),
            final_path.display()
        ))
    });
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
        let rop = tier.read_block_async(hash, dst.as_mut_ptr() as usize);
        tier.wait_for_writes();
        assert!(rop.is_ready());
        assert_eq!(dst, src);
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

        // Landed in its shard bucket, not flat under `cache_dir`.
        let shard = Path::new(&dir).join("01");
        assert_eq!(
            std::fs::read_dir(&shard).unwrap().flatten().count(),
            1,
            "block 1's file should be sharded into {}",
            shard.display()
        );

        tier.reset();

        // Index cleared and the shard's `.bin` file is unlinked.
        assert_eq!(tier.used_bytes(), 0);
        assert!(!tier.contains(&simple_hash(1)));
        let shard = Path::new(&dir).join("01");
        let remaining = std::fs::read_dir(&shard)
            .map(|it| it.flatten().count())
            .unwrap_or(0);
        assert_eq!(remaining, 0);

        // Disk is still usable after reset (shard dirs are kept).
        tier.write_block_async(simple_hash(3), src.as_ptr() as usize);
        tier.wait_for_writes();
        assert!(tier.contains(&simple_hash(3)));
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
        tier.read_block_async(hash, dst.as_mut_ptr() as usize);
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
