//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares the portable memory manager used by device contexts.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace llvm {
class raw_fd_ostream;
} // namespace llvm

namespace M::Driver {

class MemoryManagerFreeList;

// Physical/VA accounting captured when a VMM allocation fails, used to classify
// the failure and build an accurate diagnostic. The granule-state counts
// partition the budget's whole pages: a budget that is not a whole multiple of
// the page size leaves its trailing partial page outside the partition, so the
// counts sum to `budget / pageSize` pages rather than to `budget` bytes. The
// derived accessors form the "lever" partition (used / available /
// sync-recoverable / trapped) the category is built from.
struct VmmOomSnapshot {
  size_t pageSize = 0;
  size_t budget = 0; // maxCacheSize_
  size_t needed = 0; // the request that failed
  // Size of the VA reservation actually in force, so a VA shortfall can be
  // read against it. Not the configured factor: allocateVmmArena falls back to
  // a budget-sized reservation if the oversized one fails, so the config value
  // can name a factor that was never applied.
  size_t vaSize = 0;

  // Granule-state partition of the budget (counts). Each budget granule is in
  // exactly one state.
  size_t inUseGranules = 0;       // >=1 in-use byte (pinned)
  size_t pendingGranules = 0;     // pending byte(s), no in-use byte
  size_t reclaimableGranules = 0; // backed, only root-free descs
  size_t pooledGranules = 0;      // created but unmapped (freeGranulePool_)
  size_t uncommittedGranules = 0; // never created; budget headroom

  // Byte totals from the desc walk.
  size_t usedBytes = 0;         // bytes under in-use descs
  size_t pendingTotalBytes = 0; // all child-free desc bytes

  // What a drain of the queued records would really recover, which is narrower
  // than the pending totals above. Those count every child-owned desc, but a
  // free landing while its list already has a flush in flight joins the
  // primary buffer without queuing an event of its own, so nothing migrates it
  // until a later free rotates the lists. Waiting on the difference is waiting
  // for something that is not coming.
  size_t drainableBytes = 0; // bytes the queued records will migrate
  // Granules left wholly root-owned once they do, which is the physical a
  // drain can hand back. Bytes cannot answer that: one byte in a queued buffer
  // can release a whole granule, while a granule's worth spread across
  // granules that keep other child-owned bytes releases none.
  size_t drainableGranules = 0;

  // Free descs seen in the walk, root-owned and child-owned. freeCount_ is not
  // maintained in VMM mode, so this is the free-block count the diagnostics
  // report.
  size_t freeBlocks = 0;

  // VA axis: the largest contiguous free (root-owned) region that a request
  // could be carved from.
  size_t largestRootFreeVA = 0;

  // Whether the VMM arena is reserved and backed at all. False before the
  // arena exists (a failed reservation, for instance), where there is neither
  // VA nor physical to account for and no request can be served whatever the
  // budget says.
  bool arenaReserved = false;

  // Physical the request claims as things stand, rounded up to whole pages.
  // The "can it be served now" gate uses this; the lever sum is compared
  // against `needed`, since every lever changes the rounding.
  size_t neededPhysicalBytes() const {
    return pageSize == 0 ? 0 : ((needed + pageSize - 1) / pageSize) * pageSize;
  }
  // Allocatable now: whole free granules + pooled handles + budget headroom.
  // Sufficient for a request, never necessary: what a request costs depends on
  // how much of the region it lands in is already backed, so one can succeed
  // with `available()` below `neededPhysicalBytes()`.
  size_t available() const {
    return (reclaimableGranules + pooledGranules + uncommittedGranules) *
           pageSize;
  }
  // Would become allocatable after a synchronize. Whole granules only, so a
  // pending byte sharing a granule with an in-use desc does not count: that
  // granule stays pinned.
  size_t recoverableBySync() const { return pendingGranules * pageSize; }
  // Free/pending bytes stuck inside in-use granules; only a smaller page size
  // recovers them.
  size_t trapped() const {
    const size_t inUseBytes = inUseGranules * pageSize;
    return inUseBytes > usedBytes ? inUseBytes - usedBytes : 0;
  }
};

// Which step of ensureRegionBacked failed, when known. Recorded at the call
// site as a plain observation: the manager refused within its own budget, or
// the driver rejected a create or a map the manager expected to succeed.
// `None` when no specific reason was captured. Interpreting a MapRefused needs
// to know whether a graph capture was in progress, which only the freelist can
// answer, so the caller passes that in separately.
enum class VmmBackingFailure {
  None,
  CreateRefused,
  MapRefused,
};

// The distinguishable causes of a VMM allocation failure.
enum class VmmOomCategory {
  Genuine, // no lever reaches far enough: a true exhaustion
  // Not an exhaustion: the shortfall is covered by memory the manager holds
  // but cannot hand out as it stands, waiting on stream frees or stranded in
  // partially-used pages. One category, not one per lever: the message reports
  // both figures and leaves the choice to the caller.
  Avoidable,
  // The accounting shows no shortfall, yet the allocation was refused: the
  // physical is held somewhere this manager does not account for.
  Unaccounted,
  VaFragmentation,  // physical fine, no contiguous VA -> larger vmm_va_factor
  ArenaUnavailable, // the arena is not reserved -> nothing can be served
  // A map was refused while a graph capture was in progress. Mapping is
  // illegal mid-capture, so this is a pre-backing gap, not a shortage.
  IllegalDuringCapture,
};

// Whether waiting for the queued records could serve a request the free lists
// just missed, i.e. whether a retry is worth a stall. Each axis is tested on
// its own against the figure that answers it -- `drainableGranules` for
// physical, `drainableBytes` for VA -- because `available()` is sufficient but
// not necessary: a request landing on already-backed VA can be served with
// less than it says, so a physical shortfall must not rule the VA route out.
// Requiring a shortfall on the axis being tested keeps a miss with neither
// short (physical held outside this manager) from buying a stall nothing here
// can resolve. Pure.
bool pendingRetryWorthwhile(const VmmOomSnapshot &snapshot);

// Classify a failure from the snapshot and (if known) why backing failed.
// `duringCapture` says whether a graph capture was in progress when the
// backing step failed; it only changes the verdict for a refused map, which is
// illegal mid-capture. Pure.
VmmOomCategory classifyVmmOom(const VmmOomSnapshot &snapshot,
                              VmmBackingFailure backingFailure,
                              bool duringCapture = false);

// One-line human-readable summary naming the category and the recovery lever.
// Levers are phrased as bounded suggestions. Pure.
std::string describeVmmOom(const VmmOomSnapshot &snapshot,
                           VmmOomCategory category);

// Full multi-field dump of the snapshot for the verbose log. Pure.
std::string dumpVmmOom(const VmmOomSnapshot &snapshot);

class MemoryManager {
public:
  class Chunk;
  class Desc;

  class DescMap {
  public:
    DescMap(int mapSizeLog2);
    ~DescMap();

    void add(Desc *desc);
    Desc *remove(void *ptr);

    // Non-destructive lookup by exact base pointer: returns the descriptor
    // whose `ptr_` equals `ptr`, or nullptr. Does not unlink.
    Desc *find(void *ptr) const;

    // Rewrite `allocatedBy_` on every live descriptor whose current value is
    // `from`, replacing it with `to`. Used when a `MemoryManagerFreeList` is
    // about to be destroyed to redirect descriptors it allocated but does not
    // yet hold (still in use elsewhere) to a longer-lived parent freelist, so
    // a later `free()` does not dispatch through a dangling pointer.
    void redirectAllocatedBy(MemoryManagerFreeList *from,
                             MemoryManagerFreeList *to);

  private:
    static uint32_t hash(void *ptr);

    size_t bucketCount_;
    size_t bucketMask_;
    Desc **buckets_;
  };

  ~MemoryManager();

  size_t allocationAlignment() const;

  // Exact byte extent of the live allocation whose base pointer is `ptr` (the
  // aligned block size handed out for it), or 0 if `ptr` is not a current
  // allocation base from this manager. Lets a device backend map only the
  // tensor's own bytes instead of the whole containing chunk.
  size_t allocatedSize(void *ptr) const;

  bool isHost() const;

  void clear();

  MemoryManagerFreeList *freeList() const;

  // Get memory information about the memory manager
  Device::MemInfo getMemoryInfo() const;

  // Test-only: acquire `mu_` and capture the OOM accounting snapshot for
  // `needed`. Production paths compute this inline under the allocation
  // lock; this exposes it so tests can assert exact tier values (available,
  // recoverable_by_sync, trapped, granule counts) against real allocator state.
  VmmOomSnapshot oomSnapshotForTesting(size_t needed) const;

  // Check if the memory manager is enabled (has non-zero capacity)
  bool isEnabled() const;

  // Whether the VMM defragmenting allocator is active on this manager. Reflects
  // both the `memory_manager_vmm` config request and the device's
  // vmmSupported() result, as resolved by onDevice().
  bool isVmmEnabled() const;

  // Migrate to the parent free list every per-stream free buffer whose gating
  // event has completed, making that memory allocatable again. `only` restricts
  // the drain to one free list; the default drains every free list on this
  // manager, which is how an allocation miss recovers memory stranded on
  // sibling streams. Never waits on device work. Caller must NOT hold `mu_`.
  //
  // Returns the first error hit while querying gating events, after attempting
  // every record: one unusable event must not strand the others. A record whose
  // event errored stays queued for a later drain.
  ErrorOrSuccess
  drainCompletedFrees(const MemoryManagerFreeList *only = nullptr);

  // Single-arena VMM: physically back the free portion of the budget before a
  // CUDA graph capture begins. VMM map/unmap is illegal during capture, so any
  // allocation made inside the capture window must hit already-backed granules;
  // this maps the free working region up to the budget so ensureRegionBacked is
  // a no-op during capture. No-op when VMM is disabled or the arena does not
  // exist yet. Idempotent (subsequent calls map only newly-freed granules).
  // The over-mapped slack stays reclaimable (reported free; remapped on demand
  // by later eager allocations), so the steady-state footprint matches an eager
  // full-budget allocation. Must NOT be called while already capturing.
  ErrorOrSuccess preBackForCapture();

  // Allocation misses that retried after a drain, and those the retry served.
  size_t drainRetryAttempts() const {
    return drainRetryAttempts_.load(std::memory_order_relaxed);
  }
  size_t drainRetryRecoveries() const {
    return drainRetryRecoveries_.load(std::memory_order_relaxed);
  }

  // Factory methods configure the memory manager based on defaults or the
  // environment.
  // `memInfo` contains the total and free memory of the device.
  // Setting memInfo.free to 0 will disable the memory manager.
  // `memoryLimit` (bytes) is the maximum memory limit for the memory manager.
  // If -1, then the memory manager will use the default memory limit.
  // Note that MEMORY_MANAGER_SIZE_LIMIT environment variable overrides this.
  // `largeAllocAlignment` (bytes) is the start alignment given to large
  // blocks, rounded up to a power of 2; 0 disables it and -1 takes the value
  // from the config.
  static MemoryManager *onDevice(Device &device, Device::MemInfo memInfo,
                                 size_t alignment, int64_t memoryLimit = -1,
                                 int64_t largeAllocAlignment = -1);
  static MemoryManager *onHost(Device &device, size_t alignment);

private:
  friend class Chunk;
  friend class MemoryManagerFreeList;

  MemoryManager(Device &device, size_t maxSize, size_t chunkSize,
                size_t alignment, bool isHost, bool useOnlyMemoryManagerMode,
                bool vmmEnabled = false, size_t vmmPageSize = 0,
                size_t largeAllocAlignment = 0);

  Device &device() const;

  // Start-address alignment a block of `size` bytes must have, or 0 when the
  // manager-wide `alignment` already suffices. See `largeAllocAlignment_` for
  // what that alignment is for and which blocks earn it.
  size_t startAlignment(size_t size) const;

  void selfCheck() const;
  // Called once an allocation has definitely failed, so the accounting is
  // captured here rather than at the failing attempt: the retries and the
  // driver fallthrough rescue a miss often enough that measuring earlier meant
  // walking the descs on allocations that went on to succeed.
  // `backdoorError` is the driver's own failure from the fallthrough attempt,
  // reported as a note after the manager's reason.
  Error allocationMissError(const MemoryManagerFreeList *freeList,
                            size_t wrappedSize,
                            const std::string *backdoorError = nullptr,
                            bool duringCapture = false) const;
  bool allocateChunk();
  bool freeLocked(void *buf);

  // ---- VMM (defragmenting allocator) ----
  //
  // Single-arena VMM specialization of allocateChunk(): reserve one oversized
  // contiguous VA region (no eager physical backing). Returns false on failure
  // or if the arena already exists. Caller holds `mu_`.
  bool allocateVmmArena();
  // Single-arena VMM specialization of getMemoryInfo()'s free calculation:
  // physically allocatable bytes = budget minus physical committed under in-use
  // allocations (the rest is uncommitted or mapped-but-reclaimable). Caller
  // holds `mu_`.
  size_t vmmAllocatableBytes() const;
  // Acquire physical+virtual backing for a chunk of `allocSize` bytes. When
  // `vmmEnabled_`, reserves VA + creates/maps/sets-access granules (reusing the
  // free-granule pool first) and fills `handles`; otherwise calls chunkAlloc.
  // Returns false on failure (leaving nothing allocated), like a failed
  // chunkAlloc. Caller holds `mu_`.
  bool acquireChunkBacking(size_t allocSize, void **base,
                           std::vector<uint64_t> *handles, size_t *pageSize);
  // Tear down a VMM chunk's backing: unmap each granule, return handles to the
  // free-granule pool (or release if `releaseToDriver`), and free the VA.
  // Caller holds `mu_`.
  void releaseChunkBacking(Chunk *chunk, bool releaseToDriver);
  // Single-arena VMM: ensure every granule spanned by `desc` is physically
  // backed before the block is handed out, mapping unbacked granules on demand
  // (free-granule pool first, else reclaim mapped-but-free granules elsewhere
  // via reclaimMappedGranules, else vmmCreate within the physical budget). The
  // VA arena is larger than the physical budget, so an unbacked granule does
  // NOT imply free budget; when the budget is fully mapped this reclaims
  // physical from free regions and remaps it here. Fails on a driver error
  // (rolled back) or a genuine out-of-physical-budget condition. Callers should
  // gate on descHasUnbackedRegion() (this returns success() when there is
  // nothing to map). Mapping during graph capture is illegal; rather than
  // pre-checking we let the driver vmmMap call fail with its native error.
  // Caller holds `mu_`.
  ErrorOrSuccess ensureRegionBacked(Desc *desc);
  // True if any granule spanned by `desc` is currently unbacked
  // (lazy/unmapped).
  bool descHasUnbackedRegion(const Desc *desc) const;
  // NOTE: the single VMM arena is still represented by one Chunk (chunks_),
  // which owns the per-page VMM state (base_, pageSize_, vmmHandles_,
  // granuleBacked_). These helpers take it explicitly for now; folding the
  // arena state directly into MemoryManager and removing Chunk on the VMM path
  // is a follow-up.
  //
  // Mark, in `out` (sized to the arena's granule count), every granule
  // overlapped by an in-use desc. Used for free-memory accounting
  // (inUsePhysicalBytes); NOT a safe reclaim mask -- see
  // computeUnreclaimableGranules. Caller holds `mu_`.
  void computeInUseGranules(const Chunk *c, llvm::BitVector &out) const;
  // Mark, in `out`, every granule that is NOT safe to unmap/reclaim: any
  // granule overlapped by a desc that is in-use OR free-but-owned-by-a-
  // non-root (per-stream) free-list. A per-stream free only migrates to the
  // root free-list after its stream events fire, so its memory may still be
  // referenced by in-flight GPU work; unmapping it could fault a live kernel.
  // Caller holds `mu_`.
  void computeUnreclaimableGranules(const Chunk *c, llvm::BitVector &out) const;
  // Single-arena VMM (Stage 2): unmap up to `count` mapped granules that are
  // safe to reclaim (see computeUnreclaimableGranules), returning their handles
  // to the free-granule pool for remapping elsewhere. Returns the number
  // reclaimed. Caller holds `mu_`.
  size_t reclaimMappedGranules(Chunk *c, size_t count);
  // Physical bytes currently committed under in-use allocations in the single
  // VMM arena (the unreclaimable portion of the budget). Caller holds `mu_`.
  size_t inUsePhysicalBytes() const;
  // Capture the physical/VA accounting for `needed` in one desc + granule
  // pass (the granule-state counts + desc byte totals + largest free VA). Reads
  // state only; does not poll or reclaim. Caller holds `mu_`.
  VmmOomSnapshot computeVmmOomSnapshot(size_t needed) const;

  // Block until the device work gating every record naming `fl` has completed,
  // so ~MemoryManagerFreeList's unconditional flush cannot release blocks a
  // kernel may still be reading. Unbounded, so this is for teardown only: an
  // event ordered behind a host-value wait never fires on its own, and an
  // allocating thread that blocked here would be the thread that was going to
  // release it. The allocation path uses bounded retries instead.
  // Caller must NOT hold `mu_`.
  void waitForPendingFrees(const MemoryManagerFreeList *fl);

  // Avoid malloc under lock.
  Desc *popDesc();
  void pushDesc(Desc *desc);

  // Check if at least one of the devices has determined that chunk sizes should
  // grow to the maximum cache size based on the heuristics used during
  // allocation.
  static bool shouldUseMaxSize();
  // Change this process' state to use the full memory manager for future chunk
  // allocations.
  static void enableUseMaxSize();
  // Use a std::atomic to keep track of the state whether or not we use the full
  // memory manager for future chunk allocations. Since this is a boolean value
  // that can only change from 0 (false) to non-zero (true) using a mutex is not
  // necessary.
  static std::atomic<int32_t> useMaxSize_;

  // Attempt to grow the chunk size to newChunkSize. The eventual chunk size
  // will be capped so that future chunk allocations do not overflow
  // maxCacheSize_.
  void growChunkSizeTo(size_t newChunkSize);

  Device &device_;

  // This memory manager is managing "pinned" host memory.
  const bool isHost_;

  // Whether to only allocate memory from the memory manager.
  // This setting is useful for bazel to enforce the cap on gpu memory used by
  // a particular test. This enables better binpacking of tests on a CI runner.
  const bool useOnlyMemoryManagerMode_;

  // VMM (CUDA Virtual Memory Management) defragmenting allocator. When enabled,
  // chunks are backed by remappable physical granules so a large contiguous
  // allocation can be assembled from physically-scattered free memory.
  const bool vmmEnabled_;
  // Physical granule size (bytes); a multiple of the device VMM granularity.
  // All VMM reservations/mappings are rounded up to this.
  const size_t pageSize_;
  // Pool of physical granule handles that are currently unmapped and available
  // for reuse (fed by retiring wholly-free VMM chunks).
  std::vector<uint64_t> freeGranulePool_;

  // Free buffers awaiting the device work that gates their migration to the
  // parent free list. Guarded by the enclosing manager's `mu_` -- it holds no
  // lock of its own, so cancelling a record cannot race a drain of it. Every
  // method below except mayHaveRecords() requires that lock.
  class PendingFrees {
  public:
    struct Record {
      // Identity that outlives the unlocked window in drainCompletedFrees().
      uint64_t seq;
      // Device work that must complete before the buffer can migrate.
      DeviceEventRef event;
      // Free list owning the buffer; cancels its records when destroyed.
      MemoryManagerFreeList *fl;
      // Which of that free list's double-buffered lists to flush.
      int bufferId;
    };

    void add(DeviceEventRef event, MemoryManagerFreeList *fl, int bufferId);
    // Drop every record naming `fl`.
    void cancel(const MemoryManagerFreeList *fl);
    // Remove and return the record identified by `seq`, or nullopt if it is
    // gone. Identity is by `seq` rather than position so a caller that released
    // `mu_` mid-drain cannot act on a record cancelled meanwhile.
    std::optional<Record> take(uint64_t seq);
    // The (seq, event) of every record naming `fl`, or of all of them when `fl`
    // is null.
    llvm::SmallVector<std::pair<uint64_t, DeviceEventRef>>
    collect(const MemoryManagerFreeList *fl) const;
    bool empty() const { return records_.empty(); }
    // Lock-free fast path: false only when there is certainly nothing queued.
    // The one method safe to call without `mu_`.
    bool mayHaveRecords() const {
      return count_.load(std::memory_order_acquire) != 0;
    }
    const llvm::SmallVector<Record> &records() const { return records_; }
    // Highest sequence number still queued, 0 when nothing is. Sequences only
    // ever increase, so a value above a previous reading means a free was
    // queued since it.
    uint64_t maxSeq() const {
      uint64_t seq = 0;
      for (const Record &record : records_)
        seq = std::max(seq, record.seq);
      return seq;
    }

  private:
    // Keeps the lock-free counter in step with the records it mirrors.
    void syncCount() {
      count_.store(records_.size(), std::memory_order_release);
    }

    llvm::SmallVector<Record> records_;
    uint64_t nextSeq_ = 0;
    std::atomic<size_t> count_ = {0};
  };
  PendingFrees pendingFrees_;

  std::atomic<size_t> drainRetryAttempts_ = {0};
  std::atomic<size_t> drainRetryRecoveries_ = {0};

  // How hard an allocation miss tries to wait out its pending frees, read once
  // so the miss path does not take the global config lock. Zero retries turns
  // the wait off outright and restores the pre-existing behaviour of failing
  // the miss immediately, which is the switch to reach for if the stall ever
  // costs more than the restarts it avoids.
  int64_t drainRetries_ = 0;
  int64_t drainBudgetMs_ = 0;

  // Sequence high-water mark as of the last drain retry that recovered
  // nothing. A later miss whose queued records all sit at or below it is
  // waiting on the very frees that already failed to arrive -- a record gated
  // behind a host-value wait this thread releases never fires on its own -- so
  // it skips the retry instead of paying the budget again on every call.
  // Guarded by `mu_`.
  uint64_t futileDrainSeq_ = 0;

  const size_t maxCacheSize_;
  size_t chunkSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  // Start alignment for large blocks, coarser than the device's own, so a NIC
  // can pin them: amdgpu registers VRAM for RDMA at 2 MiB granularity, and
  // `ibv_reg_mr` fails with ENOMEM on a range that does not start on such a
  // boundary. Devices report a 256-byte allocation alignment, so a multi-GiB
  // KV cache group sub-allocated out of a shared chunk lands on a 2 MiB
  // boundary only by luck. 0 disables it entirely. Best-effort: allocation
  // falls back to a plain fit rather than failing
  // (MemoryManagerFreeList::allocateDesc).
  const size_t largeAllocAlignment_;
  // A block only earns that start at this multiple of it or above, which caps
  // the head padding at 1/kLargeAllocSizeFactor of the request.
  static constexpr size_t kLargeAllocSizeFactor = 32;

  mutable std::mutex mu_;
  // List of all chunks allocated from the driver and their total size.
  size_t cacheSize_ = 0;
  std::unique_ptr<Chunk> chunks_ = nullptr;

  // List of unallocated memory regions.
  std::unique_ptr<MemoryManagerFreeList> freeList_;

  // Free list of descriptors to avoid malloc under lock.
  Desc *unusedDescs_ = nullptr;

  // Map of allocated ptr->Desc*: Since the memory handled by the MemoryManager
  // is not accessible by the CPU we cannot store meta-data colocated with the
  // managed memory.
  DescMap allocatedMap_;

  // For logging and debugging.
  std::string &getDeviceDescription() const;
  void logErr(llvm::Twine msg) const;
  void logErr(llvm::Twine msg, size_t info, int64_t size,
              bool printInfoSize = true) const;

  bool doSelfCheck_ = false;
  bool doLog_ = false;
  // Free desc byte total and count. These track VA, not physical, so under
  // VMM they can exceed the physical budget; the VMM paths report
  // vmmAllocatableBytes() instead.
  size_t freeSize_ = 0;
  size_t freeCount_ = 0;
  size_t chunkCount_ = 0;
  std::string lastChunkAllocFailureDetail_;
  mutable std::string deviceDescription_;
};

class MemoryManagerFreeList {
public:
  MemoryManagerFreeList(DeviceContext *context, MemoryManager &memoryManager,
                        MemoryManagerFreeList *parent,
                        bool disableFlush = false, std::string label = "");
  ~MemoryManagerFreeList();

  // Allocate and free memory by using the `MemoryManager`. If the allocation
  // cannot be serviced by the `MemoryManager` due to allocation size or cache
  // capacity, then the allocation request gets transparently forwarded to the
  // associated `DeviceContext`.
  ErrorOr<void *> allocate(size_t size, Device::AllocMode mode);
  ErrorOrSuccess free(void *buf);

  // Return every cached free buffer to the parent, keeping this free list in
  // service. Unlike a deferred free, nothing gates the move, so the caller
  // guarantees no device work still references these buffers: it has
  // synchronized this list's stream, and no one can enqueue more on it.
  void flushCachedToParent();

  bool empty();

  // Enable out-of-bounds access checks
  void enableRedzones();

  // Enable poisoning: fill each allocation body with a NaN-ish pattern on
  // alloc, so uninitialized device reads propagate NaN/garbage into outputs and
  // are caught by existing differential tests (a complement to initcheck that
  // does not suffer its device->host-copy false positives). Independent of
  // redzones: no padding is added and no check runs on free. Combine with
  // `out-of-bounds` to also get OOB-write detection.
  void enablePoison();

  void logLargestBlocksInfo(size_t wrappedSize, std::string missReason) const;

  // Dump a detailed snapshot of this free list (and its parents) plus the
  // per-chunk allocation layout to stderr. Unlike logLargestBlocksInfo, this
  // is unconditional: it does not depend on the memory_manager_log config
  // flag. Intended for OOM diagnostic: call only when an allocation is about
  // to fail (driver memAlloc errored out, or a no-fallback path is taking an
  // Error return). Acquires the memory manager mutex; caller must not hold it.
  // `snapshot`, when non-null, is logged as-is; it is never measured here, so
  // the log and the surfaced message cannot disagree.
  void
  dumpFragmentationDiagnostic(size_t requestedSize, llvm::Twine reason,
                              const VmmOomSnapshot *snapshot = nullptr) const;

private:
  friend class MemoryManager;
  friend class MemoryManager::Chunk;

  // Define the number of "double-buffered freelists" used to implement the
  // MemoryManagerFreeList. The oldest list will be flushed to the parent once
  // the pending event handler fires.
  static constexpr int kNumFreeBufferLists = 2;
  static_assert(llvm::isPowerOf2_64(kNumFreeBufferLists),
                "Must be a non-zero power of two.");

  // Release cached free buffers to the parent freelist and reattribute any
  // still-in-use descriptors allocated through this freelist to the parent,
  // so a later `free()` does not dispatch through `desc->allocatedBy_` after
  // this freelist is destroyed. Called only from the destructor.
  void flushToParent();
  // Shared by `flushToParent` and `flushCachedToParent`. Caller holds `mu_`.
  void flushCachedToParentLocked();

  void selfCheck() const;
  bool contains(MemoryManager::Desc *desc) const;

  DeviceContext *context();
  const DeviceContext *context() const;

  MemoryManager::Desc *allocateDesc(size_t size);
  // Re-drain completed frees and retry `wrappedSize`, up to
  // `memory_manager_vmm_drain_retries` times, returning the descriptor or null.
  // For the allocation miss path, where the snapshot says the shortfall is
  // covered by frees still gated on in-flight device work: retrying lets those
  // land without blocking on an event that may never fire.
  // Caller must NOT hold `mu_`.
  MemoryManager::Desc *drainAndRetry(size_t wrappedSize);
  // Whether a graph capture was in progress when the last backing step on this
  // thread failed. Queried only on the failure path, so the driver call it
  // needs stays off the hot path.
  bool backingFailedDuringCapture() const;
  // Allocate a block of `size` bytes whose start address is a multiple of
  // `startAlign` (0 for no constraint beyond the manager's own alignment).
  MemoryManager::Desc *allocateDescAligned(size_t size, size_t startAlign);
  void freeDesc(MemoryManager::Desc *desc);

  // Free a descriptor allocated by the current free list.
  ErrorOrSuccess freeAllocatedDesc(MemoryManager::Desc *desc, void *ptr);

  void addToFreeList(MemoryManager::Desc *desc);
  void removeFromFreeList(MemoryManager::Desc *desc);

  void flushFreeBuffersToParent(int id);

  // Log helpers that check doLog_ and prepend the free list label.
  void logMsg(llvm::Twine msg) const;
  void logMsg(llvm::Twine msg, size_t info, int64_t size) const;

  struct RedzoneInfo {
    std::string stackTraceOnAlloc_;
    size_t size_;
  };

  // Apply redzone wrapping (if enabled) and/or body poisoning (if enabled) to a
  // freshly-allocated block; returns the pointer handed to the caller.
  ErrorOr<void *> finalizeDebugAlloc(void *unwrappedPtr, size_t size);
  // Remove and return the bookkeeping recorded for `ptr` at allocation, or
  // nullopt when redzones are disabled. Caller holds `mu_`.
  std::optional<RedzoneInfo> takeRedzoneInfo(void *ptr);
  // Verify both redzones still hold their fill pattern, aborting with a
  // hexdump and the allocation-time stack trace if either was overwritten.
  // The caller must NOT hold `mu_`.
  ErrorOrSuccess validateRedzones(void *wrappedPtr, const RedzoneInfo &size);
  void printRedzone(const uint8_t *dataPtr, const uint8_t *displayPtr);

  DeviceContext *ctx_;

  MemoryManager &memoryManager_;
  MemoryManagerFreeList *parent_;

  MemoryManager::Desc *freeBuffers_[kNumFreeBufferLists];
  int primaryFreeBuffers_ = 0;
  bool handlerPending_ = false;

  bool disableFlush_ = false;

  bool redzoneEnable_ = false;
  size_t redzoneSize_ = 0;
  uint8_t redzonePattern_ = 0;
  DenseMap<void *, RedzoneInfo> redzoneDeviceAllocSizeMap_;

  // Poison mode: fill the allocation body with `poisonPattern_` on alloc.
  bool poisonEnable_ = false;
  uint8_t poisonPattern_ = 0;

  std::string label_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H
