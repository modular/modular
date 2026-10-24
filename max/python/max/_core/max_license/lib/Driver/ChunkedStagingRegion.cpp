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

#include "_core/Driver/ChunkedStagingRegion.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/Chain.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/Log.h"
#include "nanobind/nanobind.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#if defined(__linux__)
#include <cerrno>
#include <fstream>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif
#include <unistd.h>
#include <utility>
#include <vector>

using namespace M;
using namespace Driver;
namespace nb = nanobind;

namespace {

using Slice = std::pair<char *, size_t>;

#if defined(__linux__)
constexpr int kPrSetVma = 0x53564d41;
constexpr int kPrSetVmaAnonName = 0;
#endif

// volatile so the compiler cannot elide the store that faults the page in.
void touchSlice(char *p, size_t len, size_t page) {
  if (len == 0)
    return;
  for (size_t off = 0; off < len; off += page)
    *reinterpret_cast<volatile char *>(p + off) = 0;
  *reinterpret_cast<volatile char *>(p + len - 1) = 0;
}

// Distinct VMA names stop the kernel merging the chunks, so touch workers
// fault them without contending on one VMA lock.
char *mapNamedContiguous(size_t total, size_t chunk, unsigned numChunks,
                         size_t page, size_t *mapLenOut) {
  size_t mapLen = ((total + page - 1) / page) * page;
  void *res = mmap(nullptr, mapLen, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (res == MAP_FAILED)
    throw std::runtime_error(
        std::string("ChunkedStagingRegion: mmap failed: ") +
        std::strerror(errno));
  char *span = static_cast<char *>(res);
#if defined(__linux__)
  constexpr size_t kVmaNameMax = 32;
  for (unsigned i = 0; i < numChunks; ++i) {
    size_t off = static_cast<size_t>(i) * chunk;
    size_t nameLen = std::min(chunk, mapLen - off);
    char nm[kVmaNameMax];
    std::snprintf(nm, sizeof(nm), "maxpinned%u", i);
    if (prctl(kPrSetVma, kPrSetVmaAnonName, span + off, nameLen, nm) != 0) {
      // All-or-nothing, and only a perf optimization.
      MLOG_WARN("ChunkedStagingRegion: naming anonymous VMA failed ({}); "
                "continuing without per-chunk VMA names",
                std::strerror(errno));
      break;
    }
  }
#else
  (void)chunk;
  (void)numChunks;
#endif
  *mapLenOut = mapLen;
  return span;
}

// Reached from a destructor, and releasing anyway is the lesser leak.
void drain(const DeviceContextRef &dctx) {
  if (ErrorOrSuccess e = dctx->synchronize(); e.isError())
    MLOG_WARN("ChunkedStagingRegion release: synchronize failed ({})",
              e.getError());
  if (ErrorOrSuccess e = dctx->pollPendingEvents(); e.isError())
    MLOG_WARN("ChunkedStagingRegion release: polling events failed ({})",
              e.getError());
}

// Dropping a wrap only queues its unregister, and handlers are polled rather
// than pushed, so each step needs its own drain. Blocking is fine in teardown.
void releaseRegion(const DeviceContextRef &dctx,
                   std::vector<M::Driver::DeviceBufferRef> &wraps, char *base,
                   size_t mapLen) {
  drain(dctx); // in-flight DMA must land before the wraps go
  wraps.clear();
  drain(dctx); // queued unregisters must run before the pages go
  munmap(base, mapLen);
}

void releaseDriverChunks(const DeviceContextRef &dctx,
                         std::vector<void *> &chunks) {
  drain(dctx); // in-flight DMA must land before the pages go
#if !MODULAR_HAL_DEVICE_CONTEXT
  for (void *chunk : chunks)
    if (ErrorOrSuccess e = dctx->device().chunkFree(chunk, /*isHost=*/true);
        e.isError())
      MLOG_WARN("ChunkedStagingRegion release: freeing a chunk failed ({})",
                e.getError());
#else
  // M::Driver::Device has no chunkFree under --config=hal-device-context;
  // unreachable in practice, since the constructor refuses driverAlloc
  // before any chunk is allocated there.
#endif
  chunks.clear();
}

// Everything below feeds `verifyDriverOwned`, called only from the
// chunkAlloc-dependent half of the constructor's `driverAlloc` branch, which
// does not exist under --config=hal-device-context (see there); unreachable
// and unused on that build without this guard.
#if !MODULAR_HAL_DEVICE_CONTEXT

#if defined(__linux__)
// Returns the VmFlags line from /proc/self/smaps for the mapping containing
// `addr`, or nullopt if no containing mapping is found or smaps couldn't be
// read (VmFlags needs CONFIG_PROC_PAGE_MONITOR, which is the common case but
// not guaranteed).
std::optional<std::string> hostMappingVmFlags(const void *addr) {
  std::ifstream smaps("/proc/self/smaps");
  if (!smaps)
    return std::nullopt;

  auto want = reinterpret_cast<uintptr_t>(addr);
  std::string line;
  bool inRange = false;
  while (std::getline(smaps, line)) {
    uintptr_t start, end;
    if (std::sscanf(line.c_str(), "%lx-%lx", &start, &end) == 2) {
      inRange = want >= start && want < end;
      continue;
    }
    if (inRange && line.compare(0, 9, "VmFlags: ") == 0)
      return line.substr(9);
  }
  return std::nullopt;
}

// Hardware-confirmed on MI355X by forced-migration testing (reproduced
// twice): a driver-owned GTT mapping (`hipHostMalloc` with
// `HSA_USERPTR_FOR_PAGED_MEM=0`) always carries the `pf` (VM_PFNMAP) flag;
// ordinary anonymous or THP-backed memory never does, regardless of `hg`,
// `ac`, `Rss`, or (confirmed against CUDA pinned host memory) `sh`/`ms`,
// which multiple unsafe mappings can show. `mm` (VM_MIXEDMAP) is kept
// alongside `pf` for tolerance against a future driver revision, though no
// measured mapping has shown it. Defense-in-depth alongside
// `movePagesVerdict` below, not the primary gate.
bool looksDriverOwned(const std::string &vmFlags) {
  std::istringstream iss(vmFlags);
  std::string flag;
  while (iss >> flag)
    if (flag == "pf" || flag == "mm")
      return true;
  return false;
}

// A missing `status` means the move_pages(2) call itself could not run
// (e.g. denied by a seccomp filter), distinct from it running and reporting
// a real NUMA node -- `syscallError` names which, for diagnostics.
struct MovePagesResult {
  std::optional<int> status;
  std::string syscallError;
};

// Queries (does not move) the NUMA node of the page at `addr` via
// move_pages(2) with nodes=nullptr.
MovePagesResult queryMovePagesStatus(void *addr) {
  void *pages[1] = {addr};
  int status[1] = {0};
  if (syscall(SYS_move_pages, /*pid=*/0, /*count=*/1, pages,
              /*nodes=*/nullptr, status, /*flags=*/0) != 0)
    return {std::nullopt, std::strerror(errno)};
  return {status[0], {}};
}

// Hardware-confirmed on MI355X: a driver-owned PFNMAP page has no `struct
// page` to report a NUMA node for, so the query faults with exactly
// `-EFAULT`. Ordinary (including userptr-fallback) memory reports a real
// node id -- but so does an ordinary page that simply has not been faulted
// in yet (per move_pages(2), `-EFAULT` also covers "this is a zero page or
// the memory area is not mapped by the process"), and `driverAlloc` has no
// touch phase, so `-EFAULT` alone cannot distinguish "driver-owned" from
// "untouched userptr memory". `looksDriverOwned`'s `pf` check is what closes
// that gap, which is why it is required even when move_pages is
// authoritative, not merely a backup for when it isn't.
//
// move_pages(2) is also a privileged NUMA syscall (requires CAP_SYS_NICE
// even to only query, per kernel/mm/migrate.c's permission check) and is
// denied outright (EPERM) under the seccomp profile BuildBuddy's remote
// executors run with, and under a hardened Kubernetes pod with dropped
// capabilities -- in which case `looksDriverOwned` is the only operative
// signal in production, not a backup. AsyncRT/unittests/AllocatorTest.cpp
// hits the same denial and already treats a negative status as "can't
// tell," not as a verdict. So: a reported status is one necessary condition
// (paired with `looksDriverOwned`); a syscall denial is inconclusive and
// falls through to `looksDriverOwned` alone, never to blind acceptance.
enum class MovePagesVerdict { kDriverOwned, kNotDriverOwned, kInconclusive };

MovePagesVerdict movePagesVerdict(const MovePagesResult &result) {
  if (!result.status.has_value())
    return MovePagesVerdict::kInconclusive;
  return *result.status == -EFAULT ? MovePagesVerdict::kDriverOwned
                                   : MovePagesVerdict::kNotDriverOwned;
}

// Verifies the first driver-allocated chunk is really driver-owned GTT
// memory rather than ordinary movable userptr memory -- the failure mode
// where `HSA_USERPTR_FOR_PAGED_MEM=0` is missing from the process
// environment, `hipHostMalloc` silently falls back to userptr memory, and a
// deployment pays the full startup cost and GTT consumption for none of the
// compaction protection it exists for. Measured: forcing migration of a
// registered (non-driver-owned) region stalled a concurrent GPU copy loop
// for roughly 0.96 s, against roughly 0.02 s with no migration forced; the
// `hipHostMalloc`-without-the-env-var fallback was also migratable and also
// stalled the copy loop, less severely. So this is a real regression to
// catch, not a missed optimization to warn about. Returns an empty string
// if the check passes, else a failure reason to report to the caller.
//
// HIP-only: the failure mode this guards against is specific to
// `hipHostMalloc`'s silent fallback to userptr memory when
// `HSA_USERPTR_FOR_PAGED_MEM` is unset. CUDA's `cuMemAllocHost` (what
// `chunkAlloc(isHost=true)` calls on that backend) genuinely page-locks the
// allocation with no equivalent opt-in or fallback, so it is pinned -- and
// therefore compaction-immune -- by construction; there is nothing to
// verify, and the AMD-specific VmFlags/move_pages signatures measured above
// do not characterize it (CUDA pinned memory is ordinary movable-looking
// anonymous memory from the kernel's point of view once you subtract the
// pin, so this check would reject it for the wrong reason).
std::string verifyDriverOwned(void *firstChunk, Device::API api) {
  if (api != Device::hipAPI)
    return {};
  std::optional<std::string> vmFlags = hostMappingVmFlags(firstChunk);
  MovePagesResult movePages = queryMovePagesStatus(firstChunk);
  MovePagesVerdict verdict = movePagesVerdict(movePages);
  bool vmFlagsOk = vmFlags && looksDriverOwned(*vmFlags);
  std::string movePagesDesc = movePages.status
                                  ? std::to_string(*movePages.status)
                                  : "call failed: " + movePages.syscallError;

  MLOG_INFO("ChunkedStagingRegion driver_alloc: first chunk at {} "
            "VmFlags=[{}] move_pages_status=[{}]",
            firstChunk, vmFlags.value_or("<unavailable>"), movePagesDesc);

  bool ok = verdict == MovePagesVerdict::kInconclusive
                ? vmFlagsOk
                : (verdict == MovePagesVerdict::kDriverOwned && vmFlagsOk);
  if (ok)
    return {};
  return "move_pages status=[" + movePagesDesc + "], VmFlags=[" +
         vmFlags.value_or("<unavailable>") + "]";
}
#else
std::string verifyDriverOwned(void *, Device::API) { return {}; }
#endif // defined(__linux__)

#endif // !MODULAR_HAL_DEVICE_CONTEXT

} // namespace

Python::ChunkedStagingRegion::ChunkedStagingRegion(
    size_t byteSize, size_t rowBytes, PyDeviceTy device, unsigned threads,
    size_t chunkBytes, bool driverAlloc, bool mlockPages)
    : device_(std::move(device)), byteSize_(byteSize), rowBytes_(rowBytes) {
  if (device_->isHost())
    throw std::invalid_argument(
        "ChunkedStagingRegion requires a non-host device "
        "(GPU/Accelerator).");
  if (byteSize == 0)
    throw std::invalid_argument("ChunkedStagingRegion: byte_size must be "
                                "> 0");
  if (rowBytes == 0)
    throw std::invalid_argument("ChunkedStagingRegion: row_bytes must be "
                                "> 0");
  if (byteSize % rowBytes != 0)
    throw std::invalid_argument(
        "ChunkedStagingRegion: byte_size " + std::to_string(byteSize) +
        " is not a multiple of row_bytes " + std::to_string(rowBytes));
  if (chunkBytes == 0)
    throw std::invalid_argument("ChunkedStagingRegion: chunk_bytes must be "
                                "> 0");
  if (driverAlloc && mlockPages)
    throw std::invalid_argument(
        "ChunkedStagingRegion: mlock_pages applies to the registered path; "
        "driver_alloc memory is already owned by the driver");

#if !defined(__linux__)
  MLOG_WARN("ChunkedStagingRegion is tuned for Linux; on this platform the "
            "per-chunk VMA-naming optimization is unavailable");
#endif

  DeviceContextRef dctx = device_->ctx;
  M::AsyncRT::CPUDeviceRef cpuDeviceRef =
      Python::getCpuDevice(0)->getAsyncRTCPUDevice();

  size_t page = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
  // A boundary inside a row breaks transfers; a boundary mid-page makes two
  // chunks share a page and the second wrap is rejected as overlapping.
  size_t alignment = std::lcm(page, rowBytes);
  chunkBytes_ =
      std::max<size_t>(alignment, (chunkBytes / alignment) * alignment);
  unsigned numChunks =
      static_cast<unsigned>((byteSize + chunkBytes_ - 1) / chunkBytes_);

  if (driverAlloc) {
#if MODULAR_HAL_DEVICE_CONTEXT
    // M::Driver::Device has no chunkAlloc/chunkFree under
    // --config=hal-device-context, so there is no way to honor this request;
    // fail loudly rather than silently falling back to the registered path
    // (which would defeat the point of asking for driver_alloc in the first
    // place).
    throw std::runtime_error(
        "ChunkedStagingRegion: driver_alloc is not available in this build "
        "configuration (--config=hal-device-context); use the registered "
        "path instead.");
#else
    std::vector<void *> chunks;
    chunks.reserve(numChunks);
    std::string allocErrMsg;
    std::string verifyErrMsg;
    {
      nb::gil_scoped_release nogil;
      auto allocChunk = [&](unsigned i) -> bool {
        size_t len = std::min(chunkBytes_, byteSize - i * chunkBytes_);
        void *chunk = nullptr;
        // `chunkAlloc` rather than `createHostBuffer`: the latter serves
        // requests below the host memory manager's cap out of the pool that
        // per-step pinned staging relies on.
        if (ErrorOrSuccess e =
                dctx->device().chunkAlloc(&chunk, len, /*isHost=*/true);
            e.isError()) {
          allocErrMsg = "chunk " + std::to_string(i) + " of " +
                        std::to_string(numChunks) + " (" + std::to_string(len) +
                        " bytes, after " + std::to_string(i * chunkBytes_) +
                        " bytes allocated): " + std::string(e.getError());
          return false;
        }
        chunks.push_back(chunk);
        return true;
      };

      // Verifying only the first chunk, before allocating the rest, matters:
      // at a production-sized region this is the difference between
      // catching a misconfiguration in milliseconds versus after minutes of
      // allocation and the whole node's shared GTT budget spent on memory
      // that was never going to pass. It also closes a narrow but real
      // window: `HSA_USERPTR_FOR_PAGED_MEM` set after HIP has already
      // initialized has no effect, so a correctly-set-looking environment
      // can still yield movable memory, and that is exactly the case this
      // check exists to catch -- it should fire as early as possible.
      if (allocChunk(0)) {
        verifyErrMsg = verifyDriverOwned(chunks[0], dctx->device().api());
        if (verifyErrMsg.empty())
          for (unsigned i = 1; i < numChunks; ++i)
            if (!allocChunk(i))
              break;
      }
      if (!allocErrMsg.empty() || !verifyErrMsg.empty())
        releaseDriverChunks(dctx, chunks);
    }
    if (!allocErrMsg.empty())
      throw std::runtime_error("ChunkedStagingRegion: driver host allocation "
                               "of " +
                               std::to_string(byteSize) + " bytes failed at " +
                               allocErrMsg);
    if (!verifyErrMsg.empty())
      throw std::runtime_error(
          "ChunkedStagingRegion: driver_alloc chunk is not driver-owned GTT "
          "memory (" +
          verifyErrMsg +
          "). HSA_USERPTR_FOR_PAGED_MEM=0 must be set in the process "
          "environment before the process starts, since HIP reads it only "
          "at initialization; hipHostRegister-based registration (the "
          "non-driver_alloc path) is unaffected by this variable.");
    driverChunks_ = std::move(chunks);
#endif
    return;
  }

  base_ = mapNamedContiguous(byteSize, chunkBytes_, numChunks, page, &mapLen_);

  if (mlockPages) {
    // Keeps these pages off the kernel's reclaim lists. Compaction still
    // migrates unevictable pages unless the node sets
    // vm.compact_unevictable_allowed=0, which the caller reports.
    if (mlock(base_, mapLen_) != 0) {
      std::string err = std::strerror(errno);
      munmap(base_, mapLen_);
      base_ = nullptr;
      throw std::runtime_error(
          "ChunkedStagingRegion: mlock of " + std::to_string(mapLen_) +
          " bytes failed: " + err +
          "; raise RLIMIT_MEMLOCK (ulimit -l) or grant CAP_IPC_LOCK");
    }
    mlocked_ = true;
  }

  std::vector<Slice> slices(numChunks);
  for (unsigned i = 0; i < numChunks; ++i) {
    size_t off = static_cast<size_t>(i) * chunkBytes_;
    slices[i] = {base_ + off, std::min(chunkBytes_, byteSize - off)};
  }

  using AsyncChain = M::AsyncRT::AsyncValueRef<M::AsyncRT::Chain>;

  std::vector<AsyncChain> touchDone;
  std::vector<AsyncChain> wrapDone;
  touchDone.reserve(numChunks);
  wrapDone.reserve(numChunks);
  for (unsigned i = 0; i < numChunks; ++i) {
    touchDone.push_back(AsyncChain::allocate(*cpuDeviceRef));
    wrapDone.push_back(AsyncChain::allocate(*cpuDeviceRef));
  }

  std::vector<M::Driver::DeviceBufferRef> wraps(numChunks);
  std::atomic<bool> anyFailed{false};
  std::string firstErrMsg;
  std::mutex errMu;

  auto wrapChunk = [&](unsigned i) {
    // Metal rejects a non-page-multiple length, which the last chunk usually
    // has. mmap already page-rounded the mapping, so this stays inside it.
    size_t wrapLen = ((slices[i].second + page - 1) / page) * page;
    ErrorOr<M::Driver::DeviceBufferRef> wrapOr =
        dctx->wrapHostMemory(slices[i].first, wrapLen);
    if (wrapOr.isError()) {
      std::lock_guard<std::mutex> lk(errMu);
      if (!anyFailed.exchange(true, std::memory_order_relaxed))
        firstErrMsg = wrapOr.getError();
      return;
    }
    wraps[i] = wrapOr.takeValue();
  };

  constexpr int kWrapWorker = 0;
  const size_t numWorkers = cpuDeviceRef->getWorkQueue()->getParallelismLevel();
  // Extra touch lanes only contend on memory bandwidth, so leave kWrapWorker.
  const size_t touchLanes =
      numWorkers > 1 ? std::min<size_t>(std::max(1u, threads), numWorkers - 1)
                     : 0;

  auto enqueueWrap = [&](unsigned i) {
    M::AsyncRT::addTask(
        *cpuDeviceRef,
        [&, i]() {
          wrapChunk(i);
          wrapDone[i].copy().emplace();
        },
        kWrapWorker);
  };

  {
    nb::gil_scoped_release nogil;

    // Gating on the previous wrap keeps wrapping serial while touch runs free.
    touchDone[0].andThenSync([&]() { enqueueWrap(0); });
    for (unsigned i = 1; i < numChunks; ++i) {
      M::AsyncRT::andThenSync(
          std::make_tuple(touchDone[i].copy(), wrapDone[i - 1].copy()),
          [&, i](AsyncChain, AsyncChain) { enqueueWrap(i); });
    }

    M::AsyncRT::parallelForEachNChainWithTaskIds(
        *cpuDeviceRef, static_cast<size_t>(numChunks),
        [touchLanes](size_t i) -> int {
          if (touchLanes == 0)
            return kDefaultTaskId;
          return static_cast<int>(1 + (i % touchLanes));
        },
        [&](size_t i) {
          touchSlice(slices[i].first, slices[i].second, page);
          touchDone[i].copy().emplace();
        });

    M::AsyncRT::await(wrapDone[numChunks - 1]);
  }

  if (anyFailed.load(std::memory_order_acquire)) {
    {
      nb::gil_scoped_release nogil;
      releaseRegion(dctx, wraps, base_, mapLen_);
    }
    base_ = nullptr;
    throw std::runtime_error("ChunkedStagingRegion: host memory wrap failed: " +
                             firstErrMsg);
  }

  wraps_ = std::move(wraps);
}

Python::ChunkedStagingRegion::ChunkedStagingRegion(
    ChunkedStagingRegion &&other) noexcept
    : device_(std::move(other.device_)), wraps_(std::move(other.wraps_)),
      driverChunks_(std::move(other.driverChunks_)), base_(other.base_),
      mapLen_(other.mapLen_), byteSize_(other.byteSize_),
      rowBytes_(other.rowBytes_), chunkBytes_(other.chunkBytes_),
      mlocked_(other.mlocked_) {
  other.driverChunks_.clear();
  other.base_ = nullptr;
}

Python::ChunkedStagingRegion::~ChunkedStagingRegion() {
  if (!driverChunks_.empty()) {
    nb::gil_scoped_release nogil;
    releaseDriverChunks(device_->ctx, driverChunks_);
    return;
  }
  if (base_ == nullptr)
    return;
  nb::gil_scoped_release nogil;
  releaseRegion(device_->ctx, wraps_, base_, mapLen_);
}

uintptr_t Python::ChunkedStagingRegion::address() const {
  if (driverAlloc())
    throw std::runtime_error("ChunkedStagingRegion: a driver-allocated "
                             "region has no single base; use "
                             "chunk_addresses");
  return reinterpret_cast<uintptr_t>(base_);
}

std::vector<uintptr_t> Python::ChunkedStagingRegion::chunkAddresses() const {
  std::vector<uintptr_t> addrs;
  if (driverAlloc()) {
    for (void *chunk : driverChunks_)
      addrs.push_back(reinterpret_cast<uintptr_t>(chunk));
    return addrs;
  }
  for (size_t i = 0; i < wraps_.size(); ++i)
    addrs.push_back(reinterpret_cast<uintptr_t>(base_) + i * chunkBytes_);
  return addrs;
}
