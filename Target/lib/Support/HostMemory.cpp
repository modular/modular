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

#include "HostMemory.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

using namespace M::Target;

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// cgroup v1 reports "unlimited" as a near-2**63 sentinel rather than "max".
constexpr size_t noCgroupLimit = size_t(1) << 62;

std::optional<size_t> parseInt(llvm::StringRef contents) {
  contents = contents.trim();
  size_t value;
  if (contents.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

std::optional<size_t> parseStatKey(llvm::StringRef stat, llvm::StringRef key) {
  while (!stat.empty()) {
    auto [line, remaining] = stat.split('\n');
    stat = remaining;
    auto [name, value] = line.split(' ');
    if (name.trim() == key)
      return parseInt(value);
  }
  return std::nullopt;
}

#ifdef __linux__
// Controller file names and memory.stat keys differ between cgroup v1 and v2.
struct Cgroup {
  bool unifiedHierarchy;
  std::string limitPath, usagePath, statPath;
  llvm::StringRef cacheKey, shmemKey;
};

std::optional<size_t> headroomInCgroup(const Cgroup &cgroup) {
  auto limit = readKernelFile(cgroup.limitPath);
  auto usage = readKernelFile(cgroup.usagePath);
  auto stat = readKernelFile(cgroup.statPath);
  if (!limit || !usage || !stat)
    return std::nullopt;
  return parseCgroupHeadroom(*limit, *usage, *stat, cgroup.cacheKey,
                             cgroup.shmemKey, cgroup.unifiedHierarchy);
}

// The mount roots miss unit-level limits such as systemd's MemoryMax=,
// since outside a container the cgroup mount is not namespaced;
// /proc/self/cgroup supplies this process's own path.
std::vector<Cgroup> cgroupsBoundingProcess() {
  auto unified = [](std::string dir) {
    return Cgroup{true,
                  dir + "/memory.max",
                  dir + "/memory.current",
                  dir + "/memory.stat",
                  "file",
                  "shmem"};
  };
  auto legacy = [](std::string dir) {
    return Cgroup{false,
                  dir + "/memory.limit_in_bytes",
                  dir + "/memory.usage_in_bytes",
                  dir + "/memory.stat",
                  "total_cache",
                  "total_shmem"};
  };
  std::vector<Cgroup> cgroups{unified("/sys/fs/cgroup"),
                              legacy("/sys/fs/cgroup/memory")};

  auto self = readKernelFile("/proc/self/cgroup");
  if (!self)
    return cgroups;
  llvm::StringRef entries = *self;
  while (!entries.empty()) {
    auto [line, remaining] = entries.split('\n');
    entries = remaining;
    auto [hierarchy, rest] = line.split(':');
    auto [controllers, path] = rest.split(':');
    auto relative = path.trim();
    if (relative.empty() || relative == "/")
      continue;
    if (hierarchy.trim() == "0") {
      cgroups.push_back(unified("/sys/fs/cgroup" + relative.str()));
      continue;
    }
    while (!controllers.empty()) {
      auto [controller, tail] = controllers.split(',');
      if (controller.trim() == "memory") {
        cgroups.push_back(legacy("/sys/fs/cgroup/memory" + relative.str()));
        break;
      }
      controllers = tail;
    }
  }
  return cgroups;
}

std::optional<size_t> minCgroupHeadroom() {
  std::optional<size_t> headroom;
  for (const auto &cgroup : cgroupsBoundingProcess())
    if (auto value = headroomInCgroup(cgroup))
      headroom = std::min(headroom.value_or(*value), *value);
  return headroom;
}
#endif // __linux__

} // namespace

std::optional<std::string> M::Target::readKernelFile(const llvm::Twine &path) {
  // procfs and cgroupfs report a size of 0, so reading by size returns
  // nothing; read until end of file instead.
  auto buffer = llvm::MemoryBuffer::getFileAsStream(path);
  if (!buffer)
    return std::nullopt;
  return std::string(buffer->get()->getBuffer());
}

llvm::Expected<HostMemory>
M::Target::parseLinuxMemInfo(llvm::StringRef meminfo) {
  std::optional<size_t> total, available, free;
  while (!meminfo.empty()) {
    auto [line, remaining] = meminfo.split('\n');
    meminfo = remaining;
    auto [key, value] = line.split(':');
    key = key.trim();
    if (key != "MemTotal" && key != "MemAvailable" && key != "MemFree")
      continue;

    value = value.trim();
    size_t kb;
    if (!value.consume_back("kB") || value.trim().getAsInteger(10, kb) ||
        kb > std::numeric_limits<size_t>::max() / 1024)
      return makeError("Invalid " + key + " field in /proc/meminfo");

    if (key == "MemTotal")
      total = kb * 1024;
    else if (key == "MemAvailable")
      available = kb * 1024;
    else
      free = kb * 1024;
  }

  if (!total || *total == 0)
    return makeError("Missing or zero MemTotal field in /proc/meminfo");
  // Older kernels may not expose MemAvailable. Do not count all cached pages:
  // the kernel's estimate excludes cache that cannot actually be reclaimed.
  if (!available)
    available = free;
  if (!available)
    return makeError(
        "Missing MemAvailable and MemFree fields in /proc/meminfo");

  return HostMemory{*total, std::min(*available, *total)};
}

std::optional<size_t> M::Target::parseCgroupHeadroom(
    llvm::StringRef limit, llvm::StringRef usage, llvm::StringRef stat,
    llvm::StringRef cacheKey, llvm::StringRef shmemKey, bool unifiedHierarchy) {
  auto limitValue = parseInt(limit);
  // cgroup v2 writes "max" for no limit; v1 writes a near-2**63 sentinel.
  if (!limitValue || *limitValue == 0)
    return std::nullopt;
  if (!unifiedHierarchy && *limitValue >= noCgroupLimit)
    return std::nullopt;
  auto usageValue = parseInt(usage);
  if (!usageValue)
    return std::nullopt;

  // Reclaimable page cache is not counted as used, so a server that just
  // read a large checkpoint is not treated as full.
  auto cache = parseStatKey(stat, cacheKey);
  auto shmem = parseStatKey(stat, shmemKey);
  if (cache && shmem && *cache > *shmem)
    *usageValue -= std::min(*cache - *shmem, *usageValue);

  // Usage can briefly exceed the limit during reclaim.
  return *usageValue >= *limitValue ? 0 : *limitValue - *usageValue;
}

HostMemory M::Target::darwinHostMemory(uint64_t total, uint64_t pageSize,
                                       uint64_t freePages,
                                       uint64_t inactivePages) {
  // Inactive pages are the kernel's first reclaim candidates: file cache and
  // anonymous memory not touched recently. Active, wired and compressed
  // memory are not counted; claiming them is what drives macOS into swap.
  uint64_t available = (freePages + inactivePages) * pageSize;
  return HostMemory{total, std::min(available, total)};
}

llvm::Expected<HostMemory> M::Target::queryHostMemory() {
#if defined(__linux__)
  auto buffer = llvm::MemoryBuffer::getFileAsStream("/proc/meminfo");
  if (!buffer)
    return makeError("Unable to read /proc/meminfo: " +
                     llvm::Twine(buffer.getError().message()));
  auto memory = parseLinuxMemInfo((*buffer)->getBuffer());
  if (!memory)
    return memory.takeError();

  // A cgroup can cap this process well below host availability (a container
  // limited to 32 GiB on a 128 GiB host). Mirrors
  // max/python/max/support/host_memory.py.
  if (auto headroom = minCgroupHeadroom())
    memory->available = std::min(memory->available, *headroom);
  return memory;
#elif defined(__APPLE__)
  uint64_t total = 0;
  size_t length = sizeof(total);
  if (sysctlbyname("hw.memsize", &total, &length, nullptr, 0) != 0 ||
      total == 0)
    return makeError("Unable to read hw.memsize");

  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t stats;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statsResult = host_statistics64(
      host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
  mach_port_deallocate(mach_task_self(), host);
  if (pageResult != KERN_SUCCESS || statsResult != KERN_SUCCESS)
    return makeError("Unable to read host VM statistics");

  return darwinHostMemory(total, pageSize, stats.free_count,
                          stats.inactive_count);
#else
  return makeError("Host memory query is not supported on this platform");
#endif
}
