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

#include "HostMemoryInfo.h"
#include "Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace M;
using namespace M::Driver;

namespace {

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

#if MODULAR_LINUX
std::optional<std::string> readSmallFile(llvm::Twine path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::nullopt;
  return std::string(buffer->get()->getBuffer());
}

// Controller file names and memory.stat keys differ between cgroup v1 and v2.
struct Cgroup {
  bool unifiedHierarchy;
  std::string limitPath, usagePath, statPath;
  llvm::StringRef cacheKey, shmemKey;
};

std::optional<size_t> headroomInCgroup(const Cgroup &cgroup) {
  auto limit = readSmallFile(cgroup.limitPath);
  auto usage = readSmallFile(cgroup.usagePath);
  auto stat = readSmallFile(cgroup.statPath);
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

  auto self = readSmallFile("/proc/self/cgroup");
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
#endif // MODULAR_LINUX

} // namespace

ErrorOr<Device::MemInfo>
M::Driver::parseHostMemoryInfo(llvm::StringRef meminfo) {
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
      return Error("Invalid " + key + " field in /proc/meminfo");

    if (key == "MemTotal")
      total = kb * 1024;
    else if (key == "MemAvailable")
      available = kb * 1024;
    else
      free = kb * 1024;
  }

  if (!total || *total == 0)
    return Error("Missing or zero MemTotal field in /proc/meminfo");
  // Older kernels may not expose MemAvailable. Do not count all cached pages:
  // the kernel's estimate excludes cache that cannot actually be reclaimed.
  if (!available)
    available = free;
  if (!available)
    return Error("Missing MemAvailable and MemFree fields in /proc/meminfo");

  Device::MemInfo info;
  info.total = *total;
  info.free = std::min(*available, *total);
  return info;
}

std::optional<size_t> M::Driver::parseCgroupHeadroom(
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

ErrorOr<Device::MemInfo>
M::Driver::getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo,
                                    Device::MemInfo hostMemoryInfo) {
  // An integrated device shares physical memory with the host only when its
  // total covers most of host RAM, allowing for carveouts. Parts like MI300A
  // are integrated but have private memory, and the driver's figures must
  // stay authoritative for them. Divide before multiplying: a malformed
  // meminfo can put the host total close to SIZE_MAX.
  if (deviceMemoryInfo.total < hostMemoryInfo.total / 5 * 4)
    return deviceMemoryInfo;

  // Preserve driver limits, including reserved memory and device-local limits
  // on systems where an integrated GPU can access only part of host memory.
  deviceMemoryInfo.free = std::min(hostMemoryInfo.free, deviceMemoryInfo.total);
  return deviceMemoryInfo;
}

ErrorOr<Device::MemInfo>
M::Driver::getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo) {
#if MODULAR_LINUX
  auto buffer = llvm::MemoryBuffer::getFileAsStream("/proc/meminfo");
  if (buffer.getError())
    return Error("Unable to read /proc/meminfo: " +
                 llvm::Twine(buffer.getError().message()));
  auto hostInfo = parseHostMemoryInfo((*buffer)->getBuffer());
  if (hostInfo.isError())
    return hostInfo.takeError();

  // A cgroup can cap this process well below host availability (a container
  // limited to 32 GiB on a 128 GiB host), so clamp to what it may still be
  // charged before substituting. Mirrors
  // max/python/max/support/host_memory.py.
  if (auto headroom = minCgroupHeadroom())
    hostInfo->free = std::min(hostInfo->free, *headroom);

  return getHostUnifiedMemoryInfo(deviceMemoryInfo, *hostInfo);
#else
  return deviceMemoryInfo;
#endif
}
