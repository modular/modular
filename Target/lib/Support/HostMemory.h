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
#ifndef TARGET_LIB_SUPPORT_HOST_MEMORY_H
#define TARGET_LIB_SUPPORT_HOST_MEMORY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace M::Target {

/// Physical host memory, in bytes. `available` is what new allocations can
/// claim once the kernel reclaims memory other processes are not actively
/// using, such as page cache.
struct HostMemory {
  size_t total = 0;
  size_t available = 0;
};

/// Returns host memory as this process sees it: system-wide availability
/// (`MemAvailable` on Linux, free plus inactive pages on macOS, the same
/// figures as psutil's `available`), clamped on Linux to the headroom of the
/// cgroup mount roots and the process's own memory cgroup.
llvm::Expected<HostMemory> queryHostMemory();

/// Reads a procfs or cgroupfs file, or nullopt if it can't be opened.
std::optional<std::string> readKernelFile(const llvm::Twine &path);

/// Parses Linux /proc/meminfo contents.
llvm::Expected<HostMemory> parseLinuxMemInfo(llvm::StringRef meminfo);

/// Parses one memory cgroup's limit, usage, and memory.stat contents and
/// returns what it may still be charged, or nullopt when it imposes no
/// limit.
std::optional<size_t>
parseCgroupHeadroom(llvm::StringRef limit, llvm::StringRef usage,
                    llvm::StringRef stat, llvm::StringRef cacheKey,
                    llvm::StringRef shmemKey, bool unifiedHierarchy);

/// Computes macOS host memory from `hw.memsize` and HOST_VM_INFO64 page
/// counts.
HostMemory darwinHostMemory(uint64_t total, uint64_t pageSize,
                            uint64_t freePages, uint64_t inactivePages);

} // namespace M::Target

#endif // TARGET_LIB_SUPPORT_HOST_MEMORY_H
