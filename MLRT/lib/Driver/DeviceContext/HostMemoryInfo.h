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

#ifndef MLRT_DRIVER_DEVICECONTEXT_HOSTMEMORYINFO_H
#define MLRT_DRIVER_DEVICECONTEXT_HOSTMEMORYINFO_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace M::Driver {

/// Parses Linux memory counters, including reclaimable memory in the free size.
ErrorOr<Device::MemInfo> parseHostMemoryInfo(llvm::StringRef meminfo);

/// Parses one memory cgroup's limit, usage, and memory.stat contents and
/// returns what it may still be charged, or nullopt when it imposes no
/// limit.
std::optional<size_t>
parseCgroupHeadroom(llvm::StringRef limit, llvm::StringRef usage,
                    llvm::StringRef stat, llvm::StringRef cacheKey,
                    llvm::StringRef shmemKey, bool unifiedHierarchy);

/// Returns host availability for a device that shares physical memory with
/// the host, capped at the device's capacity. Devices with their own memory,
/// such as integrated parts with private HBM, keep the caller's figures.
ErrorOr<Device::MemInfo>
getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo);

/// Same, with the host figures supplied by the caller, e.g. from
/// parseHostMemoryInfo.
ErrorOr<Device::MemInfo>
getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo,
                         Device::MemInfo hostMemoryInfo);

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_HOSTMEMORYINFO_H
