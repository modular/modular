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
#include "Target/lib/Support/HostMemory.h"

#include <algorithm>

using namespace M;
using namespace M::Driver;

ErrorOr<Device::MemInfo> M::Driver::getHostMemoryInfo() {
  auto memory = Target::queryHostMemory();
  if (!memory)
    return Error(llvm::toString(memory.takeError()));
  Device::MemInfo info;
  info.total = memory->total;
  info.free = memory->available;
  return info;
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
  if (hostMemoryInfo.free)
    deviceMemoryInfo.free =
        std::min(*hostMemoryInfo.free, deviceMemoryInfo.total);
  return deviceMemoryInfo;
}

ErrorOr<Device::MemInfo>
M::Driver::getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo) {
#if MODULAR_LINUX
  DEFINE_OR_RETURN_ERROR(Device::MemInfo, hostInfo, getHostMemoryInfo());
  return getHostUnifiedMemoryInfo(deviceMemoryInfo, hostInfo);
#else
  return deviceMemoryInfo;
#endif
}
