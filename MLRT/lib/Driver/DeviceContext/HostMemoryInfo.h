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

namespace M::Driver {

/// Returns host physical memory, with system-wide availability as the free
/// size. See M::Target::queryHostMemory.
ErrorOr<Device::MemInfo> getHostMemoryInfo();

/// Returns host availability for a device that shares physical memory with
/// the host, capped at the device's capacity. Devices with their own memory,
/// such as integrated parts with private HBM, keep the caller's figures.
ErrorOr<Device::MemInfo>
getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo);

/// Same, with the host figures supplied by the caller.
ErrorOr<Device::MemInfo>
getHostUnifiedMemoryInfo(Device::MemInfo deviceMemoryInfo,
                         Device::MemInfo hostMemoryInfo);

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_HOSTMEMORYINFO_H
