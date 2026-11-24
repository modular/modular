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

#ifndef TARGET_LIB_CPU_EVENT_H
#define TARGET_LIB_CPU_EVENT_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "HAL/M_driver_common.h"

namespace AsyncRT = M::AsyncRT;

/// A CPU event is a handle to a point in a queue's async chain. Recording the
/// event captures the queue's current chain value; synchronizing/querying it
/// awaits or inspects that async value. This mirrors M::Driver::CpuDeviceEvent.
struct M_driver_event {
  M_driver_event(AsyncRT::CPUDeviceRef cpuDevice,
                 AsyncRT::AnyAsyncValueRef value)
      : cpuDevice(std::move(cpuDevice)), value(std::move(value)) {}

  AsyncRT::CPUDeviceRef cpuDevice;
  AsyncRT::AnyAsyncValueRef value;
};

#endif // TARGET_LIB_CPU_EVENT_H
