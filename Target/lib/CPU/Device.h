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

#ifndef TARGET_LIB_CPU_DEVICE_H
#define TARGET_LIB_CPU_DEVICE_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "HAL/M_driver_common.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <mutex>

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver) : driver(driver) {}

  // Every queue of every context on this device registers here, so freeing
  // memory can wait for queued work that may still use it, as `cuMemFree`
  // does on CUDA. Queue operations accept memory from any context, so the
  // registry is per device rather than per context.
  void registerQueue(M_driver_queue_t queue);
  void unregisterQueue(M_driver_queue_t queue);
  // Returns the chains of the registered queues that still have work pending.
  llvm::SmallVector<M::AsyncRT::AnyAsyncValueRef, 4> pendingWork();

  M_driver_driver_t driver;

private:
  std::mutex queuesMutex;
  llvm::SmallPtrSet<M_driver_queue_t, 4> queues;
};

#endif // TARGET_LIB_CPU_DEVICE_H
