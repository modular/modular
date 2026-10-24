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

#include "Common.h"
#include "HAL/M_driver_queue.h"
#include "MetalCpp.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

struct M_driver_queue {
  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context_t context);
  static M_driver_status_t destroy(M_driver_queue_t);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                 const void *src, uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t dstOffset,
                              uint64_t size, uint8_t value);
  M_driver_status_t fillMemory(M_driver_memory_t dst, uint64_t dstOffset,
                               uint64_t size, uint64_t value,
                               uint64_t valueSize);

  M_driver_status_t synchronize();

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvents(M_driver_event_t *events, uint32_t numEvents);

  M_driver_status_t execute(M_driver_function_t function,
                            M_driver_queue_execute_config_t *config,
                            void **args, uint64_t *argSizes, uint32_t numArgs);

  // Host-blocks until everything committed before this call has completed.
  M_driver_status_t drainPriorWork();

  // Commits a blit without waiting. Offsets are absolute within each buffer
  // and must already be bounds-checked.
  M_driver_status_t enqueueBlit(MTL::Buffer *dst, uint64_t dstOffset,
                                MTL::Buffer *src, uint64_t srcOffset,
                                uint64_t size);

  M_driver_context_t context;
  NS::SharedPtr<MTL::CommandQueue> mCommandQueue;

  // Fan-in on ops: they only update it; waiting ANY fence serializes ops.
  NS::SharedPtr<MTL::Fence> mOrderingFence;

  // Fan-out: waitForEvents entries, re-encoded per CB until provably done.
  struct DonatedWait {
    NS::SharedPtr<MTL::Event> event;
    MTL::SharedEvent *sharedView = nullptr; // non-owning; set when peekable
    uint64_t value = 0;
    std::shared_ptr<std::atomic<bool>> retired; // first carrier completed
    bool handlerInstalled = false;
  };
  std::vector<DonatedWait> mDonatedWaits; // guarded by mMutex

  // Drain fast path: equal means nothing was committed since the last drain,
  // so a drain is a no-op (an idle-queue drain CB still costs a ~150us GPU
  // pass round trip). Both guarded by mMutex.
  uint64_t mCommitSeq = 0;
  uint64_t mDrainedSeq = 0;
  // Prunes satisfied entries and encodes the rest; call under mMutex.
  void encodeDonatedWaits(MTL::CommandBuffer *cb);

  std::mutex mMutex;
};
