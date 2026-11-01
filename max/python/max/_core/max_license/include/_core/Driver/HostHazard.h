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

#ifndef SDK_DRIVERAPI_PYTHON_HOSTHAZARD_H
#define SDK_DRIVERAPI_PYTHON_HOSTHAZARD_H

#include "_core/Driver/Device.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace M::Driver::Python {

/// One device-side read or write whose completion the host can wait on, or a
/// poisoned stand-in carrying the producer's error instead of an event.
struct HostHazardCompletion {
  PyDeviceEventTy event;
  PyDeviceStreamTy queue;
  std::string producerError;

  uintptr_t queueKey() const;

  // Python facing creation method for a completion where the framework needs to
  // record read / write events.
  static std::shared_ptr<HostHazardCompletion> recordOn(PyDeviceStreamTy queue);

  // Python facing creation method for a busted completion for error propagation
  static std::shared_ptr<HostHazardCompletion>
  poisoned(PyDeviceStreamTy queue, const std::string &error);

  bool isReady() const;

  void wait() const;

  void poison(const std::string &msg);
};

/// Per-buffer hazard slot: the last device write plus the outstanding reads
/// keyed by queue, so a write only orders behind its own queue's read.
struct HostHazardRecord {
  std::shared_ptr<HostHazardCompletion> lastWrite;
  llvm::SmallDenseMap<uintptr_t, std::shared_ptr<HostHazardCompletion>> readers;
  mutable std::mutex mu;
  bool tainted = false;

  void stampWrite(std::shared_ptr<HostHazardCompletion> completion);

  void stampRead(std::shared_ptr<HostHazardCompletion> completion);

  void taint();

  /// Says "this handle's queue label is wrong", which no completion can
  /// express, so the only safe answer is a full device drain.
  bool isTainted() const;

  /// Cleared only by the drain that settles it.
  void clearTaint();

  std::shared_ptr<HostHazardCompletion> takeLastWriteSnapshot();

  std::vector<std::shared_ptr<HostHazardCompletion>> takeReadersSnapshot();

  void clearSettled(std::shared_ptr<HostHazardCompletion> write,
                    std::vector<std::shared_ptr<HostHazardCompletion>> readers);
};

class HostHazardError : public std::exception {
public:
  explicit HostHazardError(std::string msg);
  const char *what() const noexcept override;

private:
  std::string message;
};

std::unique_ptr<HostHazardRecord>
cloneHostHazardRecord(const HostHazardRecord *record);

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_HOSTHAZARD_H
