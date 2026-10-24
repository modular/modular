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

#pragma once

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <string>

namespace M {
namespace MGP {

/// Generate unique kernel ID to correlate async launches and completions for
/// logging
uint64_t generateKernelId();

/// Get relative timestamp in milliseconds since first kernel launch
double getRelativeTimestampMs();

/// Thread-safe kernel logging function to prevent interleaved output
void logKernelEvent(llvm::StringRef eventType, llvm::StringRef kernelName,
                    llvm::StringRef deviceInfo, uint64_t kernelId,
                    double timestampMs, llvm::raw_ostream &os = llvm::errs());

template <typename OperandContainer>
llvm::SmallVector<M::Driver::DeviceContextRef>
extractDeviceContexts(const OperandContainer &operands);

/// Get device information string for logging (CPU vs GPU:id)
template <typename OperandContainer>
std::string getDeviceInfoString(const OperandContainer &operands);

/// Logging policies to differentiate sync vs async kernel execution
struct SyncLoggingPolicy {
  static constexpr const char *launchEvent() { return "SYNC LAUNCH"; }
  static constexpr const char *completeEvent() { return "SYNC COMPLETE"; }
  static constexpr const char *errorEvent() { return "SYNC ERROR"; }
};

struct AsyncLoggingPolicy {
  static constexpr const char *launchEvent() { return "ASYNC LAUNCH"; }
  static constexpr const char *completeEvent() { return "ASYNC COMPLETE"; }
  static constexpr const char *errorEvent() { return "ASYNC ERROR"; }
};

/// Policy-based instead of virtual to avoid a vtable, keeping overhead low
/// when logging is disabled.
template <typename OperandContainer, typename LoggingPolicy>
class KernelLogger {
public:
  KernelLogger(llvm::StringRef kernelName, const OperandContainer &operands,
               llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> outResults,
               llvm::raw_ostream &os = llvm::errs());

  ~KernelLogger();

  KernelLogger(const KernelLogger &) = delete;
  KernelLogger &operator=(const KernelLogger &) = delete;
  KernelLogger(KernelLogger &&) = delete;
  KernelLogger &operator=(KernelLogger &&) = delete;

private:
  uint64_t kernelId = 0;
  std::string kernelName;
  std::string deviceInfo;
  llvm::ArrayRef<AsyncRT::AnyAsyncValueRef> outResults;
  llvm::raw_ostream &outputStream;
};

template <typename OperandContainer>
using KernelSyncLogger = KernelLogger<OperandContainer, SyncLoggingPolicy>;

template <typename OperandContainer>
using KernelAsyncLogger = KernelLogger<OperandContainer, AsyncLoggingPolicy>;

} // namespace MGP
} // namespace M
