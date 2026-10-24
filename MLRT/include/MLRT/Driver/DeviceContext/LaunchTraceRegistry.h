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
//
// This file defines the LaunchTraceRegistry class for capturing and tracking
// GPU kernel launch traces across different device backends (CUDA, HIP, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEREGISTRY_H
#define MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEREGISTRY_H

#include "MLRT/Driver/DeviceContext/LaunchTraceEntry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace M {
namespace Driver {

// Categories of GPU operations for launch trace fingerprinting.
// These enum values are used in hash generation and must remain stable.
enum class LaunchTraceKind : uint64_t {
  kMemcpyHtoD = 1, // Host to Device memory copy
  kMemcpyDtoH = 2, // Device to Host memory copy
  kMemcpyDtoD = 3, // Device to Device memory copy
  kMemset = 4,     // Memory set operation
};

// Per-stream launch trace collection state.
struct LaunchTraceState {
  bool enabled = false;
  std::vector<KernelLaunchMetadata> entries;
};

// Global registry for launch-trace capture across device backends.
// Uses type-erased stream handles and ArrayRef for safe, generic interface.
//
// This registry tracks kernel launches and memory operations on GPU streams,
// creating deterministic hashes that exclude memory addresses but capture
// all other launch parameters. This enables detecting changes in GPU work
// patterns for regression testing and debugging.
class LaunchTraceRegistry {
public:
  // Check if launch tracing is enabled for a stream
  bool isLaunchTraceEnabled(void *stream);

  // Append a trace entry to the stream's trace log
  void appendLaunchTraceEntry(void *stream,
                              const KernelLaunchMetadata &metadata);

  // Record a kernel launch trace with full parameter hashing
  //
  // This hashes:
  // - Kernel function pointer (for identity)
  // - Grid and block dimensions
  // - Shared memory size and attribute count
  // - Launch attributes (backend-specific, passed as raw bytes)
  // - Parameter sizes (always)
  // - Parameter values (only for sub-pointer-sized scalars)
  //
  // Memory pointers are excluded to avoid false positives from different
  // allocation addresses across runs.
  void recordLaunchTrace(llvm::StringRef kernelName, uintptr_t functionPtr,
                         llvm::ArrayRef<size_t> paramSizes, uint32_t gridX,
                         uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                         uint32_t blockY, uint32_t blockZ,
                         uint32_t sharedMemBytes, llvm::ArrayRef<uint8_t> attrs,
                         uint32_t numAttrs, llvm::ArrayRef<void *> args,
                         void *stream);

  // Begin tracing for a stream (clears previous entries)
  void beginTraceForStream(void *stream);

  // Take (retrieve and clear) trace entries for a stream
  std::vector<KernelLaunchMetadata> takeTraceForStream(void *stream);

  // Begin a process-global trace that records operations across ALL streams
  // into a single list in enqueue order (clearing any previous global trace).
  // Unlike the per-stream API, no stream handle is required, so work enqueued
  // on streams the caller has no handle to (e.g. a graph's internal stream) is
  // still captured. Each entry's streamIndex identifies its stream, assigned
  // in first-seen order.
  void beginGlobalTrace();

  // End the global trace and return its entries in enqueue order.
  std::vector<KernelLaunchMetadata> takeGlobalTrace();

private:
  // Assigns a stable, first-seen index to a stream handle for the active
  // global trace. Must be called with `mu` held.
  uint32_t streamIndexForLocked(void *stream);

  std::mutex mu;
  llvm::DenseMap<void *, LaunchTraceState> states;

  // Process-global trace state: while `globalEnabled`, every stream's
  // operations are appended to `globalEntries` in enqueue order, each tagged
  // with a first-seen `streamIndices` value.
  bool globalEnabled = false;
  std::vector<KernelLaunchMetadata> globalEntries;
  llvm::DenseMap<void *, uint32_t> streamIndices;
  uint32_t nextStreamIndex = 0;
};

// Get the global launch trace registry singleton
LaunchTraceRegistry &launchTraceRegistry();

// Begin/end a process-global launch trace on the singleton registry. Thin
// wrappers so callers outside MLRT reach the same registry instance the device
// backends record into.
void beginGlobalLaunchTrace();
std::vector<KernelLaunchMetadata> takeGlobalLaunchTrace();

// Hash a memory copy operation for launch tracing.
// Excludes pointer addresses - only includes size and direction.
// This creates deterministic hashes that won't change when memory allocations
// occur at different addresses across runs.
uint64_t hashMemcpyTrace(LaunchTraceKind kind, size_t byteSize);

// Hash a memory set operation for launch tracing.
// Excludes pointer addresses - only includes size, value, and value size.
// The value is masked to only include relevant bits based on valueSize.
uint64_t hashMemsetTrace(size_t byteSize, uint64_t value, size_t valueSize);

} // namespace Driver
} // namespace M

#endif // MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEREGISTRY_H
