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
// This file defines KernelLaunchMetadata, one entry of a device launch trace.
// It lives apart from DeviceContext.h so the trace entry type and the registry
// that collects it stay available under --config=hal-device-context, where
// DeviceContext.h resolves to the HAL shim declarations.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEENTRY_H
#define MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEENTRY_H

#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llvm::json {
class Value;
}

namespace M::Driver {

/// Metadata for a trace entry (kernel launch, memcpy, or memset).
struct KernelLaunchMetadata {
  /// Type of operation that produced this trace entry
  enum class OperationKind : uint8_t {
    KernelLaunch,
    Memcpy,
    Memset,
  };

  /// Memcpy direction/kind
  enum class MemcpyKind : uint8_t { None = 0, HtoD, DtoH, DtoD };

  /// The kind of operation this metadata represents
  OperationKind kind = OperationKind::KernelLaunch;

  /// The name of the operation (kernel name, "cuMemcpyHtoD", "cuMemset", etc.)
  std::string kernelName;

  /// The hash value recorded during tracing.
  uint64_t semanticHash = 0;

  // ===== Kernel launch fields (only set for kernel launches) =====
  uint32_t gridX = 0, gridY = 0, gridZ = 0;
  uint32_t blockX = 0, blockY = 0, blockZ = 0;
  uint32_t sharedMemBytes = 0;
  uint32_t numAttrs = 0;
  uint32_t streamIndex = 0;

  // ===== Memcpy fields (only set for memcpy operations) =====
  MemcpyKind memcpyKind = MemcpyKind::None;
  size_t memcpyByteSize = 0;

  // ===== Memset fields (only set for memset operations) =====
  size_t memsetByteSize = 0;
  uint64_t memsetValue = 0;
  size_t memsetValueSize = 0;

  /// Default constructor
  KernelLaunchMetadata() = default;

  /// Constructor for simple operation (just name and hash)
  KernelLaunchMetadata(llvm::StringRef name, uint64_t hashValue)
      : kind(OperationKind::KernelLaunch), kernelName(name.str()),
        semanticHash(hashValue) {}

  /// Constructor for kernel launch with full metadata
  KernelLaunchMetadata(llvm::StringRef name, uint64_t hashValue, uint32_t gridX,
                       uint32_t gridY, uint32_t gridZ, uint32_t blockX,
                       uint32_t blockY, uint32_t blockZ,
                       uint32_t sharedMemBytes, uint32_t numAttrs,
                       uint32_t streamIndex)
      : kind(OperationKind::KernelLaunch), kernelName(name.str()),
        semanticHash(hashValue), gridX(gridX), gridY(gridY), gridZ(gridZ),
        blockX(blockX), blockY(blockY), blockZ(blockZ),
        sharedMemBytes(sharedMemBytes), numAttrs(numAttrs),
        streamIndex(streamIndex) {}

  /// Constructor for memcpy operation
  KernelLaunchMetadata(llvm::StringRef name, uint64_t hashValue,
                       MemcpyKind memcpy, size_t byteSize)
      : kind(OperationKind::Memcpy), kernelName(name.str()),
        semanticHash(hashValue), memcpyKind(memcpy), memcpyByteSize(byteSize) {}

  /// Constructor for memset operation
  KernelLaunchMetadata(llvm::StringRef name, uint64_t hashValue,
                       size_t byteSize, uint64_t value, size_t valSize)
      : kind(OperationKind::Memset), kernelName(name.str()),
        semanticHash(hashValue), memsetByteSize(byteSize), memsetValue(value),
        memsetValueSize(valSize) {}

  /// Comparison operator: sort by kernel name first, then by hash
  bool operator<(const KernelLaunchMetadata &other) const {
    if (kernelName != other.kernelName)
      return kernelName < other.kernelName;
    return semanticHash < other.semanticHash;
  }

  /// Equality operator for verification
  bool operator==(const KernelLaunchMetadata &other) const {
    return kernelName == other.kernelName && semanticHash == other.semanticHash;
  }

  bool operator!=(const KernelLaunchMetadata &other) const {
    return !(*this == other);
  }

  /// Format this trace entry as a human-readable string
  std::string format(int count = 1) const;

  /// Format this trace entry as JSON object
  llvm::json::Value formatJSON() const;
};

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_LAUNCHTRACEENTRY_H
