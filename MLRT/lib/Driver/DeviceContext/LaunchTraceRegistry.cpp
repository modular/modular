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
// This file implements the LaunchTraceRegistry class for capturing and
// tracking GPU kernel launch traces.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "Support/HashUtils.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/Sequence.h"

#include <cassert>
#include <memory>

using namespace M;
using namespace M::Driver;

// Public API to access the global registry
LaunchTraceRegistry &M::Driver::launchTraceRegistry() {
  static std::unique_ptr<LaunchTraceRegistry> registry =
      std::make_unique<LaunchTraceRegistry>();
  return *registry;
}

bool M::Driver::LaunchTraceRegistry::isLaunchTraceEnabled(void *stream) {
  std::lock_guard<std::mutex> lock(mu);
  if (globalEnabled)
    return true;
  auto it = states.find(stream);
  return it != states.end() && it->second.enabled;
}

void M::Driver::LaunchTraceRegistry::appendLaunchTraceEntry(
    void *stream, const KernelLaunchMetadata &metadata) {
  std::lock_guard<std::mutex> lock(mu);
  if (globalEnabled) {
    KernelLaunchMetadata entry = metadata;
    entry.streamIndex = streamIndexForLocked(stream);
    globalEntries.push_back(std::move(entry));
  }
  auto it = states.find(stream);
  if (it == states.end() || !it->second.enabled)
    return;
  it->second.entries.push_back(metadata);
}

void M::Driver::LaunchTraceRegistry::recordLaunchTrace(
    llvm::StringRef kernelName, uintptr_t functionPtr,
    llvm::ArrayRef<size_t> paramSizes, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemBytes, llvm::ArrayRef<uint8_t> attrs, uint32_t numAttrs,
    llvm::ArrayRef<void *> args, void *stream) {
  raw_xxhash64_stream hasher;

  // Hash deterministic launch parameters
  const uint64_t header[] = {static_cast<uint64_t>(functionPtr),
                             gridX,
                             gridY,
                             gridZ,
                             blockX,
                             blockY,
                             blockZ,
                             sharedMemBytes,
                             numAttrs};
  hasher.write(reinterpret_cast<const char *>(header), sizeof(header));

  // Hash launch attributes (deterministic)
  if (!attrs.empty())
    hasher.write(reinterpret_cast<const char *>(attrs.data()), attrs.size());

  // Hash parameter SIZES and sub-pointer-sized VALUES.
  // Conservative approach: only hash parameters smaller than pointer size.
  // This reliably detects changes in int32_t, float, uint32_t dimensions/flags
  // while avoiding false positives from pointer-sized scalars.
  assert(args.size() == paramSizes.size() &&
         "args and paramSizes must have the same length");
  constexpr size_t kPointerSize = sizeof(void *); // 8 bytes on 64-bit

  for (size_t paramIdx : llvm::seq(paramSizes.size())) {
    size_t size = paramSizes[paramIdx];

    // Always hash the size
    hasher.write(reinterpret_cast<const char *>(&size), sizeof(size));

    const void *paramPtr = args[paramIdx];

    // Hash values for parameters smaller than pointer size (definitely
    // scalars)
    if (size < kPointerSize)
      hasher.write(reinterpret_cast<const char *>(paramPtr), size);
  }

  // Create metadata with kernel name, hash, and launch parameters. streamIndex
  // is assigned at append time for global traces (see appendLaunchTraceEntry);
  // the per-stream path leaves it 0.
  KernelLaunchMetadata metadata(kernelName, hasher.hash(), gridX, gridY, gridZ,
                                blockX, blockY, blockZ, sharedMemBytes,
                                numAttrs, /*streamIndex=*/0);
  appendLaunchTraceEntry(stream, metadata);
}

void M::Driver::LaunchTraceRegistry::beginTraceForStream(void *stream) {
  std::lock_guard<std::mutex> lock(mu);
  auto &state = states[stream];
  state.enabled = true;
  state.entries.clear();
}

std::vector<KernelLaunchMetadata>
M::Driver::LaunchTraceRegistry::takeTraceForStream(void *stream) {
  std::lock_guard<std::mutex> lock(mu);
  auto it = states.find(stream);
  if (it == states.end())
    return {};

  std::vector<KernelLaunchMetadata> result = std::move(it->second.entries);
  states.erase(it);
  return result;
}

void M::Driver::LaunchTraceRegistry::beginGlobalTrace() {
  std::lock_guard<std::mutex> lock(mu);
  globalEnabled = true;
  globalEntries.clear();
  streamIndices.clear();
  nextStreamIndex = 0;
}

std::vector<KernelLaunchMetadata>
M::Driver::LaunchTraceRegistry::takeGlobalTrace() {
  std::lock_guard<std::mutex> lock(mu);
  globalEnabled = false;
  return std::move(globalEntries);
}

uint32_t M::Driver::LaunchTraceRegistry::streamIndexForLocked(void *stream) {
  auto [it, inserted] = streamIndices.try_emplace(stream, nextStreamIndex);
  if (inserted)
    ++nextStreamIndex;
  return it->second;
}

// Exported from libmax: the Python `_core` extension links the DeviceContext
// headers only and resolves these at load time (see the DriverPython target).
MODULAR_CXX_EXPORT void M::Driver::beginGlobalLaunchTrace() {
  launchTraceRegistry().beginGlobalTrace();
}

MODULAR_CXX_EXPORT std::vector<KernelLaunchMetadata>
M::Driver::takeGlobalLaunchTrace() {
  return launchTraceRegistry().takeGlobalTrace();
}

uint64_t M::Driver::hashMemcpyTrace(LaunchTraceKind kind, size_t byteSize) {
  raw_xxhash64_stream hasher;
  const uint64_t header[] = {static_cast<uint64_t>(kind),
                             static_cast<uint64_t>(byteSize)};
  hasher.write(reinterpret_cast<const char *>(header), sizeof(header));
  return hasher.hash();
}

uint64_t M::Driver::hashMemsetTrace(size_t byteSize, uint64_t value,
                                    size_t valueSize) {
  uint64_t maskedValue = value;
  if (valueSize < sizeof(value)) {
    uint64_t mask = (1ULL << (valueSize * 8)) - 1;
    maskedValue &= mask;
  }

  raw_xxhash64_stream hasher;
  const uint64_t header[] = {static_cast<uint64_t>(LaunchTraceKind::kMemset),
                             static_cast<uint64_t>(byteSize), maskedValue,
                             static_cast<uint64_t>(valueSize)};
  hasher.write(reinterpret_cast<const char *>(header), sizeof(header));
  return hasher.hash();
}
