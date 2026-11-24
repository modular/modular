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

#include "MLRT/Core/Tensor/TensorBufferRef.h"

#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "llvm/Support/Format.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

static constexpr llvm::MaybeAlign
makeMaybeAlign(std::optional<size_t> alignment) {
  return alignment.has_value() ? llvm::MaybeAlign(*alignment) : std::nullopt;
}

TensorBufferRef TensorBufferRef::create(void *buffer, size_t size,
                                        AnyAsyncValueRef storageHandle,
                                        std::optional<size_t> alignment) {
  return TensorBufferRef::create(buffer, size, std::move(storageHandle),
                                 makeMaybeAlign(alignment));
}

TensorBufferRef TensorBufferRef::create(void *buffer, size_t size,
                                        AnyAsyncValueRef storageHandle,
                                        llvm::MaybeAlign alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create", (uint64_t)size);
  return TensorBufferRef(buffer, size, storageHandle.copy(), alignment,
                         std::move(profilerEntry));
}

TensorBufferRef
TensorBufferRef::createWithNonTrackedMemory(void *buffer, size_t size,
                                            std::optional<size_t> alignment) {
  auto profilerEntry = BufferRefLifetimeProfilerEntry::create(
      "mem.buffer.create.nontracked", (uint64_t)size);
  return TensorBufferRef(buffer, size, /*storageRef=*/{},
                         makeMaybeAlign(alignment), std::move(profilerEntry));
}

TensorBufferRef TensorBufferRef::createOwned(CompactCPUDevicePtr cpuDevice,
                                             size_t size,
                                             std::optional<size_t> alignment) {
  // Special case for empty buffer
  if (size == 0) {
    return TensorBufferRef::createWithNonTrackedMemory(nullptr, size,
                                                       alignment);
  }

  AnyAsyncValueRef storageRef;
  storageRef = storageRef.createReady<OwnedMemoryBlock>(cpuDevice, size,
                                                        alignment, cpuDevice);
  void *buffer = storageRef.get<OwnedMemoryBlock>().data;
  return TensorBufferRef::create(buffer, size, std::move(storageRef),
                                 makeMaybeAlign(alignment));
}

// Create an owned reference from a taken void* pointer -- This taken
// pointer will not be freed by anyone else except by the buffer-ref now.
TensorBufferRef TensorBufferRef::take(CompactCPUDevicePtr cpuDevice,
                                      size_t size, void *takeFrom) {
  AnyAsyncValueRef storageRef;
  storageRef = storageRef.createReady<OwnedMemoryBlock>(
      cpuDevice, size, std::nullopt, cpuDevice, takeFrom);
  void *buffer = storageRef.get<OwnedMemoryBlock>().data;
  return TensorBufferRef::create(buffer, size, std::move(storageRef));
}

TensorBufferRef TensorBufferRef::borrow() const {
  return TensorBufferRef::create(
      data, size, storageRef ? storageRef.copy() : AnyAsyncValueRef{});
}

void TensorBufferRef::reset() {
  data = nullptr;
  size = 0;
  // Though the underlying buffer has not been freed, stop the profiling
  // entry since we can no longer track its lifetime.
  std::move(profilerEntry).record();
}

void TensorBufferRef::printDebug(raw_ostream &os) const {
  os << "TensorBufferRef(";
  os << llvm::format_hex(reinterpret_cast<intptr_t>(getBuffer()),
                         2 + 2 * sizeof(intptr_t));
  os << ", ";
  os << getSize();
  if (storageRef) {
    os << ", ";
    os << *storageRef.getPointer();
  }
  os << ")";
}

void TensorBufferRef::swap(TensorBufferRef &rhs) {
  std::swap(data, rhs.data);
  std::swap(size, rhs.size);
  std::swap(storageRef, rhs.storageRef);
  std::swap(profilerEntry, rhs.profilerEntry);
}

OwnedMemoryBlock::OwnedMemoryBlock(size_t size, std::optional<size_t> alignment,
                                   CompactCPUDevicePtr rt, void *takePtr)
    : data(nullptr), size(size), cpuDevice(rt) {
  if (takePtr)
    data = takePtr;
  else
    data = cpuDevice->getAllocator()->allocateBytes(
        size, alignment.value_or(kPreferredMemoryAlignment));
}

OwnedMemoryBlock::~OwnedMemoryBlock() {
  cpuDevice->getAllocator()->deallocateBytes(data, size);
}
