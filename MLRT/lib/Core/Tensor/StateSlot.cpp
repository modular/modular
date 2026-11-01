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

#include "MLRT/Core/Tensor/StateSlot.h"

#include "MLRT/Core/Tensor/TensorBufferRef.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

TensorBufferRef
TensorStateSlot::getOrCreateBuffer(CompactCPUDevicePtr cpuDevice, size_t size,
                                   std::optional<size_t> alignment) {
  updateCacheIfNeeded(bufferCache, cpuDevice, size, alignment);
  // Return a borrowed reference to the cached buffer.
  return bufferCache.borrow();
}

ErrorOr<TensorBufferRef> TensorStateSlot::getBuffer() {
  if (!bufferCache.getBuffer() && !bufferCache.getMemStorageHandleRef())
    return Error("no buffer has been cached");
  // Return a borrowed reference to the cached buffer. Since it's in the
  // cache, and the compiler follows the 'one shot' rules for reallocating
  // buffers, we can give the borrow 'execute' lifetime.
  return bufferCache.borrow();
}

ErrorOr<TensorBufferRef &> TensorStateSlot::getBufferRef() {
  if (!bufferCache.getBuffer() && !bufferCache.getMemStorageHandleRef())
    return Error("no buffer has been cached");
  return bufferCache;
}

ErrorOrSuccess TensorStateSlot::setBuffer(TensorBufferRef bufferRef) {
  if (bufferCache.getBuffer())
    return Error("buffer has already been cached");
  bufferCache = std::move(bufferRef);
  return success();
}

void TensorStateSlot::updateCacheIfNeeded(TensorBufferRef &entry,
                                          CompactCPUDevicePtr cpuDevice,
                                          size_t size,
                                          std::optional<size_t> alignment) {
  // Do nothing if size matches. The size will be 0 if the entry is
  // uninitialized, thus forcing an allocation.
  if (entry.getSize() == size)
    return;

  // Free the existing buffer if exists and create a new one.
  // Since the compiler follows the 'one shot' rules for reallocating
  // buffers, this cannot invalidate any existing borrows.
  entry = TensorBufferRef::createOwned(cpuDevice, size, alignment);
}
