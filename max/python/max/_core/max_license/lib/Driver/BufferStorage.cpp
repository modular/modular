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

#include "_core/Driver/BufferStorage.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"

#include <cstddef>
#include <memory>
#include <utility>

using namespace M;
using namespace Driver;

//===----------------------------------------------------------------------===//
// BufferStorage
//===----------------------------------------------------------------------===//

Python::BufferStorage::BufferStorage(M::Driver::DeviceBufferRef buf,
                                     TensorSpec spec, Usage usage)
    : buf(std::move(buf)), spec(std::move(spec)), usage(usage) {}

void *Python::BufferStorage::data() const { return buf->ptr(); }

size_t Python::BufferStorage::bytecount() const { return buf->byteSize(); }

const TensorSpec &Python::BufferStorage::getSpec() const { return spec; }

bool Python::BufferStorage::isHost() const {
  return buf->isHost() ||
         buf->context().device().api() == M::Driver::Device::cpuAPI;
}

bool Python::BufferStorage::isPinned() const { return buf->isHost(); }

ErrorOrSuccess Python::BufferStorage::memset(int value) {
  return buf->context().enqueueSetMemory(*buf, value, 1);
}

//===----------------------------------------------------------------------===//
// Storage views
//===----------------------------------------------------------------------===//

ErrorOr<std::shared_ptr<Python::BufferStorage>>
Python::createStorageView(const Python::BufferStorage &storage,
                          const TensorSpec &newSpec, size_t byteOffset) {
  if ((newSpec.getSizeInBytes() + byteOffset) > storage.spec.getSizeInBytes())
    RETURN_ERROR("Can't create a view larger than the initial buffer");

  // If the requested view is offsetted or has a different size, create a
  // sub-buffer.
  if (byteOffset || newSpec.getSizeInBytes() != storage.spec.getSizeInBytes()) {
    ErrorOr<M::Driver::DeviceBufferRef> newBufOr =
        storage.buf->createSubBuffer(byteOffset, newSpec.getSizeInBytes());
    if (newBufOr.isError())
      return newBufOr.takeError();
    // A view inherits the source storage's usage; it aliases the same
    // memory, so the allocation intent is unchanged.
    return std::make_shared<Python::BufferStorage>(newBufOr.takeValue(),
                                                   newSpec, storage.usage);
  }

  return std::make_shared<Python::BufferStorage>(storage.buf, newSpec,
                                                 storage.usage);
}

ErrorOr<std::shared_ptr<Python::BufferStorage>>
Python::createStorageViewOn(const Python::BufferStorage &storage,
                            const M::Driver::DeviceContextRef &targetCtx,
                            M::Driver::DeviceEvent &after) {
  ErrorOr<M::Driver::DeviceBufferRef> newBufOr =
      storage.buf->createSubBuffer(0, storage.buf->byteSize());
  if (newBufOr.isError())
    return newBufOr.takeError();
  M::Driver::DeviceBufferRef newBuf = newBufOr.takeValue();
  if (ErrorOrSuccess err = targetCtx->enqueueWaitForEvent(after); err.isError())
    return err.takeError();
  newBuf->setContext(*targetCtx);
  // Retargeting to another context is still a view over the same memory, so
  // the allocation intent carries over unchanged.
  return std::make_shared<Python::BufferStorage>(std::move(newBuf),
                                                 storage.spec, storage.usage);
}

std::shared_ptr<Python::BufferStorage>
Python::createBorrowedStorage(const std::shared_ptr<BufferStorage> &storage) {
  M::Driver::DeviceContext &ctx = storage->buf->context();
  M::Driver::DeviceBufferRef borrowedBuffer =
      storage->isPinned()
          ? ctx.createHostBufferOwning(storage->data(), storage->bytecount(),
                                       /*owning=*/false)
          : ctx.createBufferOwning(storage->data(), storage->bytecount(),
                                   /*owning=*/false);
  // A borrowed wrapper is a non-owning alias of the same memory, so it
  // inherits the source storage's usage rather than deciding one of its own.
  return std::make_shared<Python::BufferStorage>(std::move(borrowedBuffer),
                                                 storage->spec, storage->usage);
}
