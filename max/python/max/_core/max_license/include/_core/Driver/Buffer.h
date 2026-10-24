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

#ifndef SDK_DRIVERAPI_PYTHON_BUFFER_H
#define SDK_DRIVERAPI_PYTHON_BUFFER_H

#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include "_core/DLPack/Utils.h"
#include "_core/Driver/BufferStorage.h"
#include "_core/Driver/Device.h"
#include "_core/Driver/HostHazard.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace nb = nanobind;

namespace M::Driver::Python {

struct Buffer {
  //===--------------------------------------------------------------------===//
  // Construction and lifetime
  //===--------------------------------------------------------------------===//

  // Movable only
  ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;

  /// Python Constructors: These constructors are exposed to nanobind

  /// Constructor for "canonical" (i.e. contiguous) buffer. This is used when
  /// memory needs to be allocated for the given buffer.
  Buffer(M::DType dtype, ArrayRef<size_t> shape,
         std::optional<PyDeviceTy> device, Usage usage = Usage::Default);

  Buffer(M::DType dtype, ArrayRef<size_t> shape, PyDeviceStreamTy stream,
         Usage usage = Usage::Default);

  /// Constructor for creating a buffer from a Numpy array.
  Buffer(nb::ndarray<nb::ro> arr, PyDeviceTy device);

  /// Internal Constructors: These constructors will only be called from within
  /// our codebase and can use move and value semantics.

  /// Constructor for buffer slices. In this case, the underlying memory is
  /// already allocated. We just need to provide the metadata required to define
  /// the slice.
  Buffer(const TensorSpec &spec, const std::shared_ptr<BufferStorage> &storage,
         PyDeviceStreamTy stream, const std::vector<ssize_t> &strides,
         size_t startOffset, const HostHazardRecord *record);

  /// Constructor for wrapping already-allocated storage together with the
  /// device it's associated with. Used internally for values produced by
  /// `Device::allocate`/`copyTo` and similar.
  Buffer(std::shared_ptr<BufferStorage> storage, PyDeviceTy device,
         const HostHazardRecord *record);

  //===--------------------------------------------------------------------===//
  // Views and DLPack interop
  //===--------------------------------------------------------------------===//

  /// Create a new buffer with the given shape and type, sharing the underlying
  /// storage with an existing buffer. The byte size of the expected buffer must
  /// match that implied by the given shape and type.
  Buffer view(M::DType dtype, const std::vector<size_t> &shape);

  /// Implements the `from_dlpack` api as required by the dlpack contract.
  static Buffer fromDLPack(const nb::object &arrayOrTensor);

  /// Implements a `from_dlpack` api taking the `DLManagedTensor` directly.
  static Buffer
  fromDLManagedTensor(DLPack::DLManagedTensorWrapper &&managed,
                      PyDeviceTy device,
                      std::optional<intptr_t> externalStream = std::nullopt);

  /// Implements the `__dlpack_device__` api as required by the dlpack contract.
  nb::tuple dlPackDevice();

  /// Implements the `__dlpack__` api as required by the dlpack contract.
  static nb::typed<nb::object, nb::capsule>
  dlPack(nb::pointer_and_handle<Buffer> self,
         const std::optional<uintptr_t> stream, nb::kwargs kwargs);

  static nb::typed<nb::object, nb::capsule>
  getAsDlpack(nb::pointer_and_handle<Buffer> self,
              const std::optional<uintptr_t> stream, nb::kwargs kwargs);

  //===--------------------------------------------------------------------===//
  // Shape, dtype and layout
  //===--------------------------------------------------------------------===//

  /// Gets rank of buffer.
  size_t getRank() const;

  /// Gets dtype of elements in buffer.
  M::DType getDType() const;

  /// Return a sequence of shape dimensions.
  nb::tuple getShape() const;

  /// Return how far apart consecutive elements are along each dimension,
  /// counted in elements.
  ///
  /// Slicing does not move data, so a slice keeps its parent's strides: a
  /// view that drops the tail of each row still steps over the full parent
  /// row to reach the next one.
  nb::tuple getStrides() const;

  /// Returns the number of elements in the buffer.
  size_t getNumElements() const;

  /// Gets TensorSpec, which contains the shape and dtype information.
  /// For internal use.
  const TensorSpec &getSpec() const;

  /// Returns whether a buffer is a contiguous slice of a larger buffer.
  bool isContiguousSlice() const;

  /// Determines whether the buffer is contiguously allocated in memory.
  /// There are cases where buffers become non-contiguous due to needing to
  /// iterate with negative strides.
  bool isContiguous() const;

  // Returns whether the memory is aligned.
  bool aligned(int alignment) const;

  //===--------------------------------------------------------------------===//
  // Device, storage and allocation intent
  //===--------------------------------------------------------------------===//

  /// Get device on which buffer is resident.
  PyDeviceTy getDevice() const;

  // Get stream to which the buffer is bound.
  PyDeviceStreamTy getStream() const;

  /// Gets the storage backing the buffer, resolved to an offset-adjusted view
  /// when the buffer is a contiguous slice. This is what feeds the engine when
  /// executing or capturing buffer inputs.
  ///
  /// The returned storage owns a refcounted handle to the underlying
  /// `DeviceBuffer`, so callers keep the memory alive simply by holding it —
  /// no heap-allocated wrapper to free.
  ErrorOr<std::shared_ptr<BufferStorage>> getInputStorage() const;

  void *data() const;

  /// Determines whether the device is on host (i.e. on CPU).
  bool isHost() const;

  /// Whether the allocation landed in the device's host memory space.
  /// `getUsage()` reports what was requested, which can differ.
  bool isPinned() const;

  /// Allocation intent, recorded once and shared by every `Buffer` over the
  /// same storage. Independent of whether the device honored it.
  Usage getUsage() const;

  /// Whether the buffer should check host hazards before host access
  bool isHostHazardTracked() const;
  /// Whether copying `other` into this is a no-op that is safe to skip.
  bool couldElideCopyTo(const Buffer &other) const;

  //===--------------------------------------------------------------------===//
  // Element access and fills
  //===--------------------------------------------------------------------===//

  /// Sets item in buffer. Currently supports only single-element setting.
  void set(const nb::object &pyIdx, nb::object &value);

  /// Gets slice of buffer. Does not currently support slicing with indexes
  /// lower than the buffer rank.
  Buffer get(const nb::object &pyIdx);

  /// Gets underlying scalar in buffer. Currently supports only zero-rank
  /// buffers.
  nb::typed<nb::object, nb::any> item();

  /// Initializes all elements in the buffer to 0.
  void zeros();

  /// Fills a contiguous buffer with a repeated byte value.
  void memset(int64_t value);

  //===--------------------------------------------------------------------===//
  // Copies and transfers
  //===--------------------------------------------------------------------===//

  /// Copies buffer to the provided device.
  Buffer copyTo(PyDeviceTy device);

  /// Implementation of Python Buffer.to.
  Buffer to(PyDeviceTy device, const DeviceEvent *after = nullptr);
  std::vector<Buffer> to(std::vector<PyDeviceTy> devices);

  /// Copies buffer from another buffer.
  void inplaceCopyFrom(const Buffer &src);

  /// Batched ``srcs[i]`` → ``dsts[i]``, grouped by destination context so each
  /// destination device gets one submit on its own stream. Sources may be
  /// host, pinned, same-device or peer memory in any mix. Identity pairs are
  /// skipped.
  static void batchInplaceCopy(const std::vector<Buffer *> &dsts,
                               const std::vector<Buffer *> &srcs);

  /// Converts owning buffers into non-owning wrappers over the same storage.
  ///
  /// This is intended for internal graph-capture warmup paths that want to
  /// drop Python-side ownership while keeping borrowed handles to the reused
  /// memory manager allocation.
  static std::vector<Buffer>
  releaseAllToBorrowed(const std::vector<Buffer *> &buffers);

  //===--------------------------------------------------------------------===//
  // Host-hazard ordering
  //===--------------------------------------------------------------------===//

  /// Stamp a lastWrite on this buffer, no effect if not Host Hazard Tracked
  void stampWrite(std::shared_ptr<HostHazardCompletion> completion);

  /// Stamp a read on this buffer, no effect if not Host Hazard Tracked
  void stampRead(std::shared_ptr<HostHazardCompletion> completion);

  /// Drains any pending device write, device reads may occur in parallel
  void waitBeforeHostRead();

  /// Drain any pending write _and_ reads for host write permission
  void waitBeforeHostWrite();

private:
  /// Null means nothing is left to do: the buffer is untracked, or this call
  /// already drained it.
  HostHazardRecord *recordForWait();

  /// Gets an index from the storage array with a given dtype.
  nb::object getMemoryItem(size_t idx, DType dtype);

  /// Sets an index in the storage array to a given value of a given dtype.
  void setMemoryItem(size_t idx, DType dtype, nb::object &value);

  /// Returns a byte-offset-adjusted view of storage, sized to this buffer's
  /// own `spec`.
  ErrorOr<std::shared_ptr<BufferStorage>> getContiguousStorage() const;

  /// Shape and dtype of underlying buffer.
  TensorSpec spec;

  /// Underlying on-device block of memory.
  std::shared_ptr<BufferStorage> storage;

  /// Stream (aka DeviceContext) which the buffer is associated with. Implicitly
  /// also identifies the device on which the buffer is resident.
  PyDeviceStreamTy stream;

  /// Offsets required for a 1-unit jump in each dimension. Corresponds to
  /// Torch definition of strides:
  /// https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html
  std::vector<ssize_t> strides;

  /// Index of first element in buffer in underlying memory.
  size_t startOffset;

  /// HazardRecord slot owned by max.driver for ordering host side hazards.
  std::unique_ptr<HostHazardRecord> hostHazardRecord;
};

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_BUFFER_H
