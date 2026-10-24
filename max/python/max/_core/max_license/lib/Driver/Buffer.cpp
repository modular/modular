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

#include "_core/Driver/Buffer.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include "Support/RCRef.h"
#include "_core/Driver/BufferStorage.h"
#include "_core/Driver/HostHazard.h"
#include "_core/Driver/ViewStrides.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h" // IWYU pragma: keep (type caster)
#include "nanobind/stl/vector.h"   // IWYU pragma: keep (type caster)
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dlpack/dlpack.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace M;
using namespace Driver;
using M::Driver::DeviceContext;
using M::Driver::DeviceContextRef;
using M::Driver::DeviceEventRef;
using M::Driver::DeviceStreamRef;
using namespace nanobind::literals;

//===----------------------------------------------------------------------===//
// File-local helpers
//===----------------------------------------------------------------------===//

/// Converts a nanobind ndarray spec to a Max DType.
static DType dtypeFromNDArraySpec(const nb::dlpack::dtype &spec) {
  switch (spec.code) {
  case /* nb::dlpack::dtype_code::Int*/ 0:
    switch (spec.bits) {
    case 8:
      return DType::si8;
    case 16:
      return DType::si16;
    case 32:
      return DType::si32;
    case 64:
      return DType::si64;
    }

    break;
  case /* nb::dlpack::dtype_code::UInt */ 1:
    switch (spec.bits) {
    case 8:
      return DType::ui8;
    case 16:
      return DType::ui16;
    case 32:
      return DType::ui32;
    case 64:
      return DType::ui64;
    }

    break;
  case /* nb::dlpack::dtype_code::Float */ 2:
    switch (spec.bits) {
    case 32:
      return DType::f32;
    case 64:
      return DType::f64;
    }

    break;
  case /* nb::dlpack::dtype_code::Bfloat */ 4:
    switch (spec.bits) {
    case 16:
      return DType::bf16;
    }
  case /* nb::dlpack::dtype_code::Bool */ 6:
    return DType::kBool;
  }

  throw std::invalid_argument("DType not recognized");
}

// Converts a (possibly negative) index to a fixed index. Also checks for
// out of bounds errors
static size_t canonicalizeIndex(int idx, size_t dim) {
  int canonicalized;
  if (idx < 0)
    canonicalized = dim + idx;
  else
    canonicalized = idx;
  if (0 > canonicalized || (size_t)canonicalized >= dim) {
    std::string errMsg = ("index " + Twine(idx) +
                          " is out of bounds for dim with size " + Twine(dim))
                             .str();
    throw std::out_of_range(errMsg);
  }
  return canonicalized;
}

static std::vector<nb::ssize_t> ndarrayToShape(const nb::ndarray<nb::ro> &arr) {
  const auto ndims = arr.ndim();
  std::vector<nb::ssize_t> shape(ndims);
  for (std::size_t i = 0; i < ndims; ++i) {
    shape[i] = arr.shape(i);
  }
  return shape;
}

/// Return dlpack's DLDevice struct representing where the tensor lives.
static DLDevice getDLPackDevice(Python::Buffer *tensor) {
  if (tensor->getDevice()->isHost())
    return {kDLCPU, 0};
  return tensor->getDevice()->ctx->device().getDLDevice(tensor->isPinned());
}

static ErrorOrSuccess syncExternalStream(M::Driver::DeviceContextRef context,
                                         intptr_t externalStreamID) {
  auto externalStreamOr =
      context->createExternalStream((void *)externalStreamID);
  if (failed(externalStreamOr))
    return Error(llvm::formatv("{}: invalid external stream",
                               externalStreamOr.takeError()));
  auto streamOr = context->stream();
  if (failed(streamOr))
    return Error(
        llvm::formatv("{}: no stream for device", streamOr.takeError()));
  auto eventOr = externalStreamOr.takeValue()->enqueueEvent();
  if (failed(eventOr))
    return Error(llvm::formatv(
        "{}: could not enqueue event on external stream with id {}",
        eventOr.takeError(), externalStreamID));
  auto wait = (*streamOr)->enqueueWaitForEvent(**eventOr);
  if (failed(wait))
    return Error(llvm::formatv("{}: could not enqueue wait on default stream",
                               wait.takeError()));
  return success();
}

static bool copyTreatsAsHost(M::Driver::DeviceBuffer &buf) {
  return buf.isHost() ||
         buf.context().device().api() == M::Driver::Device::cpuAPI;
}

/// Mirrors MLRT's copy dispatch; keep in sync with both builds' `copyTo`.
static M::Driver::DeviceContext *copyContext(M::Driver::DeviceBuffer &src,
                                             M::Driver::DeviceBuffer &dst) {
  if (copyTreatsAsHost(src))
    return &dst.context();
  if (copyTreatsAsHost(dst))
    return &src.context();
  return &dst.context();
}

/// Reuses side's queue where possible, so the common case makes no new wrapper.
static Python::PyDeviceStreamTy queueOn(const Python::Buffer &side,
                                        M::Driver::DeviceContext *ctx) {
  Python::PyDeviceStreamTy held = side.getStream();
  if (held->stream->ctx.getPointer() == ctx)
    return held;
  return std::make_shared<Python::DeviceStream>(
      *side.getDevice(), M::Driver::DeviceContextRef::copy(ctx));
}

//===----------------------------------------------------------------------===//
// Buffer: construction and lifetime
//===----------------------------------------------------------------------===//

Python::Buffer::~Buffer() {
  // Release the GIL while destroying the storage shared_ptr.
  nb::gil_scoped_release release;
  storage.reset();
}

// Constructor for tensor requiring initial data allocation.
Python::Buffer::Buffer(M::DType dtype, ArrayRef<size_t> shape,
                       std::optional<Python::PyDeviceTy> device, Usage usage)
    : spec(shape, dtype),
      stream(std::make_shared<DeviceStream>(device ? std::move(device.value())
                                                   : Python::getCpuDevice(0))) {
  // Staging only changes the allocation where a transfer can be asynchronous.
  const bool stagingAlloc =
      usageContains(usage, Usage::Staging) && !this->stream->dev->isHost();
  storage = std::make_shared<BufferStorage>(
      this->stream->dev->allocate(spec, stagingAlloc), spec, usage);
  strides = rowMajorStrides(spec);
  startOffset = 0;
  hostHazardRecord =
      isHostHazardTracked() ? std::make_unique<HostHazardRecord>() : nullptr;
}

Python::Buffer::Buffer(M::DType dtype, ArrayRef<size_t> shape,
                       PyDeviceStreamTy stream, Usage usage)
    : spec(shape, dtype), stream(std::move(stream)) {
  const bool stagingAlloc =
      usageContains(usage, Usage::Staging) && !this->stream->dev->isHost();
  storage = std::make_shared<BufferStorage>(
      this->stream->stream->allocate(spec, stagingAlloc), spec, usage);
  strides = rowMajorStrides(spec);
  startOffset = 0;
  hostHazardRecord =
      isHostHazardTracked() ? std::make_unique<HostHazardRecord>() : nullptr;
}

// Constructor for creating tensors from numpy arrays.
Python::Buffer::Buffer(nb::ndarray<nb::ro> arr, Python::PyDeviceTy device)
    : stream(std::make_shared<DeviceStream>(std::move(device))) {
  DType type = dtypeFromNDArraySpec(arr.dtype());

  auto destructor = [arr = arr]() mutable {
    // Remove reference to underlying array storage to ensure we're not
    // destroying the original array.
    auto _ = std::move(arr);
  };
  spec = TensorSpec(ndarrayToShape(arr), type);
  // Wraps caller-owned numpy memory; no allocation intent to record.
  storage = std::make_shared<BufferStorage>(
      this->stream->dev->wrapExternalBuffer(const_cast<void *>(arr.data()),
                                            spec, std::move(destructor)),
      spec, Usage::Default);
  strides = rowMajorStrides(spec);
  startOffset = 0;
}

Python::Buffer::Buffer(const TensorSpec &spec,
                       const std::shared_ptr<BufferStorage> &storage,
                       Python::PyDeviceStreamTy stream,
                       const std::vector<ssize_t> &strides, size_t startOffset,
                       const HostHazardRecord *record)
    : spec(spec), storage(storage), stream(std::move(stream)), strides(strides),
      startOffset(startOffset),
      hostHazardRecord(cloneHostHazardRecord(record)) {}

Python::Buffer::Buffer(std::shared_ptr<BufferStorage> storage,
                       PyDeviceTy device, const HostHazardRecord *record)
    : stream(std::make_shared<DeviceStream>(std::move(device))) {
  spec = storage->getSpec();
  this->storage = std::move(storage);
  strides = rowMajorStrides(spec);
  startOffset = 0;
  hostHazardRecord = cloneHostHazardRecord(record);
}

//===----------------------------------------------------------------------===//
// Buffer: views and DLPack interop
//===----------------------------------------------------------------------===//

/// Create a new tensor with the given shape and type, sharing the underlying
/// storage with an existing tensor.
Python::Buffer Python::Buffer::view(M::DType dtype,
                                    const std::vector<size_t> &shape) {
  TensorSpec newSpec(shape, dtype);
  if (newSpec.getSizeInBytes() != spec.getSizeInBytes())
    throw std::length_error("byte sizes don't match in view creation");

  // Calculate the byte offset from the start of storage
  size_t byteOffset = spec.getEltType().getSizeInBytes() * startOffset;

  std::vector<ssize_t> newStrides =
      viewStrides(spec, strides, isContiguous(), dtype, shape, byteOffset);

  // Handle aligned case with early return
  if (byteOffset % dtype.getSizeInBytes() == 0) {
    size_t newStartOffset = byteOffset / dtype.getSizeInBytes();
    return Buffer(newSpec, storage, stream, std::move(newStrides),
                  newStartOffset, hostHazardRecord.get());
  }

  // Handle unaligned case - create a new storage that starts at the correct
  // byte offset, so the new tensor can use startOffset=0
  auto v = createStorageView(*storage, newSpec, byteOffset);
  if (v.isError())
    throw std::runtime_error(v.takeError().get());

  return Buffer(newSpec, v.takeValue(), stream, std::move(newStrides), 0,
                hostHazardRecord.get());
}

Python::Buffer Python::Buffer::fromDLPack(const nb::object &arrayOrTensor) {
  auto [device, streamID] = getPyDevice(arrayOrTensor);
  DLPack::DLManagedTensorWrapper managed =
      DLPack::callDLPack(arrayOrTensor, streamID);
  return fromDLManagedTensor(std::move(managed), device);
}

Python::Buffer
Python::Buffer::fromDLManagedTensor(DLPack::DLManagedTensorWrapper &&managed,
                                    PyDeviceTy device,
                                    std::optional<intptr_t> externalStream) {
  DLTensor &dlTensor = managed->dl_tensor;

  auto shape = ArrayRef<int64_t>(dlTensor.shape, dlTensor.ndim);
  DType type = DLPack::dtypeForDLDataType(dlTensor.dtype);
  auto spec = TensorSpec(shape, type);

  // According to the dlpack contract, the producer keeps owning the underlying
  // memory of the tensor. So in the destructor we call the deleter hook
  // provided by the producer (which usually just decrements a ref count kept by
  // the producer). This is handled by the DLManagedTensorWrapper.
  auto destructor = [m = std::move(managed)]() mutable {
    auto _ = std::move(m);
  };

  // Sync external stream first, error before attempting any wrap
  if (externalStream) {
    auto synced = syncExternalStream(device->ctx, *externalStream);
    if (failed(synced))
      throw std::runtime_error(synced.takeError().get());
  }

  // Wraps a producer-owned DLPack tensor; no allocation intent to record.
  auto storage = std::make_shared<BufferStorage>(
      device->wrapExternalBuffer(dlTensor.data, spec, std::move(destructor)),
      spec, Usage::Default);
  return Python::Buffer(std::move(storage), device, nullptr);
}

nb::tuple Python::Buffer::dlPackDevice() {
  DLDevice device = getDLPackDevice(this);
  return nb::make_tuple((int)device.device_type, device.device_id);
}

nb::typed<nb::object, nb::capsule>
Python::Buffer::dlPack(nb::pointer_and_handle<Buffer> self,
                       const std::optional<uintptr_t> stream,
                       nb::kwargs kwargs) {
  // Host-visible only on a device: staging on the CPU has no record to wait
  // on, so it takes the full synchronize below as it did before tracking.
  if (usageContains(self.p->getUsage(), Usage::Staging) && !self.p->isHost()) {
    // a waitBeforeHostWrite drains writes _and_ reads
    self.p->waitBeforeHostWrite();
    return getAsDlpack(self, stream, kwargs);
  }

  DLDevice device = getDLPackDevice(self.p);

  // For tensors on GPU devices we need to check for synchronization with the
  // consumer's stream.
  // Values are interpreted according to this documentation:
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html#array_api.array.__dlpack__
  // The value of -1 communicates no special stream synchronization is requested
  // by the consumer.
  constexpr intptr_t noStreamSync = -1;
  intptr_t consumerStreamPtr = noStreamSync;

  if ((device.device_type == kDLCUDA) || (device.device_type == kDLROCM)) {
    if (!stream.has_value()) {
      // Legacy default stream is requested if None is passed.
      if (device.device_type == kDLCUDA) {
        // For the CUDA driver the legacy default stream has the "pointer"
        // value 1.
        consumerStreamPtr = 1;
      } else {
        assert(device.device_type == kDLROCM);
        // For the HIP driver the legacy default stream has the "pointer"
        // value 0. See note below.
        consumerStreamPtr = 0;
      }
    } else {
      consumerStreamPtr = stream.value();
      if (device.device_type == kDLCUDA) {
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa
        // Check for special values. All others pass through.
        // Values of 1 and 2 pass through since they match the corresponding
        // CUDA default stream pointer values.
        if (consumerStreamPtr == 0) {
          // 0 is disallowed. Reset to default handling
          consumerStreamPtr = noStreamSync;
        }
      } else {
        assert(device.device_type == kDLROCM);
        // https://rocm.docs.amd.com/projects/HIP/en/develop/doxygen/html/group___global_defs.html#ga7f5bb5340726842f8967cd66337361d7
        // Check for special values. All others pass through.
        //
        // NOTE: Value of 0 passes through as the default legacy stream. Even if
        // `hipStreamLegacy` is documented in the AMD documentation it leads to
        // an error.
        if ((consumerStreamPtr == 1) || (consumerStreamPtr == 2)) {
          // 1 and 2 are invalid values. Reset to default handling of device
          // sync.
          consumerStreamPtr = noStreamSync;
        }
      }
    }
  }

  {
    // Release the GIL while we synchronize.
    nb::gil_scoped_release release;

    // Check if synchronization on a particular stream was requested.
    void *stream = nullptr;
    if (consumerStreamPtr != noStreamSync)
      stream = (void *)consumerStreamPtr;

    // Ensure the buffer data is ready
    ErrorOrSuccess err = self.p->storage->buf->synchronize(stream);
    if (err.isError())
      throw std::runtime_error(err.getError());
  }

  return getAsDlpack(self, stream, kwargs);
}

nb::typed<nb::object, nb::capsule>
Python::Buffer::getAsDlpack(nb::pointer_and_handle<Buffer> self,
                            const std::optional<uintptr_t> stream,
                            nb::kwargs kwargs) {
  std::vector<size_t> shape(self.p->spec.begin(), self.p->spec.end());
  DLDevice device = getDLPackDevice(self.p);

  DLDataType dlDtype = DLPack::dlDataTypeForDType(self.p->getDType());
  nb::ndarray<nb::array_api> dlPackObject{
      /*data=*/self.p->data(),
      /*ndim=*/self.p->getRank(),
      /*shape=*/shape.data(),
      /*owner=*/self.h,    // The current Python object that owns the data
      /*strides=*/nullptr, // Only necessary if not contiguous
      /*dtype=*/{dlDtype.code, dlDtype.bits, dlDtype.lanes},
      /*device_type=*/device.device_type,
      /*device_id=*/device.device_id,
  };

  // Recommended pattern by
  // https://nanobind.readthedocs.io/en/latest/ndarray.html#array-libraries
  return nb::cast(dlPackObject, nb::rv_policy::reference_internal, self.h)
      .attr("__dlpack__")("stream"_a = stream, **kwargs);
}

//===----------------------------------------------------------------------===//
// Buffer: shape, dtype and layout
//===----------------------------------------------------------------------===//

size_t Python::Buffer::getRank() const { return spec.getRank(); }

M::DType Python::Buffer::getDType() const { return spec.getEltType(); }

/// Gets the shape of a given tensor.
nb::tuple Python::Buffer::getShape() const {
  if (!spec.hasRank())
    throw std::runtime_error("unranked tensors are not supported");
  return nb::tuple(nb::cast(std::vector<size_t>(spec.begin(), spec.end())));
}

nb::tuple Python::Buffer::getStrides() const {
  if (!spec.hasRank())
    throw std::runtime_error("unranked tensors are not supported");
  return nb::tuple(nb::cast(strides));
}

size_t Python::Buffer::getNumElements() const { return spec.getNumElements(); }

const TensorSpec &Python::Buffer::getSpec() const { return spec; }

// Returns whether a tensor is contiguous slice by ensuring that the calculated
// strides are what we would calculate for a contiguous tensor of the same
// shape.
bool Python::Buffer::isContiguousSlice() const {
  return strides == rowMajorStrides(spec);
}

bool Python::Buffer::isContiguous() const {
  // Specialized check for contiguous slices.
  if (isContiguousSlice())
    return true;

  // If a buffer is not a contiguous slice, we need to check if the shape of
  // the allocated storage is the same as the shape of the buffer itself.
  if (spec != storage->getSpec())
    return false;

  // We also need to ensure that all strides are positive.
  return std::all_of(strides.begin(), strides.end(),
                     [](ssize_t elt) { return elt >= 0; });
}

bool Python::Buffer::aligned(int alignment) const {
  uintptr_t memoryAddress = reinterpret_cast<uintptr_t>(data());
  return memoryAddress % alignment == 0;
}

//===----------------------------------------------------------------------===//
// Buffer: device, storage and allocation intent
//===----------------------------------------------------------------------===//

Python::PyDeviceTy Python::Buffer::getDevice() const { return stream->dev; }

Python::PyDeviceStreamTy Python::Buffer::getStream() const { return stream; }

ErrorOr<std::shared_ptr<Python::BufferStorage>>
Python::Buffer::getInputStorage() const {
  // A contiguous slice needs its start offset folded into the view; anything
  // else already describes the whole allocation.
  if (isContiguousSlice())
    return getContiguousStorage();
  return storage;
}

void *Python::Buffer::data() const {
  return ((uint8_t *)storage->data()) +
         spec.getEltType().getSizeInBytes(startOffset);
}

bool Python::Buffer::isHost() const { return stream->dev->isHost(); }

bool Python::Buffer::isPinned() const { return storage->isPinned(); }

Python::Usage Python::Buffer::getUsage() const { return storage->usage; }

bool Python::Buffer::isHostHazardTracked() const {
  auto usage = getUsage();
  bool isStaging = usageContains(usage, Usage::Staging);
  bool isTracked = !usageContains(usage, Usage::Untracked);
  return !isHost() && isStaging && isTracked;
}

bool Python::Buffer::couldElideCopyTo(const Buffer &other) const {
  // data() already has the offset folded in. Two buffers at one address are
  // the same memory only if both are contiguous and on the same context.
  // Tracked buffers stay out: there the copy is also carrying the hazard.
  return data() == other.data() &&
         spec.getSizeInBytes() == other.spec.getSizeInBytes() &&
         isContiguous() && other.isContiguous() &&
         &storage->buf->context() == &other.storage->buf->context() &&
         !isHostHazardTracked() && !other.isHostHazardTracked();
}

//===----------------------------------------------------------------------===//
// Buffer: element access and fills
//===----------------------------------------------------------------------===//

// Set currently only supports setting scalar values with an equivalent C++
// built-in analogue. This method will be extended to support non-native types
// soon.
void Python::Buffer::set(const nb::object &pyIdx, nb::object &value) {
  // Sync pending device access before writing into the resource
  waitBeforeHostWrite();

  // Setting does not currently support slices.
  std::vector<int> indices;
  if (int result; nb::try_cast(pyIdx, result))
    indices.push_back(result);
  else if (!nb::try_cast(pyIdx, indices))
    throw nb::type_error("unsupported index type");

  // Special case when setting element in scalar.
  if (indices.size() == 1 && indices[0] == 0 && getRank() == 0) {
    setMemoryItem(0, spec.getEltType(), value);
    return;
  }

  if (indices.size() != getRank()) {
    std::string errMsg =
        ("the provided number of indices (" + Twine(indices.size()) +
         ") is not equal to the tensor rank (" + Twine(getRank()) + ")")
            .str();
    throw std::invalid_argument(errMsg);
  }

  std::vector<size_t> canonical(indices.size(), 0);
  for (size_t i = 0, len = indices.size(); i < len; ++i)
    canonical[i] = canonicalizeIndex(indices[i], spec[i]);

  size_t offset = std::inner_product(canonical.begin(), canonical.end(),
                                     strides.begin(), startOffset);
  setMemoryItem(offset, spec.getEltType(), value);
}

// Get tensor slice from tensor. Does not currently support the case where the
// indexes are less than the rank of the Buffer.
Python::Buffer Python::Buffer::get(const nb::object &pyIdx) {
  std::vector<nb::object> indices;
  if (nb::isinstance<nb::int_>(pyIdx) || nb::isinstance<nb::slice>(pyIdx))
    indices.push_back(pyIdx);
  else if (!nb::try_cast(pyIdx, indices))
    throw nb::type_error("unsupported index type in get method");

  if (indices.size() != getRank()) {
    std::string errMsg =
        ("the provided number of indices (" + Twine(indices.size()) +
         ") is not equal to the tensor rank (" + Twine(getRank()) + ")")
            .str();
    throw std::invalid_argument(errMsg);
  }

  std::vector<size_t> offsets(indices.size());
  // Since slicing can drop dimensions, we don't know the rank of the slice
  // just yet.
  std::vector<ssize_t> sliceStrides, sliceShape;
  for (size_t i = 0, len = indices.size(); i < len; ++i) {
    auto idxHandle = indices[i];
    if (int idx; nb::try_cast(idxHandle, idx)) {
      // If we get an integral index, we know the tensor is fixed at that
      // position in the given dimension. The only thing this will affect is the
      // starting offset.
      offsets[i] = canonicalizeIndex(idx, spec[i]);
    } else if (nb::slice slice; nb::try_cast(idxHandle, slice)) {
      // If we get a slice, we'll pull out the relevant numbers from the slice
      auto [start, stop, step, sliceLength] = slice.compute(spec[i]);
      offsets[i] = start;
      sliceShape.push_back(sliceLength);
      sliceStrides.push_back(step * strides[i]);
    }
  }
  size_t sliceOffset = std::inner_product(offsets.begin(), offsets.end(),
                                          strides.begin(), startOffset);
  auto sliceSpec = TensorSpec(sliceShape, spec.getEltType());
  return Buffer(sliceSpec, storage, stream, sliceStrides, sliceOffset,
                hostHazardRecord.get());
}

// Gets the scalar value contained in a zero-rank tensor.
nb::typed<nb::object, nb::any> Python::Buffer::item() {
  if (size_t numElements = spec.getNumElements(); numElements != 1)
    throw std::invalid_argument("calling `item` on a tensor with " +
                                std::to_string(numElements) +
                                " items but expected only 1");
  // `item` reads `storage->data()` directly, so untracked memory, which has
  // no record to wait on, still owes a drain.
  if (isHostHazardTracked()) {
    waitBeforeHostRead();
  } else {
    nb::gil_scoped_release release;
    if (auto err = storage->buf->synchronize())
      throw std::runtime_error(err.getError());
  }
  return getMemoryItem(startOffset, spec.getEltType());
}

void Python::Buffer::memset(int64_t value) {
  if (value < 0 || value > 255)
    throw nb::value_error("memset value must be between 0 and 255");
  if (!isContiguous())
    throw nb::value_error("memset requires a contiguous buffer");
  // A packed slice that starts or ends mid-byte shares bytes with neighbours.
  size_t bits = spec.getEltType().getWidthInBits();
  if (bits < 8 && ((startOffset * bits) % 8 || (getNumElements() * bits) % 8))
    throw nb::value_error("memset requires a byte-aligned sub-byte slice");
  if (spec.getSizeInBytes() == 0)
    return;

  nb::gil_scoped_release release;
  auto view = getContiguousStorage();
  if (view.isError())
    throw std::runtime_error(view.takeError().get());
  if (auto err = view.get()->memset(value)) {
    if (hostHazardRecord)
      hostHazardRecord->taint();
    throw std::runtime_error(err.getError());
  }
  if (hostHazardRecord) {
    auto recStream = queueOn(*this, &view.get()->buf->context());
    hostHazardRecord->stampWrite(HostHazardCompletion::recordOn(recStream));
  }
}

void Python::Buffer::zeros() { memset(0); }

//===----------------------------------------------------------------------===//
// Buffer: copies and transfers
//===----------------------------------------------------------------------===//

// Copies input tensor to the desired device.
Python::Buffer Python::Buffer::copyTo(Python::PyDeviceTy device) {
  nb::gil_scoped_release release;
  ErrorOr<M::Driver::DeviceBufferRef> dstBufOr = device->allocateBuffer(spec);
  if (dstBufOr.isError())
    throw std::runtime_error(dstBufOr.getError());
  M::Driver::DeviceBufferRef dstBuf = dstBufOr.takeValue();

  // For sliced tensors, we need to copy only the sliced portion, not the entire
  // storage
  size_t byteOffset = spec.getEltType().getSizeInBytes(startOffset);
  ErrorOrSuccess copyErrorOr =
      storage->buf->copyTo(*dstBuf, byteOffset, dstBuf->byteSize());

  if (copyErrorOr.isError())
    throw std::runtime_error(copyErrorOr.getError());

  // Track the completion for the hazard from the _running_ context, which is
  // the destination buffer for this method.
  if (hostHazardRecord) {
    M::Driver::DeviceContext *ctx = copyContext(*storage->buf, *dstBuf);
    auto tgt = ctx == &storage->buf->context()
                   ? getStream()
                   : std::make_shared<Python::DeviceStream>(
                         *device, M::Driver::DeviceContextRef::copy(ctx));
    hostHazardRecord->stampRead(HostHazardCompletion::recordOn(tgt));
  }

  // A copy destination is a fresh allocation with no requested staging
  // intent, regardless of the source buffer's usage.
  return Python::Buffer(
      std::make_shared<BufferStorage>(std::move(dstBuf), spec, Usage::Default),
      std::move(device),
      /*hostHazardRecord=*/nullptr);
}

// std::invalid_argument surfaces in Python as ValueError.
static M::Driver::DeviceEvent &requireAfter(const Python::DeviceEvent *after) {
  if (after)
    return *after->ev.getPointer();
  throw std::invalid_argument(
      "moving a buffer to another queue on the same device requires `after`, "
      "an event recorded on the source queue");
}

Python::Buffer Python::Buffer::to(Python::PyDeviceTy device,
                                  const Python::DeviceEvent *after) {
  // Pinned host memory and device memory are different memory spaces
  // even if associated with the same device. Always copy if source is pinned.
  if (*this->stream->dev != *device || this->isPinned())
    return copyTo(std::move(device));

  ErrorOr<std::shared_ptr<BufferStorage>> storageOr =
      createStorageViewOn(*storage, device->ctx, requireAfter(after));
  if (storageOr.isError())
    throw std::runtime_error(storageOr.takeError().get());
  return Python::Buffer(spec, storageOr.takeValue(),
                        std::make_shared<DeviceStream>(device), strides,
                        startOffset, hostHazardRecord.get());
}

std::vector<Python::Buffer>
Python::Buffer::to(std::vector<Python::PyDeviceTy> devices) {
  std::vector<Python::Buffer> tensors;
  tensors.reserve(devices.size());
  for (const auto &device : devices)
    tensors.emplace_back(to(device));
  return tensors;
}

// Copy the contents of another tensor into this one.
void Python::Buffer::inplaceCopyFrom(const Buffer &src) {
  nb::gil_scoped_release release;
  // Do not issue a copy if src and dst are the same.
  if (couldElideCopyTo(src))
    return;

  ErrorOr<std::shared_ptr<BufferStorage>> srcViewOr =
      src.getContiguousStorage();
  if (srcViewOr.isError())
    throw std::runtime_error(srcViewOr.takeError().get());
  ErrorOr<std::shared_ptr<BufferStorage>> dstViewOr = getContiguousStorage();
  if (dstViewOr.isError())
    throw std::runtime_error(dstViewOr.takeError().get());

  const std::shared_ptr<BufferStorage> &srcView = srcViewOr.get();
  const std::shared_ptr<BufferStorage> &dstView = dstViewOr.get();

  ErrorOrSuccess copyErrorOr = srcView->buf->copyTo(*dstView->buf);
  if (copyErrorOr.isError()) {
    // Work may already be enqueued, so an older token on either handle would
    // resolve too early.
    if (src.hostHazardRecord)
      src.hostHazardRecord->taint();
    if (hostHazardRecord)
      hostHazardRecord->taint();
    throw std::runtime_error(copyErrorOr.getError());
  }

  if (!hostHazardRecord && !src.hostHazardRecord)
    return;

  M::Driver::DeviceContext *ctx = copyContext(*srcView->buf, *dstView->buf);
  const Buffer &owner = (ctx == &srcView->buf->context()) ? src : *this;
  auto completion = HostHazardCompletion::recordOn(queueOn(owner, ctx));

  if (src.hostHazardRecord)
    src.hostHazardRecord->stampRead(completion);
  if (hostHazardRecord)
    hostHazardRecord->stampWrite(std::move(completion));
}

// Batched srcs[i] -> dsts[i], one submit per destination context. Sources are
// unconstrained; the driver resolves host, pinned, same-device and peer per
// pointer.
void Python::Buffer::batchInplaceCopy(const std::vector<Buffer *> &dsts,
                                      const std::vector<Buffer *> &srcs) {
  if (dsts.empty())
    return;
  if (dsts.size() != srcs.size())
    throw std::invalid_argument(
        "batch_inplace_copy: dsts and srcs must have the same length");

  // Pairs that name the same memory are graph-capture stable inputs — skip
  // them so we do not enqueue a no-op memcpy.
  std::vector<Buffer *> filteredDsts;
  std::vector<Buffer *> filteredSrcs;
  filteredDsts.reserve(dsts.size());
  filteredSrcs.reserve(srcs.size());

  for (size_t i = 0, n = dsts.size(); i < n; ++i) {
    Buffer *dst = dsts[i];
    Buffer *src = srcs[i];
    if (dst == src)
      continue;
    size_t dstBytes = dst->spec.getSizeInBytes();
    size_t srcBytes = src->spec.getSizeInBytes();
    if (dstBytes != srcBytes) {
      throw std::invalid_argument(
          "batch_inplace_copy: buffer size mismatch at index " +
          std::to_string(i) + ": dst has " + std::to_string(dstBytes) +
          " bytes, src has " + std::to_string(srcBytes) + " bytes");
    }
    // cuMemcpyBatchAsync and hipMemcpyBatchAsync reject a zero size.
    if (dstBytes == 0)
      continue;
    // Do not issue a copy if src and dst are the same.
    if (dst->couldElideCopyTo(*src))
      continue;
    // Records on the submit context; a host-visible dst copies on the src's.
    if (dst->isHostHazardTracked() &&
        copyContext(*src->storage->buf, *dst->storage->buf) !=
            &dst->storage->buf->context())
      throw std::invalid_argument(
          "batch_inplace_copy: tracked destination at index " +
          std::to_string(i) +
          " has its source on another queue; use inplace_copy_from instead");
    filteredDsts.push_back(dst);
    filteredSrcs.push_back(src);
  }
  if (filteredDsts.empty())
    return;

  nb::gil_scoped_release release;

  // Keep every storage view alive until the last enqueue returns.
  std::vector<std::shared_ptr<Python::BufferStorage>> views;
  views.reserve(filteredDsts.size() * 2);

  // One submit is one stream, and a stream only orders the writes it performs.
  // Group by destination context so a destination on another device gets its
  // own submit on that device's stream instead of riding on a stream that
  // gives it no ordering. Sources need no grouping: the driver brackets each
  // foreign source stream around the batch.
  struct Group {
    M::Driver::DeviceContext *ctx;
    PyDeviceTy dev;
    std::vector<M::Driver::DeviceBuffer *> dsts;
    std::vector<M::Driver::DeviceBuffer *> srcs;
    std::vector<Python::Buffer *> pyDsts;
    std::vector<Python::Buffer *> pySrcs;
  };
  std::vector<Group> groups;

  for (size_t i = 0, n = filteredDsts.size(); i < n; ++i) {
    ErrorOr<std::shared_ptr<Python::BufferStorage>> dstViewOr =
        filteredDsts[i]->getContiguousStorage();
    if (dstViewOr.isError())
      throw std::runtime_error(dstViewOr.takeError().get());
    views.push_back(dstViewOr.takeValue());
    M::Driver::DeviceBuffer *dstBuf = views.back()->buf.getPointer();

    ErrorOr<std::shared_ptr<Python::BufferStorage>> srcViewOr =
        filteredSrcs[i]->getContiguousStorage();
    if (srcViewOr.isError())
      throw std::runtime_error(srcViewOr.takeError().get());
    views.push_back(srcViewOr.takeValue());
    M::Driver::DeviceBuffer *srcBuf = views.back()->buf.getPointer();

    M::Driver::DeviceContext *dstCtx = &dstBuf->context();
    Group *group = nullptr;
    for (Group &candidate : groups) {
      if (candidate.ctx == dstCtx) {
        group = &candidate;
        break;
      }
    }
    if (!group)
      group = &groups.emplace_back(
          Group{dstCtx, filteredDsts[i]->getDevice(), {}, {}, {}, {}});
    group->dsts.push_back(dstBuf);
    group->srcs.push_back(srcBuf);
    if (filteredDsts[i]->isHostHazardTracked() &&
        filteredDsts[i]->hostHazardRecord)
      group->pyDsts.push_back(filteredDsts[i]);
    if (filteredSrcs[i]->isHostHazardTracked() &&
        filteredSrcs[i]->hostHazardRecord)
      group->pySrcs.push_back(filteredSrcs[i]);
  }

  auto taintGroup = [](const Group &group) {
    for (const Buffer *pyDst : group.pyDsts)
      pyDst->hostHazardRecord->taint();
    for (const Buffer *pySrc : group.pySrcs)
      pySrc->hostHazardRecord->taint();
  };

  // Submit back-to-back in first-appearance order so groups on distinct
  // devices can overlap.
  for (Group &group : groups) {
    ErrorOrSuccess err = group.ctx->enqueueBatchCopy(group.dsts, group.srcs);
    if (err.isError()) {
      taintGroup(group);
      throw std::runtime_error(err.getError());
    }

    // If there is a hostHazardTracked resource in the group: record the copy
    if (!group.pyDsts.empty() || !group.pySrcs.empty()) {
      auto eventOr = group.ctx->enqueueEvent();

      if (eventOr.isError()) {
        taintGroup(group);
        throw std::runtime_error(eventOr.takeError().get());
      }

      // Otherwise, make a completion out of the event and stamp
      auto completion = std::make_shared<HostHazardCompletion>();
      completion->event =
          std::make_shared<Python::DeviceEvent>(eventOr.takeValue());
      completion->queue = std::make_shared<Python::DeviceStream>(
          *group.dev, M::Driver::DeviceContextRef::copy(group.ctx));

      for (Buffer *pyDst : group.pyDsts)
        pyDst->hostHazardRecord->stampWrite(completion);
      for (Buffer *pySrc : group.pySrcs)
        pySrc->hostHazardRecord->stampRead(completion);
    }
  }
}

std::vector<Python::Buffer>
Python::Buffer::releaseAllToBorrowed(const std::vector<Buffer *> &buffers) {
  llvm::DenseMap<Python::BufferStorage *,
                 std::shared_ptr<Python::BufferStorage>>
      borrowedCache;
  std::vector<Buffer> releasedBuffers;
  releasedBuffers.reserve(buffers.size());

  for (const Buffer *buffer : buffers) {
    auto [cacheIt, inserted] = borrowedCache.insert(
        {buffer->storage.get(), std::shared_ptr<Python::BufferStorage>()});
    if (inserted)
      cacheIt->second = createBorrowedStorage(buffer->storage);

    auto borrowedStorage = cacheIt->second;
    releasedBuffers.emplace_back(
        buffer->spec, std::move(borrowedStorage), buffer->stream,
        buffer->strides, buffer->startOffset, buffer->hostHazardRecord.get());
  }

  return releasedBuffers;
}

//===----------------------------------------------------------------------===//
// Buffer: host-hazard ordering
//===----------------------------------------------------------------------===//

void Python::Buffer::stampWrite(
    std::shared_ptr<HostHazardCompletion> completion) {
  if (isHostHazardTracked() && hostHazardRecord)
    hostHazardRecord->stampWrite(completion);
}

void Python::Buffer::stampRead(
    std::shared_ptr<HostHazardCompletion> completion) {
  if (isHostHazardTracked() && hostHazardRecord)
    hostHazardRecord->stampRead(completion);
}

void Python::Buffer::waitBeforeHostRead() {
  HostHazardRecord *record = recordForWait();
  if (!record)
    return;
  std::shared_ptr<HostHazardCompletion> pending =
      record->takeLastWriteSnapshot();
  if (!pending)
    return;
  // wait releases the GIL
  pending->wait();
  record->clearSettled(std::move(pending), {});
}

void Python::Buffer::waitBeforeHostWrite() {
  HostHazardRecord *record = recordForWait();
  if (!record)
    return;
  std::shared_ptr<HostHazardCompletion> pendingWrite =
      record->takeLastWriteSnapshot();
  std::vector<std::shared_ptr<HostHazardCompletion>> pendingReads =
      record->takeReadersSnapshot();
  // these waits release the GIL
  if (pendingWrite)
    pendingWrite->wait();
  for (const auto &reader : pendingReads)
    reader->wait();
  record->clearSettled(std::move(pendingWrite), std::move(pendingReads));
}

//===----------------------------------------------------------------------===//
// Buffer: private helpers
//===----------------------------------------------------------------------===//

Python::HostHazardRecord *Python::Buffer::recordForWait() {
  if (!isHostHazardTracked())
    return nullptr;

  while (true) {
    if (hostHazardRecord && !hostHazardRecord->isTainted())
      return hostHazardRecord.get();

    // Cleared before the drain so a later taint survives it.
    if (hostHazardRecord)
      hostHazardRecord->clearTaint();

    {
      nb::gil_scoped_release release;
      if (auto err = storage->buf->synchronize()) {
        if (hostHazardRecord)
          hostHazardRecord->taint();
        throw std::runtime_error(err.getError());
      }
    }

    if (!hostHazardRecord) {
      hostHazardRecord = std::make_unique<HostHazardRecord>();
      return hostHazardRecord.get();
    }
    hostHazardRecord->clearSettled(hostHazardRecord->takeLastWriteSnapshot(),
                                   hostHazardRecord->takeReadersSnapshot());
  }
}

// Gets an item of a given dtype at a given index. Currently supports
// single-item accesses.
nb::object Python::Buffer::getMemoryItem(size_t idx, DType dtype) {
  switch (dtype.getValue()) {
#define GETTER_FOR_TYPE(kDtype)                                                \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    nb::object res = nb::cast(data[idx]);                                      \
    return res;                                                                \
  }
    GETTER_FOR_TYPE(DType::kBool)
    GETTER_FOR_TYPE(DType::si8)
    GETTER_FOR_TYPE(DType::si16)
    GETTER_FOR_TYPE(DType::si32)
    GETTER_FOR_TYPE(DType::si64)
    GETTER_FOR_TYPE(DType::ui8)
    GETTER_FOR_TYPE(DType::ui16)
    GETTER_FOR_TYPE(DType::ui32)
    GETTER_FOR_TYPE(DType::ui64)
    GETTER_FOR_TYPE(DType::f32)
    GETTER_FOR_TYPE(DType::f64)
#undef GETTER_FOR_TYPE
#define GETTER_FOR_FLOAT_SPECIAL(kDtype)                                       \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    nb::object res = nb::cast((float)data[idx]);                               \
    return res;                                                                \
  }
    GETTER_FOR_FLOAT_SPECIAL(DType::bf16)
    GETTER_FOR_FLOAT_SPECIAL(DType::f16)
#undef GETTER_FOR_FLOAT8_SPECIAL
#define GETTER_FOR_FLOAT8_SPECIAL(kDtype)                                      \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    nb::object res = nb::cast((float)data[idx]);                               \
    return res;                                                                \
  }
    GETTER_FOR_FLOAT8_SPECIAL(DType::f8e8m0fnu)
    GETTER_FOR_FLOAT8_SPECIAL(DType::f8e4m3fn)
    GETTER_FOR_FLOAT8_SPECIAL(DType::f8e4m3fnuz)
    GETTER_FOR_FLOAT8_SPECIAL(DType::f8e5m2)
    GETTER_FOR_FLOAT8_SPECIAL(DType::f8e5m2fnuz)
#undef GETTER_FOR_FLOAT8_SPECIAL
  default: {
    throw nb::type_error("__getitem__ not implemented for dtype");
  }
  }
}

// Sets an item of a given dtype at a given index. Currently supports
// single-item sets.
void Python::Buffer::setMemoryItem(size_t idx, DType dtype, nb::object &value) {
  switch (dtype.getValue()) {
  case DType::kBool: {
    // Allow setting bools from python ints or bools
    auto val = nb::int_(value);
    auto data = reinterpret_cast<uint8_t *>(storage->data());
    data[idx] = nb::cast<uint8_t>(val);
    break;
  }
#define SETTER_FOR_TYPE(kDtype)                                                \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    auto val = nb::cast<Ty>(value);                                            \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    data[idx] = val;                                                           \
    break;                                                                     \
  }
    SETTER_FOR_TYPE(DType::si8)
    SETTER_FOR_TYPE(DType::si16)
    SETTER_FOR_TYPE(DType::si32)
    SETTER_FOR_TYPE(DType::si64)
    SETTER_FOR_TYPE(DType::ui8)
    SETTER_FOR_TYPE(DType::ui16)
    SETTER_FOR_TYPE(DType::ui32)
    SETTER_FOR_TYPE(DType::ui64)
    SETTER_FOR_TYPE(DType::f32)
    SETTER_FOR_TYPE(DType::f64)
#undef SETTER_FOR_TYPE
#define SETTER_FOR_FLOAT_SPECIAL(kDtype)                                       \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    Ty val = (Ty)nb::cast<float>(value);                                       \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    data[idx] = val;                                                           \
    break;                                                                     \
  }
    SETTER_FOR_FLOAT_SPECIAL(DType::bf16)
    SETTER_FOR_FLOAT_SPECIAL(DType::f16)
#undef SETTER_FOR_FLOAT_SPECIAL
#define SETTER_FOR_FLOAT8_SPECIAL(kDtype)                                      \
  case kDtype: {                                                               \
    using Ty = typename CXXTypeForDType<kDtype>::CXXType;                      \
    auto data = reinterpret_cast<Ty *>(storage->data());                       \
    data[idx] = Ty(nb::cast<float>(value));                                    \
    break;                                                                     \
  }
    SETTER_FOR_FLOAT8_SPECIAL(DType::f8e8m0fnu)
    SETTER_FOR_FLOAT8_SPECIAL(DType::f8e4m3fn)
    SETTER_FOR_FLOAT8_SPECIAL(DType::f8e4m3fnuz)
    SETTER_FOR_FLOAT8_SPECIAL(DType::f8e5m2)
    SETTER_FOR_FLOAT8_SPECIAL(DType::f8e5m2fnuz)
#undef SETTER_FOR_FLOAT8_SPECIAL
  default: {
    throw nb::type_error("__setitem__ not implemented for dtype");
  }
  }
}

ErrorOr<std::shared_ptr<Python::BufferStorage>>
Python::Buffer::getContiguousStorage() const {
  size_t byteOffset = spec.getEltType().getSizeInBytes(startOffset);
  return createStorageView(*storage, spec, byteOffset);
}
