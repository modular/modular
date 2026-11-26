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

#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/PlatformUtils.h"
#include "Support/Preprocessor.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

// `madvise(MADV_WILLNEED)` is worth issuing on macOS only. Linux's mmap read
// path already read-aheads aggressively, and its `MADV_WILLNEED` submits the
// I/O synchronously, which serializes the reads a cold load depends on running
// in parallel: on a cold 2 GiB shard of 52 KiB tensors it measured 2.2x slower
// than the faults it replaces, and it was neutral on 64 MiB tensors.
#if MODULAR_MACOS
#include <sys/mman.h>
#define SAFETENSOR_PREFETCH_PAGES 1
#else
#define SAFETENSOR_PREFETCH_PAGES 0
#endif

using namespace M;

static constexpr size_t kHeaderBinarySize = 8;
static constexpr size_t kHeaderMaxSize = 100'000'000;

#define DEBUG_TYPE "safetensor"

//===----------------------------------------------------------------------===//
// Page-cache hints
//===----------------------------------------------------------------------===//

/// Asks the kernel to read `[offset, offset + length)` of `buffer` ahead of the
/// accesses that follow.
///
/// macOS pays a minor fault per page and issues a small readahead window per
/// fault, so a cold 6 GiB shard of 64 MiB tensors loads at 0.7 GB/s;
/// `MADV_WILLNEED` submits the range in one call instead, which takes the same
/// load to 3.6 GB/s, and 3.3 GB/s to 10 GB/s once the pages are cached.
///
/// Best-effort: the advice may be ignored, and a buffer that is not a
/// page-aligned file mapping fails the call harmlessly.
static void prefetchPages(const M::Driver::DeviceBuffer &buffer, size_t offset,
                          size_t length) {
#if SAFETENSOR_PREFETCH_PAGES
  const size_t bufferSize = buffer.byteSize();
  if (offset >= bufferSize || length == 0)
    return;
  length = std::min(length, bufferSize - offset);

  // `madvise` requires a page-aligned address, while tensor sub-buffers are
  // aligned to the tensor only, so widen to the enclosing pages.
  const size_t pageSize = llvm::sys::Process::getPageSizeEstimate();
  const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.ptr());
  const uintptr_t firstPage = base + (offset / pageSize) * pageSize;
  const uintptr_t lastPage =
      base + ((offset + length + pageSize - 1) / pageSize) * pageSize;

  if (::madvise(reinterpret_cast<void *>(firstPage), lastPage - firstPage,
                MADV_WILLNEED) != 0) {
    LLVM_DEBUG(llvm::dbgs() << "madvise(MADV_WILLNEED) failed: "
                            << std::strerror(errno) << "\n");
  }
#else
  (void)buffer;
  (void)offset;
  (void)length;
#endif
}

// f8e4m3 maps to f8e4m3fn intentionally, matching HF's `safetensor`.
#define TYPE_MAPPING(X)                                                        \
  X(kBool, BOOL)                                                               \
  X(ui8, U8)                                                                   \
  X(ui16, U16)                                                                 \
  X(ui32, U32)                                                                 \
  X(ui64, U64)                                                                 \
  X(si8, I8)                                                                   \
  X(si16, I16)                                                                 \
  X(si32, I32)                                                                 \
  X(si64, I64)                                                                 \
  X(f4e2m1fn, F4)                                                              \
  X(f8e8m0fnu, F8_E8M0)                                                        \
  X(f8e4m3fn, F8_E4M3)                                                         \
  X(f8e5m2, F8_E5M2)                                                           \
  X(f16, F16)                                                                  \
  X(f32, F32)                                                                  \
  X(f64, F64)                                                                  \
  X(bf16, BF16)

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static ErrorOr<DType> getDType(std::optional<StringRef> nameOr) {
  if (!nameOr)
    return Error("unknown dtype");

  StringRef name = *nameOr;

#define DISPATCH_DTYPE(DTYPE, NAME)                                            \
  if (name == STRINGIFY(NAME))                                                 \
    return DType::DTYPE;

  TYPE_MAPPING(DISPATCH_DTYPE)
#undef DISPATCH_DTYPE

  return Error(Twine("unknown dtype: ") + name);
}

static ErrorOr<std::string> fromDType(DType dtype) {
  switch (dtype.getValue()) {
  default:
    return Error(Twine("cannot convert dtype: ") + dtype.getAsString());

#define DISPATCH_DTYPE(DTYPE, NAME)                                            \
  case DType::DTYPE:                                                           \
    return STRINGIFY(NAME);

    TYPE_MAPPING(DISPATCH_DTYPE)
#undef DISPATCH_DTYPE
  }

  llvm_unreachable("unhandled dtype");
  return "";
}

//===----------------------------------------------------------------------===//
// SafeTensor::TensorInfo
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensor::TensorInfo>
SafeTensor::TensorInfo::get(const llvm::json::Object &tensorInfo) {
  auto dtypeData = tensorInfo.find("dtype");
  if (dtypeData == tensorInfo.end())
    return Error("dtype is not present in the tensor info");
  ErrorOr<DType> dtype = getDType(dtypeData->second.getAsString());
  if (dtype.isError())
    return dtype.takeError();

  auto shapeData = tensorInfo.find("shape");
  if (shapeData == tensorInfo.end())
    return Error("shape is not present in the tensor info");
  auto shapeArray = shapeData->second.getAsArray();
  if (!shapeArray)
    return Error("shape is not an array");

  SmallVector<size_t> shape;
  shape.reserve(shapeArray->size());
  for (const auto &shapeValue : *shapeArray) {
    std::optional<int64_t> shapeValueAsInteger = shapeValue.getAsInteger();
    if (!shapeValueAsInteger || *shapeValueAsInteger < 0)
      return Error("shape value is not a non-negative integer");
    shape.push_back(*shapeValueAsInteger);
  }

  auto dataOffsets = tensorInfo.find("data_offsets");
  if (dataOffsets == tensorInfo.end())
    return Error("data_offsets is not present in the tensor info");
  auto dataOffsetsArrayOr = dataOffsets->second.getAsArray();
  if (!dataOffsetsArrayOr)
    return Error("data_offsets is not an array");

  const llvm::json::Array &dataOffsetsArray = *dataOffsetsArrayOr;
  if (dataOffsetsArray.size() != 2)
    return Error("data_offsets must have exactly two elements");

  std::optional<int64_t> dataOffsets0 = dataOffsetsArray[0].getAsInteger();
  std::optional<int64_t> dataOffsets1 = dataOffsetsArray[1].getAsInteger();
  if (!dataOffsets0 || !dataOffsets1)
    return Error("data_offsets values are not integers");
  if (*dataOffsets0 < 0 || *dataOffsets1 < *dataOffsets0)
    return Error("data_offsets must be non-negative and ordered");

  return SafeTensor::TensorInfo(*dtype, shape, {*dataOffsets0, *dataOffsets1});
}

size_t SafeTensor::TensorInfo::getSizeInBytes() const {
  return offsets.second - offsets.first;
}

llvm::json::Value SafeTensor::TensorInfo::toJSON() const {
  llvm::json::Object json;
  json["dtype"] = *fromDType(dtype);
  json["shape"] = llvm::json::Array(shape);
  json["data_offsets"] = llvm::json::Array({offsets.first, offsets.second});
  return json;
}

//===----------------------------------------------------------------------===//
// SafeTensor
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensor> SafeTensor::get(M::Driver::DeviceContextRef device,
                                    M::Driver::DeviceBufferRef parentBuffer) {
  return get(std::move(device), std::move(parentBuffer),
             /*isFileMapping=*/false);
}

ErrorOr<SafeTensor> SafeTensor::get(M::Driver::DeviceContextRef device,
                                    M::Driver::DeviceBufferRef parentBuffer,
                                    bool isFileMapping) {
  size_t bufferSize = parentBuffer->byteSize();

  if (bufferSize < kHeaderBinarySize)
    return Error("file is too small to be a safetensor");

  // Little-endian, unaligned.
  uint64_t headerLength =
      llvm::support::endian::read<uint64_t, llvm::support::unaligned>(
          parentBuffer->ptr(), llvm::endianness::little);

  LLVM_DEBUG(llvm::outs() << "headerLength = " << headerLength << "\n");

  if (headerLength > kHeaderMaxSize)
    return Error("header length is greater than the max header size");

  if (headerLength + kHeaderBinarySize > bufferSize)
    return Error("header length is greater than the file size");

  // Start the header read while the parser walks it: the header is up to 100 MB
  // of the file and nothing else touches those pages.
  if (isFileMapping)
    prefetchPages(*parentBuffer, kHeaderBinarySize, headerLength);

  StringRef jsonString(reinterpret_cast<const char *>(parentBuffer->ptr()) +
                           kHeaderBinarySize,
                       headerLength);

  if (!jsonString.starts_with("{"))
    return Error("the header is not a valid json object");

  LLVM_DEBUG(llvm::outs() << "headerLength = " << headerLength << "\n");
  LLVM_DEBUG(llvm::outs() << "jsonString = " << jsonString << "\n");

  llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonString);
  if (!jsonValue)
    return Error("failed to deserialize safetensor");

  auto *jsonValueAsObject = jsonValue->getAsObject();
  if (!jsonValueAsObject)
    return Error("failed to convert JSON value to object");

  // The data region starts after the header and is what every tensor's
  // `data_offsets` indexes into.
  const size_t dataRegionSize = bufferSize - kHeaderBinarySize - headerLength;

  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
  tensorInfo.reserve(jsonValueAsObject->size());
  for (const auto &[key, value] : *jsonValueAsObject) {
    if (key == "__metadata__") {
      if (!metadata.empty())
        return Error("duplicate metadata while deserializing safetensor");
      auto *metadataObject = value.getAsObject();
      if (!metadataObject)
        return Error("metadata is not a JSON object");
      metadata = *metadataObject;
      continue;
    }

    auto *tensorInfoObject = value.getAsObject();
    if (!tensorInfoObject)
      return Error(Twine("tensor info is not a JSON object: ") + key.str());

    auto tensorInfoOr = TensorInfo::get(*tensorInfoObject);
    if (tensorInfoOr.isError())
      return tensorInfoOr.takeError();
    if (tensorInfoOr->offsets.second > dataRegionSize)
      return Error(Twine("tensor '") + key.str() +
                   "' has data_offsets outside the data region");

    auto [_, seen] = tensorInfo.try_emplace(key.str(), *tensorInfoOr);
    if (!seen)
      return Error(Twine("duplicate tensor name: ") + key.str());
  }

  return SafeTensor(headerLength, std::move(device), std::move(parentBuffer),
                    std::move(metadata), std::move(tensorInfo), isFileMapping);
}

ErrorOr<SafeTensor> SafeTensor::get(const std::filesystem::path &path,
                                    M::Driver::DeviceContextRef device) {
  auto result = device->memmapFile(path);
  if (result)
    return result.takeError();

  return get(std::move(device), std::move(*result), /*isFileMapping=*/true);
}

ErrorOr<SafeTensor::TensorInfo>
SafeTensor::getTensorInfo(StringRef name) const {
  auto it = tensorInfo.find(name.str());
  if (it == tensorInfo.end()) {
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor");
  }
  return it->second;
}

M::Driver::DeviceBufferRef
SafeTensor::getTensorMemory(const TensorInfo &info) const {
  const size_t byteOffset =
      kHeaderBinarySize + headerLength + info.offsets.first;
  const size_t byteSize = info.getSizeInBytes();

  // Hand the kernel the whole tensor before the caller reads it: the copy that
  // follows then runs on resident pages instead of faulting them in one at a
  // time.
  if (isFileMapping)
    prefetchPages(*parentBuffer, byteOffset, byteSize);

  auto subBuffer = parentBuffer->createSubBuffer(byteOffset, byteSize);
  assert(!subBuffer.isError() &&
         "sub-buffer dimensions should already have been validated");

  return std::move(*subBuffer);
}

//===----------------------------------------------------------------------===//
// SafeTensorBundle
//===----------------------------------------------------------------------===//

ErrorOr<SafeTensorBundle>
SafeTensorBundle::get(ArrayRef<std::filesystem::path> paths,
                      M::Driver::DeviceContextRef ctx) {
  DenseMap<std::string, size_t> tensorNameToIndex;
  SmallVector<SafeTensor> files;
  files.reserve(paths.size());

  for (const auto &[i, path] : llvm::enumerate(paths)) {
    auto safeTensorOr = SafeTensor::get(path, ctx);
    if (safeTensorOr.isError())
      return safeTensorOr.takeError();
    for (const auto &[name, tensorInfo] : safeTensorOr->getTensorInfo())
      tensorNameToIndex[name] = i;
    files.push_back(std::move(*safeTensorOr));
  }

  return SafeTensorBundle(std::move(tensorNameToIndex), std::move(files));
}

ErrorOr<M::Driver::DeviceBufferRef>
SafeTensorBundle::getTensorMemory(StringRef name) const {
  // Both maps are keyed by `std::string`, so build the key once and reuse it
  // rather than allocating one per lookup.
  const std::string key = name.str();
  auto it = tensorNameToIndex.find(key);
  if (it == tensorNameToIndex.end())
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor bundle");

  const SafeTensor &file = files[it->second];
  auto it2 = file.getTensorInfo().find(key);

  assert(it2 != file.getTensorInfo().end());

  return file.getTensorMemory(it2->second);
}
