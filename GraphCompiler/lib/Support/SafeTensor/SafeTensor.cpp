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
#include "Support/Preprocessor.h"

using namespace M;

static constexpr size_t kHeaderBinarySize = 8;
static constexpr size_t kHeaderMaxSize = 100'000'000;

#define DEBUG_TYPE "safetensor"

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
    assert(shapeValueAsInteger && "shape value is not an integer");
    shape.push_back(*shapeValueAsInteger);
  }

  auto dataOffsets = tensorInfo.find("data_offsets");
  if (dataOffsets == tensorInfo.end())
    return Error("data_offsets is not present in the tensor info");
  auto dataOffsetsArrayOr = dataOffsets->second.getAsArray();
  if (!dataOffsetsArrayOr)
    return Error("data_offsets is not an array");

  llvm::json::Array dataOffsetsArray = *dataOffsetsArrayOr;
  if (dataOffsetsArray.size() != 2)
    return Error("data_offsets must have exactly two elements");

  std::optional<int64_t> dataOffsets0 = dataOffsetsArray[0].getAsInteger();
  std::optional<int64_t> dataOffsets1 = dataOffsetsArray[1].getAsInteger();
  assert(dataOffsets0 && "data_offsets[0] is not an integer");
  assert(dataOffsets1 && "data_offsets[1] is not an integer");

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

  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
  for (const auto &[key, value] : *jsonValueAsObject) {
    if (key == "__metadata__") {
      if (!metadata.empty())
        return Error("duplicate metadata while deserializing safetensor");
      assert(value.getAsObject() && "metadata is not a JSON object");
      metadata = *value.getAsObject();
      continue;
    }

    auto tensorInfoOr = TensorInfo::get(*value.getAsObject());
    if (tensorInfoOr.isError())
      return tensorInfoOr.takeError();
    auto [_, seen] = tensorInfo.try_emplace(key.str(), *tensorInfoOr);
    if (!seen)
      return Error(Twine("duplicate tensor name: ") + key.str());
  }

  return SafeTensor(headerLength, std::move(device), std::move(parentBuffer),
                    std::move(metadata), std::move(tensorInfo));
}

ErrorOr<SafeTensor> SafeTensor::get(const std::filesystem::path &path,
                                    M::Driver::DeviceContextRef device) {
  // TODO: Read once on the CPU path. The GPU path reads twice: metadata, then
  // the data sent to the GPU.

  auto result = device->memmapFile(path);
  if (result)
    return result.takeError();

  return get(std::move(device), std::move(*result));
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
  auto subBuffer = parentBuffer->createSubBuffer(
      8 + headerLength + info.offsets.first, info.getSizeInBytes());
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
  files.reserve(files.size());

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
  auto it = tensorNameToIndex.find(name.str());
  if (it == tensorNameToIndex.end())
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the safetensor bundle");

  const SafeTensor &file = files[it->second];

  // TODO: Avoid this .str().
  auto it2 = file.getTensorInfo().find(name.str());

  assert(it2 != file.getTensorInfo().end());

  return file.getTensorMemory(it2->second);
}
