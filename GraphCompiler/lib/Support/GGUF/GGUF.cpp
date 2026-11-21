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

#include "GraphCompiler/Support/GGUF/GGUF.h"
#include "GraphCompiler/Support/GGUF/GGUFFormat.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstring>
#include <limits>

using namespace M;

#define DEBUG_TYPE "gguf"

namespace {

/// "GGUF", little-endian.
constexpr uint32_t kMagic = 0x46554747;
constexpr uint32_t kVersion2 = 2;
constexpr uint32_t kVersion3 = 3;
/// Used when the file does not set `general.alignment`.
constexpr size_t kDefaultAlignment = 32;

/// The GGUF metadata value types.
enum MetadataValueType : uint32_t {
  kUInt8 = 0,
  kInt8 = 1,
  kUInt16 = 2,
  kInt16 = 3,
  kUInt32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUInt64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

/// The size in bytes of a fixed-size metadata value, or 0 for the variable
/// sized string and array types.
size_t getMetadataValueSize(uint32_t type) {
  switch (type) {
  case kUInt8:
  case kInt8:
  case kBool:
    return 1;
  case kUInt16:
  case kInt16:
    return 2;
  case kUInt32:
  case kInt32:
  case kFloat32:
    return 4;
  case kUInt64:
  case kInt64:
  case kFloat64:
    return 8;
  default:
    return 0;
  }
}

/// A bounds-checked little-endian cursor over the file's bytes.
class Reader {
public:
  Reader(const uint8_t *data, size_t size)
      : data(data), size(size), position(0) {}

  size_t offset() const { return position; }
  size_t remaining() const { return size - position; }

  /// Reads a `numBytes`-wide little-endian unsigned integer.
  ErrorOr<uint64_t> readUInt(size_t numBytes) {
    if (numBytes > remaining())
      return Error(Twine("unexpected end of file at offset ") +
                   Twine(position));
    uint64_t value = 0;
    for (size_t i = 0; i < numBytes; ++i)
      value |= static_cast<uint64_t>(data[position + i]) << (8 * i);
    position += numBytes;
    return value;
  }

  ErrorOr<uint32_t> readUInt32() {
    auto valueOr = readUInt(4);
    if (valueOr.isError())
      return valueOr.takeError();
    return static_cast<uint32_t>(*valueOr);
  }

  /// Skips `count` bytes, failing if the file ends first.
  ErrorOrSuccess skip(uint64_t count) {
    if (count > remaining())
      return Error(Twine("unexpected end of file at offset ") +
                   Twine(position));
    position += count;
    return ErrorOrSuccess();
  }

  ErrorOr<StringRef> readString() {
    auto lengthOr = readUInt(8);
    if (lengthOr.isError())
      return lengthOr.takeError();
    uint64_t length = *lengthOr;
    if (length > remaining())
      return Error(Twine("string length ") + Twine(length) + " at offset " +
                   Twine(position) + " exceeds the file size");
    StringRef str(reinterpret_cast<const char *>(data) + position, length);
    position += length;
    return str;
  }

  /// Skips `count` strings, failing as `readString` would on the first one
  /// that does not fit.
  ErrorOrSuccess skipStrings(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      auto stringOr = readString();
      if (stringOr.isError())
        return stringOr.takeError();
    }
    return ErrorOrSuccess();
  }

  /// Reads `count` 8-byte little-endian integers into `out` in reverse order.
  ///
  /// The bounds check comes first, so a corrupt count fails without sizing
  /// `out`, and the error names the same offset as reading the integers one
  /// at a time would.
  ErrorOrSuccess readReversedUInt64s(uint64_t count,
                                     SmallVectorImpl<size_t> &out) {
    if (count > remaining() / 8)
      return Error(Twine("unexpected end of file at offset ") +
                   Twine(position + remaining() / 8 * 8));
    out.resize_for_overwrite(count);
    for (size_t i = 0; i < count; ++i) {
      uint64_t value = 0;
      for (size_t byte = 0; byte < 8; ++byte)
        value |= static_cast<uint64_t>(data[position + byte]) << (8 * byte);
      out[count - 1 - i] = static_cast<size_t>(value);
      position += 8;
    }
    return ErrorOrSuccess();
  }

private:
  const uint8_t *data;
  size_t size;
  size_t position;
};

/// Reads one metadata value.
///
/// Fixed-size values are recorded in `metadata` in their natural JSON form.
/// Arrays are skipped rather than recorded: their payload is dominated by the
/// tokenizer's vocabulary and merges tables, tens of megabytes in a typical
/// file, and reading tensors does not need them.
ErrorOrSuccess readMetadataValue(Reader &reader, uint32_t valueType,
                                 llvm::json::Object &metadata, StringRef key) {
  switch (valueType) {
  case kString: {
    auto valueOr = reader.readString();
    if (valueOr.isError())
      return valueOr.takeError();
    metadata[key] = *valueOr;
    return ErrorOrSuccess();
  }
  case kBool: {
    auto valueOr = reader.readUInt(1);
    if (valueOr.isError())
      return valueOr.takeError();
    metadata[key] = *valueOr != 0;
    return ErrorOrSuccess();
  }
  case kUInt8:
  case kUInt16:
  case kUInt32:
  case kUInt64: {
    auto valueOr = reader.readUInt(getMetadataValueSize(valueType));
    if (valueOr.isError())
      return valueOr.takeError();
    // GGUF counts and sizes fit in an int64; anything larger has no JSON
    // integer form and is reported as negative, which callers can detect.
    metadata[key] = static_cast<int64_t>(*valueOr);
    return ErrorOrSuccess();
  }
  case kInt8:
  case kInt16:
  case kInt32:
  case kInt64: {
    size_t numBytes = getMetadataValueSize(valueType);
    auto valueOr = reader.readUInt(numBytes);
    if (valueOr.isError())
      return valueOr.takeError();
    // Sign-extend from the value's width.
    unsigned shift = 64 - 8 * numBytes;
    metadata[key] = static_cast<int64_t>(*valueOr << shift) >> shift;
    return ErrorOrSuccess();
  }
  case kFloat32: {
    auto valueOr = reader.readUInt(4);
    if (valueOr.isError())
      return valueOr.takeError();
    uint32_t bits = static_cast<uint32_t>(*valueOr);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    metadata[key] = static_cast<double>(value);
    return ErrorOrSuccess();
  }
  case kFloat64: {
    auto valueOr = reader.readUInt(8);
    if (valueOr.isError())
      return valueOr.takeError();
    uint64_t bits = *valueOr;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    metadata[key] = value;
    return ErrorOrSuccess();
  }
  case kArray: {
    auto elementTypeOr = reader.readUInt32();
    if (elementTypeOr.isError())
      return elementTypeOr.takeError();
    auto countOr = reader.readUInt(8);
    if (countOr.isError())
      return countOr.takeError();
    uint32_t elementType = *elementTypeOr;
    uint64_t count = *countOr;
    if (elementType == kString)
      return reader.skipStrings(count);
    size_t elementSize = getMetadataValueSize(elementType);
    if (elementSize == 0)
      return Error(Twine("metadata array '") + key +
                   "' has an unsupported element type " + Twine(elementType));
    // Compare before multiplying: a crafted count must not wrap.
    if (count > reader.remaining() / elementSize)
      return Error(Twine("unexpected end of file at offset ") +
                   Twine(reader.offset()));
    return reader.skip(count * elementSize);
  }
  default:
    return Error(Twine("metadata entry '") + key +
                 "' has an unknown value type " + Twine(valueType));
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// GGUF::TensorInfo
//===----------------------------------------------------------------------===//

ErrorOr<GGUF::TensorInfo>
GGUF::TensorInfo::get(uint32_t tensorType, ArrayRef<size_t> shape,
                      std::pair<size_t, size_t> offsets) {
  if (offsets.second < offsets.first)
    return Error("tensor data offsets are not ordered");

  // A block-quantized tensor stores whole blocks along its contiguous
  // dimension, the last one.
  if (auto layout = getGGUFTensorLayout(tensorType)) {
    if (layout->elementsPerBlock > 1 and !shape.empty() and
        shape.back() % layout->elementsPerBlock != 0)
      return Error(Twine("shape has ") + Twine(shape.back()) +
                   " elements along its last dimension, which is not a "
                   "multiple of the " +
                   Twine(layout->elementsPerBlock) + "-element block size of " +
                   getGGUFTensorTypeName(tensorType));
  }

  return TensorInfo(tensorType, shape, offsets);
}

std::optional<size_t> GGUF::TensorInfo::getQuantizationBlockSize() const {
  auto layout = getGGUFTensorLayout(tensorType);
  if (!layout or layout->elementsPerBlock == 1)
    return std::nullopt;
  return layout->elementsPerBlock;
}

StringRef GGUF::TensorInfo::getQuantizationName() const {
  if (auto layout = getGGUFTensorLayout(tensorType))
    return layout->quantizationName;
  return "";
}

ErrorOr<DType> GGUF::TensorInfo::getDType() const {
  auto layout = getGGUFTensorLayout(tensorType);
  if (!layout)
    return Error(Twine("unsupported GGUF tensor type ") +
                 getGGUFTensorTypeName(tensorType) + " (" + Twine(tensorType) +
                 ")");
  return layout->dtype;
}

ErrorOr<SmallVector<size_t>> GGUF::TensorInfo::getByteShape() const {
  auto layout = getGGUFTensorLayout(tensorType);
  if (!layout)
    return Error(Twine("unsupported GGUF tensor type ") +
                 getGGUFTensorTypeName(tensorType) + " (" + Twine(tensorType) +
                 ")");

  SmallVector<size_t> byteShape(shape.begin(), shape.end());
  if (layout->elementsPerBlock > 1 and !byteShape.empty())
    byteShape.back() =
        byteShape.back() / layout->elementsPerBlock * layout->bytesPerBlock;
  return byteShape;
}

size_t GGUF::TensorInfo::getSizeInBytes() const {
  return offsets.second - offsets.first;
}

llvm::json::Value GGUF::TensorInfo::toJSON() const {
  llvm::json::Object json;
  json["tensor_type"] = getGGUFTensorTypeName(tensorType);
  if (auto dtypeOr = getDType())
    json["dtype"] = dtypeOr->getAsString();
  else
    json["dtype"] = "unsupported";
  if (auto byteShapeOr = getByteShape())
    json["shape"] = llvm::json::Array(*byteShapeOr);
  json["element_shape"] = llvm::json::Array(shape);
  json["data_offsets"] = llvm::json::Array({offsets.first, offsets.second});
  return json;
}

//===----------------------------------------------------------------------===//
// GGUF
//===----------------------------------------------------------------------===//

ErrorOr<GGUF> GGUF::get(M::Driver::DeviceContextRef device,
                        M::Driver::DeviceBufferRef parentBuffer) {
  size_t bufferSize = parentBuffer->byteSize();
  Reader reader(reinterpret_cast<const uint8_t *>(parentBuffer->ptr()),
                bufferSize);

  auto magicOr = reader.readUInt32();
  if (magicOr.isError())
    return magicOr.takeError();
  if (*magicOr != kMagic)
    return Error("file does not start with the GGUF magic");

  auto versionOr = reader.readUInt32();
  if (versionOr.isError())
    return versionOr.takeError();
  uint32_t version = *versionOr;
  if (version != kVersion2 and version != kVersion3)
    return Error(Twine("unsupported GGUF version ") + Twine(version) +
                 ", expected 2 or 3");

  auto tensorCountOr = reader.readUInt(8);
  if (tensorCountOr.isError())
    return tensorCountOr.takeError();
  auto metadataCountOr = reader.readUInt(8);
  if (metadataCountOr.isError())
    return metadataCountOr.takeError();
  uint64_t tensorCount = *tensorCountOr;
  uint64_t metadataCount = *metadataCountOr;

  llvm::json::Object metadata;
  for (uint64_t i = 0; i < metadataCount; ++i) {
    auto keyOr = reader.readString();
    if (keyOr.isError())
      return keyOr.takeError();
    auto valueTypeOr = reader.readUInt32();
    if (valueTypeOr.isError())
      return valueTypeOr.takeError();

    ErrorOrSuccess result =
        readMetadataValue(reader, *valueTypeOr, metadata, *keyOr);
    if (result.isError())
      return result.takeError();
  }

  size_t alignment = kDefaultAlignment;
  if (auto *alignmentValue = metadata.get("general.alignment")) {
    auto alignmentAsInteger = alignmentValue->getAsInteger();
    if (!alignmentAsInteger)
      return Error("general.alignment is not an integer");
    if (*alignmentAsInteger <= 0)
      return Error("general.alignment is not positive");
    alignment = static_cast<size_t>(*alignmentAsInteger);
  }
  if (!llvm::isPowerOf2_64(alignment))
    return Error(Twine("alignment ") + Twine(alignment) +
                 " is not a power of two");

  struct RawTensorInfo {
    /// Points into the mapped file, which outlives this function.
    StringRef name;
    uint32_t tensorType;
    SmallVector<size_t> shape;
    uint64_t start;
  };

  // The smallest tensor descriptor: an empty name, no dimensions, the type
  // and the offset. Capping the reservation by it keeps a corrupt count from
  // reserving more than the file could describe.
  constexpr uint64_t kMinTensorDescriptorSize = 8 + 4 + 4 + 8;
  llvm::SmallVector<RawTensorInfo> rawTensorInfo;
  rawTensorInfo.reserve(std::min<uint64_t>(
      tensorCount, reader.remaining() / kMinTensorDescriptorSize));
  for (uint64_t i = 0; i < tensorCount; ++i) {
    auto nameOr = reader.readString();
    if (nameOr.isError())
      return nameOr.takeError();
    auto numDimsOr = reader.readUInt32();
    if (numDimsOr.isError())
      return numDimsOr.takeError();

    RawTensorInfo &raw = rawTensorInfo.emplace_back();
    raw.name = *nameOr;
    // Dimensions are recorded fastest-varying-first, so the logical shape is
    // their reverse.
    ErrorOrSuccess shapeResult =
        reader.readReversedUInt64s(*numDimsOr, raw.shape);
    if (shapeResult.isError())
      return shapeResult.takeError();

    auto tensorTypeOr = reader.readUInt32();
    if (tensorTypeOr.isError())
      return tensorTypeOr.takeError();
    auto offsetOr = reader.readUInt(8);
    if (offsetOr.isError())
      return offsetOr.takeError();
    raw.tensorType = *tensorTypeOr;
    raw.start = *offsetOr;
  }

  uint64_t dataOffset = llvm::alignTo(reader.offset(), alignment);
  if (dataOffset > bufferSize)
    return Error(Twine("the tensor data section at offset ") +
                 Twine(dataOffset) + " starts past the end of the file");
  uint64_t dataSize = bufferSize - dataOffset;

  // Tensor descriptors carry a start offset but no size. Sizes come from the
  // tensor type's block layout; the gap to the next tensor bounds them, which
  // is also the only size information available for tensor types this reader
  // cannot size.
  //
  // Writers usually emit tensors in offset order, so the sort is skipped when
  // the starts are strictly increasing, where it could not change the order.
  // It compares only the starts, so with repeated starts it permutes exactly
  // as a sort of bare indices by start would.
  llvm::SmallVector<std::pair<uint64_t, size_t>> order;
  order.reserve(rawTensorInfo.size());
  for (auto [index, raw] : llvm::enumerate(rawTensorInfo))
    order.emplace_back(raw.start, index);
  auto startsNotIncreasing = [](const auto &a, const auto &b) {
    return a.first >= b.first;
  };
  if (llvm::adjacent_find(order, startsNotIncreasing) != order.end())
    llvm::sort(order, llvm::less_first());

  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
  for (auto [i, entry] : llvm::enumerate(order)) {
    const RawTensorInfo &raw = rawTensorInfo[entry.second];
    uint64_t gap =
        (i + 1 < order.size() ? order[i + 1].first : dataSize) - raw.start;
    if (raw.start > dataSize or gap > dataSize - raw.start)
      return Error(Twine("the tensor '") + raw.name +
                   "' starts or ends past the tensor data section");

    uint64_t size = gap;
    if (auto layout = getGGUFTensorLayout(raw.tensorType)) {
      uint64_t numElements = 1;
      for (size_t dim : raw.shape) {
        if (dim != 0 and
            numElements > std::numeric_limits<uint64_t>::max() / dim)
          return Error(Twine("the tensor '") + raw.name +
                       "' has a shape that overflows");
        numElements *= dim;
      }
      if (numElements % layout->elementsPerBlock != 0)
        return Error(
            Twine("the tensor '") + raw.name + "' holds " + Twine(numElements) +
            " elements, which is not a multiple of the " +
            Twine(layout->elementsPerBlock) + "-element block size of " +
            getGGUFTensorTypeName(raw.tensorType));
      size = numElements / layout->elementsPerBlock * layout->bytesPerBlock;
      if (size > gap)
        return Error(Twine("the tensor '") + raw.name + "' needs " +
                     Twine(size) + " bytes but only " + Twine(gap) +
                     " are left before the next tensor");
    }

    auto infoOr = TensorInfo::get(raw.tensorType, raw.shape,
                                  {raw.start, raw.start + size});
    if (infoOr.isError())
      return infoOr.takeError();
    auto [it, inserted] =
        tensorInfo.try_emplace(raw.name.str(), std::move(*infoOr));
    if (!inserted)
      return Error(Twine("duplicate tensor name: ") + raw.name);
  }

  LLVM_DEBUG(llvm::outs() << "version = " << version << "\n"
                          << "alignment = " << alignment << "\n"
                          << "dataOffset = " << dataOffset << "\n"
                          << "dataSize = " << dataSize << "\n");

  return GGUF(version, alignment, dataOffset, std::move(device),
              std::move(parentBuffer), std::move(metadata),
              std::move(tensorInfo));
}

ErrorOr<GGUF> GGUF::get(const std::filesystem::path &path,
                        M::Driver::DeviceContextRef device) {
  auto result = device->memmapFile(path);
  if (result)
    return result.takeError();

  return get(std::move(device), std::move(*result));
}

llvm::json::Value GGUF::toJSON() const {
  llvm::json::Object json;
  json["version"] = version;
  json["alignment"] = alignment;
  json["data_offset"] = dataOffset;
  // `llvm::json::Value` takes ownership, so hand it a copy.
  llvm::json::Object metadataCopy = metadata;
  json["metadata"] = llvm::json::Value(std::move(metadataCopy));
  llvm::json::Object tensors;
  for (const auto &[name, info] : tensorInfo)
    tensors[name] = info.toJSON();
  json["tensors"] = std::move(tensors);
  return json;
}

ErrorOr<GGUF::TensorInfo> GGUF::getTensorInfo(StringRef name) const {
  auto it = tensorInfo.find(name.str());
  if (it == tensorInfo.end())
    return Error(Twine("the tensor named '") + name +
                 "' was not found in the gguf file");
  return it->second;
}

M::Driver::DeviceBufferRef GGUF::getTensorMemory(const TensorInfo &info) const {
  auto subBuffer = parentBuffer->createSubBuffer(
      dataOffset + info.offsets.first, info.getSizeInBytes());
  assert(!subBuffer.isError() &&
         "sub-buffer dimensions should already have been validated");

  return std::move(*subBuffer);
}
