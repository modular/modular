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

#include "GraphCompiler/Support/GGUF/GGUFWriter.h"
#include "GraphCompiler/Support/GGUF/GGUFFormat.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <limits>

using namespace M;

namespace {

/// "GGUF", little-endian.
constexpr uint32_t kMagic = 0x46554747;

/// The GGUF metadata value types this writer emits.
enum MetadataValueType : uint32_t {
  kUInt32 = 4,
  kBool = 7,
  kString = 8,
  kUInt64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

void appendUInt(std::string &out, uint64_t value, size_t numBytes) {
  for (size_t i = 0; i < numBytes; ++i)
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

void appendString(std::string &out, StringRef value) {
  appendUInt(out, value.size(), 8);
  out.append(value.data(), value.size());
}

uint64_t alignTo(uint64_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

} // namespace

GGUFWriter::GGUFWriter(size_t alignment) : alignment(alignment) {}

void GGUFWriter::appendMetadata(MetadataEntry entry) {
  for (MetadataEntry &existing : metadata) {
    if (existing.key == entry.key) {
      existing = std::move(entry);
      return;
    }
  }
  metadata.push_back(std::move(entry));
}

void GGUFWriter::setMetadataString(StringRef key, StringRef value) {
  MetadataEntry entry{key.str(), kString, {}};
  appendString(entry.value, value);
  appendMetadata(std::move(entry));
}

void GGUFWriter::setMetadataBool(StringRef key, bool value) {
  MetadataEntry entry{key.str(), kBool, {}};
  appendUInt(entry.value, value ? 1 : 0, 1);
  appendMetadata(std::move(entry));
}

void GGUFWriter::setMetadataInt(StringRef key, int64_t value) {
  MetadataEntry entry{key.str(), kInt64, {}};
  appendUInt(entry.value, static_cast<uint64_t>(value), 8);
  appendMetadata(std::move(entry));
}

void GGUFWriter::setMetadataUInt(StringRef key, uint64_t value) {
  MetadataEntry entry{key.str(), kUInt64, {}};
  appendUInt(entry.value, value, 8);
  appendMetadata(std::move(entry));
}

void GGUFWriter::setMetadataFloat(StringRef key, double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  MetadataEntry entry{key.str(), kFloat64, {}};
  appendUInt(entry.value, bits, 8);
  appendMetadata(std::move(entry));
}

ErrorOrSuccess GGUFWriter::addTensor(StringRef name, uint32_t tensorType,
                                     ArrayRef<size_t> shape,
                                     ArrayRef<char> data) {
  if (name.empty())
    return Error("tensor name must not be empty");

  for (const TensorEntry &tensor : tensors) {
    if (tensor.name == name)
      return Error(Twine("duplicate tensor name: ") + name);
  }

  auto layout = getGGUFTensorLayout(tensorType);
  if (!layout)
    return Error(
        Twine("cannot write tensor '") + name + "' of unsupported GGUF type " +
        getGGUFTensorTypeName(tensorType) + " (" + Twine(tensorType) + ")");

  uint64_t numElements = 1;
  for (size_t dim : shape) {
    if (dim != 0 and numElements > std::numeric_limits<uint64_t>::max() / dim)
      return Error(Twine("tensor '") + name + "' has a shape that overflows");
    numElements *= dim;
  }

  if (layout->elementsPerBlock > 1 and !shape.empty() and
      shape.back() % layout->elementsPerBlock != 0)
    return Error(Twine("tensor '") + name + "' holds " + Twine(shape.back()) +
                 " elements along its last dimension, which is not a multiple "
                 "of the " +
                 Twine(layout->elementsPerBlock) + "-element block size of " +
                 getGGUFTensorTypeName(tensorType));

  uint64_t expectedBytes =
      numElements / layout->elementsPerBlock * layout->bytesPerBlock;
  if (data.size() != expectedBytes)
    return Error(Twine("tensor '") + name + "' has " + Twine(data.size()) +
                 " bytes, but its shape and type need " + Twine(expectedBytes));

  tensors.push_back(TensorEntry{
      name.str(),
      tensorType,
      SmallVector<size_t>(shape.begin(), shape.end()),
      std::string(data.begin(), data.end()),
  });
  return ErrorOrSuccess();
}

ErrorOr<std::string> GGUFWriter::serialize() const {
  if (alignment == 0 or (alignment & (alignment - 1)) != 0)
    return Error(Twine("alignment ") + Twine(alignment) +
                 " is not a power of two");

  std::string header;
  appendUInt(header, kMagic, 4);
  appendUInt(header, kVersion, 4);
  appendUInt(header, tensors.size(), 8);
  // The alignment is recorded only when it differs from the format's default,
  // which every reader assumes.
  bool writeAlignment = alignment != kDefaultAlignment;
  appendUInt(header, metadata.size() + (writeAlignment ? 1 : 0), 8);

  for (const MetadataEntry &entry : metadata) {
    appendString(header, entry.key);
    appendUInt(header, entry.valueType, 4);
    header.append(entry.value);
  }
  if (writeAlignment) {
    appendString(header, "general.alignment");
    appendUInt(header, kUInt32, 4);
    appendUInt(header, alignment, 4);
  }

  // Tensor descriptors carry the offset of the tensor within the data section,
  // so they cannot be emitted until the data section offset is known.
  std::string descriptors;
  uint64_t dataSize = 0;
  for (const TensorEntry &tensor : tensors) {
    dataSize = alignTo(dataSize, alignment);
    appendString(descriptors, tensor.name);
    appendUInt(descriptors, tensor.shape.size(), 4);
    // Dimensions are recorded fastest-varying-first.
    for (size_t i = tensor.shape.size(); i > 0; --i)
      appendUInt(descriptors, tensor.shape[i - 1], 8);
    appendUInt(descriptors, tensor.tensorType, 4);
    appendUInt(descriptors, dataSize, 8);
    dataSize += tensor.data.size();
  }

  uint64_t dataOffset = alignTo(header.size() + descriptors.size(), alignment);
  std::string out = std::move(header);
  out += descriptors;
  out.resize(dataOffset, '\0');

  uint64_t written = 0;
  for (const TensorEntry &tensor : tensors) {
    uint64_t start = alignTo(written, alignment);
    out.resize(dataOffset + start, '\0');
    out += tensor.data;
    written = start + tensor.data.size();
  }

  return out;
}

ErrorOrSuccess GGUFWriter::write(const std::filesystem::path &path) const {
  auto contentsOr = serialize();
  if (contentsOr.isError())
    return contentsOr.takeError();

  std::error_code error;
  llvm::raw_fd_ostream out(path.string(), error, llvm::sys::fs::OF_None);
  if (error)
    return Error(Twine("cannot open '") + path.string() +
                 "' for writing: " + error.message());

  out << *contentsOr;
  out.close();
  if (out.has_error()) {
    out.clear_error();
    return Error(Twine("cannot write '") + path.string() + "'");
  }
  return ErrorOrSuccess();
}
