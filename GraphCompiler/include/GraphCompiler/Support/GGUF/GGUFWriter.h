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
// GGUF writer, the counterpart of the GGUF reader.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_GGUF_WRITER_H
#define RUNTIME_GGUF_WRITER_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace M {

/// Serializes metadata and tensor data into a GGUF (GPT-Generated Unified
/// Format) file.
///
/// Metadata and tensors are added in insertion order, which is the order they
/// appear in the file. Tensor data is copied into the writer, so the caller's
/// buffers may be released before `write()`.
struct MODULAR_CXX_EXPORT GGUFWriter {
  /// The GGUF version this writer emits.
  static constexpr uint32_t kVersion = 3;
  /// The alignment of the tensor data section used when none is given.
  static constexpr size_t kDefaultAlignment = 32;

  explicit GGUFWriter(size_t alignment = kDefaultAlignment);

  size_t getAlignment() const { return alignment; }

  // The setters are named after their value type rather than overloaded: a
  // string literal is a better match for `bool` than for `StringRef`, so an
  // overloaded `setMetadata` would quietly write it as a boolean.

  /// Adds or replaces a string-valued metadata entry.
  void setMetadataString(StringRef key, StringRef value);

  /// Adds or replaces a boolean-valued metadata entry.
  void setMetadataBool(StringRef key, bool value);

  /// Adds or replaces a signed integer metadata entry, written as an `INT64`.
  void setMetadataInt(StringRef key, int64_t value);

  /// Adds or replaces an unsigned integer metadata entry, written as a
  /// `UINT64`.
  void setMetadataUInt(StringRef key, uint64_t value);

  /// Adds or replaces a floating point metadata entry, written as a `FLOAT64`.
  void setMetadataFloat(StringRef key, double value);

  /// Adds a tensor.
  ///
  /// `tensorType` is the GGML type of `data`, and `shape` is the logical shape,
  /// widest dimension first. `data` must hold exactly the tensor: its number of
  /// elements, rounded to whole quantization blocks for block-quantized types,
  /// times the size of the stored element.
  ///
  /// Returns an error for a duplicate name, a tensor type this writer does not
  /// support, a shape with a partial quantization block, or a byte count that
  /// does not match the shape.
  ErrorOrSuccess addTensor(StringRef name, uint32_t tensorType,
                           ArrayRef<size_t> shape, ArrayRef<char> data);

  /// Serializes the file into memory.
  ErrorOr<std::string> serialize() const;

  /// Writes the file to `path`, replacing it if it exists.
  ErrorOrSuccess write(const std::filesystem::path &path) const;

private:
  struct MetadataEntry {
    std::string key;
    /// The GGUF metadata value type.
    uint32_t valueType;
    /// The value, already encoded little-endian as the type requires.
    std::string value;
  };

  struct TensorEntry {
    std::string name;
    uint32_t tensorType;
    SmallVector<size_t> shape;
    std::string data;
  };

  /// Adds `entry`, replacing an entry with the same key in place.
  void appendMetadata(MetadataEntry entry);

  size_t alignment;
  SmallVector<MetadataEntry> metadata;
  SmallVector<TensorEntry> tensors;
};

} // namespace M

#endif // RUNTIME_GGUF_WRITER_H
