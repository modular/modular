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
// GGUF reader which is GPU aware.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_GGUF_H
#define RUNTIME_GGUF_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"
#include "llvm/Support/JSON.h"
#include <filesystem>
#include <optional>
#include <utility>

namespace M {

/// GGUF (GPT-Generated Unified Format) reader with GPU-aware data loading.
///
/// GGUF stores a flat table of metadata key/value pairs, then a table of
/// tensor descriptors, then the tensor data. Tensor dimensions are recorded
/// fastest-varying-first, so the logical shape of a tensor is the reverse of
/// the shape in the file.
///
/// Tensor data is either plain element data or a GGML block-quantized payload,
/// which is exposed as a `uint8` buffer whose last dimension is scaled from
/// elements to blocks. The quantization scheme itself is not modeled here.
struct MODULAR_CXX_EXPORT GGUF {
  /// The `ggml_type` of a tensor, as recorded in the file.
  enum TensorType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 36,
  };

  /// How one tensor is stored: its element type, logical shape and byte range.
  ///
  /// `dtype` is the type of the stored data, which is `ui8` for every
  /// block-quantized tensor; `shape` is the logical element shape, and
  /// `getByteShape()` is the shape of the stored data (the two differ only for
  /// quantized tensors).
  struct TensorInfo {
    TensorInfo(uint32_t tensorType, ArrayRef<size_t> shape,
               std::pair<size_t, size_t> offsets)
        : tensorType(tensorType), shape(shape), offsets(offsets) {}

    static ErrorOr<TensorInfo> get(uint32_t tensorType, ArrayRef<size_t> shape,
                                   std::pair<size_t, size_t> offsets);

    /// The number of elements in one quantization block, or `std::nullopt`
    /// when the tensor is not block-quantized.
    std::optional<size_t> getQuantizationBlockSize() const;

    bool isQuantized() const { return getQuantizationBlockSize().has_value(); }

    /// The name of the tensor's quantization scheme, such as "Q4_K"; empty
    /// when the tensor is not block-quantized.
    StringRef getQuantizationName() const;

    /// The type of the stored data: the element type for plain tensors, and
    /// `ui8` for block-quantized ones.
    ///
    /// Returns an error for tensor types this reader cannot size, which are
    /// all the quantization schemes MAX does not load.
    ErrorOr<DType> getDType() const;

    /// The shape of the stored data. Equal to `shape` unless the tensor is
    /// block-quantized, in which case the last dimension is scaled from
    /// elements to bytes.
    ErrorOr<SmallVector<size_t>> getByteShape() const;

    size_t getSizeInBytes() const;

    llvm::json::Value toJSON() const;

    uint32_t tensorType;
    SmallVector<size_t> shape;
    /// Byte range of this tensor within the file's tensor data section.
    std::pair<size_t, size_t> offsets;
  };

  /// Loads and parses a GGUF file from the given data.
  static ErrorOr<GGUF> get(M::Driver::DeviceContextRef device,
                           M::Driver::DeviceBufferRef data);

  /// Loads and parses a GGUF file from the given path.
  static ErrorOr<GGUF> get(const std::filesystem::path &path,
                           M::Driver::DeviceContextRef device);

  /// Serialize the GGUF to a JSON object.
  llvm::json::Value toJSON() const;

  uint32_t getVersion() const { return version; }

  /// The alignment of the tensor data section, from `general.alignment`.
  size_t getAlignment() const { return alignment; }

  /// Scalar metadata: the file's key/value pairs that are not arrays.
  ///
  /// Array-valued entries, which are dominated by the tokenizer's vocabulary
  /// and merges tables and are not needed to read tensors, are skipped.
  const llvm::json::Object &getMetadata() const { return metadata; }

  const llvm::DenseMap<std::string, TensorInfo> &getTensorInfo() const {
    return tensorInfo;
  }

  /// Gets metadata for a specific tensor.
  ErrorOr<GGUF::TensorInfo> getTensorInfo(StringRef name) const;

  /// Gets a tensor from given info.
  M::Driver::DeviceBufferRef getTensorMemory(const TensorInfo &info) const;

private:
  GGUF(uint32_t version, size_t alignment, uint64_t dataOffset,
       M::Driver::DeviceContextRef &&device,
       M::Driver::DeviceBufferRef &&parentBuffer, llvm::json::Object &&metadata,
       llvm::DenseMap<std::string, TensorInfo> &&tensorInfo)
      : version(version), alignment(alignment), dataOffset(dataOffset),
        device(std::move(device)), parentBuffer(std::move(parentBuffer)),
        metadata(std::move(metadata)), tensorInfo(std::move(tensorInfo)) {}

  uint32_t version;
  size_t alignment;
  /// Absolute offset of the tensor data section in the file.
  uint64_t dataOffset;
  M::Driver::DeviceContextRef device;
  // Parent buffer for all subbuffers created.
  M::Driver::DeviceBufferRef parentBuffer;
  llvm::json::Object metadata;
  llvm::DenseMap<std::string, TensorInfo> tensorInfo;
};

}; // namespace M

#include "Support/ADT/DenseStringMap.h" // IWYU pragma: keep

#endif // RUNTIME_GGUF_H
