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
// The GGML tensor layouts shared by the GGUF reader and writer.
//
//===----------------------------------------------------------------------===//

#ifndef RUNTIME_GGUF_FORMAT_H
#define RUNTIME_GGUF_FORMAT_H

#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace M {

/// How one GGML tensor type is laid out in memory.
struct GGUFTensorLayout {
  /// The element type of the stored data.
  DType dtype;
  /// The number of elements a quantization block covers; 1 for plain types.
  size_t elementsPerBlock;
  /// The size in bytes of one quantization block; the element size for plain
  /// types.
  size_t bytesPerBlock;
  /// The name of the quantization scheme, e.g. "Q4_K"; empty for plain types.
  StringRef quantizationName;
};

/// Returns the layout of `tensorType`, or `std::nullopt` for the tensor types
/// this codebase does not read or write: all the quantization schemes MAX does
/// not load.
inline std::optional<GGUFTensorLayout>
getGGUFTensorLayout(uint32_t tensorType) {
  switch (tensorType) {
  case 0: // F32
    return GGUFTensorLayout{DType::f32, 1, 4, ""};
  case 1: // F16
    return GGUFTensorLayout{DType::f16, 1, 2, ""};
  case 24: // I8
    return GGUFTensorLayout{DType::si8, 1, 1, ""};
  case 25: // I16
    return GGUFTensorLayout{DType::si16, 1, 2, ""};
  case 26: // I32
    return GGUFTensorLayout{DType::si32, 1, 4, ""};
  case 27: // I64
    return GGUFTensorLayout{DType::si64, 1, 8, ""};
  case 28: // F64
    return GGUFTensorLayout{DType::f64, 1, 8, ""};
  case 30: // BF16
    return GGUFTensorLayout{DType::bf16, 1, 2, ""};
  case 2: // Q4_0
    return GGUFTensorLayout{DType::ui8, 32, 18, "Q4_0"};
  case 12: // Q4_K
    return GGUFTensorLayout{DType::ui8, 256, 144, "Q4_K"};
  case 13: // Q5_K
    return GGUFTensorLayout{DType::ui8, 256, 176, "Q5_K"};
  case 14: // Q6_K
    return GGUFTensorLayout{DType::ui8, 256, 210, "Q6_K"};
  default:
    return std::nullopt;
  }
}

/// Returns the name of `tensorType`, matching the GGML type names, or an empty
/// string for an id this codebase does not know.
inline StringRef getGGUFTensorTypeName(uint32_t tensorType) {
  switch (tensorType) {
  case 0:
    return "F32";
  case 1:
    return "F16";
  case 2:
    return "Q4_0";
  case 3:
    return "Q4_1";
  case 6:
    return "Q5_0";
  case 7:
    return "Q5_1";
  case 8:
    return "Q8_0";
  case 9:
    return "Q8_1";
  case 10:
    return "Q2_K";
  case 11:
    return "Q3_K";
  case 12:
    return "Q4_K";
  case 13:
    return "Q5_K";
  case 14:
    return "Q6_K";
  case 15:
    return "Q8_K";
  case 16:
    return "IQ2_XXS";
  case 17:
    return "IQ2_XS";
  case 18:
    return "IQ3_XXS";
  case 19:
    return "IQ1_S";
  case 20:
    return "IQ4_NL";
  case 21:
    return "IQ3_S";
  case 22:
    return "IQ2_S";
  case 23:
    return "IQ4_XS";
  case 24:
    return "I8";
  case 25:
    return "I16";
  case 26:
    return "I32";
  case 27:
    return "I64";
  case 28:
    return "F64";
  case 29:
    return "IQ1_M";
  case 30:
    return "BF16";
  case 34:
    return "TQ1_0";
  case 35:
    return "TQ2_0";
  case 36:
    return "MXFP4";
  default:
    return "";
  }
}

} // namespace M

#endif // RUNTIME_GGUF_FORMAT_H
