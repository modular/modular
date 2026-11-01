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
// This file defines magic numbers and other structures related to the MEF file
// format.  See the BinaryExecutableFormat.md file for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEF_MEFENCODING_H
#define MLRT_GRAPHRT_MEF_MEFENCODING_H

#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"

namespace M {

namespace MEF {

//===----------------------------------------------------------------------===//
// General MEF constants
//===----------------------------------------------------------------------===//

enum {
  // Defined by the MEF_FILE production.
  kMagic1 = 0xEE,
  kMagic2 = 0xF0,
  kVersion0 = 0,

  // This is an arbitrary value used to fill out alignment padding.
  kAlignmentByte = 0xCB,

  // This indicates whether a section has an explicit alignment.
  kSectionAlignmentFlag = 0x01,

  // Primitives in MEF are 8-byte aligned.
  kPrimitiveEntryAlignment = 8,
};

// MEFInvokable.cpp addresses words in the primitive definition as a uint32_t
// array, so it needs to be aligned to at least uint32_t for this to work.
// (Attribute offsets are 64-bit, so we actually need to align to 8 bytes, but
// even if that changes again, then this restriction is still required as long
// as MEFInvokable.cpp still addresses it as a uint32_t array.)
static_assert(MEF::kPrimitiveEntryAlignment % sizeof(uint32_t) == 0,
              "size of uint32_t must evenly divide kPrimitiveEntryAlignment");

/// Return true if the magic number makes this look like a MEF file.
inline bool hasMEFMagicNumber(llvm::StringRef contents) {
  return contents.size() > 2 && uint8_t(contents[0]) == kMagic1 &&
         uint8_t(contents[1]) == kMagic2;
}

/// Parse a VBR encoded integer from the byte stream, moving the pointer for
/// the number of bytes consumed. The calling function must provide a
/// `const`-matched actual-argument to avoid compiler-error. Hence, if a
/// non-const pointer needs to be provided here, then create a `const` temporary
/// and later copy back its const_cast<>-ed updated value.
inline size_t parseVBR(const uint8_t *&pointer) {
  uint8_t nextByte = *pointer++;
  size_t result = nextByte & 0x7Fu;
  while ((nextByte & 0x80u) != 0) {
    nextByte = *pointer++;
    result = (result << 7) | size_t(nextByte & 0x7Fu);
  }
  return result;
}

/// Parse a single float from the byte stream and move the pointer accordingly.
inline float parseFloat(const uint8_t *&pointer) {
  float result = *(const float *)pointer;
  pointer += sizeof(float);
  return result;
}

/// A SignedVBR is emitted with the sign bit as the low bit and negates
/// negative numbers so we can emit things like "-2" into a single byte.
inline ssize_t parseSignedVBR(const uint8_t *&pointer) {
  size_t encoded = parseVBR(pointer);
  if (encoded & 1)
    return -(encoded >> 1);
  return encoded >> 1;
}

//===----------------------------------------------------------------------===//
// MEF section constants
//===----------------------------------------------------------------------===//

enum class SectionID {
  kInvalid = 0,

  kLocationFilenames = 1, // Strings for each filename used by a location.
  kLocationPositions = 2, // Detailed information for locations

  kStrings = 3, // All of the string data.
  kAttributes = 4,
  kPrimitiveNames = 5,
  kTopLevelOpBodies = 6,
  kTopLevelOpIndex = 7,

  kAttributeIndexMD = 8,
  kStringsMD = 9,          // Strings for metadata
  kAttributeNamesMD = 10,  // Table of primitive attribute names.
  kMLIRTypeStringsMD = 11, // String table of MLIR types.
  kMLIRTypeMD = 12,
};

//===----------------------------------------------------------------------===//
// MEF attribute constants and helpers
//===----------------------------------------------------------------------===//

/// A two-bit classification.
enum class AttrCategory : uint8_t { Float, Signed, Unsigned };

/// This bitfield encodes into a single byte metadata about the format of
/// emitted attributes.
enum class AttributeKind : uint8_t {
  kInvalid = 0,

  /// This is a bitmask, when combined with one of the basic types, indicates
  /// an array of that type.
  mArrayType = 1,

  /// The primitive types are encoded with the low bit reserved for "is array",
  /// the next two bits being their category, and the next 2 bits being the
  /// number of bytes (in log form).
  kS8 = (0 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU8 = (0 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kS16 = (1 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU16 = (1 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kS32 = (2 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU32 = (2 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kF32 = (2 << 3) | uint8_t(AttrCategory::Float) << 1,
  kS64 = (3 << 3) | uint8_t(AttrCategory::Signed) << 1,
  kU64 = (3 << 3) | uint8_t(AttrCategory::Unsigned) << 1,
  kF64 = (3 << 3) | uint8_t(AttrCategory::Float) << 1,

  // There is no special encoding to remaining enum values, they are just
  // numbered to be out of the way of the others.
  kBool = 1 << 5,
  kString = 2 << 5,
  kDictionary = 3 << 5,
  kList = 4 << 5,
  kType = 5 << 5,
  kSymbolRef = 6 << 5,
};

static inline AttrCategory getCategory(AttributeKind kind) {
  return (AttrCategory)((uint8_t(kind) >> 1) & 3);
}

static inline unsigned getWidthInBits(AttributeKind kind) {
  return (1 << ((uint8_t)kind >> 3)) * 8;
}

/// Print the AttributeKind as a string.
std::string getAsString(AttributeKind kind);

/// Given an AttributeKind for an element, return the AttributeKind for an array
/// of that element.
static inline constexpr AttributeKind getArrayOf(AttributeKind elementKind) {
  assert(elementKind <= AttributeKind::kBool &&
         "cannot make arrays of arrays or arrays of strings");
  return AttributeKind(uint8_t(elementKind) |
                       uint8_t(AttributeKind::mArrayType));
}

/// Provide a mapping from C++ types to the corresponding attribute kinds.
template <typename CXXType>
struct AttrKindForCXXType {
  // Default mapping allows a static_assert instead of template substitution
  // failure to catch invalid cases.
  static constexpr AttributeKind kind = AttributeKind::kInvalid;
};

#define DECLARE_ATTR_MAPPING(KIND, CXXTYPE)                                    \
  template <>                                                                  \
  struct AttrKindForCXXType<CXXTYPE> {                                         \
    static constexpr AttributeKind kind = AttributeKind::KIND;                 \
  }
DECLARE_ATTR_MAPPING(kBool, bool);
DECLARE_ATTR_MAPPING(kS8, int8_t);
DECLARE_ATTR_MAPPING(kU8, uint8_t);
DECLARE_ATTR_MAPPING(kS16, int16_t);
DECLARE_ATTR_MAPPING(kU16, uint16_t);
DECLARE_ATTR_MAPPING(kS32, int32_t);
DECLARE_ATTR_MAPPING(kU32, uint32_t);
DECLARE_ATTR_MAPPING(kS64, int64_t);
DECLARE_ATTR_MAPPING(kU64, uint64_t);
DECLARE_ATTR_MAPPING(kF32, float);
DECLARE_ATTR_MAPPING(kF64, double);
DECLARE_ATTR_MAPPING(kString, StringRef);
#undef DECLARE_ATTR_MAPPING

//===----------------------------------------------------------------------===//
// MEF type constants and helpers
//===----------------------------------------------------------------------===//

/// MEF Types are always compatible with GraphCompiler's DType but have some
/// additional types for good measure.
class TypeKind : public DType {
public:
  using DType::DType;
  enum ExtraCases : uint8_t {
    // In the MEF encoding of these types, they all have following bytes.
    tensor = kFirstExtendedOption,
    quantized,
    quantizedPerAxis,
    function,
    chain,
    any,
  };
};

template <>
struct AttrKindForCXXType<TypeKind> {
  static constexpr AttributeKind kind = AttributeKind::kType;
};

} // namespace MEF
} // namespace M

#endif // MLRT_GRAPHRT_MEF_MEFENCODING_H
