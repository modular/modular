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
// This file defines some helper classes that make it possible to use types
// encoded in a MEF file.  This is written in a low-dependency way that is
// suitable for use at runtime, typically through MEFAttr<MEFType>.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEF_MEFTYPES_H
#define MLRT_GRAPHRT_MEF_MEFTYPES_H

#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "mlir/IR/Builders.h"

namespace M {

/// This class represents an unknown type encoded into a MEF file, it can be
/// dynamically casted to other types for processing.
class MEFType {
public:
  MEFType(const uint8_t *pointer, const uint8_t *attrEnd)
      : pointer(pointer), attrEnd(attrEnd) {
    assert(pointer <= attrEnd &&
           "MEF datum not within bounds of the MEF Attribute Section");
  }
  MEFType(const MEFType &) = default;
  MEFType &operator=(const MEFType &) = default;

  MEF::TypeKind getKind() const { return MEF::TypeKind(*pointer); }

  /// All MEFType's are MEFType's!
  static bool classof(const MEFType *type) { return true; }

  /// Return the encoded size of this type in the MEF file, in bytes.
  size_t getEncodedSizeInBytes() const;

  const uint8_t *getPointer() const { return pointer; }
  const uint8_t *getAttrEnd() const { return attrEnd; }

protected:
  /// This is the raw pointer to the start of the type in the MEF File.
  const uint8_t *pointer;
  const uint8_t *attrEnd;
};

// Since MEFType can be constructed from a nullptr, the template
// specializations in llvm/Support/Casting.h fail to apply. As such, we
// have to explicitly define the equals and not equals operators for
// MEFType.
inline bool operator==(MEFType lhs, MEFType rhs) {
  return lhs.getPointer() == rhs.getPointer();
}

inline bool operator!=(MEFType lhs, MEFType rhs) { return !(lhs == rhs); }

//===----------------------------------------------------------------------===//
// Scalar Types
//===----------------------------------------------------------------------===//

/// This class is used to represent all the scalar types like f32, si32, etc.
template <typename CPPType>
class MEFScalarType : public MEFType {
public:
  MEFScalarType(const uint8_t *pointer, const uint8_t *attrEnd)
      : MEFType(pointer, attrEnd) {}

  static bool classof(const MEFType *type) {
    constexpr auto thisKind = DTypeForCXXType<CPPType>::kind;
    static_assert(!thisKind.isInvalid() &&
                  "C++ type isn't supported as a MEF type");
    return type->getKind() == thisKind;
  }
};

using MEFBoolType = MEFScalarType<bool>;
using MEFSI8Type = MEFScalarType<int8_t>;
using MEFUI8Type = MEFScalarType<uint8_t>;
using MEFSI16Type = MEFScalarType<int16_t>;
using MEFUI16Type = MEFScalarType<uint16_t>;
using MEFSI32Type = MEFScalarType<int32_t>;
using MEFUI32Type = MEFScalarType<uint32_t>;
using MEFSI64Type = MEFScalarType<int64_t>;
using MEFUI64Type = MEFScalarType<uint64_t>;
using MEFF32Type = MEFScalarType<float>;
using MEFF64Type = MEFScalarType<double>;

//===----------------------------------------------------------------------===//
// MEFFunctionType
//===----------------------------------------------------------------------===//

class MEFFunctionType : public MEFType {
public:
  MEFFunctionType(const uint8_t *pointer, const uint8_t *attrEnd)
      : MEFType(pointer, attrEnd) {}

  /// Decode all of the input and results types for this function type,
  /// returning them in the specified vectors.
  void decode(SmallVectorImpl<MEFType> &inputs,
              SmallVectorImpl<MEFType> &results) const;

  /// Return the encoded size of this type in the MEF file, in bytes.
  size_t getEncodedSizeInBytes() const;

  static bool classof(const MEFType *type) {
    assert(type != nullptr);
    return type->getKind() == MEF::TypeKind::function;
  }
};

//===----------------------------------------------------------------------===//
// MEFTensorType
//===----------------------------------------------------------------------===//

class MEFTensorType : public MEFType {
public:
  MEFTensorType(const uint8_t *pointer, const uint8_t *attrEnd)
      : MEFType(pointer, attrEnd) {}

  /// Decode the rank and dimensions and return the element type for the tensor.
  /// This will assert fail if the tensor is dynamically ranked or has an
  /// unknown dimension.
  MEFType decodeStatic(SmallVectorImpl<int64_t> &dims) const;

  /// Decode the rank and dimensions and return the element type for the tensor.
  /// If the tensor is dynamically ranked then hasRank will be false and dims
  /// will be left empty. Negative dimension values denote dynamic dims.
  MEFType decode(bool &hasRank, SmallVectorImpl<int64_t> &dims) const;

  /// Return the encoded size of this type in the MEF file, in bytes.
  size_t getEncodedSizeInBytes() const;

  static bool classof(const MEFType *type) {
    assert(type != nullptr);
    return type->getKind() == MEF::TypeKind::tensor;
  }
};

//===----------------------------------------------------------------------===//
// MEFQuantizedType
//===----------------------------------------------------------------------===//

class MEFQuantizedType : public MEFType {
public:
  MEFQuantizedType(const uint8_t *pointer, const uint8_t *attrEnd)
      : MEFType(pointer, attrEnd) {}

  /// Decode quantization information, returning the `<storage, expressed>`
  /// types, and filling in the scale, and zero point.
  std::pair<MEFType, MEFType> decode(float &scale, int64_t &zeroPoint) const;

  /// Return the encoded size of this type in the MEF file, in bytes.
  size_t getEncodedSizeInBytes() const;

  static bool classof(const MEFType *type) {
    assert(type != nullptr);
    return type->getKind() == MEF::TypeKind::quantized;
  }
};

//===----------------------------------------------------------------------===//
// MEFQuantizedPerAxisType
//===----------------------------------------------------------------------===//

class MEFQuantizedPerAxisType : public MEFType {
public:
  MEFQuantizedPerAxisType(const uint8_t *pointer, const uint8_t *attrEnd)
      : MEFType(pointer, attrEnd) {}

  /// Decode quantization information, returning the `<storage, expressed>`
  /// types, and filling in the scales, and zero points.
  std::pair<MEFType, MEFType> decode(SmallVectorImpl<float> &scales,
                                     SmallVectorImpl<int64_t> &zeroPoints,
                                     uint8_t &quantizedDim) const;

  /// Return the encoded size of this type in the MEF file, in bytes.
  size_t getEncodedSizeInBytes() const;

  static bool classof(const MEFType *type) {
    assert(type != nullptr);
    return type->getKind() == MEF::TypeKind::quantizedPerAxis;
  }
};

} // namespace M

namespace llvm {

template <typename T>
struct CastInfo<T, ::M::MEFType>
    : public CastIsPossible<T, ::M::MEFType>,
      public DefaultDoCastIfPossible<std::optional<T>, ::M::MEFType,
                                     CastInfo<T, ::M::MEFType>> {
  static std::optional<T> doCast(const ::M::MEFType &f) {
    return T(f.getPointer(), f.getAttrEnd());
  }
  static std::optional<T> castFailed() { return std::nullopt; }
};

} // namespace llvm

#endif // MLRT_GRAPHRT_MEF_MEFTYPES_H
