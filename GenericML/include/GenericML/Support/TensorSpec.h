//===- GenericML/Support/TensorSpec.h -------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorSpec and CompactTensorSpec classes, which hold a
// TensorShape and TensorDType together in one value.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TENSORSPEC_H
#define GENERICML_SUPPORT_TENSORSPEC_H

#include "GenericML/Support/TensorEltType.h"
#include "GenericML/Support/TensorShape.h"

namespace M {
class CompactTensorSpec;

///  This class provides helper API that is common to all *TensorSpec classes.
template <typename FinalClass>
class TensorSpecImpl {
public:
  size_t getSizeInBytes() const {
    auto *finalThis = static_cast<const FinalClass *>(this);
    return finalThis->getEltType().getSizeInBytes(finalThis->getNumElements());
  }
};

/// TensorSpec is a large but efficient to manipulate value that contains a
/// TensorShape and a TensorEltType.
class TensorSpec : public TensorSpecImpl<TensorSpec>, public TensorShape {
public:
  TensorSpec() = default;
  TensorSpec(const CompactTensorSpec &spec);
  template <typename ShapeType>
  TensorSpec(const ShapeType &shape, TensorEltType eltType)
      : TensorShape(shape), eltType(eltType) {}

  TensorEltType getEltType() const { return eltType; }
  void setEltType(TensorEltType type) { eltType = type; }

private:
  TensorEltType eltType;
};

/// CompactTensorSpec is a memory efficient representation of a shape and
/// element type, implemented using CompactTensorShape.
class CompactTensorSpec : public TensorSpecImpl<CompactTensorSpec>,
                          public CompactTensorShape {
public:
  CompactTensorSpec() : CompactTensorShape() {
    setEltType(TensorEltType::invalid);
  }
  CompactTensorSpec(const TensorSpec &other) : CompactTensorShape(other) {
    setEltType(other.getEltType());
  }
  template <typename ShapeType>
  CompactTensorSpec(const ShapeType &shape, TensorEltType eltType)
      : CompactTensorShape(shape) {
    setEltType(eltType);
  }

  // This class stores the ElementType in the auxillary storage field of the
  // underlying CompactTensorShape.
  TensorEltType getEltType() const {
    return TensorEltType(getAuxillaryStorage());
  }
  void setEltType(TensorEltType type) { setAuxillaryStorage(type.getValue()); }
};

// CompactTensorSpec should always be two words, the same as CompactTensorSpec.
static_assert(sizeof(void *) != 8 || sizeof(CompactTensorSpec) == 16);

inline TensorSpec::TensorSpec(const CompactTensorSpec &spec)
    : TensorShape(spec), eltType(spec.getEltType()) {}

} // namespace M

#endif // GENERICML_SUPPORT_TENSORSPEC_H
