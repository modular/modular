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

/// CompactTensorSpec is a memory efficient representation of a shape and
/// element type, implemented using CompactTensorShape.
class CompactTensorSpec : public CompactTensorShape {
public:
  CompactTensorSpec() : CompactTensorShape() {
    setEltType(TensorEltType::invalid);
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

  size_t getSizeInBytes() const {
    return getEltType().getSizeInBytes(getNumElements());
  }

  void print(raw_ostream &os) const;
  std::string getAsString() const;
  void dump() const;

  bool operator==(const CompactTensorSpec &rhs) const {
    return storage.equalsIncludingAux(rhs.storage);
  }
  bool operator!=(const CompactTensorSpec &rhs) const {
    return !(*this == rhs);
  }
};

// CompactTensorSpec should always be two words, the same as CompactTensorSpec.
static_assert(sizeof(void *) != 8 || sizeof(CompactTensorSpec) == 16);

} // namespace M

#endif // GENERICML_SUPPORT_TENSORSPEC_H
