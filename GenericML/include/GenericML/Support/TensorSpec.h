//===- GenericML/Support/TensorSpec.h -------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TensorSpec and TensorSpec classes, which hold a
// TensorShape and TensorDType together in one value.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_TENSORSPEC_H
#define GENERICML_SUPPORT_TENSORSPEC_H

#include "GenericML/Support/TensorShape.h"
#include "Support/DType.h"

namespace M {

/// TensorSpec is a memory efficient representation of a shape and
/// element type, implemented using TensorShape.
class TensorSpec : public TensorShape {
public:
  TensorSpec() : TensorShape() { setEltType(DType::invalid); }
  template <typename ShapeType>
  TensorSpec(const ShapeType &shape, DType eltType) : TensorShape(shape) {
    setEltType(eltType);
  }

  // This class stores the ElementType in the auxillary storage field of the
  // underlying TensorShape.
  DType getEltType() const { return DType(getAuxillaryStorage()); }
  void setEltType(DType type) { setAuxillaryStorage(type.getValue()); }

  size_t getSizeInBytes() const {
    return getEltType().getSizeInBytes(getNumElements());
  }

  void print(raw_ostream &os) const;
  std::string getAsString() const;
  void dump() const;

  bool operator==(const TensorSpec &rhs) const {
    return storage.equalsIncludingAux(rhs.storage);
  }
  bool operator!=(const TensorSpec &rhs) const { return !(*this == rhs); }
};

inline raw_ostream &operator<<(raw_ostream &os, const TensorSpec &value) {
  value.print(os);
  return os;
}

// TensorSpec should always be two words, the same as TensorSpec.
static_assert(sizeof(void *) != 8 || sizeof(TensorSpec) == 16);

} // namespace M

#endif // GENERICML_SUPPORT_TENSORSPEC_H
