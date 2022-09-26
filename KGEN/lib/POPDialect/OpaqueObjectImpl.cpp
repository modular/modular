//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace KGEN;
using namespace POP;

/// Resolve the dtype of a DTypeInterface Type. If the interface has `invalid`
/// DType, then given a `tag` attribute, if it's a DTypeConstantAttr then pull
/// out the DType and return it. Otherwise, return failure.
static FailureOr<DType> resolveDTypeWithTag(DTypeInterface itf, Location loc,
                                            Attribute tag) {
  if (Optional<DType> dtype = itf.getResolvedDType())
    return *dtype;

  if (auto dt = tag.dyn_cast<DTypeConstantAttr>())
    return dt.getDType();

  return emitError(loc) << "could not resolve dtype";
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

/// This implements `OpaqueObjectInterface::populate`. It generates a single
/// scalar according to the method prescribed by `kind`. If the dtype is
/// unknown, then this expects the tag attribute to be a type attr. Otherwise,
/// it expects UnitAttr.
LogicalResult ScalarType::populate(Location loc, InputGenKind kind,
                                   Attribute tag, void *obj) const {
  // Resolve the dtype.
  auto dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();
  DType dtype = *dtypeOr;

  return fillOpaqueElements(loc, kind, dtype, 1, obj);
}

/// This implements `OpaqueObjectInterface::destroy`. Nothing to be done for
/// ScalarType, there are no additional allocations.
void ScalarType::destroy(Attribute tag, void *obj) const {}

/// This implements `OpaqueObjectInterface::getSizeInBytes`.
FailureOr<size_t> ScalarType::getSizeInBytes(Location loc,
                                             Attribute tag) const {
  auto dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();

  return dtypeOr->getSizeInBytes(1);
}

FailureOr<bool> ScalarType::equals(Location loc, Attribute tag, void *lhsData,
                                   void *rhsData) const {
  // Check that the dtypes are equal. This only does something if the two dtypes
  // are actually different (i.e. unknown statically, dynamically carried by the
  // evaluation configuration). If the dtype is statically known by the
  // ScalarType then lhsDtype and rhsDtype will be equal.
  FailureOr<DType> dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();

  // Compare the outputs if we can.
  return compareOpaqueElements(loc, *dtypeOr, 1, lhsData, rhsData);
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

/// This implements `OpaqueObjectInterface::populate`. It generates a SIMD
/// vector (really an array of elements) according to `kind` and stores it in
/// `obj`.
LogicalResult SIMDType::populate(Location loc, InputGenKind kind, Attribute tag,
                                 void *obj) const {
  Optional<DType> dtype = getResolvedDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype && "SIMDType must have a valid dtype");

  auto sizeOr = getResolvedSize();
  if (!sizeOr.has_value())
    return failure();
  size_t numElements = *sizeOr;

  return fillOpaqueElements(loc, kind, *dtype, numElements, obj);
}

/// This implements `OpaqueObjectInterface::destroy`. Nothing to be done for
/// SIMDType, there are no additional allocations.
void SIMDType::destroy(Attribute tag, void *obj) const {}

/// This implements `OpaqueObjectInterface::getSizeInBytes`. Since a SIMD vector
/// has all its elements inline, compute the size of the array needed to hold
/// tightly-packed elements for this type.
FailureOr<size_t> SIMDType::getSizeInBytes(Location loc, Attribute tag) const {
  Optional<DType> dtype = getResolvedDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype && "SIMDType must have a valid dtype");

  // Same with the size, if it's unknown (which it should not be) then
  // we can't do anything.
  auto sizeOr = getResolvedSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return dtype->getSizeInBytes(*sizeOr);
}

FailureOr<bool> SIMDType::equals(Location loc, Attribute tag, void *lhsData,
                                 void *rhsData) const {
  // Everything in a SIMDType must be static, so we can just directly compare
  // the data.
  Optional<DType> dtype = getResolvedDType();
  assert(dtype && "SIMDType must have a valid dtype");

  Optional<int64_t> sizeOr = getResolvedSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return compareOpaqueElements(loc, *dtype, *sizeOr, lhsData, rhsData);
}
