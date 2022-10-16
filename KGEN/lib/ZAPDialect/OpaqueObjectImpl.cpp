//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MathExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/Casting.h"
#include <numeric>
#include <random>

using namespace M;
using namespace KGEN;
using namespace ZAP;

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

namespace {
LLVM_PACKED(struct OpaqueBuffer {
  void *data;
  ssize_t size;
  uint8_t dtype;
});
} // namespace

/// This implements `OpaqueObjectInterface::populate`. It generates a buffer
/// object, and furthermore allocates memory for the buffer's backing storage
/// and places that in the pointer field of the buffer structure itself.
LogicalResult BufferType::populate(Location loc, InputGenKind kind,
                                   Attribute tag, void *obj) const {
  // FIXME: This doesn't currently handle dynamic type buffers - we need the tag
  // to contain the type. Come up with a nice attribute structure that enables
  // that use case.

  // Resolve the dtype.
  Optional<DType> dtype = getResolvedDType();
  if (!dtype)
    return emitError(loc)
           << "Buffers with unbound dtype are not yet supported: " << *this;

  auto sizeOr = getResolvedSize();
  int64_t numElements;
  if (!sizeOr.has_value())
    numElements = tag.cast<IntegerAttr>().getValue().getSExtValue();
  else
    numElements = *sizeOr;

  void *ptr = alignedAlloc(kPreferredMemoryAlignment,
                           dtype->getSizeInBytes(numElements));

  OpaqueBuffer *objPtr = ((OpaqueBuffer *)obj);
  objPtr->data = ptr;
  objPtr->size = numElements;
  objPtr->dtype = dtype->getValue();

  // Do the fill.
  return fillOpaqueElements(loc, kind, *dtype, numElements, ptr);
}

/// This implements `OpaqueObjectInterface::destroy`. This deallocates any
/// memory allocated in `populate`.
void BufferType::destroy(Attribute tag, void *obj) const {
  alignedFree(((OpaqueBuffer *)obj)->data);
}

/// This implements `OpaqueObjectInterface::getSizeInBytes`. We don't care about
/// the buffer's allocation, we care about the size of the buffer itself.
FailureOr<size_t> BufferType::getSizeInBytes(Location loc,
                                             Attribute tag) const {
  return sizeof(OpaqueBuffer);
}

/// This method compares two instances of data held in a buffer of a given type.
/// This is a deep comparison.
FailureOr<bool> BufferType::equals(Location loc, Attribute tag, void *lhsData,
                                   void *rhsData) const {
  auto *lhs = (OpaqueBuffer *)lhsData;
  auto *rhs = (OpaqueBuffer *)rhsData;
  assert(lhs->dtype == rhs->dtype && lhs->size == rhs->size);
  return compareOpaqueElements(loc, DType(lhs->dtype), lhs->size, lhs->data,
                               rhs->data);
}

//===----------------------------------------------------------------------===//
// NDBufferType
//===----------------------------------------------------------------------===//

namespace {
LLVM_PACKED(struct OpaqueNDBufer {
  void *data;
  ssize_t rank;
  ssize_t shape[NDBufferType::getMaximumRank()];
  uint8_t dtype;
});
} // namespace

/// This implements `OpaqueObjectInterface::populate`. It generates an ndbuffer
/// object, and furthermore allocates memory for the ndbuffer's backing storage
/// and places that in the pointer field of the ndbuffer structure itself.
LogicalResult NDBufferType::populate(Location loc, InputGenKind kind,
                                     Attribute tag, void *obj) const {
  // FIXME: This doesn't currently handle dynamic type ndbuffer - we need the
  // tag to contain the type. Come up with a nice attribute structure that
  // enables that use case.

  // Resolve the dtype.
  Optional<DType> dtype = getResolvedDType();
  if (!dtype)
    return emitError(loc) << "Types with unbound dtype are not yet supported: "
                          << *this;

  size_t shape[NDBufferType::getMaximumRank()] = {0};
  for (const auto &[index, value] : llvm::enumerate(getShape())) {
    if (auto dim = dyn_cast_if_present<IntegerAttr>(value))
      shape[index] = dim.getInt();
    else
      shape[index] =
          tag.cast<IntArrayElementsAttr>().asArrayRef<ssize_t>()[index];
  }

  size_t numElements =
      std::accumulate(shape, shape + std::size(shape), 1, std::multiplies<>());

  void *ptr = alignedAlloc(kPreferredMemoryAlignment,
                           dtype->getSizeInBytes(numElements));

  OpaqueNDBufer *objPtr = ((OpaqueNDBufer *)obj);
  objPtr->data = ptr;
  objPtr->rank = getRank();
  memcpy(objPtr->shape, shape, sizeof(shape[0]) * std::size(shape));
  objPtr->dtype = dtype->getValue();

  // Do the fill.
  return fillOpaqueElements(loc, kind, *dtype, numElements, ptr);
}

/// This implements `OpaqueObjectInterface::destroy`. This deallocates any
/// memory allocated in `populate`.
void NDBufferType::destroy(Attribute tag, void *obj) const {
  alignedFree(((OpaqueNDBufer *)obj)->data);
}

/// This implements `OpaqueObjectInterface::getSizeInBytes`. We don't care about
/// the ndbuffer's allocation, we care about the size of the ndbuffer itself.
FailureOr<size_t> NDBufferType::getSizeInBytes(Location loc,
                                               Attribute tag) const {
  return sizeof(OpaqueNDBufer);
}

/// This method compares two instances of data held in a ndbuffer of a given
/// type. This is a deep comparison.
FailureOr<bool> NDBufferType::equals(Location loc, Attribute tag, void *lhsData,
                                     void *rhsData) const {
  auto *lhs = (OpaqueNDBufer *)lhsData;
  auto *rhs = (OpaqueNDBufer *)rhsData;
  auto lhsNumElements = std::accumulate(lhs->shape, lhs->shape + lhs->rank, 1,
                                        std::multiplies<>());
  [[maybe_unused]] auto rhsNumElements = std::accumulate(
      rhs->shape, rhs->shape + rhs->rank, 1, std::multiplies<>());
  assert(lhs->dtype == rhs->dtype && lhsNumElements == rhsNumElements);
  return compareOpaqueElements(loc, DType(lhs->dtype), lhsNumElements,
                               lhs->data, rhs->data);
}
