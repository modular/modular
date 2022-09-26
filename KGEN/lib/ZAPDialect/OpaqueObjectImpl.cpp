//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "Support/AlignedAlloc.h"
#include "Support/MathExtras.h"
#include "mlir/IR/Diagnostics.h"
#include <random>

using namespace M;
using namespace KGEN;
using namespace ZAP;

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

namespace {
LLVM_PACKED(struct OpaqueBuffer {
  intptr_t size;
  void *data;
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
  objPtr->size = numElements;
  objPtr->data = ptr;
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
