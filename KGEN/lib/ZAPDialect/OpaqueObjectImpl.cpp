//===- OpaqueObjectImpl.cpp -----------------------------------------------===//
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

/// Fill `obj` according to `kind`, `dtype`, and `numElements`. Despite `obj`
/// being suggestively named, `obj` can be any pointer - it does not have to be
/// the pointer passed to ::populate. It must have a space allocated for
/// `numElements` objects of type `dtype`, however.
static LogicalResult doFill(Location loc, InputGenKind kind, DType dtype,
                            size_t numElements, void *obj) {
  switch (kind) {
  case InputGenKind::Zeros:
    memset(obj, 0, dtype.getSizeInBytes(numElements));
    return success();
  case InputGenKind::Ones: {
    if (dtype.isComplex()) {
      unsigned widthInBytes = dtype.getWidthInBits() / 8;
      // Set the imaginary component to zero.
      memset((char *)obj + widthInBytes, 0, widthInBytes);
      dtype = dtype.stripComplex();
    }

    // Dispatch the dtype, and just fill directly with ones.
    return dtype.dispatch<LogicalResult>(obj)
        .when([&](bool *ptr) {
          std::generate(ptr, ptr + numElements, []() { return true; });
          return success();
        })
        .whenCXXInt([&](auto *ptr) { // Standard C++ integers.
          std::generate(ptr, ptr + numElements, []() { return 1; });
          return success();
        })
        .whenCXXFP([&](auto *ptr) { // float and double.
          std::generate(ptr, ptr + numElements, []() { return 1.0; });
          return success();
        })
        .otherwise([&]() { return failure(); });
  }
  case InputGenKind::Random: {
    // Fill the given buffer with random elements from the provided
    // distribution.
    auto fillWithDistribution = [&](auto *ptr, auto distribution) {
      std::default_random_engine randEngine(/*seed=*/0);
      std::generate(ptr, ptr + numElements,
                    [&]() { return distribution(randEngine); });
    };

    return dtype.dispatch<LogicalResult>(obj)
        .when([&](bool *destPtr) {
          fillWithDistribution(destPtr, std::bernoulli_distribution());
          return success();
        })
        .whenCXXInt([&](auto *destPtr) {
          fillWithDistribution(destPtr, std::uniform_int_distribution<>(
                                            dtype.isSInt() ? -10 : 0, 10));
          return success();
        })
        .whenCXXFP([&](auto *destPtr) {
          fillWithDistribution(destPtr,
                               std::uniform_real_distribution<>(-1.0, 1.0));
          return success();
        })
        .otherwise([&]() { return failure(); });
  }
  }

  return emitError(loc) << "could not fill with gen kind: "
                        << stringifyInputGenKind(kind);
}

/// Compares raw buffers `lhs` and `rhs` of type `dtype` with `numElements`
/// elements. Returns true if they are equal, false if they are not, and failure
/// if they cannot be compared.
static FailureOr<bool> dataEquals(Location loc, DType dtype, size_t numElements,
                                  void *lhs, void *rhs) {
  return dtype.dispatch<FailureOr<bool>>(lhs, rhs)
      .whenCXXArithmeticType([&](auto *lhs, auto *rhs) {
        return llvm::all_of_zip(llvm::makeArrayRef(lhs, numElements),
                                llvm::makeArrayRef(rhs, numElements),
                                [](auto a, auto b) { return isClose(a, b); });
      })
      .otherwise([&]() {
        return mlir::emitError(loc) << "unknown dtype: " << dtype.getAsString();
      });
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

namespace {
LLVM_PACKED(struct OpaqueBuffer {
  intptr_t size;
  void *data;
  int8_t dtype;
});
}

/// This implements `OpaqueObjectInterface::populate`. It generates a buffer
/// object, and furthermore allocates memory for the buffer's backing storage
/// and places that in the pointer field of the buffer structure itself.
LogicalResult BufferType::populate(Location loc, InputGenKind kind,
                                   Attribute tag, void *obj) const {
  // FIXME: This doesn't currently handle dynamic type buffers - we need the tag
  // to contain the type. Come up with a nice attribute structure that enables
  // that use case.

  // Resolve the dtype.
  Optional<DType> dtype = resolveDType();
  if (!dtype)
    return emitError(loc)
           << "Buffers with unbound dtype are not yet supported: " << *this;

  auto sizeOr = resolveSize();
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
  objPtr->dtype = (*dtype).getValue();

  // Do the fill.
  return doFill(loc, kind, *dtype, numElements, ptr);
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
  return dataEquals(loc, DType(lhs->dtype), lhs->size, lhs->data, rhs->data);
}
