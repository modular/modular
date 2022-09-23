//===- MetaTypes.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include <random>

using namespace M;
using namespace KGEN;

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
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult
SIMDType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 TypedAttr size, TypedAttr dtype) {
  if (!size || !dtype)
    return emitError() << "simd type requires size and dtype";
  if (!size.getType().isIndex())
    return emitError() << "size parameter for simd must have type `index`";
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for simd must be a !kgen.dtype";
  return success();
}

void SIMDType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDType());
}

Type SIMDType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                           ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return SIMDType::get(replAttrs[0], replAttrs[1]);
}

Optional<int64_t> SIMDType::resolveSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

/// This implements `OpaqueObjectInterface::populate`. It generates a SIMD
/// vector (really an array of elements) according to `kind` and stores it in
/// `obj`.
LogicalResult SIMDType::populate(Location loc, InputGenKind kind, Attribute tag,
                                 void *obj) const {
  DType dtype = resolveDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  auto sizeOr = resolveSize();
  if (!sizeOr.has_value())
    return failure();
  size_t numElements = *sizeOr;

  return doFill(loc, kind, dtype, numElements, obj);
}

/// This implements `OpaqueObjectInterface::destroy`. Nothing to be done for
/// SIMDType, there are no additional allocations.
void SIMDType::destroy(Attribute tag, void *obj) const { return; }

/// This implements `OpaqueObjectInterface::getSizeInBytes`. Since a SIMD vector
/// has all its elements inline, compute the size of the array needed to hold
/// tightly-packed elements for this type.
FailureOr<size_t> SIMDType::getSizeInBytes(Location loc, Attribute tag) const {
  DType dtype = resolveDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  // Same with the size, if it's unknown (which it should not be) then
  // we can't do anything.
  auto sizeOr = resolveSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return dtype.getSizeInBytes(*sizeOr);
}

FailureOr<bool> SIMDType::equals(Location loc, Attribute tag, void *lhsData,
                                 void *rhsData) const {
  // Everything in a SIMDType must be static, so we can just directly compare
  // the data.
  DType dtype = resolveDType();
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  Optional<int64_t> sizeOr = resolveSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return dataEquals(loc, dtype, *sizeOr, lhsData, rhsData);
}

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MetaDialect type support
//===----------------------------------------------------------------------===//

void MetaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"
      >();
}
