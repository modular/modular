//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/MathExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include <random>

using namespace M;

//===----------------------------------------------------------------------===//
// OpaqueObjectInterface Utility Functions
//===----------------------------------------------------------------------===//

LogicalResult M::KGEN::fillOpaqueElements(Location loc, InputGenKind kind,
                                          DType dtype, size_t numElements,
                                          void *obj) {
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

FailureOr<bool> M::KGEN::compareOpaqueElements(Location loc, DType dtype,
                                               size_t numElements, void *lhs,
                                               void *rhs) {
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
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.cpp.inc"
