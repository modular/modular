//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include <random>

using namespace M;

//===----------------------------------------------------------------------===//
// DataLayoutInterface Utility Functions
//===----------------------------------------------------------------------===//

Optional<int64_t>
KGEN::DataLayoutInterface::getTypeSizeInBytes(TargetInfoAttr target,
                                              Type type) {
  // Check for builtin types.
  auto iface = llvm::dyn_cast<DataLayoutInterface>(type);
  if (!iface) {
    // Return the integer or floating point width rounded up to the next byte.
    if (type.isIntOrFloat())
      return llvm::divideCeil(type.getIntOrFloatBitWidth(), CHAR_BIT);

    // Return the target pointer width.
    if (type.isa<FunctionType>() || type.isIndex())
      return target.getPointerSize();

    // Return the element type size multiplied by the size.
    if (auto vec = llvm::dyn_cast<VectorType>(type)) {
      Optional<int64_t> elSize =
          getTypeSizeInBytes(target, vec.getElementType());
      if (!elSize || vec.getRank() != 1)
        return {};
      return *elSize * llvm::PowerOf2Ceil(vec.getShape().back());
    }

    // No other builtin types are supported;
    return {};
  }

  return iface.getTypeSize(target);
}

Optional<int64_t>
KGEN::DataLayoutInterface::getTypeAlignInBytes(TargetInfoAttr target,
                                               Type type) {
  Builder b(target.getContext());
  auto iface = llvm::dyn_cast<DataLayoutInterface>(type);

  // Check for builtin types.
  if (!iface) {
    // Return the next power of 2 for integers and floats.
    if (type.isIntOrFloat())
      return llvm::PowerOf2Ceil(
          llvm::divideCeil(type.getIntOrFloatBitWidth(), CHAR_BIT));

    // Return the pointer size.
    if (type.isa<FunctionType>() || type.isIndex())
      return target.getPointerSize();

    // Round the vector size up to the nearest power of 2.
    if (auto vec = llvm::dyn_cast<VectorType>(type)) {
      if (Optional<int64_t> size = getTypeSizeInBytes(target, vec))
        return llvm::PowerOf2Ceil(*size);
      return {};
    }

    // No other builtin types are supported;
    return {};
  }

  return iface.getTypeAlign(target);
}

//===----------------------------------------------------------------------===//
// OpaqueObjectInterface Utility Functions
//===----------------------------------------------------------------------===//

LogicalResult KGEN::fillOpaqueElements(Location loc, InputGenKind kind,
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
