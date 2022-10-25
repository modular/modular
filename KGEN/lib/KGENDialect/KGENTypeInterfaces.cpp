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
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.cpp.inc"
