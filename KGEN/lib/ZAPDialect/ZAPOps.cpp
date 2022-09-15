//===- ZAPOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/Debug.h"

using namespace M;
using namespace M::KGEN;

static bool hasSameUnderlyingDType(Value a, Value b) {
  auto aDType = a.getType().cast<DTypeInterface>().getDType();
  auto bDType = b.getType().cast<DTypeInterface>().getDType();
  return aDType == bDType;
}

//===----------------------------------------------------------------------===//
// SIMDLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ZAP::SIMDLoadOp::verify() {
  if (hasSameUnderlyingDType(getBuffer(), getResult()))
    return success();
  return emitOpError("the buffer type (")
         << getBuffer().getType()
         << ") must have the same element type as the result simd type ("
         << getResult().getType() << ")";
}

//===----------------------------------------------------------------------===//
// SIMDStoreOp
//===----------------------------------------------------------------------===//

LogicalResult ZAP::SIMDStoreOp::verify() {
  if (hasSameUnderlyingDType(getBuffer(), getValue()))
    return success();
  return emitOpError("the buffer type (")
         << getBuffer().getType()
         << ") must have the same element type as the value simd type ("
         << getValue().getType() << ")";
}

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

void ZAP::ZAPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
