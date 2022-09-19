//===- ZAPOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/ForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace KGEN;

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
// BufferStackAllocationOp
//===----------------------------------------------------------------------===//

LogicalResult ZAP::BufferStackAllocationOp::verify() {
  if (!getType().cast<BufferType>().getSize())
    return emitOpError("cannot stack allocate a buffer of unknown size");
  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Parse the dtype of a constant buffer.
static ParseResult
parseConstantBufferDType(AsmParser &p, mlir::DenseIntOrFPElementsAttr values,
                         Type &result) {
  TypedAttr dtype;
  if (parseParamValue(p, dtype, DTypeType::get(p.getContext())))
    return failure();
  result = BufferType::get(p.getBuilder().getIndexAttr(values.size()), dtype);
  return success();
}

/// Print the dtype of a constant buffer.
static void printConstantBufferDType(AsmPrinter &p, Operation *op,
                                     mlir::DenseIntOrFPElementsAttr values,
                                     Type result) {
  printParamValue(p, result.cast<BufferType>().getDType());
}

LogicalResult ZAP::BufferConstantOp::verify() {
  auto type = getValues().getType().dyn_cast<RankedTensorType>();
  // TODO: We need an "#M.array" attribute and type.
  if (!type || type.getRank() != 1)
    return emitOpError("expected a rank 1 tensor type");
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZAPDialect/ZAP.cpp.inc"
