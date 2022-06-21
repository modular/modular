//===- MetaDialect.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Meta dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDTypeValue(AsmParser &p,
                                        FailureOr<Attribute> &result) {
  Attribute retValue;
  if (failed(parseParamValue(p, retValue, p.getBuilder().getType<DTypeType>())))
    return failure();
  result = retValue;
  return success();
}

static void printParamDTypeValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value, value.getType());
}

//===----------------------------------------------------------------------===//
// custom<OptionalParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalParamDTypeValue(AsmParser &p,
                                                FailureOr<Attribute> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = Attribute();
    return success();
  }
  return parseParamDTypeValue(p, result);
}

static void printOptionalParamDTypeValue(AsmPrinter &p, Attribute value) {
  if (!value) {
    p << '?';
    return;
  }
  printParamDTypeValue(p, value);
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

LogicalResult
ScalarType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   Attribute dtype) {
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "parameter for scalar type must be a !kgen.dtype";
  return success();
}

void ScalarType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDtype());
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult
SIMDType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 Attribute size, Attribute dtype) {
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
  walkAttrsFn(getDtype());
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   Attribute size, Attribute dtype) {
  if (size && !size.getType().isIndex())
    return emitError() << "size parameter for buffer must have type `index`";
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for buffer must be a !kgen.dtype";
  return success();
}

void BufferType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDtype());
}

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/MetaDialect/MetaDialect.cpp.inc"

void MetaDialect::initialize() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"
      >();
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/MetaDialect/Meta.cpp.inc"
      >();
}
