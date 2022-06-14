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
#include "KGEN/MetaDialect/MetaTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
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
                                        FailureOr<Attribute> &value) {
  Attribute retValue;
  if (failed(parseParamValue(p, retValue, p.getBuilder().getType<DTypeType>())))
    return failure();
  value = retValue;
  return success();
}

static void printParamDTypeValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value, value.getType());
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
}
