//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

DeclRefType LIT::getLibraryErrorType(MLIRContext *ctx) {
  // The standard library error type is `@Error` with no parameters.
  return DeclRefType::get(FlatSymbolRefAttr::get(ctx, "Error"));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.cpp.inc"
