//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/LITDialect/LITDialect.cpp.inc"

void LITDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  llvm_unreachable("no lit dialect attrs");
}
Attribute LITDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  llvm_unreachable("no lit dialect attrs");
}
void LITDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm_unreachable("no lit dialect types");
}
Type LITDialect::parseType(DialectAsmParser &p) const {
  llvm_unreachable("no lit dialect types");
}

void LITDialect::initialize() {

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/LITDialect/LIT.cpp.inc"
      >();
}
