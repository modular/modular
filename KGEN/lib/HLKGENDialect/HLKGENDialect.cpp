//===- HLKGENDialect.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HLKGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLKGENDialect/HLKGENDialect.h"
#include "KGEN/HLKGENDialect/HLKGENOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/HLKGENDialect/HLKGENDialect.cpp.inc"

void HLKGENDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  llvm_unreachable("no hlkgen dialect attrs");
}
Attribute HLKGENDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  llvm_unreachable("no hlkgen dialect attrs");
}
void HLKGENDialect::printType(Type type, DialectAsmPrinter &p) const {
  llvm_unreachable("no hlkgen dialect types");
}
Type HLKGENDialect::parseType(DialectAsmParser &p) const {
  llvm_unreachable("no hlkgen dialect types");
}

void HLKGENDialect::initialize() {

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/HLKGENDialect/HLKGEN.cpp.inc"
      >();
}
