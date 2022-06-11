//===- KGENDialect.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/KGENDialect/KGENDialect.cpp.inc"

void KGENDialect::initialize() {
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/KGENDialect/KGEN.cpp.inc"
      >();
}
