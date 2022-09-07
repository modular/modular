//===- LitOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LitDialect/LitOps.h"
#include "KGEN/LitDialect/LitDialect.h"

using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// LitDialect
//===----------------------------------------------------------------------===//

void LitDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/LitDialect/Lit.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/LitDialect/Lit.cpp.inc"
