//===- ZAPOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPOps.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M::KGEN;

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
