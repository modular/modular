//===- ZapOps.cpp ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZapDialect/ZapOps.h"
#include "KGEN/ZapDialect/ZapDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// ZapDialect
//===----------------------------------------------------------------------===//

void ZapDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/ZapDialect/Zap.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZapDialect/Zap.cpp.inc"
