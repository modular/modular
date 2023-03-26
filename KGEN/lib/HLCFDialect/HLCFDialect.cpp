//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

void M::HLCF::HLCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/HLCFDialect/HLCF.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.cpp.inc"
