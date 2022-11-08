//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/HLCFDialect/HLCFOps.h"

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

void M::HLCF::HLCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Support/HLCFDialect/HLCF.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/HLCFDialect.cpp.inc"
