//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// HLCFDialect
//===----------------------------------------------------------------------===//

void M::HLCF::HLCFDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "KGEN/HLCFDialect/HLCF.cpp.inc"
      >();
}

Operation *HLCFDialect::materializeConstant(OpBuilder &b, Attribute value,
                                            Type type, Location loc) {
  return KGEN::ParamConstantOp::create(b, loc, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.cpp.inc"
