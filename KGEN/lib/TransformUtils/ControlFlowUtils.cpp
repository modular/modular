//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/ControlFlowUtils.h"
#include "KGEN/HLCFDialect/HLCFInterfaces.h"

using namespace M;
using namespace KGEN;

bool KGEN::userCrossesFunctionCFG(Operation *op, Operation *user) {
  for (Operation *cur = user->getParentOp(), *parent = op->getParentOp();
       cur != parent; cur = cur->getParentOp()) {
    // If there is any non-control-flow operation between the user and the
    // operation, then the user crosses an unknown region.
    if (!isa<HLCF::ControlFlowNode>(cur))
      return true;
  }
  return false;
}
