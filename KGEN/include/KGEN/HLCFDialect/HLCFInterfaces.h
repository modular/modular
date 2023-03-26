//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_HLCFINTERFACES_H
#define KGEN_HLCFDIALECT_HLCFINTERFACES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Interface Verifiers
//===----------------------------------------------------------------------===//

namespace M::HLCF {
class ControlFlowNode;
class ControlFlowTerminator;

LogicalResult verifyControlFlowNode(ControlFlowNode op);
LogicalResult verifyControlFlowTerminator(ControlFlowTerminator op);
} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::HLCF {
struct ControlFlowTarget {
  ControlFlowTarget(Optional<unsigned> index, ValueRange inputs = {})
      : index(index), inputs(inputs) {}

  Optional<unsigned> index;
  ValueRange inputs;
};
} // namespace M::HLCF

#include "KGEN/HLCFDialect/HLCFInterfaces.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFINTERFACES_H
