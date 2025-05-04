//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_HLCFINTERFACES_H
#define KGEN_HLCFDIALECT_HLCFINTERFACES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// Interface Verifiers
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class FunctionLike;
} // namespace M::KGEN

namespace M::HLCF {
class ControlFlowNode;
class ControlFlowTerminator;

LogicalResult verifyControlFlowNode(ControlFlowNode op);
LogicalResult verifyControlFlowTerminator(ControlFlowTerminator op);
LogicalResult verifyControlFlow(KGEN::FunctionLike root);
} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::HLCF {
struct ControlFlowTarget {
  ControlFlowTarget(std::optional<unsigned> index, ValueRange inputs = {})
      : index(index), inputs(inputs) {}

  std::optional<unsigned> index;
  ValueRange inputs;
};
} // namespace M::HLCF

#include "KGEN/HLCFDialect/HLCFInterfaces.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFINTERFACES_H
