//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HLCFDIALECT_HLCFOPS_H
#define SUPPORT_HLCFDIALECT_HLCFOPS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Operation Traits
//===----------------------------------------------------------------------===//

namespace M::HLCF {
LogicalResult verifyControlFlowNode(Operation *op);
LogicalResult verifyControlFlowTerminator(Operation *op);
} // namespace M::HLCF

namespace mlir::OpTrait {
/// This trait marks operations whose regions form nodes an HLCF control-flow
/// tree.
template <typename ConcreteOp>
class ControlFlowNode : public TraitBase<ConcreteOp, ControlFlowNode> {
public:
  /// Perform local verification of the control-flow operation.
  static LogicalResult verifyRegionTrait(Operation *op) {
    return M::HLCF::verifyControlFlowNode(op);
  }
};

/// This trait marks operations that terminate HLCF control-flow regions.
template <typename ConcreteOp>
class ControlFlowTerminator
    : public TraitBase<ConcreteOp, ControlFlowTerminator> {
public:
  /// Perform local verification of the control-flow terminator.
  static LogicalResult verifyTrait(Operation *op) {
    return M::HLCF::verifyControlFlowTerminator(op);
  }
};
} // namespace mlir::OpTrait

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/HLCFDialect/HLCF.h.inc"

#endif // SUPPORT_HLCFDIALECT_HLCFOPS_H
