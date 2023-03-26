//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_ANALYSIS_DATAFLOW_H
#define KGEN_HLCFDIALECT_ANALYSIS_DATAFLOW_H

#include "KGEN/HLCFDialect/Analysis/ControlFlowTree.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

namespace M::HLCF {
/// Subclass the basic dead code analysis to inject a transfer function for HLCF
/// operations.
class DeadCodeAnalysis : public mlir::dataflow::DeadCodeAnalysis {
public:
  DeadCodeAnalysis(mlir::DataFlowSolver &solver,
                   ControlFlowTreeAnalysis &analysis)
      : mlir::dataflow::DeadCodeAnalysis(solver), analysis(analysis) {}

  LogicalResult visit(mlir::ProgramPoint point) override;

private:
  /// Dead code analysis on HLCF ops requires control-flow tree analysis on root
  /// operations. This analysis contains cached analyses for root operations.
  ControlFlowTreeAnalysis &analysis;
};

/// Subclass the basic sparse dataflow analysis to inject a transfer function
/// for HLCF operations.
template <typename StateT, typename AnalysisT>
class SparseDataFlowAnalysis : public AnalysisT {
public:
  using AnalysisT::AnalysisT;

  void visitOperation(Operation *op, ArrayRef<const StateT *> operandLattices,
                      ArrayRef<StateT *> resultLattices) override {
    if (!isa<ControlFlowNode>(op))
      return AnalysisT::visitOperation(op, operandLattices, resultLattices);
    auto *preds =
        this->template getOrCreateFor<mlir::dataflow::PredecessorState>(op, op);
    assert(preds->allPredecessorsKnown());
    for (Operation *pred : preds->getKnownPredecessors()) {
      for (auto [lattice, value] :
           llvm::zip(resultLattices, preds->getSuccessorInputs(pred))) {
        this->join(lattice, *this->getLatticeElementFor(op, value));
      }
    }
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const mlir::RegionSuccessor &successor,
                                    ArrayRef<StateT *> argLattices,
                                    unsigned firstIndex) override {
    auto loopOp = dyn_cast<LoopOp>(op);
    if (!loopOp)
      return AnalysisT::visitNonControlFlowArguments(op, successor, argLattices,
                                                     firstIndex);
    Block *block = &loopOp.getBody().front();
    auto *preds =
        this->template getOrCreateFor<mlir::dataflow::PredecessorState>(block,
                                                                        block);
    assert(preds->allPredecessorsKnown());
    for (Operation *pred : preds->getKnownPredecessors()) {
      for (auto [lattice, value] :
           llvm::zip(argLattices, preds->getSuccessorInputs(pred))) {
        this->join(lattice, *this->getLatticeElementFor(block, value));
      }
    }
  }
};

/// Re-implementation of SCP using HLCF-aware sparse dataflow.
/// TODO: The dataflow analysis framework needs to be improved to compose
/// better. Subclassing + copying analyses is not the right way.
class SparseConstantPropagation
    : public SparseDataFlowAnalysis<
          mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>,
          mlir::dataflow::SparseConstantPropagation> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;
};

} // namespace M::HLCF

#endif // KGEN_HLCFDIALECT_ANALYSIS_DATAFLOW_H
