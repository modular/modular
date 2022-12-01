//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/HLCFDialect/Analysis/DataFlow.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

namespace {
/// This is a pass for testing data-flow analysis on HLCF operations.
struct TestDataFlowPass
    : public mlir::PassWrapper<TestDataFlowPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataFlowPass);

  StringRef getArgument() const override { return "test-dataflow"; };

  void runOnOperation() override {
    mlir::DataFlowSolver solver;
    solver.load<HLCF::DeadCodeAnalysis>(getAnalysisManager());
    solver.load<HLCF::SparseConstantPropagation>();
    if (failed(solver.initializeAndRun(getOperation())))
      return signalPassFailure();
    getOperation()->walk([&](Operation *op) {
      auto label = op->getAttrOfType<StringAttr>("print_operand_constants");
      if (!label)
        return;
      llvm::errs() << label.getValue() << '(';
      llvm::interleaveComma(
          op->getOperands(), llvm::errs(), [&](Value operand) {
            auto *cv = solver.lookupState<
                mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(
                operand);
            if (!cv) {
              llvm::errs() << '?';
              return;
            }
            cv->print(llvm::errs());
          });
      llvm::errs() << ")\n";
    });
  }
};
} // namespace

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect, DebugInfo::DebugInfoDialect,
                  HLCF::HLCFDialect, MDialect>();
  mlir::registerCanonicalizer();
  M::HLCF::registerLowerHLCFToLLVMPass();
  DebugInfo::registerDebugInfoToLLVMPass();
  DebugInfo::registerTransformsPasses();

  // Register test passes.
  mlir::PassRegistration<TestDataFlowPass>{};

  return failed(
      mlir::MlirOptMain(argc, argv, "index optimizer driver", registry));
}
