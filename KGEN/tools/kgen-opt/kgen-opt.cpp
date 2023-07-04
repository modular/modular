//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/DataFlow.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassRegistry.h"
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
    solver.load<HLCF::DeadCodeAnalysis>(
        getAnalysis<HLCF::ControlFlowTreeAnalysis>());
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

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect>();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerConvertIndexToLLVMPass();

  // Register test passes.
  mlir::PassRegistration<TestDataFlowPass>{};

  // Register opt passes.
  KGEN::registerAlwaysInlineParametric();
  KGEN::registerCanonicalizer();
  KGEN::registerCheckLifetimes();
  KGEN::registerCheckRecursiveStructs();
  KGEN::registerCleanupCompilerGlobals();
  KGEN::registerConstraintReduction();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerFoldGlobalConstLoads();
  KGEN::registerForceInline();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerKGENCoroutinesAsync();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLowerPOPToLLVM();
  KGEN::registerLowerRuntimeClosures();
  KGEN::registerLowerSemanticCF();
  KGEN::registerLowerStructs();
  KGEN::registerMem2Reg();
  KGEN::registerOutlineClosures();
  KGEN::registerPruneImpossibleVariants();
  KGEN::registerRaiseForLoops();
  KGEN::registerSROA();
  KGEN::registerSimplifyCF();
  KGEN::registerStackReuse();
  KGEN::registerSynthesizeDebugInfo();
  KGEN::registerTweakSpilledAllocas();
  KGEN::registerVerifyParameters();
  KGEN::registerLowerToLLVMPipeline();

  return failed(
      mlir::MlirOptMain(argc, argv, "kgen optimizer driver", registry));
}
