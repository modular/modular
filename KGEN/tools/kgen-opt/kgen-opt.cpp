//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/HLCFDialect/Analysis/DataFlow.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;

namespace {

//===----------------------------------------------------------------------===//
// TestDataFlowPass
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TestGeneratePreElaboratedBody
//===----------------------------------------------------------------------===//

/// This pass generates a kgen.func that clones the body and adds a
/// "_elaborated" suffix to the name for all the specified lit.funcs in the
/// module. It's used to test the logic in LowerLIT that handles pre-elaborated
/// funcs.
struct TestGeneratePreElaboratedBody
    : public mlir::PassWrapper<TestGeneratePreElaboratedBody,
                               OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGeneratePreElaboratedBody);

  StringRef getArgument() const override {
    return "test-generate-elaborated-body";
  };

  void runOnOperation() override {
    ModuleOp theModule = getOperation();
    TargetInfoAttr target = M::lookupTargetInfo(theModule);

    // Attach a kgen.func to the lit.func. This dummy function simply contains
    // exactly the same operations as the lit.func, but has a slightly different
    // name.
    for (auto func : theModule.getOps<KGEN::LIT::FuncOp>()) {
      // Allow the test to skip functions if they have the doNotExtern attr.
      if (func->hasAttr("doNotExtern"))
        continue;

      // Allow some functions to specify that they use an incompatible target.
      TargetInfoAttr funcTarget = target;
      if (auto newTarget = func->getAttrOfType<TargetInfoAttr>("test.target"))
        funcTarget = newTarget;

      OpBuilder b(func.getContext());
      OwningOpRef<ModuleOp> fakeModule = b.create<ModuleOp>(func.getLoc());
      OpBuilder fakeBuilder = OpBuilder::atBlockEnd(fakeModule->getBody());
      KGEN::FuncOp fakeCompiledBody = fakeBuilder.create<KGEN::FuncOp>(
          func.getLoc(), b.getStringAttr(func.getSymName() + "_precompiled"),
          func.getSignature(), KGEN::InlineLevel::Automatic);

      // Just clone the body in.
      mlir::IRMapping map;
      func.getBodyRegion().cloneInto(&fakeCompiledBody.getBodyRegion(), map);

      // Generate the bytecode for the module bytecode.
      SmallVector<char> buffer;
      llvm::raw_svector_ostream stream(buffer);
      if (failed(mlir::writeBytecodeToFile(*fakeModule, stream)))
        return signalPassFailure();

      // Externalize the function and attach the post elaboration metadata.
      func.getBody()->clear();
      OpBuilder::atBlockBegin(func.getBody())
          .create<KGEN::LIT::ExternFuncOp>(func.getLoc());
      StringAttr linkName = b.getStringAttr("link_" + func.getSymName());
      func.setPreCompiledModuleRefAttr(FlatSymbolRefAttr::get(linkName));
      func.setLinkageName(fakeCompiledBody.getSymNameAttr());

      // Generate a package link to the fake module.
      OpBuilder linkBuilder(func);
      auto bytecodeBufferAttr = createResourceAttr(
          &getContext(), buffer, func.getSymName() + "_generated_body_attr");
      linkBuilder.create<KGEN::LIT::PackageLinkOp>(
          func.getLoc(), linkName, bytecodeBufferAttr, funcTarget,
          M::KGEN::EnvAttr::parseDefines(func.getContext(), {}).takeValue(),
          M::KGEN::LIT::PackageArchiveAttr::get(bytecodeBufferAttr,
                                                bytecodeBufferAttr));
    }
  }
};

//===----------------------------------------------------------------------===//
// TestAlwaysFailPass
//===----------------------------------------------------------------------===//

/// This is a pass that always fails for the purpose of debugging reproducers.
struct TestAlwaysFailPass
    : public mlir::PassWrapper<TestAlwaysFailPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlwaysFailPass);

  StringRef getArgument() const override { return "test-always-fail"; };

  void runOnOperation() override { return signalPassFailure(); }
};

} // namespace

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register all KGEN dialects.
  registerAllKGENDialects(registry);

  // Initialize the host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize LLVM exporters.
  registerKGENToLLVMTranslation(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerConvertIndexToLLVMPass();

  // Register test passes.
  mlir::PassRegistration<TestDataFlowPass>{};
  mlir::PassRegistration<TestGeneratePreElaboratedBody>{};
  mlir::PassRegistration<TestAlwaysFailPass>{};

  // Register opt passes.
  KGEN::registerCanonicalizer();
  KGEN::registerCheckLifetimes();
  KGEN::registerCheckRecursiveStructs();
  KGEN::registerCleanupCompilerGlobals();
  KGEN::registerConstraintReduction();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerExternalizePrecompiledFunctions();
  KGEN::registerFoldGlobalConstLoads();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerCallingConvention();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerLoops();
  KGEN::registerLowerKGENCoroutinesAsync();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLowerPreElaboratedLIT();
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
  KGEN::registerStripParserMetadata();
  DebugInfo::registerDebugInfoToLLVM();
  DebugInfo::registerDebugInfoStrip();

  // Register passes that require a runtime.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createThreadPoolWorkQueue());
  mlir::registerPass(
      [&] { return KGEN::createElaborateGeneratorsWithDefaultJIT(runtime); });
  mlir::registerPass([&] { return KGEN::createForceInline(runtime); });
  mlir::registerPass(
      [&] { return KGEN::createAlwaysInlineParametric(runtime); });
  mlir::registerPass([&] { return KGEN::createAutomaticInline(runtime); });

  return failed(
      mlir::MlirOptMain(argc, argv, "kgen optimizer driver", registry));
}
