//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/STLExtras.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

/// If requested, attach XRay instrumentation to the given module.
/// TODO: Eventually we should explore attaching this information at a higher
/// level of the stack.
static void attachXRayAttributes(llvm::Module &module,
                                 const CompilationOptions &options) {
  if (!options.enableXRayInstrumentation)
    return;

  for (llvm::Function &f : module.functions()) {
    if (f.isDeclaration())
      continue;
    f.addFnAttr("function-instrument", "xray-always");
  }
}

std::unique_ptr<llvm::Module>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx) {
  OwningOpRef<ModuleOp> module = produceStandaloneModule();
  return lowerAllFuncsToLLVM(ctx, *module);
}

std::unique_ptr<llvm::Module>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  mlir::PassManager pm(module->getContext());

  // TODO (#7846): Remove this once the elaborator does inlining. Maybe keep
  //   `force-inline`.
  pm.addPass(createForceInline());
  pm.addNestedPass<KGEN::FuncOp>(createCleanupCompilerGlobals());
  pm.addNestedPass<KGEN::FuncOp>(mlir::createCanonicalizerPass());

  // If we aren't generating debug information, make sure it's been stripped.
  if (options.debugLevel == CompilationOptions::kNoDebug)
    pm.addPass(DebugInfo::createDebugInfoStrip());

  LowerToLLVMOptions llvmOptions(options.getDIEmissionKind(),
                                 options.debugAtLevel);
  pm.addPass(createLowerZAPToPOP());
  buildLowerToLLVMPipeline(pm, llvmOptions);
  if (failed(pm.run(module)))
    return nullptr;

  // Translate the operation into an LLVM module.
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx);
  if (!llvmModule)
    return nullptr;

  // Attach any necessary instrumentation to the module.
  attachXRayAttributes(*llvmModule, options);
  return llvmModule;
}

//===----------------------------------------------------------------------===//
// EmitLLVMPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_EMITLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class EmitLLVMPass : public M::KGEN::impl::EmitLLVMBase<EmitLLVMPass> {
public:
  using EmitLLVMBase::EmitLLVMBase;
  EmitLLVMPass(Runtime &rt) : EmitLLVMBase::EmitLLVMBase(), runtime(&rt) {}

  void runOnOperation() override;

private:
  Runtime *runtime;
};
} // namespace

void EmitLLVMPass::runOnOperation() {
  // If no runtime was provided, create one.
  auto rt = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
      runtime, createLeakCheckAllocator(createMallocAllocator()),
      createSingleThreadWorkQueue());

  // TODO: Populate compilation options from pass options.
  auto compiler = ObjectCompiler::create(
      *rt, ".kgen_cache",
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable(),
      CompilationOptions());
  if (failed(compiler)) {
    getOperation()->emitError()
        << "failed to create object compiler: " << compiler.getError();
    return signalPassFailure();
  }

  // Lower all functions to LLVM.
  llvm::LLVMContext ctx;
  auto llvmModule = compiler->lowerAllFuncsToLLVM(ctx);
  if (!llvmModule)
    return signalPassFailure();

  // We might have an output file.
  std::unique_ptr<llvm::ToolOutputFile> outputFile = nullptr;
  if (!output.empty()) {
    std::string err;
    outputFile = mlir::openOutputFile(output.getValue(), &err);
    if (!outputFile) {
      mlir::emitError(getOperation()->getLoc()) << err;
      return signalPassFailure();
    }
  }

  if (outputFile) {
    llvmModule->print(outputFile->os(), nullptr);
    outputFile->keep();
    return;
  }

  llvmModule->print(llvm::outs(), nullptr);
}

std::unique_ptr<Pass> M::KGEN::createEmitLLVMPass(Runtime &rt) {
  return std::make_unique<EmitLLVMPass>(rt);
}

void M::KGEN::registerEmitLLVMPass(Runtime &rt) {
  mlir::registerPass([&]() { return createEmitLLVMPass(rt); });
}
