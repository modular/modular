//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENCompiler.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
ObjectCompiler::lowerAllFuncsToLLVM(const SymbolTable &symtab,
                                    const ExportMap &exportedSymbols,
                                    llvm::LLVMContext &ctx) {
  OwningOpRef<ModuleOp> module =
      produceStandaloneModule(symtab, exportedSymbols);
  return lowerAllFuncsToLLVM(ctx, *module);
}

std::unique_ptr<llvm::Module>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  TimeTraceScope<> traceScope("lower-to-llvm");
  mgr->clear();

  // We only need to run the post-elaboration passes if we are searching. In
  // non-search mode, we know the passes have already been run.
  if (isSearch)
    populatePostElaborationPasses(*mgr, runtime, options);

  LowerToLLVMOptions llvmOptions(options.getDIEmissionKind(),
                                 options.debugAtLevel);
  llvmOptions.isJIT = isJIT;
  // Use KGENCompilerRT allocators.
  llvmOptions.alignedAllocFnName = "KGEN_CompilerRT_AlignedAlloc";
  llvmOptions.alignedFreeFnName = "KGEN_CompilerRT_AlignedFree";
  buildLowerToLLVMPipeline(*mgr, llvmOptions);
  if (failed(mgr->run(module)))
    return nullptr;

  // Use the input filename for the module name if possible.
  StringRef moduleName = "LLVMDialectModule";
  if (auto moduleLoc = module.getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = moduleLoc.getFilename();

  // Translate the operation into an LLVM module.
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return nullptr;

  // Attach any necessary instrumentation to the module.
  attachXRayAttributes(*llvmModule, options);
  return llvmModule;
}
