//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGENToLLVMPipeline.h"
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
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace KGEN;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

/// If requested, attach sanitizer/XRay/etc. instrumentations to the given
/// module.
/// TODO: Eventually we should explore attaching this information at a higher
/// level of the stack.
static void attachInstrumentationAttributes(llvm::Module &module,
                                            const CompilationOptions &options) {
  if (!options.enableXRayInstrumentation && !options.sanitizers)
    return;

  for (llvm::Function &f : module.functions()) {
    if (f.isDeclaration())
      continue;
    if (options.enableXRayInstrumentation)
      f.addFnAttr("function-instrument", "xray-always");
    if (options.sanitizers.has(Sanitizers::kAddress))
      f.addFnAttr(llvm::Attribute::SanitizeAddress);
    if (options.sanitizers.has(Sanitizers::kThread))
      f.addFnAttr(llvm::Attribute::SanitizeThread);
  }
}

/// HACK HACK HACK https://github.com/modularml/modular/issues/27478
/// Using LineTables for NVPTX backend disables optimizations in cuda JIT. Use
/// DebugDirectives instead for equivalent performance to no-debug.
static void adaptDebugEmissionKind(ModuleOp module, StringRef targetTriple,
                                   DebugInfo::EmissionKind debugLevel) {
  bool generatingPtx = targetTriple.contains("nvptx");
  if (generatingPtx && debugLevel == DebugInfo::EmissionKind::LineTablesOnly) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](DebugInfo::DICompileUnitAttr CU) -> std::optional<Attribute> {
          if (CU.getEmissionKind() == DebugInfo::EmissionKind::LineTablesOnly) {
            return DebugInfo::DICompileUnitAttr::get(
                CU.getSourceLanguage(), CU.getFile(), CU.getProducer(),
                CU.getIsOptimized(),
                DebugInfo::EmissionKind::DebugDirectivesOnly,
                CU.getNameTableKind());
          }
          return std::nullopt;
        });
    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/true);
  }
}

ErrorOr<std::unique_ptr<llvm::Module>>
ObjectCompiler::lowerAllFuncsToLLVM(const SymbolTable &symtab,
                                    const ExportMap &exportedSymbols,
                                    llvm::LLVMContext &ctx) {
  OwningOpRef<ModuleOp> module =
      produceStandaloneModule(symtab, exportedSymbols);
  return lowerAllFuncsToLLVM(ctx, *module);
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  mlir::MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

ErrorOr<std::unique_ptr<llvm::Module>>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  CompilerTimeTraceScope traceScope("lower-to-llvm");

  mlir::PassManager mgr = createPassManager(pmOptions.operationName, &context);

  ErrorOrSuccess configPM = pmOptions.configurePassManager(mgr);
  if (configPM)
    return configPM.takeError();

  adaptDebugEmissionKind(module, options.targetTriple,
                         options.getDIEmissionKind());

  LowerToLLVMOptions llvmOptions(
      options.getDIEmissionKind(), options.debugAtLevel,
      static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage));
  llvmOptions.globalCtorFnName = ExecutionEngine::getGlobalCtorFnName();
  llvmOptions.globalDtorFnName = ExecutionEngine::getGlobalDtorFnName();
  // Use KGENCompilerRT allocators.
  llvmOptions.alignedAllocFnName = "KGEN_CompilerRT_AlignedAlloc";
  llvmOptions.alignedFreeFnName = "KGEN_CompilerRT_AlignedFree";

  buildLowerToLLVMPipeline(mgr, llvmOptions);

  if (failed(mgr.run(module)))
    return Error("run LowerToLLVMPipeline failed");

  // Use the input filename for the module name if possible.
  StringRef moduleName = "LLVMDialectModule";
  if (auto moduleLoc = module.getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Translate the operation into an LLVM module.
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return Error("translate module to LLVMIR failed");

  // Attach any necessary instrumentation to the module.
  attachInstrumentationAttributes(*llvmModule, options);

  return llvmModule;
}
