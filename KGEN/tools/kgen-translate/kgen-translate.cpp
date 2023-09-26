//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

int main(int argc, char *argv[]) {
  KGENCommonOptions clOptions;

  cl::opt<bool> disableBuiltinModule{
      "mojo-disable-builtins",
      cl::desc("Don't auto-import the builtin module. WARNING: A bunch of "
               "stuff will break!"),
      cl::init(false)};

  cl::opt<bool> disableParserCaching{
      "mojo-disable-parser-caching",
      cl::desc("Disable caching when parsing the input Mojo file."),
      cl::init(false)};

  cl::opt<bool> enablePrebuiltPackages{
      "mojo-enable-prebuilt-packages",
      cl::desc("Use prebuilt packages when parsing the input Mojo file."),
      cl::init(false)};

  cl::opt<bool> warnMissingDocStrings{
      "mojo-warn-missing-doc-strings",
      cl::desc("Emit warnings for partial or missing doc strings."),
      cl::init(false)};

  cl::opt<bool> experimentalLifetimes{
      "mojo-experimental-lifetimes",
      cl::desc("Enable experimental new lifetimes generation."),
      cl::init(false)};

  cl::opt<unsigned> maxNotesPerDiagnostic{
      "max-notes-per-diagnostic",
      cl::desc("Maximum number of notes emitted per diagnostic."),
      cl::init(10)};

  cl::opt<bool> useMLIRDiagnostics{"use-mlir-diagnostics",
                                   cl::desc("Whether to use MLIR diagnostics."),
                                   cl::init(true)};

  mlir::TranslateToMLIRRegistration fromMojo(
      "import-mojo", "Import 'mojo' from source",
      [&](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        sourceMgr.setIncludeDirs(clOptions.getIncludePaths());

        TraceProfiler profiler(clOptions.timeTrace,
                               clOptions.timeTraceGranularity);

        DialectRegistry registry;
        registerAllKGENDialects(registry);
        context->appendDialectRegistry(registry);

        // Set up the runtime.
        std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();
        mlir::TimingScope ts;
        CompilationOptions options = clOptions.getCompilationOptions();
        LIT::ParserConfig config(context, *runtime, options);
        config.useMLIRDiagnostics = useMLIRDiagnostics;
        config.warnMissingDocStrings = warnMissingDocStrings;
        config.experimentalLifetimes = experimentalLifetimes;
        config.maxNotesPerDiagnostic = maxNotesPerDiagnostic;
        config.parsingStandardLibrary = !enablePrebuiltPackages;
        config.useBuiltinModule = !disableBuiltinModule;
        if (disableParserCaching)
          config.moduleCachingLevel = LIT::ParserConfig::kCacheNone;
        return LIT::importMojoFile(sourceMgr, config, ts);
      });

  // Register LLVM IR generation.
  mlir::TranslateFromMLIRRegistration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(os, nullptr);
        return success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<MDialect>();
        registerKGENToLLVMTranslation(registry);
      });

  // Run the tool driver.
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
