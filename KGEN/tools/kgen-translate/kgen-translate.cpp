//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;

int main(int argc, char *argv[]) {
  KGEN::KGENCommonOptions clOptions;

  cl::opt<bool> disableParserCaching{
      "mojo-disable-parser-caching",
      cl::desc("Disable caching when parsing the input Mojo file."),
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

        // Set up the runtime.
        std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();
        mlir::TimingScope ts;
        KGEN::CompilationOptions options = clOptions.getCompilationOptions();
        MojoParserConfig config(context, *runtime, options);
        config.useMLIRDiagnostics = useMLIRDiagnostics;
        config.warnMissingDocStrings = warnMissingDocStrings;
        config.experimentalLifetimes = experimentalLifetimes;
        config.maxNotesPerDiagnostic = maxNotesPerDiagnostic;
        // Disable binary packages when using `kgen-translate`.
        config.parsingStandardLibrary = true;
        if (disableParserCaching)
          config.moduleCachingLevel = MojoParserConfig::kCacheNone;
        return importMojoFile(sourceMgr, config, ts);
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
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
      });

  // Run the tool driver.
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
