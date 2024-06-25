//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Init/Init.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

int main(int argc, char *argv[]) {
  KGENCommonOptions clOptions;
  KGENCommonCLOptions parser(clOptions);

  KGEN::registerKGENCommandLineOptions();

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-translate",
      Init::Options().withRuntimeOptions(
          LLCL::RuntimeOptions().withMainWillNotDonate().withCPUAffinity(
              false)));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }

  M::cl::MOpt<bool> disableBuiltinModule{
      "mojo-disable-builtins",
      cl::desc("Don't auto-import the builtin module. WARNING: A bunch of "
               "stuff will break!"),
      cl::init(false)};

  M::cl::MOpt<bool> enablePrebuiltPackages{
      "mojo-enable-prebuilt-packages",
      cl::desc("Use prebuilt packages when parsing the input Mojo file."),
      cl::init(false)};

  M::cl::MOpt<bool> diagnoseMissingDocStrings{
      "mojo-diagnose-missing-doc-strings",
      cl::desc("Diagnose partial or missing doc strings."), cl::init(false)};

  M::cl::MOpt<unsigned> maxNotesPerDiagnostic{
      "max-notes-per-diagnostic",
      cl::desc("Maximum number of notes emitted per diagnostic."),
      cl::init(10)};

  M::cl::MOpt<bool> useMLIRDiagnostics{
      "use-mlir-diagnostics", cl::desc("Whether to use MLIR diagnostics."),
      cl::init(true)};

  M::cl::MOpt<std::string> parserBytecodeOutput{
      "bytecode-output",
      cl::desc("If specified, the parser output is also printed as bytecode."),
      cl::init("")};

  cl::opt<std::string> parserSearchPaths{
      "mojo-search-paths",
      cl::desc("If specified, the `mojo.import_path` field of modular.cfg is "
               "overwritten with this value."),
      cl::init("")};

  mlir::TranslateToMLIRRegistration fromMojo(
      "import-mojo", "Import 'mojo' from source",
      [&](llvm::SourceMgr &sourceMgr,
          MLIRContext *context) -> OwningOpRef<ModuleOp> {
        sourceMgr.setIncludeDirs(clOptions.getIncludePaths());

        clOptions.useSingleThreadedWorkqueue();
        TraceProfiler profiler(clOptions.timeTrace,
                               clOptions.timeTraceGranularity);

        DialectRegistry registry;
        registerAllKGENDialects(registry);
        registerContext(registry, *ctxOr);
        context->appendDialectRegistry(registry);

        mlir::TimingScope ts;
        CompilationOptions options = clOptions.getCompilationOptions();
        options.searchPaths = parserSearchPaths.getValue();
        LIT::ParserConfig config(context, options);
        config.useMLIRDiagnostics = useMLIRDiagnostics;
        config.diagnoseMissingDocStrings = diagnoseMissingDocStrings;
        config.maxNotesPerDiagnostic = maxNotesPerDiagnostic;
        config.disablePrebuiltPackages = !enablePrebuiltPackages;
        config.useBuiltinModule = !disableBuiltinModule;
        OwningOpRef<ModuleOp> output = LIT::importMojoFile(
            *(*ctxOr)->get<LLCL::Runtime>(), sourceMgr, config, ts);

        if (output && !parserBytecodeOutput.getValue().empty()) {
          std::string message;
          auto out =
              mlir::openOutputFile(parserBytecodeOutput.getValue(), &message);
          if (!out) {
            llvm::errs() << "failed to open file: " << message << "\n";
            return {};
          }
          if (failed(mlir::writeBytecodeToFile(*output, out->os())))
            return {};
          out->keep();
        }

        return output;
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
