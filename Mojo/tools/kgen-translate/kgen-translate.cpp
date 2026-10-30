//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Init/Init.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoTooling/DocGen.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/DriverOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/ToolCommon/OptionNormalizers.h"
#include "Mojo/ToolCommon/TargetOptions.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/Driver/OptionMarshalling.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

#define DRIVER_OPTIONS_PATH "Mojo/tools/kgen-translate/KGENTranslateOptions.inc"
#include "Mojo/tools/kgen-translate/KGENTranslateOptionsHelpTexts.inc"
#include "Support/Driver/OptTable.inc"

namespace {
/// A headless parser listener that mirrors how the Mojo LSP server exercises
/// the parser. It reports interest only in locations within the main file
/// (matching `LSPParserListener::isInterestedInLoc` in the LSP server), which
/// gates the parser's eager, listener-driven resolution to the same scope the
/// LSP uses. The `on*` notification callbacks are intentionally left as
/// base-class no-ops: kgen-translate only needs the parser to *do the work*,
/// not to build a symbol index itself.
class MainFileParserListener : public LIT::ParserListener {
public:
  explicit MainFileParserListener(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr) {}

  bool isInterestedInLoc(llvm::SMLoc loc) override {
    return sourceMgr.FindBufferContainingLoc(loc) == sourceMgr.getMainFileID();
  }

private:
  const llvm::SourceMgr &sourceMgr;
};

/// The options that kgen-translate parsed from its command line.
struct TranslateOptions {
  CompilationOptions compilationOptions;
  FrontendOptions frontendOptions;
  DriverOptions driverOptions;
  std::string outputPath = "-";
  std::string bytecodeOutput;
  bool lspMode = false;
  bool splitInputFile = false;
  std::string inputSplitMarker;
  std::string outputSplitMarker;
  mlir::SourceMgrDiagnosticVerifierHandler::Level verifyDiagnosticsLevel =
      mlir::SourceMgrDiagnosticVerifierHandler::Level::None;
  bool allowUnregisteredDialects = false;
  bool errorDiagnosticsOnly = false;
  bool noImplicitModule = false;
  bool enablePrebuiltPackages = false;
};

/// Forwards the `-mllvm` values, and the printing flags that kgen-translate
/// owns but LLVM and MLIR read, to `llvm::cl`.
bool parseForwardedCLOptions(const State &state,
                             const llvm::opt::InputArgList &args) {
  KGEN::registerKGENCommandLineOptions();
  KGENPassCLOptions::registerOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Each translation's table gives these options its own IDs.
  static constexpr llvm::StringLiteral printingFlags[] = {
      "kgen-print-inline-type-values", "mlir-print-debuginfo",
      "mlir-print-op-generic"};
  SmallVector<const char *> clArgs = {state.programName};
  for (const llvm::opt::Arg *arg : args) {
    StringRef name = arg->getOption().getName();
    if (name == "mllvm")
      clArgs.push_back(arg->getValue());
    else if (llvm::is_contained(printingFlags, name))
      clArgs.push_back(args.getArgString(arg->getIndex()));
  }
  return llvm::cl::ParseCommandLineOptions(clArgs.size(), clArgs.data(),
                                           "kgen-translate", &llvm::errs());
}

/// Reports an error and returns a failure exit code if more than one input was
/// given.
int rejectExtraInputs(const State &state, ArrayRef<std::string> inputs) {
  if (inputs.size() <= 1)
    return EXIT_SUCCESS;
  return state.reportError(
      llvm::formatv("too many input files, cannot process both '{0}' and '{1}'",
                    inputs[0], inputs[1]));
}

ErrorOr<ContextRef> createContext() {
  return Init::createContext(
      "kgen-translate",
      Init::Options().withCPUDeviceOptions(
          AsyncRT::CPUDeviceOptions().withMainWillNotDonate().withCPUAffinity(
              false)));
}

/// Runs `translation` on the input file, or stdin when there is none.
int runTranslation(const State &state, const mlir::Translation &translation,
                   ArrayRef<std::string> inputs,
                   const TranslateOptions &options) {
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input =
      mlir::openInputFile(inputs.empty() ? "-" : inputs.front(), &errorMessage);
  if (!input)
    return state.reportError(errorMessage);
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(options.outputPath, &errorMessage);
  if (!output)
    return state.reportError(errorMessage);

  mlir::MlirTranslateMainConfig config;
  config.allowUnregisteredDialects(options.allowUnregisteredDialects)
      .errorDiagnosticsOnly(options.errorDiagnosticsOnly)
      .outputSplitMarker(options.outputSplitMarker)
      .verifyDiagnostics(options.verifyDiagnosticsLevel);
  if (!options.inputSplitMarker.empty())
    config.splitInputFile(options.inputSplitMarker);
  else if (options.splitInputFile)
    config.splitInputFile();

  if (failed(mlir::mlirTranslateMain(std::move(input), output->os(),
                                     {&translation}, config)))
    return EXIT_FAILURE;
  output->keep();
  return EXIT_SUCCESS;
}

/// Imports Mojo source as MLIR and prints it to `output`.
LogicalResult importMojo(const State &state,
                         const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                         llvm::raw_ostream &output, MLIRContext *context,
                         ContextRef &ctxRef, TranslateOptions &options) {
  configureSourceMgr(state, options.driverOptions, *sourceMgr);

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerContext(registry, ctxRef);
  context->appendDialectRegistry(registry);

  mlir::TimingScope ts;
  CompilationOptions &compilationOptions = options.compilationOptions;
  FrontendOptions frontendOptions = options.frontendOptions;

  OwningOpRef<ModuleOp> module;
  if (options.lspMode) {
    // Mirror the language server: build a single-threaded parser context
    // over this SourceMgr and run the same lazy, error-tolerant parse the
    // LSP uses for an open document.
    context->disableMultithreading();

    // Install a parser listener the same way the LSP server does, so the
    // parse exercises the parser's eager listener-driven resolution paths
    // (member lookups, reference resolution, import resolution) that are
    // otherwise skipped when no listener is present.
    MainFileParserListener lspListener(*sourceMgr);
    frontendOptions.parserListener = &lspListener;

    MojoParserContext parserContext(*sourceMgr, context, compilationOptions,
                                    frontendOptions);
    LIT::ASTDecl *decl =
        parserContext.parseFileForLSP(sourceMgr->getMainFileID());
    parserContext.ensureSignaturesResolved();
    if (!decl)
      return failure();

    // Clone the parsed module out: the parser context owns the original
    // IR and finalizes imported bytecode in its destructor, so we
    // snapshot the pre-finalization "LSP view" the server sees.
    module = OwningOpRef<ModuleOp>(parserContext.getModule().clone());
  } else {
    module = LIT::importMojoFile(ctxRef, *sourceMgr, context,
                                 compilationOptions, frontendOptions, ts,
                                 /*includedFiles=*/nullptr);
  }

  if (!module)
    return failure();

  context->loadDialect<KGENDialect>();
  ErrorOr<EnvAttr> env = compilationOptions.parseDefinesWithDefaults(
      context, options.driverOptions.defines);
  if (env.isError())
    return failure(state.reportError(env.getError()));
  (*module)->setAttr(EnvAttr::getEnvAttrName(), *env);

  // To make sure the parser did a good thing, we run the
  // verify-parameters pass. We skip this in LSP mode: parseFileForLSP
  // intentionally leaves empty-body FnOp stubs for imported bytecode and
  // skips DCE/verification, so the verifier would reject IR that is
  // legitimately shaped the way the language server sees it.
  if (!options.lspMode) {
    mlir::PassManager pm(context);
    pm.addPass(createVerifyParameters());
    if (failed(pm.run(*module)))
      return failure(state.reportError("mojo parser created invalid IR"));
  }

  if (!options.bytecodeOutput.empty()) {
    std::string message;
    std::unique_ptr<llvm::ToolOutputFile> out =
        mlir::openOutputFile(options.bytecodeOutput, &message);
    if (!out)
      return failure(state.reportError("failed to open file: " + message));

    if (failed(mlir::writeBytecodeToFile(*module, out->os())))
      return failure();
    out->keep();
  }

  if (failed(verify(*module)))
    return failure();
  module->print(output);
  return success();
}

/// Parses MLIR from `sourceMgr` and prints its translation to LLVM IR.
LogicalResult
translateToLLVMIR(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                  llvm::raw_ostream &output, MLIRContext *context,
                  bool insertImplicitModule) {
  DialectRegistry registry;
  registry.insert<MDialect>();
  registerKGENToLLVMTranslation(registry);
  context->appendDialectRegistry(registry);

  OwningOpRef<Operation *> op =
      parseSourceFileForTool(sourceMgr, context, insertImplicitModule);
  if (!op || failed(verify(*op)))
    return failure();

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*op, llvmContext);
  if (!llvmModule)
    return failure();

  llvmModule->print(output, nullptr);
  return success();
}

} // namespace

// Each translation other than -import-mojo parses its arguments again with a
// table of only the options it heeds, in its own namespace so that the tables'
// definitions don't clash.
namespace EmitMojoDoc {
#define DRIVER_OPTIONS_PATH                                                    \
  "Mojo/tools/kgen-translate/KGENTranslateDocOptions.inc"
#include "Mojo/tools/kgen-translate/KGENTranslateDocOptionsHelpTexts.inc"
#include "Support/Driver/OptTable.inc"

/// Writes the documentation of a Mojo file or package as JSON.
int run(State state) {
  options::Table optTable;
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode =
          state.parseArguments(optTable, options::helpTexts(), args))
    return *exitCode;

  TranslateOptions options;
  MarshallingErrors errors;
  {
    DriverOptions &driverOptions = options.driverOptions;
    FrontendOptions &frontendOptions = options.frontendOptions;
#define MOJO_OPTION_WITH_MARSHALLING(...)                                      \
  PARSE_OPTION_WITH_MARSHALLING(args, errors, __VA_ARGS__)
#include "Mojo/tools/kgen-translate/KGENTranslateDocOptions.inc"
#undef MOJO_OPTION_WITH_MARSHALLING
  }
  if (int result = state.rejectInvalidArguments(args, errors))
    return result;

  if (args.hasArg(options::OPT_validate_doc_strings) &&
      !args.hasArg(options::OPT_werror) &&
      !args.hasArg(options::OPT_wno_error)) {
    state.reportWarning(
        "--validate-doc-strings is deprecated, use -Werror instead");
    state.warningsAsErrors = true;
  }

  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (inputs.empty())
    return state.reportError("no input file provided");
  if (int result = rejectExtraInputs(state, inputs))
    return result;
  setWarningPolicy(state, options.frontendOptions);

  ErrorOr<ContextRef> ctxOr = createContext();
  if (ctxOr.isError())
    return state.reportError(
        llvm::formatv("failed to create context: {0}", ctxOr.getError()));

  ErrorOr<std::filesystem::path> resolvedPath =
      resolveMojoInputFileOrPackage(inputs.front());
  if (resolvedPath.isError())
    return state.reportError(resolvedPath.getError());

  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(options.outputPath, &outputError);
  if (!output)
    return state.reportError(outputError);

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerContext(registry, *ctxOr);
  MLIRContext context(registry);

  if (!generateMojoDocJSON(*resolvedPath, context, options.frontendOptions,
                           options.driverOptions.includePaths,
                           state.diagnosticFormat, output->os()))
    return state.reportError("could not generate documentation");
  output->keep();
  return EXIT_SUCCESS;
}
} // namespace EmitMojoDoc

namespace MLIRToLLVMIR {
#define DRIVER_OPTIONS_PATH                                                    \
  "Mojo/tools/kgen-translate/KGENTranslateLLVMIROptions.inc"
#include "Mojo/tools/kgen-translate/KGENTranslateLLVMIROptionsHelpTexts.inc"
#include "Support/Driver/OptTable.inc"

/// Translates MLIR to LLVM IR.
int run(State state) {
  options::Table optTable;
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode =
          state.parseArguments(optTable, options::helpTexts(), args))
    return *exitCode;

  TranslateOptions options;
  MarshallingErrors errors;
  {
    DriverOptions &driverOptions = options.driverOptions;
#define MOJO_OPTION_WITH_MARSHALLING(...)                                      \
  PARSE_OPTION_WITH_MARSHALLING(args, errors, __VA_ARGS__)
#include "Mojo/tools/kgen-translate/KGENTranslateLLVMIROptions.inc"
#undef MOJO_OPTION_WITH_MARSHALLING
  }
  if (int result = state.rejectInvalidArguments(args, errors))
    return result;

  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (int result = rejectExtraInputs(state, inputs))
    return result;

  if (!parseForwardedCLOptions(state, args))
    return EXIT_FAILURE;

  ErrorOr<ContextRef> ctxOr = createContext();
  if (ctxOr.isError())
    return state.reportError(
        llvm::formatv("failed to create context: {0}", ctxOr.getError()));

  TraceProfiler profiler(options.driverOptions.timeTrace,
                         options.driverOptions.timeTraceGranularity);
  mlir::Translation translation(
      [&](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
          llvm::raw_ostream &output, MLIRContext *context) {
        return translateToLLVMIR(sourceMgr, output, context,
                                 !options.noImplicitModule);
      },
      "Translate MLIR to LLVMIR", /*inputAlignment=*/std::nullopt);
  return runTranslation(state, translation, inputs, options);
}
} // namespace MLIRToLLVMIR

int main(int argc, char *argv[]) {
  State state("kgen-translate",
              llvm::ArrayRef<const char *>(argv + 1, argv + argc));
  options::Table optTable;

  // This table accepts every translation's options, so it finds the
  // translation wherever it is given. The other tables accept only their own
  // translation, so a missing or repeated one is left to this table to report.
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList selectionArgs =
      optTable.ParseArgs(state.arguments, missingIndex, missingCount);
  unsigned numTranslations = selectionArgs.hasArg(options::OPT_import_mojo) +
                             selectionArgs.hasArg(options::OPT_emit_mojo_doc) +
                             selectionArgs.hasArg(options::OPT_mlir_to_llvmir);
  if (numTranslations == 1) {
    if (selectionArgs.hasArg(options::OPT_emit_mojo_doc))
      return EmitMojoDoc::run(state);
    if (selectionArgs.hasArg(options::OPT_mlir_to_llvmir))
      return MLIRToLLVMIR::run(state);
  }

  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode =
          state.parseArguments(optTable, options::helpTexts(), args))
    return *exitCode;

  TranslateOptions options;
  TargetArgs targetArgs;
  MarshallingErrors errors;
  {
    DriverOptions &driverOptions = options.driverOptions;
    FrontendOptions &frontendOptions = options.frontendOptions;
    CompilationOptions &compilationOptions = options.compilationOptions;
#define MOJO_OPTION_WITH_MARSHALLING(...)                                      \
  PARSE_OPTION_WITH_MARSHALLING(args, errors, __VA_ARGS__)
#include "Mojo/tools/kgen-translate/KGENTranslateOptions.inc"
#undef MOJO_OPTION_WITH_MARSHALLING
  }
  if (int result = state.rejectInvalidArguments(args, errors))
    return result;

  if (numTranslations != 1) {
    return state.reportError("exactly one of -import-mojo, -emit-mojo-doc "
                             "and -mlir-to-llvmir must be given");
  }

  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (int result = rejectExtraInputs(state, inputs))
    return result;

  setWarningPolicy(state, options.frontendOptions);
  if (ErrorOrSuccess err =
          validateCompilationOptions(options.compilationOptions))
    return state.reportError(err.getError());

  if (args.hasArg(options::OPT_TargetOptionGroup)) {
    MLIRContext targetContext;
    targetContext.loadDialect<MDialect>();
    TargetInfoAttr target;
    if (ErrorOrSuccess err = parseTargetOptions(
            targetArgs, options.compilationOptions, targetContext, target))
      return state.reportError(err.getError());
  }

  bool useMLIRDiagnostics = true;
  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_use_mlir_diagnostics,
                          options::OPT_use_mlir_diagnostics_eq);
      arg && arg->getOption().matches(options::OPT_use_mlir_diagnostics_eq)) {
    std::optional<bool> value =
        llvm::StringSwitch<std::optional<bool>>(arg->getValue())
            .Cases({"true", "TRUE", "True", "1"}, true)
            .Cases({"false", "FALSE", "False", "0"}, false)
            .Default(std::nullopt);
    if (!value)
      return state.reportError(
          llvm::formatv("invalid value '{0}' for --use-mlir-diagnostics, "
                        "expected a boolean",
                        arg->getValue()));
    useMLIRDiagnostics = *value;
  }
  if (state.diagnosticFormat == DiagnosticFormat::JSON && useMLIRDiagnostics) {
    return state.reportError("--diagnostic-format=json is incompatible "
                             "with --use-mlir-diagnostics=true");
  }
  options.frontendOptions.useMLIRDiagnostics = useMLIRDiagnostics;
  options.frontendOptions.disablePrebuiltPackages =
      !options.enablePrebuiltPackages;

  if (!parseForwardedCLOptions(state, args))
    return EXIT_FAILURE;

  ErrorOr<ContextRef> ctxOr = createContext();
  if (ctxOr.isError())
    return state.reportError(
        llvm::formatv("failed to create context: {0}", ctxOr.getError()));

  TraceProfiler profiler(options.driverOptions.timeTrace,
                         options.driverOptions.timeTraceGranularity);
  mlir::Translation translation(
      [&](const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
          llvm::raw_ostream &output, MLIRContext *context) {
        return importMojo(state, sourceMgr, output, context, *ctxOr, options);
      },
      "Import 'mojo' from source", /*inputAlignment=*/std::nullopt);
  return runTranslation(state, translation, inputs, options);
}
