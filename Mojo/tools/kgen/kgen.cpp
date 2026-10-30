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
#include "Config/Version.h"
#include "Init/Init.h"
#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/Precompile/Precompile.h"
#include "Mojo/Support/CLOptionUtils.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/Support/ForceLinkMLIRC.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/ToolCommon/DriverOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Mojo/ToolCommon/OptionNormalizers.h"
#include "Mojo/ToolCommon/TargetOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/Diags.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/Driver/OptionMarshalling.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/Paths.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/Process.h"
#include "Target/TargetTraits.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace KGEN;
using namespace mlir;

#define DRIVER_OPTIONS_PATH "Mojo/tools/kgen/KGENOptions.inc"
#include "Mojo/tools/kgen/KGENOptionsHelpTexts.inc"
#include "Support/Driver/OptTable.inc"

namespace {
/// Parser listener used for the `lsp` command: restricts interest to the main
/// file's locations, mirroring how the language server parses an open document
/// (and matching kgen-translate's -lsp listener). Installing a listener also
/// exercises the parser's eager listener-driven resolution paths.
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

/// The EmitAs outputs kgen writes; their spellings are accepted directly,
/// and target-declared kinds canonicalize onto them.
static constexpr EmitAs kEmitAsKinds[] = {
    EmitAs::OBJECT, EmitAs::LLVM,     EmitAs::LLVM_BITCODE,
    EmitAs::ASM,    EmitAs::LLVM_OPT, EmitAs::LLVM_OPT_BITCODE};

/// kgen products with no EmitAs identity, named only here.
static constexpr llvm::StringRef kAsmVerboseKind = "asm-verbose";
static constexpr llvm::StringRef kHeaderKind = "header";
static constexpr llvm::StringRef kSharedLibKind = "shared-lib";
static constexpr llvm::StringRef kProductKinds[] = {
    kAsmVerboseKind, kHeaderKind, kSharedLibKind};

/// True if `cmd` is the emit command for `kind`.
static bool isEmitCmd(llvm::StringRef cmd, llvm::StringRef kind) {
  return cmd.consume_front("emit=") && cmd == kind;
}
static bool isEmitCmd(llvm::StringRef cmd, EmitAs kind) {
  return isEmitCmd(cmd, stringifyEmitAs(kind));
}

/// The options that kgen parsed from its command line.
struct CLOptions : public CommonOptions {
  /// Canonical command string: "elaborate[=<mode>]", "emit=<kind>",
  /// "execute", "lsp[=no-dump]", or "precompile".
  std::string cmd;
  /// The requested `-emit` kind, or empty when no emit command was given.
  std::string emissionKind;
  SmallVector<CommandLineFunc> funcs;
  /// The input files after the first, which `inputFilename` holds.
  std::vector<std::string> inputFiles;
  CompilationOptions compilationOptions;
  FrontendOptions frontendOptions;
  AsyncRT::CPUDeviceOptions cpuDeviceOptions;
  TargetArgs targetArgs;
  DriverOptions driverOptions;
  std::string dependencyFilename;
  bool emitTextualAsm = false;
  bool ignoreFailures = false;
  /// The driver state, which formats the errors and diagnostics.
  const State *state = nullptr;

  std::optional<CommandLineFunc> shouldExecuteFunc(StringRef func) const {
    auto found = llvm::find_if(
        funcs, [&](const CommandLineFunc &ek) { return ek.name == func; });
    if (found == funcs.end())
      return std::nullopt;
    return *found;
  }

  /// Add all the input files provided on the command line to the SourceMgr.
  /// This is how MLIR parses multiple files.
  ErrorOrSuccess addInputFilesToSourceMgr(llvm::SourceMgr &mgr);
  void addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr);
};

/// Parses the command that `arg` gives into `clOptions`.
ErrorOrSuccess parseCommand(const llvm::opt::Arg &arg, CLOptions &clOptions) {
  StringRef value = arg.getNumValues() ? arg.getValue() : "";
  switch (arg.getOption().getID()) {
  case options::OPT_elaborate:
    clOptions.cmd = "elaborate";
    return M::success();
  case options::OPT_elaborate_eq:
    if (value != "use-parametric-interpreter" &&
        value != "no-use-parametric-interpreter")
      return Error("unsupported 'elaborate' option value '" + value + "'");
    clOptions.cmd = ("elaborate=" + value).str();
    return M::success();
  case options::OPT_emit:
  case options::OPT_emit_eq:
    // Any kind is recorded; it is validated against kgen's emission kinds and
    // the target's traits once the targets are initialized.
    clOptions.emissionKind =
        value.empty() ? stringifyEmitAs(EmitAs::OBJECT).str() : value.str();
    clOptions.cmd = "emit=" + clOptions.emissionKind;
    return M::success();
  case options::OPT_execute:
    clOptions.cmd = "execute";
    return M::success();
  case options::OPT_lsp:
    clOptions.cmd = "lsp";
    return M::success();
  case options::OPT_lsp_eq:
    if (value != "no-dump")
      return Error("unsupported 'lsp' option value '" + value + "'");
    clOptions.cmd = "lsp=no-dump";
    return M::success();
  case options::OPT_precompile:
    clOptions.cmd = "precompile";
    return M::success();
  default:
    llvm_unreachable("not a command");
  }
}

void printVersion(llvm::raw_ostream &os) {
  ProjectVersion version = getMojoVersion();
  os << "KGEN compiler:\n  ";
  os << "Mojo version: " << version.major << '.' << version.minor << '.'
     << version.patch << version.label << "\n  ";
  os << "Git SHA: " << version.revision << "\n  ";
  os << "Build config: " << version.buildType << "\n\n";

  // Print the host target config.
  llvm::sys::printDefaultTargetAndDetectedCPU(os);
  // Print all registered targets.
  llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
}

/// Forwards the `-mllvm` values to `llvm::cl`, after registering the LLVM,
/// MLIR and KGEN options they may name.
bool parseForwardedCLOptions(const char *programName,
                             const llvm::opt::InputArgList &args) {
  M::registerCommandFlags();
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  KGEN::registerDefaultKGENPasses("kgen");
  // Passes contributed by whichever backend plugins this build links.
  KGEN::TargetTraitsRegistry::get().registerAllPasses();
  registerPassManagerCLOptions();
  KGEN::initializeDebugOptions();
  KGEN::KGENPassCLOptions::registerOptions();

  SmallVector<const char *> clArgs = {programName};
  for (const llvm::opt::Arg *arg : args.filtered(options::OPT_mllvm))
    clArgs.push_back(arg->getValue());
  return llvm::cl::ParseCommandLineOptions(clArgs.size(), clArgs.data(), "kgen",
                                           &llvm::errs());
}
} // namespace

ErrorOrSuccess CLOptions::addInputFilesToSourceMgr(llvm::SourceMgr &mgr) {
  if (inputFiles.empty())
    mgr.AddNewSourceBuffer(openInputFileOrExit(), llvm::SMLoc());

  for (StringRef in : inputFiles) {
    std::error_code ec;
    std::filesystem::path fullPath = std::filesystem::absolute(in.str(), ec);
    if (ec) {
      return Error(
          llvm::formatv("failed to resolve the absolute path for '{0}': {1}",
                        in.str(), ec.message()));
    }
    std::string errorMsg;
    auto result = mlir::openInputFile(fullPath.string(), &errorMsg);
    if (!result)
      return Error(errorMsg);

    mgr.AddNewSourceBuffer(std::move(result), llvm::SMLoc());
  }

  return M::success();
}

void CLOptions::addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr) {
  if (auto err = addInputFilesToSourceMgr(mgr))
    exit(reportError(err.getError()));
}

/// Emit the IR for `theModule` to a file.
static LogicalResult emitModuleIR(ModuleOp theModule, const CLOptions &opts) {
  CompilerTimeTraceScope traceScope("emit-module",
                                    theModule.getSymName().value_or(""));
  if (opts.emitTextualAsm) {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/false, ".mlir");
    if (!outFile)
      return failure();

    theModule.print(outFile->os());
    // `print` does not insert a newline, so add one here.
    outFile->os() << "\n";
    outFile->keep();
  } else {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/true, ".mlirbc");
    if (!outFile)
      return failure();

    if (failed(mlir::writeBytecodeToFile(theModule, outFile->os())))
      return failure();
    outFile->keep();
  }

  // Try to save the textual IR as an intermediate file.
  if (auto irFile = opts.getIntermediateFile(opts.outputFilename, ".mlir")) {
    theModule.print(irFile->os());
    irFile->keep();
  }

  return mlir::success();
}

/// Create a dependency file for the `-d` option.
///
/// This functionality is generally only for the benefit of the build system,
/// and informs it of the dependencies of the input files.
static LogicalResult createDependencyFile(const CLOptions &clOptions,
                                          ArrayRef<std::string> includedFiles) {
  // It only makes sense to output a dependency file that can map inputs to
  // outputs. If the output file already exists and is not a regular file --
  // like `"-"` for stdout, or a character file like `/dev/null` -- then fail.
  if (clOptions.outputFilename == "-" ||
      (llvm::sys::fs::exists(clOptions.outputFilename) &&
       !llvm::sys::fs::is_regular_file(clOptions.outputFilename))) {
    return failure(clOptions.reportError(
        "can only create dependency file for outputs written to files"));
  }

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(clOptions.dependencyFilename, &errorMessage);
  if (!outputFile)
    return failure(clOptions.reportError(errorMessage));

  // Resolve each of the dependencies and add them to the file.
  outputFile->os() << clOptions.outputFilename << ":";
  for (StringRef includeFile : includedFiles)
    outputFile->os() << ' ' << includeFile;

  outputFile->os() << "\n";
  outputFile->keep();
  return mlir::success();
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     CLOptions &clOptions) {
  DialectRegistry registry;
  TraceProfiler tracer(clOptions.driverOptions.timeTrace,
                       clOptions.driverOptions.timeTraceGranularity);

  if (KGEN::debugFlag) {
    ctx->disableMultithreading();
    clOptions.cpuDeviceOptions.withSingleThreaded();
  }

  // `-lsp` already forces MLIR pass-manager multithreading off below (there is
  // exactly one file, one clone, one check pipeline run: no parallel work to
  // exploit). Without this, the AsyncRT work queue still defaults to a full
  // thread pool, so every invocation pays thread-pool startup/teardown cost
  // for threads that never do anything -- a measured ~15-20% of wall time on
  // real kernel files, matching the real LSP server's `-mojo-test` mode, which
  // does the same.
  if (clOptions.cmd == "lsp" || clOptions.cmd == "lsp=no-dump")
    clOptions.cpuDeviceOptions.withSingleThreaded();

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);

  // Create our context, with a cpuDevice; this should not fail.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen", Init::Options().withCPUDeviceOptions(
                  clOptions.cpuDeviceOptions.withCPUAffinity(false)));
  if (ctxOr.isError())
    return failure();
  registerContext(registry, *ctxOr);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);

  CompilationOptions options = clOptions.compilationOptions;
  if (clOptions.saveTemps)
    options.saveTempsPrefix = clOptions.tempsDir;
  options.verboseOutput = isEmitCmd(clOptions.cmd, kAsmVerboseKind);
  options.cacheBaseExtra = "kgen";

  OwningOpRef<ModuleOp> theModule;
  auto inputFileName = llvm::StringRef(clOptions.inputFilename);

  // Initialize the timing manager.
  std::unique_ptr<mlir::TimingManager> timingManager;
  if (clOptions.driverOptions.timeTrace) {
    timingManager = std::make_unique<TimeProfilerTimingManager>();
  } else {
    auto defaultManager = std::make_unique<mlir::DefaultTimingManager>();
    applyDefaultTimingManagerCLOptions(*defaultManager);
    timingManager = std::move(defaultManager);
  }

  // Resolve the target from the target options. A target that the input
  // module carries is overwritten by this one.
  ctx->loadDialect<MDialect>();
  TargetInfoAttr target;
  if (ErrorOrSuccess err =
          parseTargetOptions(clOptions.targetArgs, options, *ctx, target))
    return failure(clOptions.reportError(err.getError()));

  TimingScope timing = timingManager->getRootScope();

  PassManagerConfigOptions pmOptions;
  pmOptions.applyPassManagerCLOptions = true;
  pmOptions.enableTiming = false;
  pmOptions.timingScope = &timing;

  if (clOptions.cmd == "elaborate=use-parametric-interpreter")
    options.useParametricInterpreter = true;
  else if (clOptions.cmd == "elaborate=no-use-parametric-interpreter")
    options.useParametricInterpreter = false;

  KGENCompiler compiler(*ctx, options, std::move(pmOptions));

  // The `lsp`/`lsp=no-dump` commands reproduce how the language server
  // processes an open document (MojoDocument::checkModuleSemantics), all
  // in-process: the lazy, error-tolerant parse (parseFileForLSP) followed by
  // the check pipeline (runCheckLITPipeline) on the cloned module.
  // Diagnostics are reported on stderr and (for plain `-lsp`) the resulting
  // checked module IR is printed to stdout; it then returns before the
  // normal import/elaborate/emit path (no elaboration or lowering happens,
  // matching the server, which stops at semantic checking). Note: docstring
  // code blocks are not checked here (that is the server's processDocStrings
  // step).
  //
  // Unlike the server (a long-running process with no exit code, for which
  // diagnostics are the only signal), this CLI command reports any
  // error-severity diagnostic as a non-zero exit -- useful for scripts/tests
  // that want a pass/fail signal without scraping stderr.
  if (clOptions.cmd == "lsp" || clOptions.cmd == "lsp=no-dump") {
    if (!inputFileName.ends_with(".mojo"))
      return failure(
          clOptions.reportError("lsp command requires a .mojo file"));

    ctx->disableMultithreading();
    // Route both parse and check diagnostics through one handler so they print
    // to stderr with source locations, mirroring checkModuleSemantics.
    mlir::SourceMgrDiagnosticHandler diagHandler(mgr, ctx);

    // Track whether any error-severity diagnostic was emitted, without
    // changing how diagHandler prints it. Handlers run most-recently-
    // registered-first and stop at the first success(); returning failure()
    // here always defers to diagHandler, which is registered first.
    bool sawError = false;
    mlir::ScopedDiagnosticHandler errorTracker(
        ctx, [&](mlir::Diagnostic &diag) {
          if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            sawError = true;
          return failure();
        });

    FrontendOptions frontendOptions = clOptions.frontendOptions;
    frontendOptions.useMLIRDiagnostics = true;
    MainFileParserListener lspListener(mgr);
    frontendOptions.parserListener = &lspListener;

    MojoParserContext parserContext(mgr, ctx, options, frontendOptions);
    LIT::ASTDecl *decl = parserContext.parseFileForLSP(mgr.getMainFileID());
    if (!decl)
      return failure(clOptions.reportError("could not parse the module"));

    // Skip checking on a failed parse, like the actual server would do.
    if (!sawError) {
      // Run the real check pipeline on the same DCE'd per-decl clone the server
      // checks. Pipeline failure is non-fatal (the server only debug-logs it).
      OwningOpRef<ModuleOp> clone = LIT::cloneDeclModuleForCompilation(*decl);
      (void)compiler.runCheckLITPipeline(*clone);

      if (clOptions.cmd != "lsp=no-dump") {
        clone->print(llvm::outs());
        llvm::outs() << "\n";
      }
    }

    parserContext.ensureSignaturesResolved();

    return failure(sawError);
  }

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  if (inputFileName.ends_with(".mojo")) {
    TimingScope litScope = timing.nest("Import Mojo source");
    theModule = LIT::importMojoFile(*ctxOr, mgr, ctx, options,
                                    clOptions.frontendOptions, litScope,
                                    &includedFiles);
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // Tag the module with the environment parsed from the defines.
  ctx->loadDialect<KGENDialect>();

  // Populate the module with the user-provided -D options.
  ErrorOr<EnvAttr> env =
      options.parseDefinesWithDefaults(ctx, clOptions.driverOptions.defines);
  if (env.isError())
    return failure(clOptions.reportError(env.takeError().get()));
  theModule.get()->setAttr(EnvAttr::getEnvAttrName(), env.takeValue());

  // Extend the module with the Module env-attrs.
  extendWithModularEnvAttr(theModule.get(),
                           (*ctxOr)->get<CompilationContext>());

  // If we are generating a dependency file, do so now.
  if (!clOptions.dependencyFilename.empty()) {
    if (failed(createDependencyFile(clOptions, includedFiles)))
      return failure(
          clOptions.reportError("failed to create a dependency file"));
  }

  if (TargetInfoAttr moduleTarget = getTargetInfo(*theModule)) {
    if (!clOptions.targetArgs.march.empty()) {
      mlir::emitWarning(theModule->getLoc(),
                        "overriding module target specification with -march");
    } else if (moduleTarget.getTripleStr() != target.getTripleStr() ||
               moduleTarget.getArch() != target.getArch() ||
               moduleTarget.getFeatures() != target.getFeatures()) {
      mlir::emitWarning(theModule->getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
    }
  }
  options.targetTriple = target.getTripleStr();
  options.targetCpu = target.getArch();
  options.targetFeatures = target.getFeatures();
  options.targetAbi = target.getAbi();

  auto compilerOr =
      ObjectCompiler::create(kMojoCacheBaseDirName, options,
                             clOptions.cmd == "execute", *ctx, pmOptions);
  if (failed(compilerOr))
    return failure(clOptions.reportError(compilerOr.getError()));
  ObjectCompiler &objCompiler = **compilerOr;

  // Compiles the module through KGEN compiler pipeline.
  // We don't need to try to look anything up.
  if (ErrorOrSuccess err = compiler.runKGENPipeline(*theModule, target))
    return failure(clOptions.reportError(err.getError()));

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == "elaborate" ||
      clOptions.cmd == "elaborate=use-parametric-interpreter" ||
      clOptions.cmd == "elaborate=no-use-parametric-interpreter")
    return emitModuleIR(*theModule, clOptions);

  // Construct the symbol table and the export map.
  SymbolTable symtab(*theModule);
  ExportMap exportedSymbols = getExportedSymbols(*theModule);

  // Handle LLVM output.
  if (isEmitCmd(clOptions.cmd, EmitAs::LLVM) ||
      isEmitCmd(clOptions.cmd, EmitAs::LLVM_BITCODE)) {
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objCompiler.lowerAllFuncsToLLVM(llvmCtx, *theModule);

    if (llvmModuleOr)
      return failure(clOptions.reportError(
          Twine("could not lower funcs to LLVM, ") + llvmModuleOr.getError()));

    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    if (isEmitCmd(clOptions.cmd, EmitAs::LLVM_BITCODE)) {
      if (ErrorOrSuccess err =
              objCompiler.emitBitcode(*llvmModule, outFile->os()))
        return failure(clOptions.reportError(err.takeError().get()));
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();
    return mlir::success();
  }

  if (isEmitCmd(clOptions.cmd, EmitAs::LLVM_OPT) ||
      isEmitCmd(clOptions.cmd, EmitAs::LLVM_OPT_BITCODE)) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    LLVMModuleAndContext llvmModule;
    if (ErrorOrSuccess err = objCompiler.lowerAllFuncsToLLVMAndOptimize(
            *theModule, llvmModule)) {
      return failure(clOptions.reportError("failed to generate LLVMIR: " +
                                           Twine(err.getError())));
    }
    if (isEmitCmd(clOptions.cmd, EmitAs::LLVM_OPT_BITCODE)) {
      if (ErrorOrSuccess err =
              objCompiler.emitBitcode(*llvmModule, outFile->os()))
        return failure(clOptions.reportError(err.takeError().get()));
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (isEmitCmd(clOptions.cmd, EmitAs::ASM) ||
      isEmitCmd(clOptions.cmd, kAsmVerboseKind)) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return failure(clOptions.reportError("could not open .s output file"));

    ErrorOrSuccess standaloneOr =
        objCompiler.emitAssembly(std::move(theModule), outFile->os());
    if (failed(standaloneOr))
      return failure(
          clOptions.reportError("could not produce standalone asm: " +
                                Twine(standaloneOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // Handle header emission, we don't need to generate an archive for this.
  if (isEmitCmd(clOptions.cmd, kHeaderKind)) {
    LogicalResult result = failure();
    auto writeFn = [&](raw_ostream &os) {
      result =
          objCompiler.emitCXXHeader(*theModule, clOptions.outputFilename, os);
    };
    if (clOptions.outputFilename == "-") {
      auto writeContents = [&](raw_ostream &os) {
        writeFn(os);
        os.flush();
        return llvm::Error::success();
      };
      if (llvm::Error err =
              llvm::writeToOutput(clOptions.outputFilename, writeContents)) {
        return failure(
            clOptions.reportError(toModularError(std::move(err)).get()));
      }

      // Safely process creating the header, taking into account that we may
      // have different processes trying to produce this header in parallel.
    } else if (ErrorOr<std::filesystem::path> err =
                   writeFileUnderLock(clOptions.outputFilename, writeFn);
               err.isError()) {
      return failure(clOptions.reportError(err.getError()));
    }
    return mlir::success();
  }

  // If there are no exported symbols, there's nothing to codegen. Report this
  // as an error.
  if (exportedSymbols.empty()) {
    return failure(
        clOptions.reportError("module does not `@export` any symbols or define "
                              "a `main` function; nothing to codegen"));
  }

  // If we need to execute, grab the function metadata before the module is
  // consumed.
  struct FunctionExecution {
    StringAttr name;
    Location loc;
    FunctionType type;
    CommandLineFunc clFunc;
  };
  SmallVector<FunctionExecution> funcExecs;
  StringSet<> foundFuncs;
  if (clOptions.cmd == "execute") {
    for (auto fn : theModule->getOps<FuncOp>()) {
      StringAttr name = fn.getSymNameAttr();
      // See if we were asked to execute this function.
      if (std::optional<CommandLineFunc> clFunc =
              clOptions.shouldExecuteFunc(name)) {
        funcExecs.push_back(FunctionExecution{name, fn.getLoc(),
                                              fn.getFunctionType(), *clFunc});
        foundFuncs.insert(name);
        if (auto err = clFunc->verifyFuncSignature(funcExecs.back().type)) {
          mlir::emitError(fn.getLoc(), err.getError());
          if (!clOptions.ignoreFailures)
            return failure();
        }
      }
    }
    // If we didn't find a function the user asked to execute, emit an error.
    for (const auto &fn : clOptions.funcs) {
      if (!foundFuncs.count(fn.name)) {
        return mlir::emitError(theModule->getLoc(),
                               "could not find func '@" + fn.name + "'");
      }
    }
  }

  if (isEmitCmd(clOptions.cmd, kSharedLibKind)) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/true, ".so");
    if (!outFile)
      return failure(clOptions.reportError("could not open .so output file"));

    ErrorOrSuccess sharedObjOr =
        objCompiler.emitSharedObject(std::move(theModule), outFile->os());
    if (failed(sharedObjOr))
      return failure(clOptions.reportError(
          "could not produce standalone shared object binary: " +
          Twine(sharedObjOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // -emit and -execute both require compiled objects.
  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(std::move(theModule));
  if (failed(archiveOr)) {
    return failure(clOptions.reportError("failed to emit archive: " +
                                         Twine(archiveOr.getError())));
  }
  BufferRef archive = archiveOr.takeValue();

  // If we're emitting the archive, do it.
  if (isEmitCmd(clOptions.cmd, EmitAs::OBJECT)) {
    // Look up the first item in the exported symbols to trigger archive
    // generation.
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".o");
    if (!outFile)
      return failure(clOptions.reportError("could not open .o output file"));

    outFile->os() << archive->getBuffer();
    outFile->keep();
    return mlir::success();
  }

  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  // Detect cross-compilation by checking whether the target CPU is the same as
  // the host CPU.
  eeOptions.crossCompiling = options.targetCpu != llvm::sys::getHostCPUName();

  auto engineOr = initializeExecutionEngine(*ctx, options, std::move(eeOptions),
                                            /*isJIT=*/true, pmOptions);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  ExecutionEngine &engine = **engineOr;

  // Helper to execute a func.
  auto execFunc = [&](const FunctionExecution &func, StringAttr name,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    CompilerTimeTraceScope traceScope("execute-function", name);
    // Trigger compilation so we can pull out the archive.
    ErrorOr<CompiledFunc> funcOr = engine.lookup(name);
    if (failed(funcOr))
      return failure(clOptions.reportError(funcOr.getError()));

    if (auto err = clFunc.executeAndPrint(*funcOr)) {
      mlir::emitError(func.loc, err.getError());
      return failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  // Pass the compiled archive to the execution engine.
  if (ErrorOrSuccess err =
          engine.addIfAbsent<StaticArchiveLayer>("exec", std::move(archive)))
    return failure(clOptions.reportError(err.getError()));

  // Loop over the functions, executing as necessary.
  for (const FunctionExecution &func : funcExecs) {
    if (failed(execFunc(func, func.name, func.clFunc))) {
      return failure(
          clOptions.reportError("failed to execute " + func.name.strref()));
    }
  }

  return mlir::success();
}

/// Precompiles the package directory given as the input, as `mojo precompile`
/// does. The package parses under its directory's name, and takes the name of
/// the `.mojoc` file it is written to.
static LogicalResult runPrecompile(CLOptions &clOptions) {
  if (!clOptions.inputFiles.empty())
    return failure(clOptions.reportError(
        "precompile expects a single package directory as input"));
  StringRef input = clOptions.inputFilename;
  if (!Filesystem::isMojoSourcePackagePath(input.str()))
    return failure(clOptions.reportError(
        "'" + input + "' does not correspond to a Mojo package"));
  std::error_code ec;
  std::filesystem::path packagePath =
      std::filesystem::weakly_canonical(input.str(), ec);
  if (ec)
    return failure(clOptions.reportError("cannot canonicalize '" + input +
                                         "': " + ec.message()));
  std::string sourceName = packagePath.filename().string();
  std::filesystem::path outputPath(clOptions.outputFilename);
  std::string outputName = outputPath.extension() == ".mojoc"
                               ? outputPath.stem().string()
                               : sourceName;

  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen", Init::Options().withCPUDeviceOptions(
                  clOptions.cpuDeviceOptions.withCPUAffinity(false)));
  if (ctxOr.isError())
    return failure(clOptions.reportError(ctxOr.getError()));
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerContext(registry, *ctxOr);
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);

  llvm::SourceMgr sourceMgr;
  configureSourceMgr(*clOptions.state, clOptions.driverOptions, sourceMgr);
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);
  ScopedMLIRWarningHandler warningHandler(
      &context, clOptions.frontendOptions.disableWarnings,
      clOptions.frontendOptions.warningsAsErrors);

  CompilationOptions options = clOptions.compilationOptions;
  context.loadDialect<MDialect>();
  TargetInfoAttr target;
  if (ErrorOrSuccess err =
          parseTargetOptions(clOptions.targetArgs, options, context, target))
    return failure(clOptions.reportError(err.getError()));
  // Precompiled packages are built to be agnostic of how their importers
  // compile them, so match `mojo precompile`'s conservative settings.
  options.debugLevel = CompilationOptions::kFullDebugInfo;
  options.optimizationLevel = 0;

  TimingScope ts;
  auto [module, packageOp] = LIT::importMojoPackage(
      *ctxOr, packagePath.string(), sourceName, sourceMgr, &context, options,
      clOptions.frontendOptions, ts);
  if (!module)
    return failure(clOptions.reportError("could not parse the package"));

  ErrorOr<OwningOpRef<ModuleOp>> built = buildPrecompiledPackage(
      *module, packageOp, sourceName, outputName, options.bitcodeLibs);
  if (failed(built))
    return failure(clOptions.reportError(built.getError()));

  KGENCompiler compiler(context, options);
  if (failed(compiler.runCheckLITPipeline(*module)))
    return failure(clOptions.reportError("errors occurred during compilation"));

  bool textual = clOptions.emitTextualAsm;
  auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/!textual);
  if (textual) {
    (*built)->print(outFile->os());
    outFile->os() << "\n";
  } else if (failed(writePrecompiledFile((*built)->getOperation(),
                                         outFile->os()))) {
    return failure(clOptions.reportError("failed to write the package"));
  }
  outFile->keep();
  return failure(warningHandler.wasErrorEmitted());
}

int main(int argc, char **argv) {
  KGEN::forceLinkMLIRC();

  CLOptions clOptions;
  CLOptionsBase clOptionsBase(argc, argv, clOptions);
  State state("kgen", llvm::ArrayRef<const char *>(argv + 1, argv + argc));
  clOptions.state = &state;
  options::Table optTable;

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  auto printHelpWithEmissionKinds = [&](std::string_view helpText) {
    state.printHelp(helpText);
    // The emission-kind table is target-dependent, so it is rendered from the
    // registered TargetTraits rather than baked into the option declaration.
    llvm::outs() << "\nEMISSION KINDS (--emit=<kind>):\n";
    KGEN::printSupportedEmissionKinds(llvm::outs());
    llvm::outs() << "  Additional kgen-only kinds, accepted for every target:\n"
                 << "    " << kAsmVerboseKind
                 << " - verbose assembly with preserved comments\n"
                 << "    " << kHeaderKind
                 << " - a C header with declarations of exported functions\n"
                 << "    " << stringifyEmitAs(EmitAs::LLVM_OPT)
                 << " - optimized LLVM IR\n"
                 << "    " << stringifyEmitAs(EmitAs::LLVM_OPT_BITCODE)
                 << " - bitcode of optimized LLVM IR\n";
    return EXIT_SUCCESS;
  };
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode = state.parseArguments(
          optTable, options::helpTexts(), args,
          /*stopAtFirstInput=*/false, printHelpWithEmissionKinds))
    return *exitCode;
  if (args.hasArg(options::OPT_version)) {
    printVersion(llvm::outs());
    return EXIT_SUCCESS;
  }

  MarshallingErrors errors;
  {
    DriverOptions &driverOptions = clOptions.driverOptions;
    FrontendOptions &frontendOptions = clOptions.frontendOptions;
    CompilationOptions &compilationOptions = clOptions.compilationOptions;
    TargetArgs &targetArgs = clOptions.targetArgs;
    AsyncRT::CPUDeviceOptions &cpuDeviceOptions = clOptions.cpuDeviceOptions;
#define MOJO_OPTION_WITH_MARSHALLING(...)                                      \
  PARSE_OPTION_WITH_MARSHALLING(args, errors, __VA_ARGS__)
#include "Mojo/tools/kgen/KGENOptions.inc"
#undef MOJO_OPTION_WITH_MARSHALLING
  }
  if (int result = state.rejectInvalidArguments(args, errors))
    return result;

  SmallVector<llvm::opt::Arg *> commands = llvm::to_vector(args.filtered(
      options::OPT_elaborate, options::OPT_elaborate_eq, options::OPT_emit,
      options::OPT_emit_eq, options::OPT_execute, options::OPT_lsp,
      options::OPT_lsp_eq, options::OPT_precompile));
  if (commands.size() != 1)
    return state.reportError("exactly one of --elaborate, --emit, --execute, "
                             "--lsp and --precompile must be given");
  if (ErrorOrSuccess err = parseCommand(*commands.front(), clOptions))
    return state.reportError(err.getError());

  setWarningPolicy(state, clOptions.frontendOptions);
  if (ErrorOrSuccess err =
          validateCompilationOptions(clOptions.compilationOptions))
    return state.reportError(err.getError());
  if (size_t numThreads = clOptions.compilationOptions.numThreads) {
    size_t &maxThreads = clOptions.cpuDeviceOptions.maxThreads;
    clOptions.cpuDeviceOptions.numThreads = 0;
    maxThreads = maxThreads ? std::min(maxThreads, numThreads) : numThreads;
  }

  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (!inputs.empty()) {
    clOptions.inputFilename = inputs.front();
    clOptions.inputFiles.assign(inputs.begin() + 1, inputs.end());
  }
  for (StringRef func : args.getAllArgValues(options::OPT_func))
    clOptions.funcs.push_back(CommandLineFunc::parse(func));

  if (!parseForwardedCLOptions(argv[0], args))
    return EXIT_FAILURE;

  // The requested emission kind must be one kgen implements; traits-declared
  // kinds must also be accepted by the compilation target.
  if (StringRef kind = clOptions.emissionKind; !kind.empty()) {
    std::optional<EmitAs> emitAs = symbolizeEmitAs(kind);
    // The spelling must name an EmitAs kgen actually writes.
    if (emitAs && !llvm::is_contained(kEmitAsKinds, *emitAs))
      emitAs = std::nullopt;
    bool isProduct = llvm::is_contained(kProductKinds, kind);
    if (!emitAs && !isProduct && !KGEN::isKnownEmissionKind(kind)) {
      llvm::errs() << "kgen: error: unsupported 'emit' option value '" << kind
                   << "'; see the EMISSION KINDS section of --help for the "
                      "accepted kinds\n";
      return EXIT_FAILURE;
    }
    if (KGEN::isKnownEmissionKind(kind)) {
      StringRef triple = clOptions.targetArgs.triple.empty()
                             ? clOptions.compilationOptions.targetTriple
                             : clOptions.targetArgs.triple;
      ErrorOr<const TargetTraits *> traitsOr =
          TargetTraitsRegistry::get().lookup(llvm::Triple(triple));
      ErrorOrSuccess err = traitsOr.isError()
                               ? ErrorOrSuccess(Error(traitsOr.getError()))
                               : (*traitsOr)->validateEmissionKind(kind);
      if (err) {
        llvm::errs() << "kgen: error: " << err.getError() << "\n";
        return EXIT_FAILURE;
      }
      if (!emitAs && !isProduct) {
        // A target-declared kind canonicalizes onto the EmitAs its traits
        // map it to; link products need a linker kgen does not have.
        ErrorOr<EmitAs> targetEmitAsOr = (*traitsOr)->emitAsForKind(kind);
        if (targetEmitAsOr.isError()) {
          llvm::errs() << "kgen: error: " << targetEmitAsOr.getError() << "\n";
          return EXIT_FAILURE;
        }
        if (*targetEmitAsOr == EmitAs::OBJECT ||
            !llvm::is_contained(kEmitAsKinds, *targetEmitAsOr)) {
          llvm::errs() << "kgen: error: emission kind '" << kind
                       << "' is not supported by kgen\n";
          return EXIT_FAILURE;
        }
        clOptions.cmd = ("emit=" + stringifyEmitAs(*targetEmitAsOr)).str();
      }
    }
  }

  // The input is a package directory, which the input setup below cannot open.
  if (clOptions.cmd == "precompile")
    return failed(runPrecompile(clOptions));

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  configureSourceMgr(state, clOptions.driverOptions, sourceManager);
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        ctx->printOpOnDiagnostic(true);
        ScopedMLIRWarningHandler warningHandler(
            ctx, clOptions.frontendOptions.disableWarnings,
            clOptions.frontendOptions.warningsAsErrors);
        if (failed(runToolPipeline(ctx, sourceManager, clOptions)))
          return failure();
        return failure(warningHandler.wasErrorEmitted());
      }));
}
