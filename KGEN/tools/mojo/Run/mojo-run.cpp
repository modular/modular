//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-run.h"
#include "../Common/Compilation.h"
#include "../Common/XlinkerResolution.h"

#include "Init/Init.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Compiler/Diags.h"
#include "Support/Config.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <optional>
#include <signal.h>

#ifdef KGEN_ENABLE_PASS_OPTIONS
#include "KGEN/ToolCommon/CLOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#endif // KGEN_ENABLE_PASS_OPTIONS

using namespace M;
using namespace KGEN;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Command line argument parsing
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Run/RunOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct RunOptTable : public llvm::opt::PrecomputedOptTable {
  RunOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};

} // namespace

/// Parses the command line arguments from the given `state` object.
/// Command line argument parsing is unique for this command:
///
/// - For all arguments passed in *before* the input argument ("Foo.mojo", for
///   example), we parse those "normally." That is, `--help` prints the command
///   help text, unrecognized options cause a failure, and recognized options
///   impact the behavior of the command.
/// - We ignore all arguments passed in *after* the input argument, and instead
///   just pass those along verbatim to the underlying Mojo program. This
///   includes options like `--help`, since the user may be invoking
///   `mojo MyHelpfulProgram.mojo --help` in order to print the program's help
///   text.
///
/// As a result, this function has 2 results:
/// 1. Its return value: either an integer exit code signaling that program
///    execution should exit immediately with that code, or nullopt, signifying
///    program execution should continue.
/// 2. `args`: the command line arguments preceding and including the input
///    argument ("Foo.mojo" from the example above).
static std::optional<int> parseArgs(State &state, llvm::opt::InputArgList &args,
                                    llvm::SourceMgr &sourceManager,
                                    CompilationOptions &compilationOptions,
                                    MLIRContext &ctx, TargetInfoAttr &target,
                                    RunOptTable &options) {
  // First, parse all arguments to find the input file location.
  // For `mojo run`, we need special handling: arguments BEFORE the input file
  // are parsed as options to mojo run, but arguments AFTER are passed to the
  // program itself.
  unsigned unused = 0;
  llvm::opt::InputArgList allArgs =
      options.ParseArgs(state.arguments, unused, unused);

  // Find the input file position.
  auto inputArgs = allArgs.filtered(options::OPT_INPUT);
  if (inputArgs.empty()) {
    // No input file - check if user wanted help.
    if (allArgs.hasArg(options::OPT_help)) {
      return state.printHelp(
#include "Run/RunOptionsHelpText.inc"
      );
    } else if (allArgs.hasArg(options::OPT_help_hidden)) {
      return state.printHelp(
#include "Run/RunOptionsHelpHiddenText.inc"
      );
    }
    // No input and no help request is an error, which will be caught below.
  } else {
    // We have an input file. Parse only the arguments up to and including it.
    llvm::opt::InputArgList argsBeforeInput = options.ParseArgs(
        state.arguments.slice(0, (*inputArgs.begin())->getIndex() + 1), unused,
        unused);

    // Check if help was requested BEFORE the input file.
    if (argsBeforeInput.hasArg(options::OPT_help)) {
      return state.printHelp(
#include "Run/RunOptionsHelpText.inc"
      );
    } else if (argsBeforeInput.hasArg(options::OPT_help_hidden)) {
      return state.printHelp(
#include "Run/RunOptionsHelpHiddenText.inc"
      );
    }
  }

  // Set up common option IDs.
  CommonOptionIDs optionIDs{
      .help = options::OPT_help,
      .helpHidden = options::OPT_help_hidden,
      .diagnosticFormat = options::OPT_diagnostic_format,
      .disableWarnings = options::OPT_disable_warnings,
      .warningsAsErrors = options::OPT_werror,
      .noWarningsAsErrors = options::OPT_wno_error,
      .ignoreIncompatiblePackageErrors =
          options::OPT_ignore_incompatible_package_errors,
      .unknown = options::OPT_UNKNOWN,
      .input = options::OPT_INPUT,
      .includeDirs = options::OPT_I,
      .optimizationLevel = options::OPT_optimization_level,
      .debugLevel = options::OPT_debug_level,
      .sanitize = options::OPT_sanitize,
      .sharedLibasan = options::OPT_shared_libasan,
      .externalLibasan = options::OPT_external_libasan,
      .bitcodeLibs = options::OPT_bitcode_libs,
      .debugInfoLanguage = options::OPT_debug_info_language,
      .numThreads = options::OPT_num_threads,
      .mojoSearchPaths = options::OPT_mojo_search_paths,
      .loopUnrollingWarnThreshold = options::OPT_loop_unrolling_warn_threshold,
      .elaborationErrorLimit = options::OPT_elaboration_error_limit,
      .elaborationErrorIncludePrelude =
          options::OPT_elaboration_error_include_prelude,
      .elaborationErrorVerbose = options::OPT_elaboration_error_verbose,
      .elaborationMaxDepth = options::OPT_elaboration_max_depth,
      .targetTriple = options::OPT_target_triple,
      .targetCpu = options::OPT_target_cpu,
      .targetFeatures = options::OPT_target_features,
      .march = options::OPT_march,
      .mcpu = options::OPT_mcpu,
      .mtune = options::OPT_mtune,
      .targetAccelerator = options::OPT_target_accelerator,
      .mcmodel = options::OPT_mcmodel,
      .largeDataThreshold = options::OPT_large_data_threshold,
      .relocationModel = options::OPT_relocation_model,
      .diagnoseMissingDocStrings = options::OPT_diagnose_missing_doc_strings,
      .maxNotes = options::OPT_max_notes,
      .defines = options::OPT_D,
      .stripFilePrefix = options::OPT_strip_file_prefix,
      .disableBuiltins = options::OPT_disable_builtins,
      .fixit = options::OPT_fixit,
      .exportFixit = options::OPT_export_fixit,
      .warnOnUnstableAPIs = options::OPT_warn_on_unstable_apis,
  };

  // Configure parsing for `mojo run` - only parse args up to the input file.
  CommonParseConfig config{
      .parseAllArguments = false,
      .requireSingleInput = true,
  };

  // Parse common arguments.
  ErrorOr<CommonParseResult> result = parseCommonMojoArguments(
      state, sourceManager, ctx, options, optionIDs, config);
  if (failed(result))
    return state.reportError(result.getError());

  if (result->exitCode)
    return *result->exitCode;

  // Extract results.
  args = std::move(result->args);
  compilationOptions = std::move(result->compilationOptions);
  target = std::move(result->target);
  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

/// Executes the given module's `main` function, or returns an error indicating
/// why it could not be executed.
static ErrorOrSuccess executeMain(ExecutionEngine &engine,
                                  MLRT::Runtime &runtime,
                                  ArrayRef<const char *> arguments) {
  auto runFn = [arguments](void *fnPtr) -> ErrorOrSuccess {
    using FnType = int (*)(int, const char *const *);

    if (int result = ((FnType)fnPtr)(arguments.size(), arguments.data()))
      return Error("execution exited with a non-zero result: " + Twine(result));
    return M::success();
  };
  llvm::CrashRecoveryContext crc;
  crc.Enable();
  ErrorOrSuccess result;
  if (!crc.RunSafely(
          [&]() { result = engine.runProgram("exec", "main", runFn); })) {
    // With JIT compilation, printed stack trace is not useful. Recommend user
    // to use build + run mode to get symbolicated stack trace.
    return Error("execution crashed\nTo get a symbolicated stack trace, "
                 "compile your program using `mojo build` with debug info "
                 "enabled (e.g., `-debug-level=line-tables`) and execute it "
                 "separately.");
  }
  return result;
}

/// Ensures that the context's profiler, if there is one, copies any outstanding
/// references into its own arena so that the profiler is fully self-contained.
static void internTimeTraceProfile(M::Context &maxContext) {
  std::optional<TimeTraceProfiler> &profilerOr =
      maxContext.get<MLRT::Runtime>()->getProfiler();
  if (profilerOr)
    profilerOr->intern();
}

/// Given a module representing a Mojo program, and a set of `arguments` to pass
/// along to that program, initializes an execution engine and executes the
/// program. Returns a successful exit code if the program was executed
/// successfully, and an unsuccessful exit code otherwise.
static int executeModule(const State &state, MLRT::Runtime &runtime,
                         MLIRContext &context,
                         const CompilationOptions &options,
                         OwningOpRef<ModuleOp> module, TargetInfoAttr target,
                         ArrayRef<const char *> arguments,
                         M::Context &maxContext,
                         ArrayRef<std::string> additionalLibraries) {
  // Compile the Mojo module to the end of the KGEN pipeline.
  KGENCompiler compiler(context, options);
  if (ErrorOrSuccess err = compiler.runKGENPipeline(*module, target))
    return state.reportError(err.getError());

  // Validate that `main` was defined in the module.
  SymbolTable symtab(*module);
  ExportMap exports = getExportedSymbols(*module);
  if (exports.find(StringAttr::get(&context, "main")) == exports.end())
    return state.reportError("module does not define a `main` function");

  // Create the object compiler and compile the module to an archive.
  auto objCompilerOr =
      ObjectCompiler::create(".mojo_cache", options, /*isJIT=*/true, context);
  if (failed(objCompilerOr))
    return state.reportError(objCompilerOr.getError());
  ObjectCompiler &objCompiler = **objCompilerOr;

  // Extract and set bitcode libraries from the module before compilation.
  if (auto arrayAttr =
          module->getOperation()->getAttrOfType<LLVMBitcodeLibArrayAttr>(
              LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName()))
    arrayAttr.externalize(objCompiler.getBitcodeLibs());

  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(std::move(module));
  if (failed(archiveOr))
    return state.reportError(archiveOr.getError());

  // Setup the execution engine.
  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  for (const std::string &libPath : additionalLibraries)
    eeOptions.libraryPaths.emplace_back(libPath);
  ErrorOr<std::unique_ptr<ExecutionEngine>> execEngineOr =
      initializeExecutionEngine(context, options, std::move(eeOptions),
                                /*isJIT=*/true);
  if (failed(execEngineOr))
    return state.reportError(execEngineOr.getError());
  ExecutionEngine &engine = **execEngineOr;

  // Load the compiled archive into the execution engine.
  if (ErrorOrSuccess err =
          engine.addIfAbsent<StaticArchiveLayer>("exec", archiveOr.takeValue()))
    return state.reportError(err.getError());

  ErrorOr<CompiledFunc> funcOr = engine.lookup("main");
  if (failed(funcOr))
    return state.reportError(funcOr.getError());

  // Finally, execute the 'main' function of the Mojo program.
  CompilerTimeTraceScope traceScope("execute-main");
  ErrorOrSuccess result = executeMain(engine, runtime, arguments);
  if (failed(result))
    return state.reportError(result.getError());

  // If the context contains a profiler, invoke profiler->intern()
  // before we kill the context.
  internTimeTraceProfile(maxContext);
  return EXIT_SUCCESS;
}

/// Given the path to a Mojo source file, opens that file, compiles it, and
/// executes it. Returns an integer representing a successful exit code if
/// the source file could be compiled and if it executed without raising an
/// error, otherwise returns a failure code.
static int run(const State &subcommandState) {
  // Parse arguments.
  State state = subcommandState;
  RunOptTable optionsTable;
  llvm::opt::InputArgList args;
  llvm::SourceMgr sourceManager;
  CompilationOptions options;
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  TargetInfoAttr target;
  if (std::optional<int> exitCode = parseArgs(
          state, args, sourceManager, options, mlirCtx, target, optionsTable))
    return *exitCode;

#ifdef KGEN_ENABLE_PASS_OPTIONS
  const char *cKGENOptions = "KGEN_OPTIONS";
  KGEN::KGENPassCLOptions::registerOptions();
  llvm::cl::ParseCommandLineOptions(0, &cKGENOptions, "", nullptr, nullptr,
                                    cKGENOptions);
#endif // KGEN_ENABLE_PASS_OPTIONS

  warnBuildingForDebugWithDebugBuiltCompiler(state, options.debugLevel);

  MLRT::RuntimeOptions runtimeOptions;
  configureRuntimeOptions(runtimeOptions, options);

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(runtimeOptions), "run");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(mlirCtx, ctx);

  // Lower the input file to an MLIR module.
  MLRT::Runtime &runtime = *ctx->get<MLRT::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &mlirCtx);
  ScopedMLIRWarningHandler warningHandler(&mlirCtx, options.disableWarnings,
                                          options.warningsAsErrors);

  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, &mlirCtx, runtime,
      options::OPT_diagnose_missing_doc_strings, options::OPT_max_notes,
      options::OPT_D, options::OPT_strip_file_prefix,
      options::OPT_disable_builtins, options::OPT_mojo_search_paths,
      options::OPT_fixit, options::OPT_export_fixit,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        return LIT::importMojoFile(ctx, sourceManager, parserConfig, ts);
      });
  if (failed(moduleOp))
    return state.reportError(moduleOp.getError());

  if (!moduleOp.get()->getOperation()) {
    // Only --experimental-fixit returns a null module (after applying fixes).
    // --experimental-export-fixit continues normal execution after writing
    // YAML.
    assert(args.hasArg(options::OPT_fixit));
    return EXIT_SUCCESS;
  }

  // Resolve any user-supplied `-Xlinker` flags into shared libraries the JIT
  // should load. `mojo run` has no native linker — the program is JIT'd in
  // process — so `-Xlinker -L<dir> -Xlinker -l<name>` is translated into a
  // set of dlopen-able paths rather than passed verbatim to a system linker.
  SmallVector<std::string> additionalLibraries = resolveXlinkerLibraries(
      state, args.getAllArgValues(options::OPT_Xlinker));

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Execute the Mojo program.
  int result = executeModule(
      state, runtime, mlirCtx, options, moduleOp.takeValue(), target,
      state.arguments.slice(args.getLastArg(options::OPT_INPUT)->getIndex()),
      *ctx, additionalLibraries);
  if (result != EXIT_SUCCESS)
    return result;

  // Check if any warnings were promoted to errors via -Werror.
  return warningHandler.wasErrorEmitted() ? EXIT_FAILURE : EXIT_SUCCESS;
}

void M::registerRunSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("run", run);
}
