//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-run.h"
#include "../../common/Telemetry.h"
#include "../Common/Compilation.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ExecutionEngine/JIT/ObjectCompilerLayer.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Config.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/Init/Init.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Telemetry/Telemetry.h"

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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

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
  RunOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};

} // namespace

/// Validate that the requested sanitizers are compatible with the running
/// process. We currently rely on pulling sanitizer symbols from the running
/// process, so we need to ensure that the requested sanitizers are compatible
/// with the running process.
static std::optional<int> validateSanitizers(const State &state,
                                             const Sanitizers &sanitizers) {
  if (!sanitizers)
    return std::nullopt;
  auto emitError = [&](StringRef sanitizer) {
    return state.reportError(
        "This build of `mojo` does not support `mojo run` with `--sanitize " +
        sanitizer +
        "`, consider generating a sanitized "
        "executable using `mojo build` instead.");
  };

  // Check that the running process has the requested sanitizers.
#if !LLVM_ADDRESS_SANITIZER_BUILD
  if (sanitizers.has(Sanitizers::kAddress))
    return emitError("address");
#endif
#if !LLVM_THREAD_SANITIZER_BUILD
  if (sanitizers.has(Sanitizers::kThread))
    return emitError("thread");
#endif
  return std::nullopt;
}

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
  // First, parse all arguments, in order to find the index of the input
  // argument.
  unsigned unused = 0;
  llvm::opt::InputArgList allArgs =
      options.ParseArgs(state.arguments, unused, unused);

  // LLVMOption treats all "positional arguments" (arguments that do not have a
  // "-" or "--" prefix) as `INPUT`. The very first of these is our Mojo source
  // file, and each remaining positional argument is an argument being passed to
  // the Mojo executable produced from that source file.
  auto inputArgs = allArgs.filtered(options::OPT_INPUT);
  if (inputArgs.empty()) {
    // If we have no input argument, then that's normally an error -- unless the
    // user is invoking `--help`.
    if (allArgs.hasArg(options::OPT_help)) {
      return state.printHelp(
#include "Run/RunOptionsHelpText.inc"
      );
    }
    return state.reportError("no input file provided");
  }

  // We now have the index of the Mojo source file argument, so we can parse
  // the arguments up to and including that argument "normally."
  args = options.ParseArgs(
      state.arguments.slice(0, (*inputArgs.begin())->getIndex() + 1), unused,
      unused);

  // If those arguments include `--help`, print help before checking any other
  // arguments.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Run/RunOptionsHelpText.inc"
    );
  }

  // Otherwise, within this subset of arguments that appear before the input,
  // unknown or invalid arguments are rejected.
  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Open the provided input file path, or exit with an error if it's not a
  // valid argument that can be opened.
  auto bufferOrErr =
      openMojoInputFile(args.getLastArgValue(options::OPT_INPUT));
  if (failed(bufferOrErr))
    return state.reportError(bufferOrErr.getError());

  // Initialize the source manager with the input file buffer, as well as the
  // appropriate diagnostic handler.
  sourceManager.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  sourceManager.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());

  if (ErrorOrSuccess err = parseCompilationOptions(
          state, args, compilationOptions, sourceManager, ctx, options::OPT_I,
          options::OPT_no_optimization, options::OPT_debug_level,
          options::OPT_sanitize, options::OPT_debug_info_language))
    return state.reportError(err.getError());
  if (ErrorOrSuccess err = parseTargetOptions(
          state, args, compilationOptions, sourceManager, ctx, target,
          options::OPT_target_triple, options::OPT_target_cpu,
          options::OPT_target_features, options::OPT_march, options::OPT_mcpu,
          options::OPT_mtune))
    return state.reportError(err.getError());

  // Validate the requested sanitizers.
  if (std::optional<int> exitCode =
          validateSanitizers(state, compilationOptions.sanitizers))
    return exitCode;
  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

/// Returns whether the given module exports a `main` function.
static bool moduleExportsMain(ModuleOp theModule, const SymbolTable &symtab) {
  for (auto funcOp : theModule.getOps<ExportInterface>())
    if (funcOp.getLinkageNameAttr() == "main")
      return true;
  return false;
}

/// Executes the given module's `main` function, or returns an error indicating
/// why it could not be executed.
static ErrorOrSuccess executeMain(ModuleOp moduleOp, const SymbolTable &symtab,
                                  ExecutionEngine *engine,
                                  LLCL::Runtime &runtime,
                                  ArrayRef<const char *> arguments) {
  if (!moduleExportsMain(moduleOp, symtab))
    return Error("could not find a 'main' function to execute");
  [[maybe_unused]] auto timeScope =
      runtime.context->get<M::Telemetry::TelemetryContext>()
          ->createUInt64Timer<std::chrono::milliseconds>(
              "mojo.run.time", M::Telemetry::Level::L2);

  auto runFn = [arguments](void *fnPtr) -> ErrorOrSuccess {
    using FnType = int (*)(int, const char *const *);

    if (int result = ((FnType)fnPtr)(arguments.size(), arguments.data()))
      return Error("execution exited with a non-zero result: " + Twine(result));
    return M::success();
  };
  return engine->runProgram("exec", "main", runFn);
}

/// Inserts the `M::Context` in the KGEN globals table so that Mojo code can
/// pass this `M::Context` to the `EngineContext`.
static ErrorOrSuccess insertMaxContextInGlobals(ExecutionEngine &engine,
                                                M::Context &maxContext) {
  // Get the insertion function from the KGENCompilerRT JIT Dylib.
  auto insertGlobalFnOr = engine.lookup("KGEN_CompilerRT_InsertGlobal");
  if (insertGlobalFnOr.isError())
    return insertGlobalFnOr.takeError();
  CompiledFunc &insertGlobalFn = *insertGlobalFnOr;

  // Call the function to insert the `M::Context`.
  insertGlobalFn.invoke<void, StringRef, void *>(StringRef("MaxContext"),
                                                 (void *)&maxContext);

  return M::success();
}

/// Writes the time trace profile to a file if the profiler is present.
static ErrorOrSuccess writeTimeTraceProfile(M::Context &maxContext) {
  // FIXME(#36219): Write all traces, since the runtime won't be flushed.
  // This should be removed when the runtime is properly flushed when the
  // context goes out of scope.
  std::optional<TimeTraceProfiler> &profilerOr =
      maxContext.get<LLCL::Runtime>()->getProfiler();
  if (profilerOr) {
    auto writeErr = profilerOr->write("-");
    if (writeErr.isError())
      return writeErr.takeError();
  }

  return M::success();
}

/// Given a module representing a Mojo program, and a set of `arguments` to pass
/// along to that program, initializes an execution engine and executes the
/// program. Returns a successful exit code if the program was executed
/// successfully, and an unsuccessful exit code otherwise.
static int executeModule(const State &state, LLCL::Runtime &runtime,
                         MLIRContext &context,
                         const CompilationOptions &options, ModuleOp moduleOp,
                         TargetInfoAttr target,
                         ArrayRef<const char *> arguments,
                         M::Context &maxContext) {
  KGENCompiler compiler(context, options);
  mlir::PassManager &pm = compiler.getPassManager();

  configurePassManager(pm);
  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  ErrorOr<std::unique_ptr<ExecutionEngine>> execEngineOr =
      initializeExecutionEngine(pm, options, std::move(eeOptions),
                                /*isJIT=*/true, target);
  if (failed(execEngineOr))
    return state.reportError(execEngineOr.getError());
  std::unique_ptr<ExecutionEngine> engine = std::move(*execEngineOr);

  // Insert the `M::Context` into the KGENCompilerRT globals.
  // The MAX engine uses this mechanism to share globals with Mojo code.
  if (auto errOr = insertMaxContextInGlobals(*engine, maxContext))
    return state.reportError(errOr.getError());

  auto &objectCompilerLayer = engine->getLayer<ObjectCompilerLayer>();

  // Compile the moduleOp down to the post-elaboration phase,
  // because before that phase we don't have flat symbols.
  if (ErrorOrSuccess err = compiler.runKGENPipeline(moduleOp, target))
    return state.reportError(err.getError());

  if (ErrorOrSuccess err = objectCompilerLayer.add("exec", moduleOp))
    return state.reportError(err.getError());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(moduleOp);
  ExportMap exports = getExportedSymbols(moduleOp);
  if (exports.empty())
    return state.reportError("module does not define a `main` function");

  // Trigger compilation so we can pull out the archive.
  // Start with `main` because mojo-run should always have `main`, and this
  // sets up ORC JIT first query to be pending on the root of the function call
  // stack so that materialization ordering is correct.
  if (exports.find(StringAttr::get(&context, Twine("main"))) == exports.end())
    return state.reportError("could not find a 'main' function to execute");
  ErrorOr<CompiledFunc> funcOr = engine->lookup("main");
  if (failed(funcOr))
    return state.reportError(funcOr.getError());

  // Finally, execute the 'main' function of the Mojo program.
  CompilerTimeTraceScope traceScope("execute-main");
  ErrorOrSuccess result =
      executeMain(moduleOp, symtab, engine.get(), runtime, arguments);
  if (failed(result))
    return state.reportError(result.getError());

  // Write out the time trace profile if the context contains a profiler.
  if (auto errOr = writeTimeTraceProfile(maxContext))
    return state.reportError(errOr.getError());

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

  // Create our context (including the runtime).
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(LLCL::RuntimeOptions()));
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);
  registerContext(mlirCtx, ctx);

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args,
      /*privateArgs=*/{options::OPT_D, options::OPT_I});

  // Lower the input file to an MLIR module.
  LLCL::Runtime &runtime = *ctx->get<LLCL::Runtime>();
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &mlirCtx);
  ErrorOr<OwningOpRef<ModuleOp>> moduleOp = invokeMojoParser(
      state, args, options, &mlirCtx, runtime,
      options::OPT_diagnose_missing_doc_strings,
      options::OPT_validate_doc_strings, options::OPT_max_notes, options::OPT_D,
      [&](LIT::ParserConfig &parserConfig, mlir::TimingScope &ts) {
        return LIT::importMojoFile(runtime, sourceManager, parserConfig, ts);
      });
  if (failed(moduleOp))
    return state.reportError(moduleOp.getError());

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Execute the Mojo program.
  return executeModule(
      state, runtime, mlirCtx, options, **moduleOp, target,
      state.arguments.slice(args.getLastArg(options::OPT_INPUT)->getIndex()),
      *ctx);
}

void M::registerRunSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("run", run);
}
