//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-run.h"

#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
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
static std::optional<int> parseArgs(const State &state,
                                    llvm::opt::InputArgList &args) {
  // First, parse all arguments, in order to find the index of the input
  // argument.
  RunOptTable options;
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
    if (allArgs.hasArg(options::OPT_help, options::OPT_help_text)) {
      return state.printHelp(
          /*plainText=*/allArgs.hasArg(options::OPT_help_text),
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
  if (args.hasArg(options::OPT_help, options::OPT_help_text)) {
    return state.printHelp(/*plainText=*/args.hasArg(options::OPT_help_text),
#include "Run/RunOptionsHelpText.inc"
    );
  }

  // Otherwise, within this subset of arguments that appear before the input,
  // unknown arguments are rejected.
  if (args.hasArg(options::OPT_UNKNOWN)) {
    int result = 1;
    for (llvm::opt::Arg *arg : args.filtered(options::OPT_UNKNOWN))
      result = state.reportError("unrecognized argument '" +
                                 arg->getSpelling() + "'\n");
    return result;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Mojo to MLIR compilation
//===----------------------------------------------------------------------===//

/// Given a list of arguments that includes an input file path, reads the Mojo
/// file at that path and translates it into an MLIR module. Returns an failure
/// exit code should an error occur, and nullopt otherwise.
static std::optional<int>
compileModule(const State &state, const llvm::opt::InputArgList &args,
              LLCL::Runtime &runtime, MLIRContext &context,
              CompilationOptions &options, OwningOpRef<ModuleOp> &moduleOp) {
  // We're done parsing arguments, and can move on to actually building the
  // input: start by opening the input file, or exiting with an error.
  auto bufferOrErr =
      openMojoInputFile(args.getLastArgValue(options::OPT_INPUT));
  if (failed(bufferOrErr))
    return state.reportError(bufferOrErr.getError());

  // Initialize the source manager with the input file buffer and all includes.
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());
  sourceManager.setIncludeDirs(args.getAllArgValues(options::OPT_I));

  // Initialize the MLIR context.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  index::IndexDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
  // Reset the context's diagnostic handler.
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);

  // We don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timing = timingManager.getRootScope();

  // Build the compilation options based on the provided arguments.
  options.targetTriple = llvm::Triple::normalize(args.getLastArgValue(
      options::OPT_targetTriple, llvm::sys::getDefaultTargetTriple()));
  options.targetCpu =
      args.getLastArgValue(options::OPT_targetCpu, llvm::sys::getHostCPUName());
  options.targetFeatures =
      args.getLastArgValue(options::OPT_targetFeatures, getHostCPUFeatures());
  if (args.hasArg(options::OPT_no_optimization))
    options.optimizationLevel = 0;
  options.linkDirs = args.getAllArgValues(options::OPT_L);

  // Parse the input Mojo file into an MLIR module.
  MojoParserConfig parseConfig(&context, runtime, options);
  parseConfig.validateDocStrings = args.hasArg(options::OPT_doc_validate);
  int maxNotes = 0;
  if (!args.getLastArgValue(options::OPT_max_notes).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;

  TimingScope mojoScope = timing.nest("Import Mojo");
  moduleOp = importMojoFile(sourceManager, parseConfig, mojoScope);
  if (!moduleOp)
    return state.reportError("could not parse the module");

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  context.loadDialect<KGENDialect>();
  ErrorOr<EnvAttr> envOrErr =
      EnvAttr::parseDefines(&context, args.getAllArgValues(options::OPT_D));
  if (failed(envOrErr))
    return state.reportError(
        llvm::formatv("an internal error occurred when initializing the Mojo "
                      "MLIR module: {0}",
                      envOrErr.getError()));
  moduleOp.get()->setAttr(EnvAttr::getEnvAttrName(), *envOrErr);

  return {};
}

//===----------------------------------------------------------------------===//
// Mojo program execution
//===----------------------------------------------------------------------===//

/// Either extract the target info from the given module or, if the info isn't
/// available, construct target info based on the given compilation options. If
/// target info could not be constructed, a diagnostic is emitted and a null
/// attribute is returned.
static TargetInfoAttr
getOrConstructTargetInfo(MLIRContext &context, ModuleOp moduleOp,
                         const CompilationOptions &options) {
  TargetInfoAttr target = getTargetInfo(moduleOp);
  if (target) {
    if (target.getTripleStr() != options.targetTriple ||
        target.getCpu() != options.targetCpu ||
        target.getFeatures() != options.targetFeatures) {
      mlir::emitWarning(moduleOp.getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
      target = nullptr;
    }
  }

  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr =
        getTargetInfoFor(&context, options.targetTriple, options.targetCpu,
                         options.targetFeatures);
    if (failed(targetOr)) {
      mlir::emitError(moduleOp.getLoc(), targetOr.getError());
      return {};
    }
    target = targetOr.takeValue();
  }

  return target;
}

/// Invoke the KGEN compiler runtime setter for argv to the underlying Mojo
/// program, to pass along the given `arguments`.
static ErrorOrSuccess setArgV(ExecutionEngine *engine,
                              ArrayRef<const char *> arguments) {
  ErrorOr<CompiledFunc> setterOrErr = engine->lookup("KGEN_CompilerRT_SetArgV");
  if (failed(setterOrErr))
    return Error(llvm::formatv("an internal error occurred when initializing "
                               "arguments to the underlying Mojo program: {0}",
                               setterOrErr.getError()));
  setterOrErr->invoke<void>(arguments.size(), arguments.data());
  return {};
}

namespace {
/// Whether a module exports a `main` function.
enum class ExportsMain {
  /// The module does not export a `main` function.
  NoMain,
  /// The module exports a `def main` function.
  IsDef,
  /// The module exports a `fn main` function.
  IsFn,
};
} // namespace

/// Returns whether the given module exports a `main` function. If it doesn't,
/// error diagnostics are emitted to point out why certain candidate functions
/// are not viable.
static ExportsMain moduleExportsMain(ModuleOp theModule,
                                     const SymbolTable &symtab) {
  MLIRContext *ctx = theModule.getContext();
  auto noneType = POP::ArrayType::get(0, IntegerType::get(ctx, 1));

  // Iterate over exported symbols named "main".
  for (ExportOp exportOp :
       llvm::make_filter_range(theModule.getOps<ExportOp>(), [](ExportOp op) {
         return op.getAlias() == "main";
       })) {
    // It needs to be a function.
    FuncOp funcOp =
        symtab.lookup<FuncOp>(exportOp.getExported().getRootReference());
    if (!funcOp)
      continue;
    // And that function needs to take zero arguments, and return a single
    // result.
    FunctionType fnType = funcOp.getFunctionType();
    if (fnType.getNumInputs() != 0 || fnType.getNumResults() != 1)
      continue;

    // If it returns `None`, it's the `fn main` we're looking for.
    if (fnType.getResult(0) == noneType)
      return ExportsMain::IsFn;

    // Otherwise, it could be a `def main()`, which returns a variant
    // composed of the error and none types.
    Type resultType = fnType.getResult(0);
    auto emitInvalidReturnTypeError = [&]() -> ExportsMain {
      mlir::emitError(funcOp.getLoc(),
                      "'main' function has invalid return type '")
          << resultType << "'; it must return 'None' or 'Error'";
      return ExportsMain::NoMain;
    };

    auto varType = dyn_cast<POP::VariantType>(resultType);
    if (!varType)
      return emitInvalidReturnTypeError();

    ArrayRef<TypedAttr> varElemTypes = varType.getTypes();
    if (varElemTypes.size() != 2)
      return emitInvalidReturnTypeError();

    // The error type is `!pop.struct<pointer<scalar<si8>>, index>`.
    Type errorType = POP::StructType::get(
        ArrayRef<Type>{POP::PointerType::get(POP::SIMDType::get(
                           1, DTypeConstantAttr::get(ctx, KGENDType::si8))),
                       IndexType::get(ctx)});
    if (varElemTypes[0] != ConcreteTypeConstantAttr::get(errorType))
      return emitInvalidReturnTypeError();
    if (varElemTypes[1] == ConcreteTypeConstantAttr::get(noneType))
      return ExportsMain::IsDef;
    return emitInvalidReturnTypeError();
  }

  return ExportsMain::NoMain;
}

/// Executes the given module's `main` function, or returns an error indicating
/// why it could not be executed.
static ErrorOrSuccess executeMain(ModuleOp moduleOp, const SymbolTable &symtab,
                                  ExecutionEngine *engine) {
  ExportsMain exportsMain = moduleExportsMain(moduleOp, symtab);
  if (exportsMain == ExportsMain::NoMain)
    return Error("could not find a 'main' function to execute");

  auto runFn = [exportsMain](void *fnPtr) -> ErrorOrSuccess {
    // `fn main` is simple to handle, with no arguments and void result.
    if (exportsMain == ExportsMain::IsFn) {
      using FnType = void (*)();
      ((FnType)fnPtr)();
      return M::success();
    }

    // The `variant<Error, None>` result is decomposed in the arguments. The
    // error type just contains a Mojo `StringRef`.
    struct MojoError {
      const char *data;
      ssize_t length;
    };
    MojoError err;
    uint8_t isNormalResult = false;
    // The last argument is the discrminant, set to 0 if the result is an
    // error variant.
    using ErrorFnType = void (*)(MojoError *, uint8_t *);
    ((ErrorFnType)fnPtr)(&err, &isNormalResult);
    if (!isNormalResult) {
      // Read out and report the error message.
      StringRef errStr(err.data, err.length);
      return Error("main function threw an error: " + errStr);
    }
    return M::success();
  };
  return engine->runProgram("exec", "main", runFn);
}

/// Given a module representing a Mojo program, and a set of `arguments` to pass
/// along to that program, initializes an execution engine and executes the
/// program. Returns a successful exit code if the program was executed
/// successfully, and an unsuccessful exit code otherwise.
static int executeModule(const State &state, LLCL::Runtime &runtime,
                         MLIRContext &context,
                         const CompilationOptions &options, ModuleOp moduleOp,
                         ArrayRef<const char *> arguments) {
  // Now, move on to lowering the MLIR module to LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Instantiate an execution engine for JIT compilation.
  auto machineOrErr = createTargetMachine(options,
                                          /*isJIT=*/true);
  if (failed(machineOrErr))
    return state.reportError(machineOrErr.getError());
  std::unique_ptr<llvm::TargetMachine> machine = std::move(*machineOrErr);

  auto engineOrErr = ExecutionEngine::createWithStandardLayers(
      {/*registerDebugPlugins=*/options.debugLevel !=
       CompilationOptions::DebugInfoLevel::kNoDebug},
      *machine);
  if (failed(engineOrErr))
    return state.reportError(engineOrErr.getError());
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOrErr);

  // Add the object compiler layer to the execution engine.
  mlir::PassManager passManager(&context);
  ErrorOr<ObjectCompiler> objectCompiler =
      ObjectCompiler::create(runtime, passManager, ".kgen_cache", options);
  if (failed(objectCompiler))
    return state.reportError(Twine("could not create object compiler: ") +
                             objectCompiler.getError());
  ObjectCompilerLayer &objectCompilerLayer =
      engine->addLayer<ObjectCompilerLayer>(std::move(*objectCompiler),
                                            engine->getLinkingLayer());

  // Add the KGEN compiler layer. To do so, first get the backend chains to pass
  // into the compile layer.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (failed(transformCacheBackend))
    return state.reportError(transformCacheBackend.getError());

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      runtime, (std::filesystem::path(".kgen_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (failed(regionCacheBackend))
    return state.reportError(transformCacheBackend.getError());

  // Next, find a target specification.
  TargetInfoAttr target = getOrConstructTargetInfo(context, moduleOp, options);
  if (!target)
    return EXIT_FAILURE;

  // Finally, instantiate the compiler layer, using the build info from the
  // current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(&context);
  KGENCompilerLayer &compilerLayer = engine->addLayer<KGENCompilerLayer>(
      passManager, runtime, target, build, options, objectCompilerLayer,
      std::move(*transformCacheBackend), std::move(*regionCacheBackend));

  // Add the module into the layer. This will actually compile it down to the
  // post-elaboration phase, because before that phase we don't have flat
  // symbols.
  if (ErrorOrSuccess err = compilerLayer.add("exec", moduleOp))
    return state.reportError(err.getError());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(moduleOp);
  ExportMap exports = getExportedSymbols(moduleOp);
  if (exports.empty())
    return state.reportError(
        "module does not `@export` any symbols; nothing to codegen");

  // Trigger compilation so we can pull out the archive.
  ErrorOr<CompiledFunc> funcOr = engine->lookup(exports.front().second.alias);
  if (failed(funcOr))
    return state.reportError(funcOr.getError());

  // Initialize the command line arguments to pass to the Mojo program.
  ErrorOrSuccess argv = setArgV(engine.get(), arguments);
  if (failed(argv))
    return state.reportError(argv.getError());

  // Finally, execute the 'main' function of the Mojo program.
  TimeTraceScope<> traceScope("execute-main");
  ErrorOrSuccess result = executeMain(moduleOp, symtab, engine.get());
  if (failed(result))
    return state.reportError(result.getError());

  return EXIT_SUCCESS;
}

/// Given the path to a Mojo source file, opens that file, compiles it, and
/// executes it. Returns an integer representing a successful exit code if
/// the source file could be compiled and if it executed without raising an
/// error, otherwise returns a failure code.
static int run(const State &state) {
  // Parse arguments.
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode = parseArgs(state, args))
    return *exitCode;

  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  // Lower the input file to an MLIR module.
  MLIRContext context;
  CompilationOptions options;
  OwningOpRef<ModuleOp> moduleOp;
  if (std::optional<int> exitCode =
          compileModule(state, args, runtime, context, options, moduleOp))
    return *exitCode;

  // Execute the Mojo program.
  return executeModule(
      state, runtime, context, options, *moduleOp,
      state.arguments.slice(args.getLastArg(options::OPT_INPUT)->getIndex()));
}

void M::registerRunSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("run", run);
}
