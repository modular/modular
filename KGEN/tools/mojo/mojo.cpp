//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Config/Version.h"
#include "KGEN/CLOptions.h"
#include "KGEN/EmitFuncHeader.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
/// What to do with a given Mojo file.
enum class MojoCommand {
  kEmit,
  kEmitHeader,
  kExecute,
};

/// Top-level options for the `mojo` executable.
class CLOptions : public KGENCommonOptions, public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<MojoCommand> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(MojoCommand::kEmit, "emit", "Emit funcs as object files."),
          clEnumValN(
              MojoCommand::kEmitHeader, "emit-header",
              "Emit a C header file with declarations of exported functions."),
          clEnumValN(MojoCommand::kExecute, "execute",
                     "Execute the main function.")),
      cl::init(MojoCommand::kExecute)};

  cl::opt<bool> validateDocStrings{
      "doc-validate", cl::desc("Validate doc strings in the input Mojo file."),
      cl::init(false)};
};
} // namespace

/// Returns true if the given module exports a main() function, false otherwise.
static bool moduleExportsMain(ModuleOp theModule, SymbolTable &symtab,
                              bool &isDef) {
  MLIRContext *ctx = theModule.getContext();
  auto noneType = POP::ArrayType::get(0, IntegerType::get(ctx, 1));
  for (auto exportOp : theModule.getOps<ExportOp>()) {
    // Is there an exported "main"?
    if (exportOp.getAlias() != "main")
      continue;
    FuncOp func =
        symtab.lookup<FuncOp>(exportOp.getExported().getRootReference());
    if (!func)
      continue;
    FunctionType funcType = func.getFunctionType();
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1)
      continue;
    if (funcType.getResult(0) == noneType) {
      isDef = false;
      return true;
    }

    // Else, it it could be a `def main()` which returns an optional void.
    auto variantTy = dyn_cast<POP::VariantType>(funcType.getResult(0));
    if (!variantTy)
      return false;
    auto variantElementTys = variantTy.getTypes();
    if (variantElementTys.size() != 2)
      return false;
    // The Error type is "!pop.struct<pointer<scalar<si8>>, index>"
    Type errorType = POP::StructType::get(
        ArrayRef<Type>{POP::PointerType::get(POP::SIMDType::get(
                           1, DTypeConstantAttr::get(ctx, KGENDType::si8))),
                       IndexType::get(ctx)});
    if (variantElementTys[0] != ConcreteTypeConstantAttr::get(errorType))
      return false;
    if (variantElementTys[1] == ConcreteTypeConstantAttr::get(noneType)) {
      isDef = true;
      return true;
    }
    return false;
  }
  return false;
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static int runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                           CLOptions &clOptions) {
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  CompilationOptions compilationOptions = clOptions.getCompilationOptions();
  OwningOpRef<ModuleOp> theModule;
  llvm::StringRef inputFileName(clOptions.inputFilename.getValue());

  // Set up the runtime.
  std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();

  // Initialize the timing manager.
  std::unique_ptr<mlir::TimingManager> timingManager;
  if (clOptions.timeTrace) {
    timingManager = std::make_unique<TimeProfilerTimingManager>();
  } else {
    auto defaultManager = std::make_unique<mlir::DefaultTimingManager>();
    applyDefaultTimingManagerCLOptions(*defaultManager);
    timingManager = std::move(defaultManager);
  }
  TimingScope timing = timingManager->getRootScope();

  mlir::PassManager pm(ctx);
  if (failed(applyPassManagerCLOptions(pm)))
    return EXIT_FAILURE;
  pm.enableTiming(timing);
  if (clOptions.enableMLIRCrashReproducer.getValue()) {
    ctx->disableMultithreading();
    pm.enableCrashReproducerGeneration(clOptions.inputFilename.getValue() +
                                           ".repro.mlir",
                                       /*genLocalReproducer=*/true);
  }

  if (!inputFileName.ends_with(".mojo") && !inputFileName.ends_with(".🔥"))
    return clOptions.reportError("expected a Mojo file");
  TimingScope mojoScope = timing.nest("Import Mojo");

  MojoParserConfig parseConfig(ctx, *runtime, compilationOptions);
  parseConfig.validateDocStrings = clOptions.validateDocStrings;

  theModule = importMojoFile(mgr, parseConfig, mojoScope);
  if (!theModule)
    return clOptions.reportError("could not parse the module");

  // Initialize the host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Find a target specification or construct one using the commandline options.
  TargetInfoAttr target = getTargetInfo(*theModule);
  if (target) {
    if (target.getTripleStr() != clOptions.targetTriple ||
        target.getCpu() != clOptions.targetCpu ||
        target.getFeatures() != clOptions.targetFeatures) {
      mlir::emitWarning(theModule->getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
      target = nullptr;
    }
  }
  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr =
        getTargetInfoFor(ctx, clOptions.targetTriple, clOptions.targetCpu,
                         clOptions.targetFeatures);
    if (targetOr.isError()) {
      mlir::emitError(theModule->getLoc(), targetOr.getError());
      return EXIT_FAILURE;
    }
    target = targetOr.takeValue();
  }

  // Get the build info from the current build.
  BuildInfoAttr build = BuildInfoAttr::getForCurrentBuild(ctx);

  // Now create the execution engine so we can JIT.
  auto tmOr =
      createTargetMachine(compilationOptions,
                          /*isJIT=*/clOptions.cmd == MojoCommand::kExecute);
  if (tmOr.isError())
    return clOptions.reportError(tmOr.getError());

  auto engineOr = ExecutionEngine::createWithStandardLayers(
      {/*registerDebugPlugins=*/compilationOptions.debugLevel !=
       CompilationOptions::DebugInfoLevel::kNoDebug},
      **tmOr);
  if (failed(engineOr))
    return clOptions.reportError(engineOr.getError());
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOr);

  // Add the object compiler layer.
  auto compiler =
      ObjectCompiler::create(*runtime, pm, ".kgen_cache", compilationOptions);
  if (failed(compiler))
    return clOptions.reportError(Twine("could not create object compiler: ") +
                                 compiler.getError());
  auto &objLayer = engine->addLayer<ObjectCompilerLayer>(
      std::move(*compiler), engine->getLinkingLayer());

  // Add the KGEN compiler layer.
  // First though, get the backend chains to pass into the compile layer.
  auto transformCacheBackend = Cache::getLocalDefaultBackendChain(
      *runtime, (std::filesystem::path(".kgen_cache") / "transform").string(),
      KGEN_VERSION_STRING);
  if (transformCacheBackend.isError())
    return clOptions.reportError(transformCacheBackend.getError());

  auto regionCacheBackend = Cache::getLocalDefaultBackendChain(
      *runtime, (std::filesystem::path(".kgen_cache") / "region").string(),
      KGEN_VERSION_STRING);
  if (regionCacheBackend.isError())
    return clOptions.reportError(transformCacheBackend.getError());

  auto &compileLayer = engine->addLayer<KGENCompilerLayer>(
      pm, *runtime, target, build, clOptions.getCompilationOptions(), objLayer,
      std::move(*transformCacheBackend), std::move(*regionCacheBackend));

  // And add the module into the layer. This will actually compile it down to
  // the post-elaboration phase because before that phase we don't have flat
  // symbols.
  if (auto err = compileLayer.add("exec", *theModule))
    return clOptions.reportError(err.getError());

  // Generate a symbol table and an export map for the module post-compile.
  SymbolTable symtab(*theModule);
  ExportMap exports = getExportedSymbols(*theModule);

  // Handle header emission, we don't need to generate an archive for this.
  if (clOptions.cmd == MojoCommand::kEmitHeader) {
    if (failed(
            emitHeader(symtab, exports, *compiler, clOptions.outputFilename)))
      return clOptions.reportError("failed to emit header file");
    return EXIT_SUCCESS;
  }

  // No ops, we can't actually do anything.
  auto symbolRange = theModule->getOps<mlir::SymbolOpInterface>();
  if (symbolRange.empty())
    return clOptions.reportError(
        "no functions were left in the module after compiling, this usually "
        "means that there was no `@export`ed function to use as a root - did "
        "you forget an `@export`?");

  // Look up the first item in the exported symbols to trigger compilation.
  // TODO(#10893): This behavior is sketchy. We should be exporting the roots of
  //   callstacks we want codegen'd. This requires updating tests.
  if (exports.empty()) {
    StringAttr name = (*symbolRange.begin()).getNameAttr();
    exports.insert({name, {name, false}});
  }

  // Trigger compilation so we can pull out the archive.
  ErrorOr<CompiledFunc> funcOr = engine->lookup(exports.front().second.alias);
  if (funcOr.isError())
    return clOptions.reportError(funcOr.getError());

  // If we're emitting the archive, do it.
  if (clOptions.cmd == MojoCommand::kEmit) {
    // Notify the object layer that we don't need immediate execution.
    objLayer.notForImmediateExecution();
    // And lookup the archive.
    std::optional<Cache::BufferRef> archive =
        objLayer.lookupArchive(*theModule);
    if (!archive.has_value())
      return clOptions.reportError("no compiled archive for the module");
    return failed(clOptions.emitArchive((*archive)->getBuffer()));
  }

  bool isDef = false;
  if (!moduleExportsMain(*theModule, symtab, isDef))
    return clOptions.reportError("could not find 'fn main()' or 'def main()', "
                                 "please provide a main function with no "
                                 "arguments / return values.");

  TimeTraceScope<> traceScope("execute-main");
  auto compiledFuncOr = engine->lookup("main");
  if (failed(compiledFuncOr))
    return clOptions.reportError(compiledFuncOr.getError());
  if (isDef) {
    size_t dummy[2] = {0, 0};
    uint8_t isNormalResult = false;
    compiledFuncOr->invoke<void>(dummy, &isNormalResult);
    if (!isNormalResult)
      return clOptions.reportError("main function threw an error");
  } else {
    compiledFuncOr->invoke<void>();
  }
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << "Mojo compiler:\n  ";
    os << "Modular version " << version.major << '.' << version.minor << '.'
       << version.patch << "\n  ";
    os << "Git SHA " << version.revision << "\n  ";
    os << "Build config " << version.buildType << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerPassManagerCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext context;
  // Set up the dialects so we can use it while demangling.
  DialectRegistry registry;
  TraceProfiler tracer(clOptions);

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  index::IndexDialect, LLVM::LLVMDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  context.appendDialectRegistry(registry);

  // Set up the input file.
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.getIncludePaths());
  sourceManager.AddNewSourceBuffer(clOptions.openInputFileOrExit(),
                                   llvm::SMLoc());

  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);
  return runToolPipeline(&context, sourceManager, clOptions);
}
