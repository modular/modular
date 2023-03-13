//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/EmitFuncHeader.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ParseLit.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
/// What to do with a given Mojo file.
enum class MojoCommand {
  kDocGen,
  kEmit,
  kExecute,
};

class CLOptions : public KGENCommonOptions, public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<MojoCommand> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(MojoCommand::kDocGen, "doc-gen",
                     "Generate markdown documentation."),
          clEnumValN(MojoCommand::kEmit, "emit", "Emit funcs as object files."),
          clEnumValN(MojoCommand::kExecute, "execute",
                     "Execute the main function.")),
      cl::init(MojoCommand::kExecute)};
};
} // namespace

/// Returns true if the given module exports a main() function, false otherwise.
static bool moduleExportsMain(ModuleOp theModule, SymbolTable &symtab) {
  auto emptyListType =
      KGEN::ListType::get(IntegerType::get(theModule.getContext(), 1), 0);
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
    return dyn_cast<KGEN::ListType>(funcType.getResult(0)) == emptyListType;
  }
  return false;
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static int runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                           const CLOptions &clOptions) {
  DialectRegistry registry;
  TraceProfiler tracer(clOptions);

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  HLCF::HLCFDialect, index::IndexDialect, LLVM::LLVMDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
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
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);
  if (clOptions.enableMLIRCrashReproducer.getValue()) {
    ctx->disableMultithreading();
    pm.enableCrashReproducerGeneration(clOptions.inputFilename.getValue() +
                                           ".repro.mlir",
                                       /*genLocalReproducer=*/true);
  }

  if (!inputFileName.ends_with(".lit") && !inputFileName.ends_with(".mojo"))
    return clOptions.reportError("expected a .mojo file");
  TimingScope mojoScope = timing.nest("Import Mojo");

  if (clOptions.cmd == MojoCommand::kDocGen) {
    std::unique_ptr<llvm::ToolOutputFile> os =
        clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".md");
    if (failed(generateLitDoc(mgr, ctx, os->os(), mojoScope, compilationOptions,
                              *runtime)))
      return clOptions.reportError("could not generate documentation");
    os->keep();
    return EXIT_SUCCESS;
  }

  theModule = importLitFile(mgr, ctx, mojoScope, compilationOptions,
                            /*useMLIRDiagnostics=*/false, *runtime);

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

  elaborateModule(pm, *runtime, target, {clOptions.enableSearch});

  if (failed(pm.run(*theModule)))
    return clOptions.reportError("compilation failed");

  SymbolTable symtab(*theModule);
  auto compiler = ObjectCompiler::create(*runtime, pm, ".kgen_cache", symtab,
                                         compilationOptions);
  if (failed(compiler))
    return clOptions.reportError(Twine("could not create object compiler: ") +
                                 compiler.getError());

  // This produces a standalone object for all the objects we requested.
  auto standaloneOr = compiler->produceStandaloneObject(
      /*isJIT=*/clOptions.cmd == MojoCommand::kExecute);
  if (failed(standaloneOr))
    return clOptions.reportError("compiler error");
  Cache::BufferRef standaloneObject = std::move(*standaloneOr);

  // If we're emitting the object, do it.
  if (clOptions.cmd == MojoCommand::kEmit) {
    if (failed(clOptions.emitObject(standaloneObject->getBuffer())))
      return clOptions.reportError("unable to emit object file");

    auto headerPath = clOptions.getHeaderOutputPath();
    // If we have no output path, we can't emit headers so return.
    if (!headerPath)
      return clOptions.reportError("please provide an output filename");

    // Finish off by producing a header file with the decls.
    if (failed(emitHeader(*compiler, *headerPath)))
      return clOptions.reportError("failed to emit the header file");
    return EXIT_SUCCESS;
  }

  assert(clOptions.cmd == MojoCommand::kExecute);
  // We can only execute on the host machine.
  if (llvm::sys::getDefaultTargetTriple() != target.getTripleStr())
    return clOptions.reportError("can only execute on host target");

  // Now create the execution engine so we can JIT.
  auto engineOr = ExecutionEngine::create(compilationOptions);
  if (failed(engineOr))
    return clOptions.reportError(engineOr.getError());
  ExecutionEngine engine = std::move(*engineOr);

  // TODO (8082): This should not be necessary.
  std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
  KGEN::registerIntelAMX(compilerRTFunctions);
  KGEN::registerLLCL(compilerRTFunctions);
  KGEN::registerMemory(compilerRTFunctions);
  KGEN::registerPrint(compilerRTFunctions);
  KGEN::registerSystem(compilerRTFunctions);
  KGEN::registerTracing(compilerRTFunctions);
  for (auto [name, ptr] : compilerRTFunctions)
    if (auto err = engine.add("exec", name, ptr))
      return clOptions.reportError(err.getError());

  if (auto err = engine.add("exec", std::move(standaloneObject)))
    return clOptions.reportError(err.getError());

  if (!moduleExportsMain(*theModule, symtab))
    return clOptions.reportError(
        "could not find 'fn main():', please provide a main function with no "
        "arguments / return values.");

  TimeTraceScope<> traceScope("execute-main");
  auto compiledFuncOr = engine.lookup("exec", "main");
  if (failed(compiledFuncOr))
    return clOptions.reportError(compiledFuncOr.getError());
  compiledFuncOr->invoke<void>();
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Initialize the compiler runtime.
  KGEN_CompilerRT_Initialize();

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerPassManagerCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.searchPaths);
  sourceManager.AddNewSourceBuffer(clOptions.openInputFileOrExit(),
                                   llvm::SMLoc());

  mlir::MLIRContext context;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);
  return runToolPipeline(&context, sourceManager, clOptions);
}
