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
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {
class CLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<Command> cmd{cl::desc("The command to execute"),
                       cl::values(clEnumValN(Command::kEmit, "emit",
                                             "Emit funcs as object files."),
                                  clEnumValN(Command::kExecute, "execute",
                                             "Execute the main function.")),
                       llvm::cl::Required};

  cl::list<std::string> searchPaths{
      "I", cl::desc("Path to use to search for included files.")};

  cl::opt<bool> enableSearch{
      "enable-search", cl::init(false),
      cl::desc("Do search when an evaluator is provided.")};
};
} // namespace

/// Look for a main() -> Int in the module. Return it if found, otherwise
/// return a nullptr.
static FuncOp findMain(ModuleOp theModule) {
  for (auto fn : theModule.getOps<FuncOp>()) {
    if (!fn.getName().endswith("main()"))
      continue;

    // Check main's signature.
    mlir::FunctionType funcType = fn.getFunctionType();
    Type resType = funcType.getResult(0);

    // Is the return type a struct with one field?
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        !funcType.getResult(0).isa<KGEN::POP::StructType>() ||
        cast<KGEN::POP::StructType>(resType).getNumElements() != 1)
      return nullptr;

    Type fieldType =
        cast<KGEN::POP::StructType>(resType).getConcreteElementType(0);

    // Is the field a scalar?
    if (!isa<POP::SIMDType>(fieldType) ||
        !cast<POP::SIMDType>(fieldType).isScalar() ||
        !isa<DTypeConstantAttr>(cast<POP::SIMDType>(fieldType).getDType()))
      return nullptr;

    DTypeConstantAttr fieldDType =
        cast<DTypeConstantAttr>(cast<POP::SIMDType>(fieldType).getDType());
    if (!fieldDType.getDType().isIndex())
      return nullptr;
    return fn;
  }
  // No main found.
  return nullptr;
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static int runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                           const CLOptions &clOptions) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  HLCF::HLCFDialect, index::IndexDialect, LLVM::LLVMDialect,
                  scf::SCFDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  CompilationOptions compilationOptions;
  OwningOpRef<ModuleOp> theModule;
  llvm::StringRef inputFileName(clOptions.inputFilename.getValue());

  // Initialize the timing manager.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  mlir::PassManager pm(ctx);
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);

  if (!inputFileName.ends_with(".lit"))
    return clOptions.reportError("expected a .lit file");
  TimingScope litScope = timing.nest("Import Lit");
  theModule = importLitFile(mgr, ctx, litScope, compilationOptions, false);
  pm.addPass(createLowerLITTerminators());

  if (!theModule)
    return clOptions.reportError("could not parse the module");

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  // Set up the runtime.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  elaborateModule(pm, runtime, {clOptions.searchPaths, clOptions.enableSearch},
                  includedFiles);

  if (failed(pm.run(*theModule)))
    return clOptions.reportError("compilation failed");

  SymbolTable symtab(*theModule);
  auto compiler = ObjectCompiler::create(runtime, ".kgen_cache", symtab,
                                         compilationOptions);
  if (failed(compiler))
    return clOptions.reportError(Twine("could not create object compiler: ") +
                                 compiler.getError());

  TargetInfoAttr attr = TargetInfoAttr::getForHost(ctx);

  // This produces a standalone object for all the objects we requested.
  auto standaloneOr = compiler->produceStandaloneObject(
      attr, /*isJIT=*/clOptions.cmd == Command::kExecute);
  if (failed(standaloneOr))
    return clOptions.reportError("compiler error");
  Cache::BufferRef standaloneObject = std::move(*standaloneOr);

  // If we're emitting the object, do it.
  if (clOptions.cmd == Command::kEmit) {
    if (failed(clOptions.emitObject(standaloneObject->getBuffer())))
      return clOptions.reportError("unable to emit object file");

    auto headerPath = clOptions.getHeaderOutputPath();
    // If we have no output path, we can't emit headers so return.
    if (!headerPath)
      return clOptions.reportError("please provide an output filename");

    // Finish off by producing a header file with the decls.
    if (failed(emitHeader(*compiler, *headerPath)))
      return clOptions.reportError("failed to emit the header file");
    return 0;
  }

  assert(clOptions.cmd == Command::kExecute);
  // Now create the execution engine so we can JIT.
  auto engineOr = ExecutionEngine::create(compilationOptions);
  if (failed(engineOr))
    return clOptions.reportError(engineOr.getError());
  ExecutionEngine engine = std::move(*engineOr);

  // TODO (8082): This should not be necessary.
  std::vector<std::pair<StringLiteral, void *>> compilerRTFunctions;
  KGEN::registerBenchmark(compilerRTFunctions);
  KGEN::registerDebugAssert(compilerRTFunctions);
  KGEN::registerIntelAMX(compilerRTFunctions);
  KGEN::registerLLCL(compilerRTFunctions);
  KGEN::registerPrint(compilerRTFunctions);
  KGEN::registerRandom(compilerRTFunctions);
  KGEN::registerSystem(compilerRTFunctions);
  KGEN::registerTracing(compilerRTFunctions);
  for (auto [name, ptr] : compilerRTFunctions)
    if (auto err = engine.add("exec", name, ptr))
      return clOptions.reportError(err.getError());

  if (auto err = engine.add("exec", std::move(standaloneObject)))
    return clOptions.reportError(err.getError());

  // Helper to execute a func.
  auto execMain = [&](FuncOp theFunc) -> int {
    TimeTraceScope<> traceScope("execute-function", theFunc.getSymName());
    auto compiledFuncOr = engine.lookup("exec", theFunc.getNameAttr());
    if (failed(compiledFuncOr))
      return clOptions.reportError(compiledFuncOr.getError());
    return compiledFuncOr->invoke<ssize_t>();
  };

  FuncOp mainFunc = findMain(*theModule);
  if (!mainFunc)
    return clOptions.reportError("could not find fn main() -> Int");

  return (execMain(mainFunc));
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Initialize the compiler runtime.
  KGEN_CompilerRT_Initialize();

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerPassManagerCLOptions();
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
