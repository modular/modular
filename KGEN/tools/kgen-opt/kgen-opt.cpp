//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Init/Init.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/ForceLinkMLIRC.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/Debug.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;

//===----------------------------------------------------------------------===//
// TestAlwaysFailPass
//===----------------------------------------------------------------------===//

namespace {
/// This is a pass that always fails for the purpose of debugging reproducers.
struct TestAlwaysFailPass
    : public mlir::PassWrapper<TestAlwaysFailPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlwaysFailPass)

  StringRef getArgument() const override { return "test-always-fail"; }

  void runOnOperation() override { return signalPassFailure(); }
};
} // namespace

int main(int argc, char **argv) {
  // Force linking of MLIR C symbols to JIT Mojo code relying on the mlir
  // bindings.
  KGEN::forceLinkMLIRC();

  // HACK: Read in the option early.
  bool asyncrtSingleThread = false;
  if (argc >= 2 && StringRef(argv[1]) == "--asyncrt-single-thread")
    asyncrtSingleThread = true;

  DialectRegistry registry;

  // Register all KGEN dialects.
  registerAllKGENDialects(registry);

  // Initialize all targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Initialize the host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize LLVM exporters.
  registerKGENToLLVMTranslation(registry);

  // Register test passes.
  mlir::PassRegistration<TestAlwaysFailPass>{};

  // Create our context.
  AsyncRT::RuntimeOptions asyncrtOpts;
  asyncrtOpts.withLeakCheckedAllocator();
  if (asyncrtSingleThread)
    asyncrtOpts.withSingleThreaded();
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-opt", Init::Options().withRuntimeOptions(asyncrtOpts));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  if (asyncrtSingleThread) {
    // Defend against upstream errors.
    [[maybe_unused]] auto &runtime = *(*ctxOr)->get<AsyncRT::Runtime>();
    assert(runtime.getWorkQueue()->getParallelismLevel() == 1);
  }
  registerContext(registry, *ctxOr);

  // Register passes.
  KGEN::registerDefaultKGENPasses();
  DebugInfo::registerTransformsPasses();

  // Register cl options.
  static llvm::cl::opt<bool> dummyOpt{"asyncrt-single-thread"};

  static llvm::cl::opt<bool> timeTrace{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory.")};

  static llvm::cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::init(0)};

  KGEN::registerKGENCommandLineOptions();
  KGEN::initializeDebugOptions();
  KGEN::KGENPassCLOptions::registerOptions();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "kgen optimizer driver", registry);
  if (KGEN::debugFlag)
    llvm::errs() << "WARNING: `kgen-debug-only` may work incorrectly with "
                    "multithreading enabled\n";

  KGEN::TraceProfiler tracer(timeTrace, timeTraceGranularity);

  return failed(
      MlirOptMain(argc, argv, inputFilename, outputFilename, registry));
}
