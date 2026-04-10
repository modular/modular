//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// MLIR processing path for kgen-opt.
//
//===----------------------------------------------------------------------===//

#include "KGEN/tools/kgen-opt/MLIRDriver.h"

#include "Init/Init.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/ForceLinkMLIRC.h"
#include "KGEN/Support/MojoPackage.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/Debug.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Debug/Counter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

//===----------------------------------------------------------------------===//
// TestAlwaysFailPass
//===----------------------------------------------------------------------===//

namespace {
/// A pass that always fails, useful for debugging reproducers.
struct TestAlwaysFailPass
    : public mlir::PassWrapper<TestAlwaysFailPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlwaysFailPass)
  StringRef getArgument() const override { return "test-always-fail"; }
  void runOnOperation() override { signalPassFailure(); }
};
} // namespace

//===----------------------------------------------------------------------===//
// CL options (owned by this translation unit)
//===----------------------------------------------------------------------===//

namespace {
struct MLIRCLOptions {
  // Dummy opt so that the early-scan "--asyncrt-single-thread" flag (which is
  // read before option parsing) is not rejected as unknown by the CL parser.
  llvm::cl::opt<bool> asyncrtSingleThread{"asyncrt-single-thread"};

  llvm::cl::opt<bool> timeTrace{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory.")};

  llvm::cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::init(0)};

  llvm::cl::opt<bool> ignoreIncompatiblePackageErrors{
      "ignore-incompatible-package-errors",
      llvm::cl::desc(
          "Ignore errors encountered when loading incompatible Mojo packages."),
      llvm::cl::init(false)};
};

/// Lazily constructed singleton that owns all MLIR-path CL options.
/// Construction registers the options with the global CL state.
MLIRCLOptions &getCLOptions() {
  static MLIRCLOptions opts;
  return opts;
}
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void M::KGEN::Tool::registerMLIRDialectsAndPasses(
    mlir::DialectRegistry &registry) {
  KGEN::forceLinkMLIRC();
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);
  mlir::PassRegistration<TestAlwaysFailPass>{};
  KGEN::registerDefaultKGENPasses("kgen-opt");
  DebugInfo::registerTransformsPasses();
}

bool M::KGEN::Tool::registerMLIRPathCLOptions(mlir::DialectRegistry &registry,
                                              int argc, char **argv) {
  // HACK: Read --asyncrt-single-thread early; context creation must precede
  // option registration.
  bool asyncrtSingleThread = false;
  for (int i = 1; i < argc; ++i) {
    if (llvm::StringRef(argv[i]) == "--asyncrt-single-thread") {
      asyncrtSingleThread = true;
      break;
    }
  }
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create and register the AsyncRT context before MlirOptMainConfig so that
  // dialect plugins can see it.
  AsyncRT::RuntimeOptions asyncrtOpts;
  asyncrtOpts.withLeakCheckedAllocator();
  if (asyncrtSingleThread)
    asyncrtOpts.withSingleThreaded();

  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-opt", Init::Options().withRuntimeOptions(asyncrtOpts));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return false;
  }

  if (asyncrtSingleThread) {
    [[maybe_unused]] auto &runtime = *(*ctxOr)->get<AsyncRT::Runtime>();
    assert(runtime.getWorkQueue()->getParallelismLevel() == 1);
  }

  registerContext(registry, *ctxOr);

  // Trigger construction of all MLIR-path cl::opt objects.
  getCLOptions();

  // Register MLIR framework options (pass pipeline parser, diagnostics, …).
  mlir::MlirOptMainConfig::registerCLOptions(registry);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugCounter::registerCLOptions();

  // Register KGEN dialect and pass CL options.
  KGEN::registerKGENCommandLineOptions();
  KGEN::initializeDebugOptions();
  KGEN::KGENPassCLOptions::registerOptions();
  return true;
}

mlir::LogicalResult
M::KGEN::Tool::runMLIRPath(llvm::StringRef inputFile,
                           llvm::StringRef outputFile,
                           mlir::DialectRegistry &registry) {
  MLIRCLOptions &clOpts = getCLOptions();

  if (KGEN::debugFlag)
    llvm::errs() << "WARNING: `kgen-debug-only` may work incorrectly with "
                    "multithreading enabled\n";

  KGEN::TraceProfiler tracer(clOpts.timeTrace, clOpts.timeTraceGranularity);

  // When reading from stdin and the input is a tty, warn the user.
  if (inputFile == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  // If this is a Mojo package file, verify the header and skip past it to get
  // to the MLIR within.
  llvm::MemoryBufferRef mlirBuffer = *file;
  std::unique_ptr<llvm::MemoryBuffer> decompressedPkgData;
  if (KGEN::isMojoPackage(*file)) {
    ErrorOr<KGEN::MojoPackageMLIRBuffer> mlirBufOrErr =
        M::KGEN::getMLIRBufferFromPackage(
            *file, clOpts.ignoreIncompatiblePackageErrors);
    if (mlirBufOrErr.isError()) {
      llvm::errs() << mlirBufOrErr.takeError().get() << "\n";
      return mlir::failure();
    }
    mlirBuffer = mlirBufOrErr->buffer;
    decompressedPkgData = std::move(mlirBufOrErr->ownedData);
  }

  auto mlirBuff = llvm::MemoryBuffer::getMemBuffer(
      mlirBuffer, /*RequiresNullTerminator=*/true);

  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputFile, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();
  if (failed(MlirOptMain(output->os(), std::move(mlirBuff), registry, config)))
    return mlir::failure();

  output->keep();
  return mlir::success();
}
