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
#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/Context.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;

//===----------------------------------------------------------------------===//
// TestGeneratePreElaboratedBody
//===----------------------------------------------------------------------===//

/// Write the operation to bytecode and return it as a dense resource.
static DenseResourceElementsAttr serializeToResource(Operation *op,
                                                     const Twine &name) {
  SmallVector<char> buffer;
  llvm::raw_svector_ostream stream(buffer);
  if (failed(mlir::writeBytecodeToFile(op, stream)))
    return {};
  return createResourceAttr(op->getContext(), buffer, name);
}

/// Generate a dummy module with the same function body.
template <typename OpT>
static OwningOpRef<ModuleOp> cloneIntoFakeModule(KGEN::LIT::FuncOp func) {
  OpBuilder b(func.getContext());
  OwningOpRef<ModuleOp> fakeModule = b.create<ModuleOp>(func.getLoc());
  OpBuilder fakeBuilder = OpBuilder::atBlockEnd(fakeModule->getBody());
  OpT fakeCompiledBody;
  if constexpr (!std::is_same_v<OpT, KGEN::LIT::FuncOp>) {
    fakeCompiledBody = fakeBuilder.create<OpT>(
        func.getLoc(), func.getSymNameAttr(), func.getSignature());

    // Just clone the body in.
    mlir::IRMapping map;
    func.getBodyRegion().cloneInto(&fakeCompiledBody.getBodyRegion(), map);
  } else {
    fakeCompiledBody = cast<OpT>(fakeBuilder.clone(*func));
  }
  fakeCompiledBody.setPackageExported();

  return fakeModule;
}

namespace {

/// This pass generates a kgen.func that clones the body and adds a
/// "_elaborated" suffix to the name for all the specified lit.funcs in the
/// module. It's used to test the logic in LowerLIT that handles pre-elaborated
/// funcs.
struct TestGeneratePreElaboratedBody
    : public mlir::PassWrapper<TestGeneratePreElaboratedBody,
                               OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGeneratePreElaboratedBody)

  StringRef getArgument() const override {
    return "test-generate-elaborated-body";
  }

  void runOnOperation() override {
    ModuleOp theModule = getOperation();

    // Attach a kgen.func to the lit.func. This dummy function simply contains
    // exactly the same operations as the lit.func, but has a slightly different
    // name.
    for (auto func : theModule.getOps<KGEN::LIT::FuncOp>()) {
      // Allow the test to skip functions if they have the doNotExtern attr.
      if (func->hasAttr("doNotExtern"))
        continue;

      // This pass generates `package.link` ops with archives for each target
      // specified by functions that define attributes named
      // "test.target.[0-9]".
      SmallVector<TargetInfoAttr> targets;
      for (size_t i = 0; i < 10; ++i)
        if (auto target = func->getAttrOfType<TargetInfoAttr>(
                llvm::formatv("test.target.{0}", i).str()))
          targets.push_back(target);

      OpBuilder b(func.getContext());
      OwningOpRef<ModuleOp> fakeCopyModule =
          cloneIntoFakeModule<KGEN::LIT::FuncOp>(func);
      OwningOpRef<ModuleOp> fakeModule =
          cloneIntoFakeModule<KGEN::GeneratorOp>(func);
      OwningOpRef<ModuleOp> fakeCompiledModule =
          cloneIntoFakeModule<KGEN::FuncOp>(func);

      // Externalize the function and attach the post elaboration metadata.
      func.getBody()->clear();
      OpBuilder::atBlockBegin(func.getBody())
          .create<KGEN::LIT::ExternFuncOp>(func.getLoc());
      StringRef funcName = *func.getSymName();
      StringAttr linkName = b.getStringAttr("link_" + funcName);
      func.setPreCompiledModuleRefAttr(FlatSymbolRefAttr::get(linkName));
      func.setPreElaborationNameAttr(func.getSymNameAttr());
      func.setLinkageName(func.getSymNameAttr());

      DenseResourceElementsAttr postParseBytecode = serializeToResource(
          *fakeCopyModule, funcName + "_generated_post_parse_attr");
      if (!postParseBytecode)
        return signalPassFailure();

      // Generate a package link to the fake module.
      OpBuilder linkBuilder(func);
      linkBuilder.create<KGEN::PackageLinkOp>(
          func.getLoc(), linkName, postParseBytecode, /*dependencies=*/nullptr);
    }
  }
};

//===----------------------------------------------------------------------===//
// TestAlwaysFailPass
//===----------------------------------------------------------------------===//

/// This is a pass that always fails for the purpose of debugging reproducers.
struct TestAlwaysFailPass
    : public mlir::PassWrapper<TestAlwaysFailPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlwaysFailPass)

  StringRef getArgument() const override { return "test-always-fail"; }

  void runOnOperation() override { return signalPassFailure(); }
};

} // namespace

int main(int argc, char **argv) {
  // HACK: Read in the option early.
  bool llclSingleThread = false;
  if (argc >= 2 && StringRef(argv[1]) == "--llcl-single-thread")
    llclSingleThread = true;

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
  mlir::PassRegistration<TestGeneratePreElaboratedBody>{};
  mlir::PassRegistration<TestAlwaysFailPass>{};

  // Create our context.
  AsyncRT::RuntimeOptions llclOpts;
  llclOpts.withLeakCheckedAllocator();
  if (llclSingleThread)
    llclOpts.withSingleThreaded();
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-opt", Init::Options().withRuntimeOptions(llclOpts));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  if (llclSingleThread) {
    // Defend against upstream errors.
    [[maybe_unused]] auto &runtime = *(*ctxOr)->get<AsyncRT::Runtime>();
    assert(runtime.getWorkQueue()->getParallelismLevel() == 1);
  }
  registerContext(registry, *ctxOr);

  // Register passes.
  KGEN::registerDefaultKGENPasses();

  // Register cl options.
  static llvm::cl::opt<bool> dummyOpt{"llcl-single-thread"};

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

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "kgen optimizer driver", registry);

  KGEN::TraceProfiler tracer(timeTrace, timeTraceGranularity);

  return failed(
      MlirOptMain(argc, argv, inputFilename, outputFilename, registry));
}
