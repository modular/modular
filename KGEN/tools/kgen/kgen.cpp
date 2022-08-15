//===- kgen.cpp -----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/CommonCLOptions.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace mlir;

namespace {
enum class PipelineStage {
  kUnknown = 0,
  kGenericMLIR, ///< Some MLIR file format, don't know what.
  kHLKGEN,
  kKGEN,
  kElaborated,
  kLLVM
};

class CLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  cl::list<std::string> searchPaths{
      "I", cl::desc("Path to use to search for included files.")};

  cl::list<ExecutableKernel, bool, ExecutableKernelParser> exec{
      "execute", cl::desc("Specifies the kernels to execute. Defaults to an "
                          "empty list, which will not execute any kernel.")};

  cl::list<EmittableKernel, bool, EmittableKernelParser> emit{
      "emit",
      cl::desc("Specifies the kernels to emit. Defaults to an empty list, "
               "which will emit a file for each kernel in the input file.")};

  Optional<EmittableKernel>
  shouldEmitKernel(mlir::LLVM::LLVMFuncOp kernel) const {
    if (emit.empty())
      return EmittableKernel{kernel.getName().str(),
                             (kernel.getName() + ".o").str()};

    auto found = llvm::find_if(emit, [&](const EmittableKernel &ek) {
      return ek.name == kernel.getName();
    });
    if (found == emit.end())
      return None;
    return *found;
  }

  Optional<ExecutableKernel>
  shouldExecuteKernel(mlir::LLVM::LLVMFuncOp kernel) const {
    auto found = llvm::find_if(exec, [&](const ExecutableKernel &ek) {
      return ek.name == kernel.getName();
    });

    if (found == exec.end())
      return None;
    return *found;
  }
};
} // namespace

static PipelineStage sniffInputFormat(llvm::MemoryBufferRef inputFile) {
  // If there's nothing in the file, then we can't sniff anything.
  if (inputFile.getBuffer().empty())
    return PipelineStage::kUnknown;

  // It's some kind of MLIR file.
  if (inputFile.getBufferIdentifier().endswith(".mlir"))
    return PipelineStage::kGenericMLIR;

  // Don't know what it is.
  return PipelineStage::kUnknown;
}

static bool hasOpWithDialect(llvm::iterator_range<Region::OpIterator> &&range,
                             llvm::SmallPtrSetImpl<Dialect *> &cache,
                             Dialect *dialect) {
  if (cache.contains(dialect))
    return true;

  auto found = llvm::find_if(range, [&](auto &op) {
    Dialect *thisOpDialect = op.getName().getDialect();
    // No dialect, so bail.
    if (!thisOpDialect)
      return false;

    // Insert the dialect into the cache.
    cache.insert(thisOpDialect);

    // No nested regions, just check if this op has it or not.
    if (op.getNumRegions() == 0)
      return thisOpDialect == dialect;

    // Nested regions, we have to check ops within this op.
    for (Region &region : op.getRegions())
      if (hasOpWithDialect(region.getOps(), cache, dialect))
        return true;

    // No match found.
    return false;
  });
  return found != range.end();
}

static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     PipelineStage stage,
                                     const CLOptions &clOptions) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<mlir::arith::ArithmeticDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();

  mlir::registerLLVMDialectTranslation(registry);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
  // Allow unregistered dialects, we will verify we know what to do with it
  // later.
  ctx->allowUnregisteredDialects();

  OwningOpRef<ModuleOp> theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  llvm::SmallPtrSet<Dialect *, 4> seenDialectCache;

  // Prime the cache with the arith dialect - all the other dialects are/have
  // container ops so we are very likely to hit them before we would hit an
  // arith op, so this enables us to run this walk once and prime the cache so
  // the other calls to hasOpWithDialect are more likely to hit the cache rather
  // than walking the IR.
  bool hasArithDialect =
      hasOpWithDialect(theModule->getOps(), seenDialectCache,
                       ctx->getLoadedDialect<mlir::arith::ArithmeticDialect>());

  // Finish sensing the contents.
  if (stage == PipelineStage::kGenericMLIR) {
    if (hasOpWithDialect(theModule->getOps(), seenDialectCache,
                         ctx->getLoadedDialect<KGEN::HLKGENDialect>())) {
      stage = PipelineStage::kHLKGEN;
    } else if (hasOpWithDialect(theModule->getOps(), seenDialectCache,
                                ctx->getLoadedDialect<KGEN::KGENDialect>())) {
      if (theModule->getOps<KGEN::GeneratorInterfaceOp>().empty())
        stage = PipelineStage::kElaborated;
      else
        stage = PipelineStage::kKGEN;
    } else if (!theModule->getOps<mlir::LLVM::LLVMFuncOp>().empty()) {
      stage = PipelineStage::kLLVM;
    }
  }

  if (stage == PipelineStage::kGenericMLIR)
    return mlir::emitError(
        theModule->getLoc(),
        "could not sense the contents of this file, cannot proceed");

  // Set up the pass pipeline.
  mlir::PassManager pm(ctx);
  if (stage == PipelineStage::kHLKGEN) {
    pm.addPass(KGEN::createLowerHLKGENPass());
    pm.addPass(mlir::createCanonicalizerPass());
    stage = PipelineStage::kKGEN;
  }

  if (stage == PipelineStage::kKGEN) {
    auto elaborate = KGEN::createElaborateKernelsPass();
    std::string includes;
    llvm::raw_string_ostream includeStr(includes);
    for (StringRef include : clOptions.searchPaths)
      includeStr << "search-path=" << include << " ";

    if (failed(elaborate->initializeOptions(includeStr.str())))
      return failure(
          clOptions.reportError("unable to initialize elaborator options"));

    pm.addPass(std::move(elaborate));
    pm.addPass(mlir::createCanonicalizerPass());
    stage = PipelineStage::kElaborated;
  }

  if (stage == PipelineStage::kElaborated) {
    OpPassManager &kpm = pm.nest<KGEN::KernelOp>();
    if (hasArithDialect)
      kpm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    if (hasOpWithDialect(theModule->getOps(), seenDialectCache,
                         ctx->getLoadedDialect<mlir::scf::SCFDialect>())) {
      kpm.addPass(mlir::createConvertSCFToCFPass());
      kpm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
    }
    kpm.addPass(KGEN::createConvertPOPToLLVMPass());
    pm.addPass(KGEN::createConvertKGENToLLVMPass());

    // And finally canonicalize.
    pm.addPass(mlir::createCanonicalizerPass());
    stage = PipelineStage::kLLVM;
  }

  assert(stage == PipelineStage::kLLVM &&
         "expected LLVM at this stage of the pipeline");

  // Now create the execution engine so we can JIT.
  auto engineOr = KGEN::ExecutionEngine::create();
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));

  KGEN::ExecutionEngine engine = std::move(*engineOr);

  // Helper to emit the object for a kernel.
  auto emitObjectForKernel = [&](mlir::LLVM::LLVMFuncOp k,
                                 const Twine &filename) -> LogicalResult {
    // If the filename is not provided, then default to the current working
    // directory.
    std::filesystem::path objPath = filename.str();
    if (!objPath.is_absolute())
      objPath = std::filesystem::current_path() / filename.str();

    // Open the output file so we can emit to it.
    std::string err;
    auto outFile = mlir::openOutputFile(objPath.string(), &err);
    if (!outFile)
      return mlir::emitError(k.getLoc(), err);

    auto objOr = engine.getObject(k);
    if (failed(objOr))
      return mlir::emitError(k.getLoc(),
                             "could not get the object for the kernel '@" +
                                 k.getName() + "': " + objOr.getError());

    std::unique_ptr<llvm::MemoryBuffer> obj = std::move(*objOr);
    outFile->os().write(obj->getBufferStart(), obj->getBufferSize());
    outFile->keep();
    return mlir::success();
  };

  // Run the pass manager. This will ensure that the module has been fully
  // lowered to LLVM.
  if (failed(pm.run(*theModule)))
    return failure(clOptions.reportError("compilation failed"));

  // Loop over the kernels and (1) add them to the engine and (2) maybe emit the
  // kernel as an object file.
  for (auto k : theModule->getOps<mlir::LLVM::LLVMFuncOp>()) {
    // First add the kernel to the engine.
    if (ErrorOrSuccess err = engine.add(k))
      return mlir::emitError(k.getLoc(), err.getError());

    // If we were asked to emit this kernel, do so.
    if (Optional<EmittableKernel> emittableKernel =
            clOptions.shouldEmitKernel(k))
      if (failed(emitObjectForKernel(k, emittableKernel->outputFilename)))
        return failure();
  }

  // Now, if we were asked to execute any kernels, do so.
  for (const auto &exec : clOptions.exec) {
    auto k = theModule->lookupSymbol<mlir::LLVM::LLVMFuncOp>(exec.name);
    if (!k) {
      mlir::emitError(theModule->getLoc())
          << "could not find kernel '@" << exec.name << "'";
      if (!clOptions.ignoreFailures)
        return failure();
      continue;
    }

    if (auto err = exec.verifyKernelSignature(k.getFunctionType())) {
      mlir::emitError(k.getLoc(), err.getError());
      if (!clOptions.ignoreFailures)
        return failure();
      continue;
    }

    if (auto err = exec.executeAndPrint(engine)) {
      mlir::emitError(k.getLoc(), err.getError());
      if (!clOptions.ignoreFailures)
        return failure();
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv);

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  PipelineStage stage = sniffInputFormat(*inputFile);

  if (stage == PipelineStage::kUnknown)
    return clOptions.reportError("could not sniff the input file format");

  return failed(clOptions.configureMLIRContextAndSourceMgrAndExecute(
      std::move(inputFile),
      [&](MLIRContext *ctx, llvm::SourceMgr &mgr) -> LogicalResult {
        return runToolPipeline(ctx, mgr, stage, clOptions);
      }));
}
