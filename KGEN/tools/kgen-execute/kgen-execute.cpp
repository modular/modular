//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/CompilerRT.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

//===--------------------------------------------------------------------===//
// ProcessBuffer
//===--------------------------------------------------------------------===//

namespace {
/// This struct essentially provides the body of the execution flow that we pass
/// to configureMLIRContextAndSourceMgrAndExecute. It's long enough that we
/// don't want to have it inline, and pulling it out into a functor makes it
/// more readable.
struct ProcessBuffer {
  LLCL::Runtime &runtime;
  KGENCLOptions &clOptions;

  LogicalResult operator()(MLIRContext *ctx, llvm::SourceMgr &sourceMgr) const {
    KGEN::CompilationOptions compilationOptions =
        clOptions.getCompilationOptions();
    DialectRegistry registry;
    // Don't need LIT here.
    registry.insert<KGEN::KGENDialect, KGEN::POP::POPDialect, HLCF::HLCFDialect,
                    mlir::index::IndexDialect>();
    mlir::registerLLVMDialectTranslation(registry);

    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();

    // Open the input file.
    OwningOpRef<ModuleOp> module;
    if (compilationOptions.getDebugInfoLevelForInput())
      module = DebugInfo::parseSourceFileWithDebugInfo(
          sourceMgr, ctx, compilationOptions.getDIEmissionKind());
    else
      module = parseSourceFile<ModuleOp>(sourceMgr, ctx);
    if (!module)
      return failure(clOptions.reportError("could not parse input file"));

    // The IR module is being compiled to an object file. Find a target
    // specification or use the host target.
    TargetInfoAttr target = getTargetInfo(*module);
    if (!target) {
      ErrorOr<TargetInfoAttr> hostTarget =
          getTargetInfoFor(ctx, clOptions.targetTriple, clOptions.targetCpu,
                           clOptions.targetFeatures);
      if (hostTarget.isError())
        return mlir::emitError(module->getLoc(), hostTarget.getError());
      target = hostTarget.takeValue();
      setTargetInfo(*module, target);
    } else {
      if (target.getTripleStr() != clOptions.targetTriple) {
        mlir::emitWarning(module->getLoc(), "module target ")
            << target.getTripleStr() << " does not match command line option "
            << clOptions.targetTriple
            << ", command line target will be ignored";
      }
    }

    SymbolTable symtab(*module);
    mlir::PassManager mgr(ctx);
    auto compiler = KGEN::ObjectCompiler::create(runtime, mgr, ".kgen_cache",
                                                 symtab, compilationOptions);
    if (failed(compiler))
      return failure(clOptions.reportError("could not create compiler: " +
                                           Twine(compiler.getError())));

    // Produce a single standalone .a
    auto standaloneOr = compiler->produceStandaloneArchive(
        /*isJIT=*/clOptions.cmd == Command::kExecute);
    if (failed(standaloneOr))
      return failure();
    Cache::BufferRef standaloneArchive = std::move(*standaloneOr);

    if (clOptions.cmd == Command::kEmit)
      return clOptions.emitArchive(standaloneArchive->getBuffer());

    llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols =
        KGEN::getExportedSymbols(*module);
    DenseMap<StringAttr, KGEN::FuncOp> exportedFuncs;
    for (auto &p : exportedSymbols)
      if (auto func = symtab.lookup<KGEN::FuncOp>(p.first))
        exportedFuncs.insert({p.second.alias, func});

    auto lookupFunc = [&](StringRef funcName) -> ErrorOr<KGEN::FuncOp> {
      StringAttr funcNameAttr = StringAttr::get(ctx, funcName);

      // Check the exported symbols.
      auto it = exportedFuncs.find(funcNameAttr);
      if (it != exportedFuncs.end())
        return it->second;

      // Fallback to checking the module.
      auto func = symtab.lookup<KGEN::FuncOp>(funcNameAttr);
      if (!func)
        return Error("could not find func '" + funcName + "'.");
      return func;
    };

    auto engineOr =
        KGEN::ExecutionEngine::create(clOptions.getCompilationOptions());
    if (engineOr.isError()) {
      clOptions.reportError(engineOr.getError());
      return failure();
    }

    auto execEngine = std::move(*engineOr);

    // Add the module to the execution engine.
    if (auto err = execEngine.add("exec", std::move(standaloneArchive)))
      return failure(clOptions.reportError(err.getError()));

    for (const auto &k : clOptions.funcs) {
      auto funcOr = lookupFunc(k.name);
      if (funcOr.isError())
        return failure(clOptions.reportError(funcOr.getError()));

      KGEN::FuncOp func = *funcOr;
      auto compiledFuncOr = execEngine.lookup("exec", k.name);
      if (failed(compiledFuncOr))
        return failure(clOptions.reportError(compiledFuncOr.getError()));

      // And now we diverge.
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
      case Command::kEmitLLVM:
      case Command::kEmitAssembly:
      case Command::kEmit:
        break;
      case Command::kExecute: {
        if (auto err = k.verifyFuncSignature(func.getFunctionType()))
          return failure(clOptions.reportError(err.getError()));

        if (auto err = k.executeAndPrint(*compiledFuncOr))
          return failure(clOptions.reportError(err.getError()));
        break;
      }
      }
    }

    return mlir::success();
  }
};
} // namespace

//===--------------------------------------------------------------------===//
// main
//===--------------------------------------------------------------------===//

int main(int argc, char **argv) {
  KGENCLOptions clOptions(argc, argv);

  // Initialize the compiler runtime.
  KGEN_CompilerRT_Initialize();

  // Initialize the LLCL runtime.
  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  // Initialize the host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Provide a tool function that runs the requested ops, again, so we can
  // re-use it.
  auto toolFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                    raw_ostream &os) {
    return clOptions.configureMLIRContextAndSourceMgrAndExecute(
        std::move(chunkBuffer), ProcessBuffer{runtime, clOptions});
  };

  // Process the file.
  return failed(
      splitAndProcessBuffer(std::move(inputFile), toolFn, llvm::outs(), false));
}
