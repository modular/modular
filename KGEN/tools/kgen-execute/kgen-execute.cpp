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
#include "Support/CommonCLOptions.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
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
  KGEN::ExecutionEngine &execEngine;
  KGENCLOptions &clOptions;

  LogicalResult operator()(MLIRContext *ctx, llvm::SourceMgr &sourceMgr) const {
    DialectRegistry registry;
    // Don't need LIT here.
    registry.insert<KGEN::KGENDialect, KGEN::POP::POPDialect,
                    mlir::index::IndexDialect, mlir::scf::SCFDialect>();
    mlir::registerLLVMDialectTranslation(registry);

    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();

    // Open the input file.
    OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, ctx);
    if (!module)
      return failure(clOptions.reportError("could not parse input file"));

    KGEN::ObjectCompiler compiler(".kgen_cache", *module);

    // Lower the input to an object.
    KGEN::TargetInfoAttr attr = KGEN::TargetInfoAttr::getForHost(ctx);

    // Produce a single standalone .o
    auto standaloneOr = compiler.produceStandaloneObject(
        attr,
        /*isJIT=*/clOptions.cmd == Command::kExecute);
    if (failed(standaloneOr))
      return failure();
    std::unique_ptr<llvm::MemoryBuffer> standaloneObject =
        std::move(*standaloneOr);

    if (clOptions.cmd == Command::kEmit)
      return clOptions.emitObject(std::move(standaloneObject));

    SymbolTable &symtab = compiler.getSymbolTable();
    auto lookupFunc = [&](StringRef funcName) -> ErrorOr<KGEN::FuncOp> {
      auto func = symtab.lookup<KGEN::FuncOp>(funcName);
      if (!func)
        return Error("could not find func '" + funcName + "'.");
      return func;
    };

    // Add the module to the execution engine.
    if (auto err = execEngine.add("exec", std::move(standaloneObject)))
      return failure(clOptions.reportError(err.getError()));

    for (const auto &k : clOptions.funcs) {
      auto funcOr = lookupFunc(k.name);
      if (funcOr.isError())
        return failure(clOptions.reportError(funcOr.getError()));

      KGEN::FuncOp func = *funcOr;
      auto compiledFuncOr = execEngine.lookup("exec", func);
      if (failed(compiledFuncOr))
        return failure(clOptions.reportError(compiledFuncOr.getError()));

      // And now we diverge.
      switch (clOptions.cmd) {
      case Command::kGenLibraryFile:
      case Command::kElaborate:
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

  // Enable command line options for various MLIR internals.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      clOptions.openInputFileOrExit();

  auto engineOr = KGEN::ExecutionEngine::create();
  if (engineOr.isError())
    clOptions.reportError(engineOr.getError());

  auto execEngine = std::move(*engineOr);

  // Provide a tool function that runs the requested ops, again, so we can
  // re-use it.
  auto toolFn = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                    raw_ostream &os) {
    return clOptions.configureMLIRContextAndSourceMgrAndExecute(
        std::move(chunkBuffer), ProcessBuffer{execEngine, clOptions});
  };

  // Process the file.
  return failed(
      splitAndProcessBuffer(std::move(inputFile), toolFn, llvm::outs(), false));
}
