//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Cache/CachePasses/CachePasses.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/Context.h"
#include "Support/Init/Init.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace LLCL;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  Cache::CacheDialect>();
  mlir::registerCanonicalizer();

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "cache-opt", Init::Options().withRuntimeOptions(
                       LLCL::RuntimeOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(registry, *ctxOr);

  Cache::registerCachePasses(*(*ctxOr)->get<LLCL::Runtime>());

  return failed(mlir::MlirOptMain(argc, argv, "cache-opt", registry));
}
