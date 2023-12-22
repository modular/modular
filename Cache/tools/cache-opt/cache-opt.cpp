//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "Cache/CachePasses/CachePasses.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
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

  std::unique_ptr<Runtime> runtime = createRuntime(RuntimeOptions().forDebug());

  Cache::registerCachePasses(*runtime);

  return failed(mlir::MlirOptMain(argc, argv, "cache-opt", registry));
}
