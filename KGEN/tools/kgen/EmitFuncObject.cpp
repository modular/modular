//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EmitFuncObject.h"
#include "KGEN/ExecutionEngine.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

LogicalResult M::KGEN::emitObjectForFunc(ExecutionEngine &engine, FuncOp fn,
                                         const std::filesystem::path &objPath) {
  // Open the output file so we can emit to it.
  std::string err;
  auto outFile = mlir::openOutputFile(objPath.string(), &err);
  if (!outFile)
    return mlir::emitError(fn.getLoc(), err);

  auto funcOr = engine.lookup(fn);
  if (failed(funcOr))
    return mlir::emitError(fn.getLoc(), "could not lookup the func '@" +
                                            fn.getName() +
                                            "': " + funcOr.getError());

  auto objOr = funcOr->getObject();
  if (failed(objOr))
    return mlir::emitError(fn.getLoc(),
                           "could not get the object for the func '@" +
                               fn.getName() + "': " + objOr.getError());

  std::unique_ptr<llvm::MemoryBuffer> obj = std::move(*objOr);
  outFile->os().write(obj->getBufferStart(), obj->getBufferSize());
  outFile->keep();

  return mlir::success();
}
