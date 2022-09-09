//===- EmitKernelObject.cpp -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EmitKernelObject.h"
#include "KGEN/ExecutionEngine.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

LogicalResult
M::KGEN::emitObjectForKernel(ExecutionEngine &engine, FuncOp k,
                             const std::filesystem::path &objPath) {
  // Open the output file so we can emit to it.
  std::string err;
  auto outFile = mlir::openOutputFile(objPath.string(), &err);
  if (!outFile)
    return mlir::emitError(k.getLoc(), err);

  auto kernelOr = engine.lookup(k);
  if (failed(kernelOr))
    return mlir::emitError(k.getLoc(), "could not lookup the kernel '@" +
                                           k.getName() +
                                           "': " + kernelOr.getError());

  auto objOr = kernelOr->getObject();
  if (failed(objOr))
    return mlir::emitError(k.getLoc(),
                           "could not get the object for the kernel '@" +
                               k.getName() + "': " + objOr.getError());

  std::unique_ptr<llvm::MemoryBuffer> obj = std::move(*objOr);
  outFile->os().write(obj->getBufferStart(), obj->getBufferSize());
  outFile->keep();

  return mlir::success();
}
