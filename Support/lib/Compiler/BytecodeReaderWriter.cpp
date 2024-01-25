//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include <mlir/IR/BuiltinOps.h>

using namespace M;

OwningOpRef<Operation *>
M::readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                          const mlir::ParserConfig &config) {
  Block b;
  if (failed(mlir::readBytecodeFile(buffer, &b, config)) ||
      !llvm::hasSingleElement(b))
    return nullptr;

  // Take ownership of the op from the block.
  Operation *op = &b.front();
  op->remove();
  return op;
}
