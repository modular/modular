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

LogicalResult M::writeAttrToBytecodeFile(Attribute attr, raw_ostream &os) {
  // There isn't an easy way to do this other than create a temporary operation
  // and write it out.
  OwningOpRef<ModuleOp> tempBytecodeOp =
      ModuleOp::create(UnknownLoc::get(attr.getContext()));
  (*tempBytecodeOp)->setAttr("bytecode.attr", attr);
  return mlir::writeBytecodeToFile(tempBytecodeOp.get(), os);
}

Attribute M::readAttrFromBytecodeFile(llvm::MemoryBufferRef buffer,
                                      MLIRContext *ctx) {
  OwningOpRef<Operation *> rawOp = readOpFromBytecodeFile(
      buffer, mlir::ParserConfig(ctx, /*verifyAfterParse=*/false));
  if (!rawOp)
    return nullptr;
  // Pull out the encoded attribute from the operation.
  return rawOp->getAttr("bytecode.attr");
}
