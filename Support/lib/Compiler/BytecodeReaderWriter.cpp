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

//===----------------------------------------------------------------------===//
// loadSymbolsFromBytecode
//===----------------------------------------------------------------------===//

LogicalResult M::loadSymbolsFromBytecode(
    Operation *op, mlir::BytecodeReader &reader,
    function_ref<bool(StringAttr)> existsFn,
    function_ref<void(Operation *, Operation *)> insertFn,
    const SymbolTable &bytecodeSymTab) {
  if (reader.isMaterializable(op)) {
    if (failed(reader.materialize(op, [&](Operation *op) { return true; })))
      return failure();
  }

  // Extract a dependency from the bytecode module and move it into the main
  // module, if it doesn't already exist there. If a symbol was moved, return
  // it.
  auto extractDependency = [&](StringAttr name) -> Operation * {
    // Don't move the symbol if it already exists in the main module.
    if (existsFn(name))
      return nullptr;
    Operation *symbol = bytecodeSymTab.lookup(name);
    assert(symbol && "expected valid symbol reference");

    // Move the symbol into the main module.
    insertFn(symbol, op);
    return symbol;
  };

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) -> WalkResult {
    if (Operation *decl = extractDependency(ref.getAttr()))
      return loadSymbolsFromBytecode(decl, reader, existsFn, insertFn,
                                     bytecodeSymTab);
    return WalkResult::advance();
  });
  auto result = op->walk([&](Operation *op) {
    // Extract references to type declarations.
    if (walker.walk(op->getAttrDictionary()).wasInterrupted())
      return WalkResult::interrupt();
    for (Type type : op->getResultTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        if (walker.walk(type).wasInterrupted())
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult M::loadSymbolsFromBytecode(Operation *op,
                                         mlir::BytecodeReader &reader,
                                         SymbolTable &symTab,
                                         const SymbolTable &bytecodeSymTab) {
  return loadSymbolsFromBytecode(
      op, reader, [&](StringAttr name) -> bool { return symTab.lookup(name); },
      [&](Operation *op, Operation *after) {
        op->moveAfter(after);
        symTab.insert(op);
      },
      bytecodeSymTab);
}
