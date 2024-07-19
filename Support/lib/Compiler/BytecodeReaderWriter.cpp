//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Buffer.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/SourceMgr.h"

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

OwningOpRef<Operation *>
M::readOpFromBytecodeFile(DenseResourceElementsAttr bytecodeAttr,
                          const mlir::ParserConfig &config) {
  mlir::AsmResourceBlob *blob = bytecodeAttr.getRawHandle().getBlob();
  if (!blob)
    return OwningOpRef<Operation *>();
  ArrayRef<char> bytecode = blob->getData();
  llvm::MemoryBufferRef bufferRef(StringRef(bytecode.begin(), bytecode.size()),
                                  "");

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::BytecodeReader reader(bufferRef, config, /*lazyLoad=*/false, sourceMgr);
  Block b;
  if (failed(reader.readTopLevel(&b)) || !llvm::hasSingleElement(b))
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
// writeModuleToBytecodeAttr
//===----------------------------------------------------------------------===//

DenseResourceElementsAttr M::writeModuleToBytecodeAttr(ModuleOp module) {
  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(module, *str)))
    return nullptr;

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  return createResourceAttr(module->getContext(), std::move(str),
                            "bytecode_" +
                                llvm::toHex(hash, /*LowerCase=*/true));
}

//===----------------------------------------------------------------------===//
// loadSymbolsFromBytecode
//===----------------------------------------------------------------------===//

LogicalResult M::loadSymbolsFromBytecode(
    Operation *op, mlir::BytecodeReader &reader,
    function_ref<bool(StringAttr)> existsFn,
    function_ref<void(Operation *, Operation *)> insertFn,
    const SymbolTable &bytecodeSymTab) {

  // Process the dependencies using a worklist.
  std::vector<Operation *> worklist;
  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();

    if (reader.isMaterializable(op)) {
      if (failed(reader.materialize(op, [&](Operation *op) { return true; })))
        return failure();
    }

    mlir::AttrTypeWalker walker;
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
    walker.addWalk([&](FlatSymbolRefAttr ref) {
      if (Operation *decl = extractDependency(ref.getAttr()))
        worklist.push_back(decl);
    });
    op->walk([&](Operation *op) {
      // Extract references to type declarations.
      walker.walk(op->getAttrDictionary());
      for (Type type : op->getResultTypes())
        walker.walk(type);
      for (Region &region : op->getRegions()) {
        for (Type type : region.getArgumentTypes())
          walker.walk(type);
      }
    });
  }

  return success();
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
