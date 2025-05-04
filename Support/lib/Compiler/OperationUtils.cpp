//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"

std::string M::getUniqueSymbolName(std::string baseName, SymbolTable &symtab,
                                   unsigned &counter) {
  std::string uniqueName = baseName;
  while (symtab.lookup(uniqueName))
    uniqueName = (baseName + "_" + Twine(counter++)).str();
  return uniqueName;
}

std::string M::getUniqueSymbolName(std::string baseName, SymbolTable &symtab) {
  unsigned counter = 0;
  return getUniqueSymbolName(baseName, symtab, counter);
}

std::string M::getFlattenedSymbolName(SymbolRefAttr symbol) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(symbol))
    return flatSym.getValue().str();

  // Flatten the symbol name into a single string.
  SmallString<32> name = symbol.getRootReference().getValue();
  llvm::raw_svector_ostream nameOS(name);
  for (FlatSymbolRefAttr sym : symbol.getNestedReferences())
    nameOS << "::" << sym.getValue();
  return nameOS.str().str();
}

bool M::isCIdentifier(StringRef ident) {
  if (ident.empty() || !(llvm::isAlpha(ident[0]) || ident[0] == '_'))
    return false;
  return llvm::all_of(ident.drop_front(),
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

M::WalkResult M::OpRegionBlockWalker::walk(Operation *op) {
  if (walkOp) {
    WalkResult opWalkResult = walkOp(op);
    if (opWalkResult.wasInterrupted() || opWalkResult.wasSkipped())
      return opWalkResult;
  }

  for (Region &region : llvm::make_early_inc_range(op->getRegions())) {
    WalkResult regionWalkResult = walk(&region);
    if (regionWalkResult.wasInterrupted())
      return regionWalkResult;
  }
  return WalkResult::advance();
}

M::WalkResult M::OpRegionBlockWalker::walk(Region *region) {
  if (walkRegion) {
    WalkResult regionWalkResult = walkRegion(region);
    if (regionWalkResult.wasInterrupted() || regionWalkResult.wasSkipped())
      return regionWalkResult;
  }

  for (Block &block : llvm::make_early_inc_range(region->getBlocks())) {
    WalkResult blockWalkResult = walk(&block);
    if (blockWalkResult.wasInterrupted())
      return blockWalkResult;
  }
  return WalkResult::advance();
}

M::WalkResult M::OpRegionBlockWalker::walk(Block *block) {
  if (walkBlock) {
    WalkResult blockWalkResult = walkBlock(block);
    if (blockWalkResult.wasInterrupted() || blockWalkResult.wasSkipped())
      return blockWalkResult;
  }

  for (Operation &op : llvm::make_early_inc_range(block->getOperations())) {
    WalkResult opWalkResult = walk(&op);
    if (opWalkResult.wasInterrupted())
      return opWalkResult;
  }
  return WalkResult::advance();
}
