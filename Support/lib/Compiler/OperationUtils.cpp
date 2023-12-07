//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
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
