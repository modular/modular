//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "JITSupport.h"
#include "KGEN/KGENDialect/KGENUtils.h"

using namespace M;
using namespace KGEN;

llvm::JITSymbolFlags
KGEN::getFlagsForExportedSymbol(const ExportedSymbol &symbol) {
  return (symbol.isData ? llvm::JITSymbolFlags::None
                        : llvm::JITSymbolFlags::Callable) |
         llvm::JITSymbolFlags::Exported;
}

llvm::JITSymbolFlags KGEN::getGlobalFnSymbolFlags() {
  return llvm::JITSymbolFlags::Callable | llvm::JITSymbolFlags::Exported |
         llvm::JITSymbolFlags::Weak;
}
