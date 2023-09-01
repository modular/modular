//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef EMIT_FUNC_HEADER_H
#define EMIT_FUNC_HEADER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/MapVector.h"

namespace M::KGEN {
class ObjectCompiler;
struct ExportedSymbol;

/// Emit the header for a set of exported functions.
LogicalResult
emitHeader(SymbolTable &symtab,
           const llvm::MapVector<StringAttr, ExportedSymbol> &exportedSymbols,
           ObjectCompiler &compiler, StringRef filename);
} // namespace M::KGEN

#endif // EMIT_FUNC_HEADER_H
