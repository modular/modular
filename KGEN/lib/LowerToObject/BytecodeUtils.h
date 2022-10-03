//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_BYTECODE_UTILS_H
#define KGEN_BYTECODE_UTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace M::KGEN {
/// Replace a symbol in the IR with a symbol of the same name read from the
/// bytecode in `buf`.
FailureOr<Operation *>
replaceSymbolFromBytecode(mlir::SymbolOpInterface toReplace,
                          mlir::SymbolTable &symtab, llvm::MemoryBufferRef buf);
} // namespace M::KGEN

#endif // KGEN_BYTECODE_UTILS_H
