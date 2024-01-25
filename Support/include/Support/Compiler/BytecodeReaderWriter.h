//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_BYTECODEREADERWRITER_H
#define SUPPORT_COMPILER_BYTECODEREADERWRITER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace M {
/// Read a single operation from the given bytecode file. Returns nullptr in the
/// case of failure.
OwningOpRef<Operation *>
readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                       const mlir::ParserConfig &config);
template <typename T>
OwningOpRef<T> readOpFromBytecodeFile(llvm::MemoryBufferRef buffer,
                                      const mlir::ParserConfig &config) {
  OwningOpRef<Operation *> rawOp = readOpFromBytecodeFile(buffer, config);
  if (OwningOpRef<T> op = dyn_cast_if_present<T>(*rawOp)) {
    rawOp.release();
    return std::move(op);
  }
  return OwningOpRef<T>();
}
} // namespace M

#endif // SUPPORT_COMPILER_BYTECODEREADERWRITER_H
