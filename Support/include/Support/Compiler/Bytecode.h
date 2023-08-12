//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_BYTECODE_H
#define SUPPORT_COMPILER_BYTECODE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Bytecode/BytecodeImplementation.h"

namespace M {
/// ODS helper for parsing enums.
template <typename T>
LogicalResult readEnum(mlir::DialectBytecodeReader &reader, T &result) {
  uint64_t value;
  if (failed(reader.readVarInt(value)))
    return failure();
  result = static_cast<T>(value);
  return success();
}
} // namespace M

#endif // SUPPORT_COMPILER_BYTECODE_H
