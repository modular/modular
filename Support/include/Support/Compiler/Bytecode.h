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
/// ODS helper for parsing an enum.
template <typename T>
LogicalResult readEnum(mlir::DialectBytecodeReader &reader, T &result) {
  uint64_t value;
  if (failed(reader.readVarInt(value)))
    return failure();
  result = static_cast<T>(value);
  return success();
}

/// ODS helper for parsing an array of enums.
template <typename T>
LogicalResult readEnumArray(mlir::DialectBytecodeReader &reader,
                            SmallVectorImpl<T> &result) {
  return reader.readList(
      result, [&](T &value) { return M::readEnum<T>(reader, value); });
}

/// ODS helper for printing an enum.
template <typename T>
void writeEnum(mlir::DialectBytecodeWriter &writer, T value) {
  writer.writeVarInt(static_cast<uint64_t>(value));
}

/// ODS helper for printing an array of enums.
template <typename T>
void writeEnumArray(mlir::DialectBytecodeWriter &writer, ArrayRef<T> values) {
  writer.writeList(values, [&](T value) { writeEnum(writer, value); });
}

} // namespace M

#endif // SUPPORT_COMPILER_BYTECODE_H
