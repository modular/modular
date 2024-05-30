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

//===----------------------------------------------------------------------===//
// Readers and Writers
//===----------------------------------------------------------------------===//

/// ODS helper for parsing an enum.
template <typename T>
LogicalResult readIntegral(mlir::DialectBytecodeReader &reader, T &result) {
  uint64_t value;
  if (failed(reader.readVarInt(value)))
    return failure();
  result = static_cast<T>(value);
  return success();
}

/// ODS helper for parsing an array of enums.
template <typename T>
LogicalResult readIntegralArray(mlir::DialectBytecodeReader &reader,
                                SmallVectorImpl<T> &result) {
  return reader.readList(
      result, [&](T &value) { return M::readIntegral<T>(reader, value); });
}

/// ODS helper for printing an enum.
template <typename T>
void writeIntegral(mlir::DialectBytecodeWriter &writer, T value) {
  writer.writeVarInt(static_cast<uint64_t>(value));
}

/// ODS helper for printing an array of enums.
template <typename T>
void writeIntegralArray(mlir::DialectBytecodeWriter &writer,
                        ArrayRef<T> values) {
  writer.writeList(values, [&](T value) { writeIntegral(writer, value); });
}

//===----------------------------------------------------------------------===//
// WrappedAttrType
//===----------------------------------------------------------------------===//

/// This class provides a bytecode specific wrapper that invokes a special
/// bytecode `get` method of an attribute or type. This is useful for types that
/// override their `get` methods to perform additional logic (which we want to
/// avoid for bytecode, where we already know the values are in the canonical
/// form).
template <typename T>
struct WrappedAttrType : public T {
  using T::T;

  template <typename... Ts>
  static T get(Ts &&...ts) {
    return T::getFromBytecode(std::forward<Ts>(ts)...);
  }
};

} // namespace M

#endif // SUPPORT_COMPILER_BYTECODE_H
