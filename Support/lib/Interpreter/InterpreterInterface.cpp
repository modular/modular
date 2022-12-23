//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

size_t InterpreterState::allocateMemory(unsigned numElements, Type type) {
  size_t addr = memory.size();
  memory.insert(memory.end(), numElements, {TypedAttr(), type});
  return addr;
}

ErrorOr<TypedAttr> InterpreterState::readMemory(size_t addr, Type type) const {
  if (addr >= memory.size())
    return Error("read at address " + Twine(addr) + " is out of bounds");
  const std::pair<TypedAttr, Type> result = memory[addr];
  if (result.second != type)
    return Error("read of type " + mlir::debugString(type) +
                 " on memory of type " + mlir::debugString(result.second));
  return result.first;
}

ErrorOrSuccess InterpreterState::writeMemory(size_t addr, TypedAttr value) {
  if (addr >= memory.size())
    return Error("write at address " + Twine(addr) + " is out of bounds");
  std::pair<TypedAttr, Type> &result = memory[addr];
  if (result.second != value.getType())
    return Error("write of type " + mlir::debugString(value.getType()) +
                 " to memory of type " + mlir::debugString(result.second));
  result.first = value;
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.cpp.inc"
