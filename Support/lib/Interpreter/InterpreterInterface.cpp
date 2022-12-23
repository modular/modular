//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.h"
#include "Support/MDialect/MTypeInterfaces.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

/// Provide a virtual "invalid space" for the interpreter's memory. This is so
/// that an address of 0 can actually be considered a null pointer.
static constexpr intptr_t invalidMemOffset = 0x10000000;

intptr_t InterpreterState::allocateMemory(size_t size) {
  intptr_t addr = memory.size() + invalidMemOffset;
  memory.resize(memory.size() + size);
  return addr;
}

ErrorOr<void *> InterpreterState::getMemory(intptr_t addr, size_t size) {
  if (!addr)
    return Error("null address");
  if (addr < invalidMemOffset ||
      static_cast<size_t>(addr - invalidMemOffset) > memory.size())
    return Error("address is out-of-bounds");
  addr -= invalidMemOffset;
  // Accessing memory at the end iterator is okay if the access size is zero.
  // For example, if the memory size is 2, a memory access at index 2 is only
  // valid if the access size is 0. This is because the index points to the
  // beginning of the byte in memory.
  if (addr + size > memory.size())
    return Error("memory access size " + Twine(size) + " is out-of-bounds");
  return reinterpret_cast<void *>(memory.data() + addr);
}

ErrorOrSuccess InterpreterState::writeAttributeToMemory(intptr_t addr,
                                                        TypedAttr value) {
  Optional<int64_t> size =
      DataLayoutInterface::getTypeSizeInBytes(target, value.getType());
  if (!size)
    return Error("could not query the size of the type to write");
  ErrorOr<void *> mem = getMemory(addr, *size);
  if (mem.isError())
    return mem.takeError();

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    llvm::StoreIntToMemory(intAttr.getValue(),
                           reinterpret_cast<uint8_t *>(*mem), *size);
    return success();
  }

  if (auto fpAttr = dyn_cast<FloatAttr>(value)) {
    llvm::StoreIntToMemory(fpAttr.getValue().bitcastToAPInt(),
                           reinterpret_cast<uint8_t *>(*mem), *size);
    return success();
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(value.getType()))
    return itf.writeTo(value, *size, *mem);

  return Error(mlir::debugString(value.getType()) +
               " does not implement MemoryableTypeInterface");
}

ErrorOr<TypedAttr> InterpreterState::readAttributeFromMemory(intptr_t addr,
                                                             Type type) {
  Optional<int64_t> size =
      DataLayoutInterface::getTypeSizeInBytes(target, type);
  if (!size)
    return Error("could not query the size of the type to write");
  ErrorOr<void *> mem = getMemory(addr, *size);
  if (mem.isError())
    return mem.takeError();

  if (isa<IndexType>(type)) {
    APInt value(64, 0);
    llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), *size);
    return IntegerAttr::get(type, value);
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    APInt value(intType.getWidth(), 0);
    llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), *size);
    return IntegerAttr::get(type, value);
  }

  if (auto fpType = dyn_cast<FloatType>(type)) {
    APInt intVal(fpType.getWidth(), 0);
    llvm::LoadIntFromMemory(intVal, reinterpret_cast<uint8_t *>(*mem), *size);
    APFloat value(fpType.getFloatSemantics(), intVal);
    return FloatAttr::get(fpType, value);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(type))
    return itf.readFrom(*size, *mem);

  return Error(mlir::debugString(type) +
               " does not implement MemoryableTypeInterface");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.cpp.inc"
#include "Support/Interpreter/MemoryableTypeInterface.cpp.inc"
