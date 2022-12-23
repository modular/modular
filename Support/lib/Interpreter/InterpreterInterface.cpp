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

/// Provide a virtual "invalid space" for the interpreter's memory. This is so
/// that an address of 0 can actually be considered a null pointer.
static constexpr intptr_t invalidMemOffset = 0x10000000;

intptr_t InterpreterState::allocateMemory(size_t size) {
  intptr_t addr = memory.size() + invalidMemOffset;
  memory.resize(memory.size() + size);
  return addr;
}

ErrorOr<MemoryReference> InterpreterState::getMemory(intptr_t addr) {
  if (!addr)
    return Error("null address");
  if (addr < invalidMemOffset ||
      static_cast<size_t>(addr - invalidMemOffset) > memory.size())
    return Error("address is out-of-bounds");
  return MemoryReference(*this, addr);
}

ErrorOr<void *> InterpreterState::materializeReference(intptr_t addr,
                                                       size_t size) {
  assert(addr >= invalidMemOffset ||
         static_cast<size_t>(addr - invalidMemOffset) <= memory.size() &&
             "invalid memory reference");
  addr -= invalidMemOffset;
  // Accessing memory at the end iterator is okay if the access size is zero.
  if (addr + size > memory.size())
    return Error("memory access size " + Twine(size) + " is out-of-bounds");
  return reinterpret_cast<void *>(memory.data() + addr);
}

//===----------------------------------------------------------------------===//
// MemoryableTypeInterface
//===----------------------------------------------------------------------===//

ErrorOr<void *> MemoryReference::get(size_t size) {
  return state.materializeReference(addr, size);
}

ErrorOrSuccess M::writeAttributeToMemory(TypedAttr value, MemoryReference ref) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    unsigned size;
    if (isa<IndexType>(intAttr.getType()))
      size = 8; // FIXME: Target abstraction needs to be factored out from KGEN.
    else
      size = llvm::divideCeil(cast<IntegerType>(intAttr.getType()).getWidth(),
                              CHAR_BIT);

    ErrorOr<void *> mem = ref.get(size);
    if (mem.isError())
      return mem.takeError();
    llvm::StoreIntToMemory(intAttr.getValue(),
                           reinterpret_cast<uint8_t *>(*mem), size);
    return success();
  }

  if (auto fpAttr = dyn_cast<FloatAttr>(value)) {
    unsigned size = cast<FloatType>(fpAttr.getType()).getWidth() / CHAR_BIT;
    ErrorOr<void *> mem = ref.get(size);
    if (!mem.isError())
      return mem.takeError();
    llvm::StoreIntToMemory(fpAttr.getValue().bitcastToAPInt(),
                           reinterpret_cast<uint8_t *>(*mem), size);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(value.getType()))
    return itf.writeTo(value, ref);

  return Error(mlir::debugString(value.getType()) +
               " does not implement MemoryableTypeInterface");
}

ErrorOr<TypedAttr> M::readAttributeFromMemory(Type type, MemoryReference ref) {
  if (isa<IndexType>(type)) {
    // FIXME: Target abstraction needs to be factored out from KGEN.
    ErrorOr<void *> mem = ref.get(8);
    if (failed(mem))
      return mem.takeError();
    APInt value(64, 0);
    llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), 8);
    return IntegerAttr::get(type, value);
  }

  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned size = llvm::divideCeil(intType.getWidth(), CHAR_BIT);
    ErrorOr<void *> mem = ref.get(size);
    if (mem.isError())
      return mem.takeError();
    APInt value(intType.getWidth(), 0);
    llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), size);
    return IntegerAttr::get(type, value);
  }

  if (auto fpType = dyn_cast<FloatType>(type)) {
    unsigned size = fpType.getWidth() / CHAR_BIT;
    ErrorOr<void *> mem = ref.get(size);
    if (mem.isError())
      return mem.takeError();
    APInt intVal(fpType.getWidth(), 0);
    llvm::LoadIntFromMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
    APFloat value(fpType.getFloatSemantics(), intVal);
    return FloatAttr::get(fpType, value);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(type))
    return itf.readFrom(ref);

  return Error(mlir::debugString(type) +
               " does not implement MemoryableTypeInterface");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.cpp.inc"
#include "Support/Interpreter/MemoryableTypeInterface.cpp.inc"
