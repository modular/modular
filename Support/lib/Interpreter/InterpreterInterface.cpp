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
  // Check for a null pointer access.
  if (!addr)
    return Error("null address");
  // Otherwise, check for an address that we know is invalid.
  if (addr < invalidMemOffset)
    return Error("address is invalid: " + Twine(addr));
  addr -= invalidMemOffset;
  // Accessing memory at the end iterator is okay if the access size is zero.
  // For example, if the memory size is 2, a memory access at index 2 is only
  // valid if the access size is 0. This is because the index points to the
  // beginning of the byte in memory. Check if the address exceeds the size of
  // allocated memory.
  if (static_cast<size_t>(addr) > memory.size())
    return Error("address is out-of-bounds: " + Twine(addr));
  // Now check if the access size will still be in-bounds.
  if (addr + size > memory.size())
    return Error("memory access size " + Twine(size) + " is out-of-bounds");
  return reinterpret_cast<void *>(memory.data() + addr);
}

ErrorOrSuccess InterpreterState::writeAttributeToMemory(intptr_t addr,
                                                        TypedAttr value) {
  if (isa<IntegerAttr, FloatAttr>(value)) {
    int64_t size =
        *DataLayoutInterface::getTypeSizeInBytes(target, value.getType());
    ErrorOr<void *> mem = getMemory(addr, size);
    if (mem.isError())
      return mem.takeError();

    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      llvm::StoreIntToMemory(intAttr.getValue(),
                             reinterpret_cast<uint8_t *>(*mem), size);
      return success();
    }

    llvm::StoreIntToMemory(cast<FloatAttr>(value).getValue().bitcastToAPInt(),
                           reinterpret_cast<uint8_t *>(*mem), size);
    return success();
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(value.getType()))
    return itf.writeTo(value, addr, *this);

  return Error(mlir::debugString(value.getType()) +
               " does not implement MemoryableTypeInterface");
}

ErrorOr<TypedAttr> InterpreterState::readAttributeFromMemory(intptr_t addr,
                                                             Type type) {
  if (isa<IndexType, IntegerType, FloatType>(type)) {
    int64_t size = *DataLayoutInterface::getTypeSizeInBytes(target, type);
    ErrorOr<void *> mem = getMemory(addr, size);
    if (mem.isError())
      return mem.takeError();

    if (auto intType = dyn_cast<IntegerType>(type)) {
      APInt value(intType.getWidth(), 0);
      llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), size);
      return IntegerAttr::get(type, value);
    }

    if (auto fpType = dyn_cast<FloatType>(type)) {
      APInt intVal(fpType.getWidth(), 0);
      llvm::LoadIntFromMemory(intVal, reinterpret_cast<uint8_t *>(*mem), size);
      APFloat value(fpType.getFloatSemantics(), intVal);
      return FloatAttr::get(fpType, value);
    }

    APInt value(64, 0);
    llvm::LoadIntFromMemory(value, reinterpret_cast<uint8_t *>(*mem), size);
    return IntegerAttr::get(type, value);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(type))
    return itf.readFrom(addr, *this);

  return Error(mlir::debugString(type) +
               " does not implement MemoryableTypeInterface");
}

/// Report an error with folding an operation.
static ErrorTree reportFoldError(Operation &op, ArrayRef<Attribute> operands,
                                 const Twine &prefix,
                                 const Twine &suffix = "") {
  std::string note;
  llvm::raw_string_ostream os(note);
  os << prefix << op.getName() << '(';
  llvm::interleaveComma(operands, os);
  os << ')' << suffix;
  return {op.getLoc(), Error(os.str())};
}

ErrorTreeOr<InterpreterState::RegionResult>
InterpreterState::evaluateRegion(DenseMap<Value, Attribute> &values,
                                 ArrayRef<TypedAttr> arguments,
                                 Region &region) {
  assert(llvm::hasSingleElement(region) && "TODO: support CFG regions");
  Block &body = region.front();

  // Map the region argument values.
  for (auto [arg, input] : llvm::zip(region.getArguments(), arguments))
    values.try_emplace(arg, input);

  // Interpret the IR in the single-block region without evaluating the
  // terminator.
  SmallVector<Attribute> operands;
  SmallVector<OpFoldResult> results;
  for (Operation &op : body.without_terminator()) {
    operands.clear();
    results.clear();
    for (Value operand : op.getOperands())
      operands.push_back(values.lookup(operand));

    // Check for an interpreter interface implementation.
    if (auto interpItf = dyn_cast<InterpreterOpInterface>(op)) {
      ErrorOrSuccess err = interpItf.interpret(operands, *this, results);
      if (err.isError()) {
        return std::move(
            reportFoldError(op, operands, "failed to interpret operation ")
                .addCause(op.getLoc(), err.takeError()));
      }
    } else {
      // Otherwise, try to use the operation folder.
      if (failed(op.fold(operands, results)))
        return reportFoldError(op, operands, "failed to fold operation ");
    }
    for (auto [i, result, output] :
         llvm::zip(llvm::seq<unsigned>(0, op.getNumResults()), results,
                   op.getResults())) {
      auto value = result.dyn_cast<Attribute>();
      if (!value) {
        return reportFoldError(op, operands, "operation evaluation ",
                               " did not return a value for result #" +
                                   Twine(i));
      }
      values.try_emplace(output, value);
    }
  }

  // Collect the constant values of the operands and return them.
  RegionResult result;
  result.terminator = body.getTerminator();
  for (Value operand : body.getTerminator()->getOperands())
    result.operands.push_back(values.lookup(operand));
  return result;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.cpp.inc"
#include "Support/Interpreter/MemoryableTypeInterface.cpp.inc"
