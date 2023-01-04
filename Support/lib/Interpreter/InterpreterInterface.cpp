//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

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
static ErrorTree reportFoldError(Operation *op, ArrayRef<Attribute> operands,
                                 const Twine &prefix,
                                 const Twine &suffix = "") {
  std::string note;
  llvm::raw_string_ostream os(note);
  os << prefix << op->getName() << '(';
  llvm::interleaveComma(operands, os);
  os << ')' << suffix;
  return {op->getLoc(), Error(os.str())};
}

//===----------------------------------------------------------------------===//
// Interpreter Implementation

/// Interpret a call operation.
static void interpretCallOp(mlir::CallOpInterface op,
                            ArrayRef<Attribute> operands,
                            InterpreterState &state) {
  auto callee = op.getCallableForCallee().get<SymbolRefAttr>();
  Region &body = state.lookupFunctionBody(callee);

  // Function regions are isolated from above, so push a new stack frame. Then,
  // transfer control flow to the beginning of the function body.
  state.pushFrame(op);
  state.transferControlFlowTo(&body.front(), operands);
}

/// Interpreter a return-like operation.
static void interpretReturnOp(Operation *op, ArrayRef<Attribute> operands,
                              InterpreterState &state) {
  // Pop the current frame and transfer control flow back to the call operation,
  // using the operands of the return as the results of the call.
  Operation *call = state.popFrame();
  state.setReturnValues(operands);
  state.transferControlFlowTo(call);
}

/// Interpreter a generic operation by trying to use its operation folder.
static ErrorTreeOr<SuccessType>
interpretOpWithFolder(Operation *op, ArrayRef<Attribute> operands,
                      InterpreterState &state) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(operands, results)))
    return reportFoldError(op, operands, "failed to fold operation ");
  for (auto [i, result, output] : llvm::zip(
           llvm::seq<int>(0, op->getNumResults()), results, op->getResults())) {
    auto value = result.dyn_cast<Attribute>();
    if (!value)
      return reportFoldError(op, operands, "operation evaluation ",
                             " did not return a value for result #" + Twine(i));
    state.mapOrOverwrite(output, value);
  }
  return success();
}

ErrorTreeOr<SmallVector<Attribute>>
InterpreterState::startInterpreterAt(Region &region,
                                     ArrayRef<Attribute> arguments) {
  // Push an empty stack frame and map the region arguments.
  stack.emplace_back(nullptr);
  transferControlFlowTo(&region.front(), arguments);

  // Run the interpreter.
  return runInterpreter();
}

ErrorTreeOr<SmallVector<Attribute>> InterpreterState::runInterpreter() {
  SmallVector<Attribute> operands;
  while (op) {
    Operation *prev = op;

    operands.clear();
    // Lookup the operands of the current operation.
    for (Value operand : op->getOperands())
      operands.push_back(lookupValue(operand));

    // Check for a builtin interface.
    if (auto call = dyn_cast<mlir::CallOpInterface>(op)) {
      interpretCallOp(call, operands, *this);
    } else if (op->hasTrait<OpTrait::ReturnLike>()) {
      interpretReturnOp(op, operands, *this);

      // Check for an interpreter interface implementation.
    } else if (auto interpItf = dyn_cast<InterpreterOpInterface>(op)) {
      ErrorTreeOr<SuccessType> err = interpItf.interpret(operands, *this);
      if (err.isError())
        return reportFoldError(op, operands, "failed to interpret operation ")
            .addCause(err.takeError());

      // Otherwise, try to use the operation folder.
    } else {
      ErrorTreeOr<SuccessType> result =
          interpretOpWithFolder(op, operands, *this);
      if (result.isError())
        return result.takeError();
    }

    // If the operation has not changed, advance to the next operation. If the
    // current operation is a terminator, return an error.
    if (prev == op) {
      if (!op->getNextNode())
        return ErrorTree(op->getLoc(),
                         "terminator did not transfer control flow");
      op = op->getNextNode();
    }
  }

  // The stack frame must be empty.
  assert(stack.empty() && "exiting interpreter with remaining stack frames");
  return takeReturnValues();
}

void InterpreterState::transferControlFlowTo(Operation *target) {
  op = target;
  if (op) {
    mapResults(takeReturnValues());
    op = op->getNextNode();
  }
}

void InterpreterState::transferControlFlowTo(Block *target,
                                             ArrayRef<Attribute> arguments) {
  for (auto [arg, value] : llvm::zip(target->getArguments(), arguments))
    mapOrOverwrite(arg, value);
  op = &target->front();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.cpp.inc"
#include "Support/Interpreter/MemoryableTypeInterface.cpp.inc"
