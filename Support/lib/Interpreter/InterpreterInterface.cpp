//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

InterpreterState::InterpreterState(MLIRContext *ctx, TargetInfoAttr target)
    : target(target),
      blobMgr(*ctx->getOrLoadDialect<MDialect>()
                   ->getRegisteredInterface<MBlobManagerInterface>()) {}

InterpreterState::InterpreterState(TargetInfoAttr target)
    : InterpreterState(target.getContext(), target) {}

/// Provide a virtual "invalid space" for the interpreter's memory. This is so
/// that an address of 0 can actually be considered a null pointer.
static constexpr int64_t invalidMemOffset = 10'000'000;

int64_t InterpreterState::allocateMemory(size_t size, size_t align,
                                         MemoryKind kind) {
  // Pick the base address of the new blob.
  int64_t baseAddr = invalidMemOffset;
  if (!memory.empty()) {
    MemoryBlob &last = memory.back();
    baseAddr = last.baseAddr + last.size;
  }
  // Ensure the base address is aligned.
  baseAddr = llvm::alignTo(baseAddr, align);

  // Create the blob with aligned memory.
  MemoryBlob blob{
      kind, baseAddr, size, align, {alignedAlloc(align, size), &alignedFree}};
  memset(blob.memory.get(), 0, size);
  memory.push_back(std::move(blob));
  return baseAddr;
}

ErrorOr<InterpreterState::MemoryBlob &>
InterpreterState::getBlob(int64_t addr) {
  // Check for a null pointer access.
  if (!addr)
    return Error("null address");
  // Otherwise, check for an address that we know is invalid.
  if (addr < invalidMemOffset)
    return Error("address is invalid: " + Twine(addr));

  // Binary search for the blob that corresponds to the address.
  auto it =
      llvm::lower_bound(memory, addr, [](const MemoryBlob &blob, int64_t addr) {
        return blob.baseAddr + blob.size <= static_cast<size_t>(addr);
      });
  if (it == memory.end())
    return Error("address is out-of-bounds: " + Twine(addr));
  return *it;
}

ErrorOr<void *> InterpreterState::getMemory(int64_t addr, size_t size) {
  ErrorOr<MemoryBlob &> blob = getBlob(addr);
  if (blob.isError())
    return blob.takeError();
  int64_t offset = addr - blob->baseAddr;

  // Accessing memory at the end iterator is okay if the access size is zero.
  // For example, if the memory size is 2, a memory access at index 2 is only
  // valid if the access size is 0. This is because the index points to the
  // beginning of the byte in memory. Check if the address exceeds the size of
  // allocated memory.
  if (static_cast<size_t>(offset) > blob->size)
    return Error("address is out-of-bounds: " + Twine(addr));
  // Now check if the access size will still be in-bounds.
  if (offset + size > blob->size)
    return Error("memory access size " + Twine(size) + " is out-of-bounds");
  return (uint8_t *)blob->memory.get() + offset;
}

ErrorOrSuccess InterpreterState::writeAttributeToMemory(int64_t addr,
                                                        TypedAttr value) {
  if (isa<IntegerAttr, FloatAttr>(value)) {
    int64_t size =
        *DataLayoutInterface::getTypeStoreSize(target, value.getType());
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

ErrorOr<TypedAttr> InterpreterState::readAttributeFromMemory(int64_t addr,
                                                             Type type) {
  if (isa<IndexType, IntegerType, FloatType>(type)) {
    int64_t size = *DataLayoutInterface::getTypeStoreSize(target, type);
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

ErrorOrSuccess InterpreterState::exchangeInterpreterMemory(
    Region &entry, MutableArrayRef<Attribute> results) {
  // Use the parent function name as the base name for materialized resources.
  auto symbol = cast<mlir::SymbolOpInterface>(entry.getParentOp());
  std::string baseName = (symbol.getName() + "_mem").str();

  // Deduplicate references to the same allocation blob.
  DenseMap<MemoryBlob *, MemoryHandle> blobs;
  auto getOrMaterialize = [&](MemoryBlob &blob) {
    if (auto it = blobs.find(&blob); it != blobs.end())
      return it->second;
    MemoryHandle hdl = blobMgr.insert(
        baseName,
        mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
            ArrayRef<char>((char *)blob.memory.get(), blob.size), blob.align));
    blobs.try_emplace(&blob, hdl);
    return hdl;
  };

  // Replace raw pointers in the results except for null pointers. Error if a
  // reference to invalid memory is returned from the function.
  mlir::AttrTypeReplacer replacer;
  std::optional<Error> err;
  replacer.addReplacement(
      [&](PointerAttr ptr) -> std::pair<Attribute, WalkResult> {
        // Allow null pointers to persist.
        if (ptr.getAddr() == 0)
          return {ptr, WalkResult::advance()};
        // Find the memory point this address points to.
        ErrorOr<MemoryBlob &> blob = getBlob(ptr.getAddr());
        if (blob.isError()) {
          err = blob.takeError();
          return {ptr, WalkResult::interrupt()};
        }
        // Returning stack memory invalid.
        if (blob->kind == MemoryKind::Stack) {
          err = Error("returning reference to stack memory");
          return {ptr, WalkResult::interrupt()};
        }
        // Materialize the blob.
        MemoryHandle hdl = getOrMaterialize(*blob);
        return {MemRefAttr::get(hdl, ptr.getAddr() - blob->baseAddr, blob->kind,
                                ptr.getType()),
                WalkResult::advance()};
      });

  for (Attribute &result : results) {
    result = replacer.replace(result);
    if (err)
      return std::move(*err);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interpreter Implementation

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

/// Interpret a call operation.
static ErrorTreeOrSuccess interpretCallOp(mlir::CallOpInterface op,
                                          ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  auto callee = op.getCallableForCallee().get<SymbolRefAttr>();
  auto bodyOr = state.lookupFunctionBody(callee);
  if (bodyOr.isError())
    return ErrorTree(op->getLoc(), bodyOr.takeError());

  Region &body = **bodyOr;

  // Function regions are isolated from above, so push a new stack frame. Then,
  // transfer control flow to the beginning of the function body.
  state.pushFrame(op, body.getParentOp());
  state.transferControlFlowTo(&body.front(), operands);
  return success();
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
static ErrorTreeOrSuccess interpretOpWithFolder(Operation *op,
                                                ArrayRef<Attribute> operands,
                                                InterpreterState &state) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(operands, results)))
    return reportFoldError(op, operands, "failed to fold operation ");
  for (auto [i, result, output] : llvm::zip(
           llvm::seq<int>(0, op->getNumResults()), results, op->getResults())) {
    if (auto value = result.dyn_cast<Attribute>())
      state.mapOrOverwrite(output, value);
    else
      state.mapOrOverwrite(output, state.lookupValue(result.get<Value>()));
  }
  return success();
}

ErrorTreeOr<SmallVector<Attribute>>
InterpreterState::startInterpreterAt(Region &region,
                                     ArrayRef<Attribute> arguments) {
  // Push an empty stack frame and map the region arguments.
  pushFrame(nullptr, region.getParentOp());
  transferControlFlowTo(&region.front(), arguments);

  // Reset the interpret to a clean state.
  auto resetState = llvm::make_scope_exit([&] {
    stack.clear();
    memory.clear();
    returnValues.reset();
  });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result = runInterpreter();
  if (result) {
    SmallVector<Attribute> results = result.takeValue();
    if (ErrorOrSuccess err = exchangeInterpreterMemory(region, results);
        err.isError())
      return ErrorTree(region.getLoc(), err.takeError());
    return results;
  }

  // The interpreter ran into an error. Report an error using a stacktrace.
  ErrorTree error = result.takeError();
  for (const StackFrame &frame : llvm::reverse(stack)) {
    StringRef funcName = cast<mlir::SymbolOpInterface>(frame.func).getName();
    error = ErrorTree(frame.func->getLoc(),
                      Error("failed to interpret function @" + funcName),
                      std::move(error));
    if (frame.origin)
      error = ErrorTree(frame.origin->getLoc(),
                        Error("failed to evaluate call"), std::move(error));
  }
  return std::move(error);
}

ErrorTreeOr<SmallVector<Attribute>> InterpreterState::runInterpreter() {
  SmallVector<Attribute> operands;
  while (pc) {
    Operation *prev = pc;

    operands.clear();
    // Lookup the operands of the current operation.
    for (Value operand : pc->getOperands())
      operands.push_back(lookupValue(operand));

    // Check for a builtin interface.
    if (auto call = dyn_cast<mlir::CallOpInterface>(pc)) {
      auto err = interpretCallOp(call, operands, *this);
      if (err.isError()) {
        return reportFoldError(pc, operands, "failed to interpret call ")
            .addCause(err.takeError());
      }
    } else if (pc->hasTrait<OpTrait::ReturnLike>()) {
      interpretReturnOp(pc, operands, *this);

      // Check for an interpreter interface implementation.
    } else if (auto interpItf = dyn_cast<InterpreterOpInterface>(pc)) {
      ErrorTreeOrSuccess err = interpItf.interpret(operands, *this);
      if (err.isError())
        return reportFoldError(pc, operands, "failed to interpret operation ")
            .addCause(err.takeError());

      // Otherwise, try to use the operation folder.
    } else {
      ErrorTreeOrSuccess result = interpretOpWithFolder(pc, operands, *this);
      if (result.isError())
        return result.takeError();
    }

    // If the operation has not changed, advance to the next operation. If the
    // current operation is a terminator, return an error.
    if (prev == pc) {
      if (!pc->getNextNode())
        return ErrorTree(pc->getLoc(),
                         "terminator did not transfer control flow");
      pc = pc->getNextNode();
    }
  }

  // The stack frame must be empty.
  assert(stack.empty() && "exiting interpreter with remaining stack frames");
  return takeReturnValues();
}

void InterpreterState::transferControlFlowTo(Operation *target) {
  pc = target;
  if (pc) {
    mapResults(takeReturnValues());
    pc = pc->getNextNode();
  }
}

void InterpreterState::transferControlFlowTo(Block *target,
                                             ArrayRef<Attribute> arguments) {
  for (auto [arg, value] : llvm::zip(target->getArguments(), arguments))
    mapOrOverwrite(arg, value);
  pc = &target->front();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.cpp.inc"
#include "Support/Interpreter/MemoryableTypeInterface.cpp.inc"
