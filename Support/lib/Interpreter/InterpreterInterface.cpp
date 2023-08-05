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

/// Provide a virtual "invalid space" for the interpreter's memory. This is so
/// that an address of 0 can actually be considered a null pointer.
static constexpr int64_t kHeapBaseAddr = 1'000'000'000;
/// Give the memory segments a healthy amount of maximum memory.
static constexpr int64_t kTableSize = 4'000'000'000;
/// Put the stack after the heap.
/// FIXME: The stack pointer does not grow downwards like in every system.
static constexpr int64_t kStackBaseAddr = kHeapBaseAddr + kTableSize;

InterpreterState::InterpreterState(MLIRContext *ctx, TargetInfoAttr target)
    : target(target), blobMgr(MemoryHandle::getManagerInterface(ctx)),
      heapMemory(MemoryKind::Heap, kHeapBaseAddr, kHeapBaseAddr + kTableSize),
      stackMemory(MemoryKind::Stack, kStackBaseAddr,
                  kStackBaseAddr + kTableSize) {}

InterpreterState::InterpreterState(TargetInfoAttr target)
    : InterpreterState(target.getContext(), target) {}

InterpreterState::MemoryBlob::MemoryBlob(int64_t baseAddr, size_t size,
                                         size_t align)
    : baseAddr(baseAddr), size(size), align(align),
      memory(alignedAlloc(align, size), &alignedFree) {}

ErrorOrSuccess InterpreterState::MemoryBlob::setPointerRegion(
    int64_t offset, int64_t regionSize, int64_t pointerSize,
    bool writePointer) {
  if (!pointerRegions) {
    if (!writePointer)
      return success();
    pointerRegions.emplace(size);
    pointerRegions->set(offset);
    return success();
  }

  // The write clobbers a pointer region if a bit is set between
  // `(offset - pointerSize, offset)` or between
  // `(offset + size - pointerSize, offset + size)`, indicating partial
  // overwrite of a pointer region.
  if (pointerRegions->find_first_in(
          std::max<int64_t>(0, offset - pointerSize + 1), offset) != -1 ||
      pointerRegions->find_first_in(
          std::max<int64_t>(0, offset + regionSize - pointerSize + 1),
          offset + regionSize) != -1)
    return Error("write clobbers a pointer region");

  if (!writePointer)
    pointerRegions->reset(offset, offset + regionSize);
  else
    pointerRegions->set(offset);
  return success();
}

ErrorOr<InterpreterState::MemoryBlob &>
InterpreterState::MemoryTable::addBlob(size_t size, size_t align) {
  // Pick the base address of the new blob.
  int64_t baseAddr = minAddr;
  if (!blobs.empty()) {
    MemoryBlob &last = blobs.back();
    baseAddr = last.baseAddr + last.size;
  }
  // Ensure the base address is aligned.
  baseAddr = llvm::alignTo(baseAddr, align);

  // Ensure the new blob does not exceed the maximum address. The table sizes
  // are big so this is purely defensive.
  if (LLVM_UNLIKELY(baseAddr + static_cast<int64_t>(size) >= maxAddr))
    return Error("interpreter is out of memory!");

  // Create the blob with aligned memory.
  MemoryBlob blob(baseAddr, size, align);
  memset(blob.memory.get(), 0, size);
  blobs.push_back(std::move(blob));
  return blobs.back();
}

ErrorOr<InterpreterState::MemoryBlob &>
InterpreterState::MemoryTable::getBlob(int64_t addr) {
  // Check for a null pointer access.
  if (!addr)
    return Error("null address");
  // Otherwise, check for an address that we know is invalid.
  if (addr < minAddr || addr >= maxAddr)
    return Error("address is out-of-bounds: " + Twine(addr));

  // Binary search for the blob that corresponds to the address.
  auto it =
      llvm::lower_bound(blobs, addr, [](const MemoryBlob &blob, int64_t addr) {
        return blob.baseAddr + blob.size <= static_cast<size_t>(addr);
      });
  if (it == blobs.end())
    return Error("address is out-of-bounds: " + Twine(addr));

  // If the blob has been freed, then return an error.
  if (!it->memory)
    return Error("accessing memory that was freed");

  return *it;
}

ErrorOr<int64_t> InterpreterState::allocateStackMemory(size_t size,
                                                       size_t align) {
  // Track the additional stack allocation on the current frame.
  ++getCurrentFrame().numStackAllocs;

  ErrorOr<MemoryBlob &> blob = stackMemory.addBlob(size, align);
  if (blob.isError())
    return blob.takeError();
  return blob->baseAddr;
}

ErrorOr<int64_t> InterpreterState::allocateHeapMemory(size_t size,
                                                      size_t align) {
  ErrorOr<MemoryBlob &> blob = heapMemory.addBlob(size, align);
  if (blob.isError())
    return blob.takeError();
  return blob->baseAddr;
}

ErrorOrSuccess InterpreterState::freeHeapMemory(int64_t addr) {
  ErrorOr<MemoryBlob &> blob = heapMemory.getBlob(addr);
  if (blob.isError())
    return blob.takeError();
  // Don't do anything fancy here. Just free the underlying memory and mark the
  // blob as freed.
  blob->memory.reset();
  return success();
}

ErrorOr<std::pair<InterpreterState::MemoryBlob &, int64_t>>
InterpreterState::getMemory(int64_t addr, size_t size) {
  // Determine which table the address belongs to and then lookup the blob.
  MemoryTable &table = stackMemory.contains(addr) ? stackMemory : heapMemory;
  ErrorOr<MemoryBlob &> blob = table.getBlob(addr);
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
  return std::pair<MemoryBlob &, int64_t>(*blob, offset);
}

ErrorOr<void *> InterpreterState::getWritableMemory(int64_t addr, size_t size,
                                                    bool writePointer) {
  ErrorOr<std::pair<MemoryBlob &, int64_t>> memref = getMemory(addr, size);
  if (memref.isError())
    return memref.takeError();
  auto [blob, offset] = memref.takeValue();

  // If the access is a pointer write, then mark the region as a pointer. The
  // pointer write size must be equal to the target pointer size.
  size_t pointerSize = target.getDataLayout().getPointerSize();
  if (writePointer && size != pointerSize)
    return Error("pointer write size is not equal to pointer bitwidth");
  if (ErrorOrSuccess err =
          blob.setPointerRegion(offset, size, pointerSize, writePointer);
      err.isError())
    return err.takeError();

  return (uint8_t *)blob.memory.get() + offset;
}

ErrorOr<const void *> InterpreterState::getReadableMemory(int64_t addr,
                                                          size_t size) {
  ErrorOr<std::pair<MemoryBlob &, int64_t>> memref = getMemory(addr, size);
  if (memref.isError())
    return memref.takeError();
  auto [blob, offset] = memref.takeValue();
  return (uint8_t *)blob.memory.get() + offset;
}

ErrorOrSuccess InterpreterState::writeAttributeToMemory(int64_t addr,
                                                        TypedAttr value) {
  if (isa<IntegerAttr, FloatAttr>(value)) {
    int64_t size =
        *DataLayoutInterface::getTypeStoreSize(target, value.getType());
    ErrorOr<void *> mem = getWritableMemory(addr, size);
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
    ErrorOr<const void *> mem = getReadableMemory(addr, size);
    if (mem.isError())
      return mem.takeError();

    if (auto intType = dyn_cast<IntegerType>(type)) {
      APInt value(intType.getWidth(), 0);
      llvm::LoadIntFromMemory(value, (const uint8_t *)*mem, size);
      return IntegerAttr::get(type, value);
    }

    if (auto fpType = dyn_cast<FloatType>(type)) {
      APInt intVal(fpType.getWidth(), 0);
      llvm::LoadIntFromMemory(intVal, (const uint8_t *)*mem, size);
      APFloat value(fpType.getFloatSemantics(), intVal);
      return FloatAttr::get(fpType, value);
    }

    APInt value(64, 0);
    llvm::LoadIntFromMemory(value, (const uint8_t *)*mem, size);
    return IntegerAttr::get(type, value);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(type))
    return itf.readFrom(addr, *this);

  return Error(mlir::debugString(type) +
               " does not implement MemoryableTypeInterface");
}

ErrorOrSuccess
InterpreterState::externalizeMemory(Region &entry,
                                    MutableArrayRef<Attribute> results) {
  // Use the parent function name as the base name for materialized resources.
  auto symbol = cast<mlir::SymbolOpInterface>(entry.getParentOp());
  std::string baseName = (symbol.getName() + "_mem").str();

  // Lazily materialize interpreter memory.
  MemorySpaceAttr interpreterMemorySpace;
  DenseMap<const MemoryBlob *, int64_t> blobIndices;
  auto getOrMaterializeMemory = [&] {
    if (interpreterMemorySpace)
      return interpreterMemorySpace;

    // First map all the blobs to indices so that pointers can be unmapped.
    int64_t blobIndex = 0;
    for (const MemoryTable *table : {&heapMemory, &stackMemory})
      for (const MemoryBlob &blob : table->blobs)
        blobIndices.try_emplace(&blob, blobIndex++);

    // Now unmap the memory.
    std::vector<M::MemoryBlob> blobs;
    for (const MemoryTable *table : {&heapMemory, &stackMemory}) {
      for (const MemoryBlob &blob : table->blobs) {
        MemoryHandle hdl = blobMgr.insert(
            baseName, mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
                          ArrayRef<char>((char *)blob.memory.get(), blob.size),
                          blob.align));
        SmallVector<M::MemoryBlob::PointerRegion> pointerRegions;
        if (blob.pointerRegions) {
          int index = -1;
          // Iterate over the pointer regions indices set in the bitvector.
          while ((index = blob.pointerRegions->find_next(index)) != -1) {
            // Read out the address and map it to a blob.
            APInt addrInt(target.getDataLayout().getPointerBitWidth(), 0);
            llvm::LoadIntFromMemory(addrInt,
                                    (uint8_t *)blob.memory.get() + index,
                                    target.getDataLayout().getPointerSize());
            ErrorOr<std::pair<MemoryBlob &, int64_t>> mem =
                getMemory(addrInt.getSExtValue(), 0);
            // If the address is garbage, just ignore it and let it live.
            if (mem.isError())
              continue;
            auto [memBlob, offset] = mem.takeValue();
            pointerRegions.push_back(M::MemoryBlob::PointerRegion{
                index, blobIndices.at(&memBlob), offset});
          }
        }
        blobs.emplace_back(hdl, table->kind, std::move(pointerRegions));
      }
    }
    interpreterMemorySpace = MemorySpaceAttr::get(target.getContext(), blobs);
    return interpreterMemorySpace;
  };

  // Replace raw pointers in the results except for null pointers. Error if a
  // reference to invalid memory is returned from the function.
  mlir::AttrTypeReplacer replacer;
  std::optional<Error> err;
  replacer.addReplacement([&](PointerAttr ptr) -> Attribute {
    ErrorOr<std::pair<MemoryBlob &, int64_t>> mem = getMemory(ptr.getAddr(), 0);
    // If the memory is garbage, let it live.
    if (mem.isError())
      return ptr;
    auto [blob, offset] = mem.takeValue();
    MemorySpaceAttr space = getOrMaterializeMemory();
    return MemRefAttr::get(space, blobIndices.at(&blob), offset, ptr.getType());
  });

  for (Attribute &result : results) {
    result = replacer.replace(result);
    if (err)
      return std::move(*err);
  }
  return success();
}

ErrorOrSuccess
InterpreterState::internalizeMemory(MutableArrayRef<Attribute> args) {
  // This struct represents an interned memory space and allows index + offset
  // pairs to be mapped to addresses.
  struct InternedMemorySpace {
    std::vector<MemoryBlob *> blobs;
  };
  DenseMap<MemorySpaceAttr, InternedMemorySpace> interned;

  // This functor deduplicates incoming memory spaces and maps the contained
  // memory into the interpreter.
  auto getOrInternSpace =
      [&](MemorySpaceAttr space) -> ErrorOr<InternedMemorySpace &> {
    if (auto it = interned.find(space); it != interned.end())
      return it->second;
    // Process and intern the blobs.
    InternedMemorySpace map;
    for (const M::MemoryBlob &blob : space.getValue()) {
      MemoryTable &table =
          blob.getKind() == MemoryKind::Heap ? heapMemory : stackMemory;
      mlir::AsmResourceBlob *asmBlob = blob.getHandle().getBlob();

      // Initialize the memory.
      ErrorOr<MemoryBlob &> mem =
          table.addBlob(asmBlob->getData().size(), asmBlob->getDataAlignment());
      if (mem.isError())
        return mem.takeError();
      memcpy(mem->memory.get(), asmBlob->getData().data(),
             asmBlob->getData().size());
      map.blobs.push_back(&*mem);
    }

    // Now that all the blobs have been processed, map any pointer values.
    for (auto [blob, interned] : llvm::zip(space.getValue(), map.blobs)) {
      for (const M::MemoryBlob::PointerRegion &ptr : blob.getPointerRegions()) {
        // Map the pointer to an interpreter address.
        int64_t addr = map.blobs[ptr.blobIndex]->baseAddr + ptr.blobOffset;
        // Write the address in memory.
        APInt addrInt(target.getDataLayout().getPointerBitWidth(), addr);
        llvm::StoreIntToMemory(addrInt,
                               (uint8_t *)interned->memory.get() + ptr.offset,
                               target.getDataLayout().getPointerSize());
      }
    }

    return interned.try_emplace(space, std::move(map)).first->second;
  };

  // Intern each resource once into the interpreter.
  auto getOrIntern = [&](MemRefAttr ref) -> ErrorOr<int64_t> {
    ErrorOr<InternedMemorySpace &> space = getOrInternSpace(ref.getMemory());
    if (space.isError())
      return space.takeError();
    return space->blobs[ref.getIndex()]->baseAddr + ref.getOffset();
  };

  // Replace memory references in the inputs with interpreter memory pointers.
  mlir::AttrTypeReplacer replacer;
  std::optional<Error> err;
  replacer.addReplacement(
      [&](MemRefAttr ref) -> std::pair<Attribute, WalkResult> {
        ErrorOr<int64_t> addr = getOrIntern(ref);
        if (addr.isError()) {
          err = addr.takeError();
          return {ref, WalkResult::interrupt()};
        }
        return {PointerAttr::get(*addr, ref.getType()), WalkResult::advance()};
      });

  for (Attribute &arg : args) {
    arg = replacer.replace(arg);
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
  // Internal memory references.
  SmallVector<Attribute> args = llvm::to_vector(arguments);
  if (ErrorOrSuccess err = internalizeMemory(args); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());

  // Push an empty stack frame and map the region arguments.
  pushFrame(nullptr, region.getParentOp());
  transferControlFlowTo(&region.front(), args);

  // Reset the interpret to a clean state.
  auto resetState = llvm::make_scope_exit([&] {
    stack.clear();
    heapMemory.reset();
    stackMemory.reset();
    returnValues.reset();
  });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result = runInterpreter();
  if (result) {
    SmallVector<Attribute> results = result.takeValue();
    // Externalize references to interpreter memory.
    if (ErrorOrSuccess err = externalizeMemory(region, results); err.isError())
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

Operation *InterpreterState::popFrame() {
  // Drop all stack memory on the current frame.
  for (size_t i = 0, e = getCurrentFrame().numStackAllocs; i != e; ++i)
    stackMemory.blobs.pop_back();

  Operation *origin = getCurrentFrame().origin;
  stack.pop_back();
  return origin;
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
