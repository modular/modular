//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterInterface.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

/// Give the memory segments a healthy amount of maximum memory.
static constexpr int64_t kTableSize = 1'000'000'000;
/// Provide a virtual "invalid space" for the interpreter's memory. This is so
/// that an address of 0 can actually be considered a null pointer.
static constexpr int64_t kHeapBaseAddr = 1'000'000'000;
/// Put the stack after the heap.
/// FIXME: The stack pointer does not grow downwards like in every system.
static constexpr int64_t kStackBaseAddr = kHeapBaseAddr + kTableSize;
/// The start of constant global memory.
static constexpr int64_t kConstGlobalBaseAddr = kStackBaseAddr + kTableSize;
/// The start of persistent memory.
static constexpr int64_t kPersistentBaseAddr =
    kConstGlobalBaseAddr + kTableSize;

InterpreterState::InterpreterState(MLIRContext *ctx, TargetInfoAttr target)
    : ctx(ctx), target(target),
      memory{{MemoryKind::Heap, kHeapBaseAddr, kHeapBaseAddr + kTableSize},
             {MemoryKind::Stack, kStackBaseAddr, kStackBaseAddr + kTableSize},
             {MemoryKind::ConstGlobal, kConstGlobalBaseAddr,
              kConstGlobalBaseAddr + kTableSize},
             {MemoryKind::Persistent, kPersistentBaseAddr,
              kPersistentBaseAddr + kTableSize}} {}

InterpreterState::InterpreterState(TargetInfoAttr target)
    : InterpreterState(target.getContext(), target) {}

InterpreterState::MemoryBlob::MemoryBlob(int64_t baseAddr, size_t size,
                                         size_t align, MemoryHandleAttr hdl)
    : baseAddr(baseAddr), size(size), align(align),
      memory(
          hdl ? MemoryT(hdl)
              : MemoryT(OwnedMemory(alignedAlloc(align, size), &alignedFree))) {
}

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
InterpreterState::MemoryTable::addBlob(size_t size, size_t align,
                                       MemoryHandleAttr hdl) {
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
  blobs.emplace_back(baseAddr, size, align, hdl);
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
  if (it == blobs.end()) {
    // Handle zero-size blobs at the end of the list.
    if (!blobs.empty() && blobs.back().baseAddr == addr)
      it = std::prev(blobs.end());
    else
      return Error("address is out-of-bounds: " + Twine(addr));
  }

  // If the blob has been freed, then return an error.
  if (it->isFreed())
    return Error("accessing memory that was freed");

  return *it;
}

ErrorOr<int64_t> InterpreterState::allocateStackMemory(size_t size,
                                                       size_t align) {
  // Track the additional stack allocation on the current frame if there is one.
  if (stackIdx)
    ++getCurrentFrame().numStackAllocs;

  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Stack).addBlob(size, align, /*hdl=*/{});
  if (blob.isError())
    return blob.takeError();
  // Zero-initialize the memory.
  memset(blob->getOwned(), 0, size);
  return blob->baseAddr;
}

ErrorOr<int64_t> InterpreterState::allocateHeapMemory(size_t size,
                                                      size_t align) {
  // Return nullptr for zero-sized allocations. This is valid but mainly is to
  // prevent address collisions of zero-sized allocations.
  if (!size)
    return 0;

  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Heap).addBlob(size, align, /*hdl=*/{});
  if (blob.isError())
    return blob.takeError();
  // Zero-initialize the memory.
  memset(blob->getOwned(), 0, size);
  return blob->baseAddr;
}

ErrorOr<int64_t> InterpreterState::allocatePersistentMemory(size_t size,
                                                            size_t align) {
  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Persistent).addBlob(size, align, /*hdl=*/{});
  if (blob.isError())
    return blob.takeError();
  // Zero-initialize the memory.
  memset(blob->getOwned(), 0, size);
  return blob->baseAddr;
}

ErrorOrSuccess InterpreterState::freeHeapMemory(int64_t addr) {
  // Free functions tolerate null pointers.
  if (addr == 0)
    return success();
  ErrorOr<MemoryBlob &> blob = getTable(MemoryKind::Heap).getBlob(addr);
  if (blob.isError())
    return blob.takeError();
  // Don't do anything fancy here. Just free the underlying memory and mark the
  // blob as freed.
  blob->free();
  return success();
}

ErrorOr<int64_t> InterpreterState::mapConstGlobalMemory(MemoryHandleAttr hdl) {
  // Look for an existing mapped blob for the handle.
  MemoryTable &table = getTable(MemoryKind::ConstGlobal);
  for (const MemoryBlob &blob : table.blobs)
    if (blob.getHandle() == hdl)
      return blob.baseAddr;

  // Otherwise, try to map it in.
  ErrorOr<MemoryBlob &> blob =
      table.addBlob(hdl.getSize(), hdl.getAlign(), hdl);
  if (blob.isError())
    return blob.takeError();
  return blob->baseAddr;
}

ErrorOr<std::pair<InterpreterState::MemoryBlob &, int64_t>>
InterpreterState::getMemory(int64_t addr, size_t size) {
  // Determine which table the address belongs to and then lookup the blob.
  MemoryTable &table = [&]() -> MemoryTable & {
    for (MemoryTable &table : memory)
      if (table.contains(addr))
        return table;
    return *memory;
  }();

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
  if (!blob.isOwned())
    return Error("cannot write to constant global memory");

  // If the access is a pointer write, then mark the region as a pointer. The
  // pointer write size must be equal to the target pointer size.
  size_t pointerSize = target.getDataLayout().getPointerSize();
  if (writePointer && size != pointerSize)
    return Error("pointer write size is not equal to pointer bitwidth");
  if (ErrorOrSuccess err =
          blob.setPointerRegion(offset, size, pointerSize, writePointer);
      err.isError())
    return err.takeError();

  return (uint8_t *)blob.getOwned() + offset;
}

ErrorOr<const void *> InterpreterState::getReadableMemory(int64_t addr,
                                                          size_t size) {
  ErrorOr<std::pair<MemoryBlob &, int64_t>> memref = getMemory(addr, size);
  if (memref.isError())
    return memref.takeError();
  auto [blob, offset] = memref.takeValue();
  return (uint8_t *)blob.getMemory() + offset;
}

ErrorOrSuccess InterpreterState::writeAttributeToMemory(int64_t addr,
                                                        TypedAttr value) {
  if (!target)
    return Error("attribute write requires a target model");

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
  if (!target)
    return Error("attribute read requires a target model");

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

uint64_t InterpreterState::allocateSymbolicMemory(TypedAttr init) {
  uint64_t result = symbolicMemory.size();
  symbolicMemory.push_back(init);
  // Track the allocation on the current frame if there is one.
  if (stackIdx)
    ++getCurrentFrame().numSymbolicAllocs;
  return result;
}

ErrorOr<TypedAttr &> InterpreterState::getSymbolicMemory(uint64_t slot) {
  if (slot >= symbolicMemory.size())
    return Error("symbolic memory slot is out-of-bounds: " + Twine(slot));
  return symbolicMemory[slot];
}

ErrorOr<Attribute>
InterpreterState::readAttributeFromPointer(Attribute pointer,
                                           Type elementType) {
  if (auto sym = dyn_cast_or_null<SymbolicPointerAttr>(pointer))
    return getSymbolicMemory(sym.getSlot());
  if (auto ptr = dyn_cast_or_null<PointerAttr>(pointer))
    return readAttributeFromMemory(ptr.getAddr(), elementType);
  return Error("not a pointer constant");
}

ErrorOrSuccess
InterpreterState::externalizeMemory(MutableArrayRef<Attribute> results) {
  // Lazily materialize interpreter memory.
  MemorySpaceAttr interpreterMemorySpace;
  DenseMap<const MemoryBlob *, int64_t> blobIndices;
  auto getOrMaterializeMemory = [&] {
    if (interpreterMemorySpace)
      return interpreterMemorySpace;

    // First map all the blobs to indices so that pointers can be unmapped.
    int64_t blobIndex = 0;
    for (const MemoryTable &table : memory) {
      for (const MemoryBlob &blob : table.blobs) {
        // Don't extern freed blobs.
        if (blob.isFreed())
          continue;
        blobIndices.try_emplace(&blob, blobIndex++);
      }
    }

    // Now unmap the memory.
    std::vector<MemoryBlobAttr> blobs;
    for (const MemoryTable &table : memory) {
      for (const MemoryBlob &blob : table.blobs) {
        if (blob.isFreed())
          continue;

        SmallVector<PointerRegion> pointerRegions;
        if (blob.pointerRegions) {
          assert(blob.isOwned() && "const memory cannot have pointers");
          int index = -1;
          // Iterate over the pointer regions indices set in the bitvector.
          while ((index = blob.pointerRegions->find_next(index)) != -1) {
            // Read out the address and map it to a blob.
            APInt addrInt(target.getDataLayout().getPointerBitWidth(), 0);
            llvm::LoadIntFromMemory(addrInt, (uint8_t *)blob.getOwned() + index,
                                    target.getDataLayout().getPointerSize());
            ErrorOr<std::pair<MemoryBlob &, int64_t>> mem =
                getMemory(addrInt.getSExtValue(), 0);
            // If the address is garbage, just ignore it and let it live.
            if (mem.isError())
              continue;
            auto [memBlob, offset] = mem.takeValue();
            pointerRegions.push_back(
                PointerRegion{index, blobIndices.at(&memBlob), offset});
          }
        }

        // Add the new blob value to the blob manager if one with the same value
        // does not already exist.
        MemoryHandleAttr hdl;
        if (blob.isOwned()) {
          ArrayRef<char> dataRef((const char *)blob.getOwned(), blob.size);
          hdl = MemoryHandleAttr::get(ctx, blob.align, dataRef);
        } else {
          hdl = blob.getHandle();
        }
        blobs.push_back(MemoryBlobAttr::get(hdl, table.kind, pointerRegions));
      }
    }
    interpreterMemorySpace = MemorySpaceAttr::get(ctx, blobs);
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
    std::vector<std::pair<MemoryTable *, size_t>> blobs;
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
    for (MemoryBlobAttr blob : space.getValue()) {
      MemoryTable &table = getTable(blob.getKind());

      // Constant global is mapped directly into the interpreter.
      MemoryHandleAttr hdl;
      if (blob.getKind() == MemoryKind::ConstGlobal)
        hdl = blob.getHandle();

      // Initialize the memory.
      MemoryHandleAttr asmBlob = blob.getHandle();
      map.blobs.emplace_back(&table, table.blobs.size());
      ErrorOr<MemoryBlob &> mem =
          table.addBlob(asmBlob.getSize(), asmBlob.getAlign(), hdl);
      if (mem.isError())
        return mem.takeError();
      if (!hdl)
        memcpy(mem->getOwned(), asmBlob.getData(), asmBlob.getSize());
    }

    // Now that all the blobs have been processed, map any pointer values.
    for (auto [blob, tabIdx] : llvm::zip(space.getValue(), map.blobs)) {
      for (const PointerRegion &ptr : blob.getPointerRegions()) {
        auto [tab, blobIdx] = tabIdx;
        MemoryBlob *interned = &tab->blobs[blobIdx];
        assert(interned->isOwned() && "const memory cannot have pointers");
        // Map the pointer to an interpreter address.
        auto [ptrTab, ptrBlobIdx] = map.blobs[ptr.blobIndex];
        int64_t addr = ptrTab->blobs[ptrBlobIdx].baseAddr + ptr.blobOffset;
        // Write the address in memory.
        APInt addrInt(target.getDataLayout().getPointerBitWidth(), addr);
        llvm::StoreIntToMemory(addrInt,
                               (uint8_t *)interned->getOwned() + ptr.offset,
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
    auto [tab, blobIdx] = space->blobs[ref.getIndex()];
    return tab->blobs[blobIdx].baseAddr + ref.getOffset();
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

  mlir::AttrTypeReplacer liftStore;
  liftStore.addReplacement(
      [&](StoreToMemAttr store) -> std::pair<Attribute, WalkResult> {
        Type valueType = store.getValue().getType();
        if (!getTarget()) {
          auto ptr =
              SymbolicPointerAttr::get(symbolicMemory.size(), store.getType());
          symbolicMemory.push_back(store.getValue());
          return {ptr, WalkResult::advance()};
        }

        ErrorOr<PointerAttr> ptr =
            allocateInternalStackFor(valueType, store.getType());
        if (ptr.isError()) {
          err = ptr.takeError();
          return {store, WalkResult::interrupt()};
        }
        if (ErrorOrSuccess err =
                writeAttributeToMemory(ptr->getAddr(), store.getValue());
            err.isError()) {
          err = ptr.takeError();
          return {store, WalkResult::interrupt()};
        }
        return {ptr.takeValue(), WalkResult::advance()};
      });

  for (Attribute &arg : args) {
    arg = replacer.replace(arg);
    if (err)
      return std::move(*err);
    arg = liftStore.replace(arg);
    if (err)
      return std::move(*err);
  }
  return success();
}

ErrorOr<TypedAttr> InterpreterState::loadAttributeFromMemRef(MemRefAttr memref,
                                                             Type type) {
  // Reset memory upon exit.
  auto resetState = llvm::make_scope_exit([&] { reset(); });
  Attribute attr = memref;
  if (ErrorOrSuccess err = internalizeMemory(attr))
    return err.takeError();
  return readAttributeFromMemory(cast<PointerAttr>(attr).getAddr(), type);
}

ErrorOr<PointerAttr> InterpreterState::allocateInternalStackFor(Type type,
                                                                Type ptrType) {
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(getTarget(), type);
  std::optional<int64_t> align =
      DataLayoutInterface::getTypeABIAlign(getTarget(), type);
  if (!size || !align)
    return Error("could not get result slot type size or alignment");

  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Stack).addBlob(*size, *align, /*hdl=*/{});
  if (blob.isError())
    return blob.takeError();
  return PointerAttr::get(blob->baseAddr, ptrType);
}

//===----------------------------------------------------------------------===//
// Interpreter Implementation

/// Report an error with folding an operation.
static ErrorTree reportFoldError(Operation *op, ArrayRef<Attribute> operands,
                                 const Twine &prefix,
                                 const Twine &suffix = "") {
  std::string note;
  llvm::raw_string_ostream os(note);
  os << prefix << op->getName();
  if (!op->getAttrs().empty()) {
    os << '{';
    llvm::interleaveComma(op->getAttrs(), os, [&](const NamedAttribute &attr) {
      os << attr.getName().getValue() << ": " << attr.getValue();
    });
    os << '}';
  }
  os << '(';
  llvm::interleaveComma(operands, os);
  os << ')' << suffix;
  return {op->getLoc(), Error(os.str())};
}

/// Interpret a generic operation by trying to use its operation folder.
static ErrorTreeOrSuccess interpretOpWithFolder(Operation *op,
                                                ArrayRef<Attribute> operands,
                                                InterpreterState &state) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(operands, results)))
    return reportFoldError(op, operands, "failed to fold operation ");
  for (auto [result, output] : llvm::zip(results, op->getResults())) {
    if (auto value = llvm::dyn_cast<Attribute>(result))
      state.mapOrOverwrite(output, value);
    else
      state.mapOrOverwrite(output, state.lookupValue(result.get<Value>()));
  }
  return success();
}

void InterpreterState::reset() {
  block = nullptr;
  pc = Block::iterator();
  stackIdx = 0;
  for (MemoryTable &table : memory)
    table.reset();
}

ErrorTree InterpreterState::addStackTrace(ErrorTree error) {
  for (const StackFrame &frame :
       llvm::reverse(ArrayRef(stack).take_front(stackIdx))) {
    StringRef funcName = cast<mlir::SymbolOpInterface>(frame.func).getName();
    error = ErrorTree(frame.func->getLoc(),
                      Error("failed to interpret function @" + funcName),
                      std::move(error));
    if (frame.origin)
      error = ErrorTree(frame.origin->getLoc(),
                        Error("failed to evaluate call"), std::move(error));
  }
  return error;
}

ErrorTreeOr<SmallVector<Attribute>>
InterpreterState::executeRegion(Region &region, ArrayRef<Attribute> arguments) {
  // Internalize memory inside function arguments.
  SmallVector<Attribute> args = llvm::to_vector(arguments);
  if (ErrorOrSuccess err = internalizeMemory(args); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());

  // Reset the interpret to a clean state.
  auto resetState = llvm::make_scope_exit([&] { reset(); });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result = interpretFunction(region, args);
  if (result) {
    SmallVector<Attribute> results = result.takeValue();
    // Externalize references to interpreter memory.
    if (ErrorOrSuccess err = externalizeMemory(results); err.isError())
      return ErrorTree(region.getLoc(), err.takeError());
    return results;
  }

  // The interpreter ran into an error. Report an error using a stacktrace.
  return addStackTrace(result.takeError());
}

/// Execute a region that has a ByRefResult or InitSelf argument.
ErrorTreeOr<TypedAttr> InterpreterState::executeRegionWithResultSlot(
    Region &region, ArrayRef<Attribute> arguments, bool isInitSelf,
    SmartVariant<Type, TypedAttr> resultValue) {
  Location loc = region.getLoc();
  if (region.getArguments().empty())
    return ErrorTree(loc, "internal error: region has no arguments");

  // Allocate the result slot.
  Type resultPtrType =
      (isInitSelf ? region.getArgument(0) : region.getArguments().back())
          .getType();
  TypedAttr resultSlotAttr;

  if (!getTarget()) {
    uint64_t slot = symbolicMemory.size();
    symbolicMemory.push_back(cast<TypedAttr>(resultValue));
    resultSlotAttr = SymbolicPointerAttr::get(slot, resultPtrType);
  } else {
    ErrorOr<PointerAttr> resultSlotAttrOr =
        allocateInternalStackFor(cast<Type>(resultValue), resultPtrType);
    if (resultSlotAttrOr.isError())
      return ErrorTree(loc, resultSlotAttrOr.takeError());
    resultSlotAttr = resultSlotAttrOr.takeValue();
  }

  SmallVector<Attribute> allArgs;
  if (isInitSelf)
    allArgs.push_back(resultSlotAttr);
  llvm::append_range(allArgs, arguments);
  if (!isInitSelf)
    allArgs.push_back(resultSlotAttr);

  // Internalize memory inside function arguments.
  if (ErrorOrSuccess err = internalizeMemory(allArgs); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());

  // Reset the interpret to a clean state.
  auto resetState = llvm::make_scope_exit([&] { reset(); });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result =
      interpretFunction(region, allArgs);

  // The interpreter ran into an error. Report an error using a stacktrace.
  if (!result)
    return addStackTrace(result.takeError());

  TypedAttr value;
  if (!getTarget()) {
    value = symbolicMemory[cast<SymbolicPointerAttr>(resultSlotAttr).getSlot()];
  } else {
    ErrorOr<TypedAttr> resultOr = readAttributeFromMemory(
        cast<PointerAttr>(resultSlotAttr).getAddr(), cast<Type>(resultValue));
    if (resultOr.isError())
      return ErrorTree(loc, resultOr.takeError());
    value = resultOr.takeValue();
  }

  if (ErrorOrSuccess err = externalizeMemory(value); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());
  return value;
}

ErrorTreeOr<SmallVector<Attribute>>
InterpreterState::interpretFunction(Region &body,
                                    ArrayRef<Attribute> arguments) {
  callFunctionBody(body, arguments);

  SmallVector<Attribute> operands;
  while (block) {
    // Advance the iterator.
    if (pc.isValid())
      ++pc;
    else
      pc = block->begin();

    operands.clear();
    // Lookup the operands of the current operation.
    for (Value operand : pc->getOperands())
      operands.push_back(lookupValue(operand));

    // Check for an interpreter interface implementation.
    if (auto interpItf = dyn_cast<InterpreterOpInterface>(*pc)) {
      ErrorTreeOrSuccess err = interpItf.interpret(operands, *this);
      if (err.isError())
        return reportFoldError(&*pc, operands, "failed to interpret operation ")
            .addCause(err.takeError());

      // Otherwise, try to use the operation folder.
    } else {
      ErrorTreeOrSuccess result = interpretOpWithFolder(&*pc, operands, *this);
      if (result.isError())
        return result.takeError();
    }
  }

  // The stack frame must be empty.
  if (stackIdx) {
    llvm::report_fatal_error(
        "exiting interpreter with remaining stack frames " + Twine(stackIdx));
  }
  return std::move(exitValues);
}

Operation *InterpreterState::getOrigin(size_t depth) {
  if (depth >= stack.size())
    return nullptr;
  return stack[stackIdx - 1 - depth].origin;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterOpInterface.cpp.inc"
#include "KGEN/Interpreter/MemoryableTypeInterface.cpp.inc"
