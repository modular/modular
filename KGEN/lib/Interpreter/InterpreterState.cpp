//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/Interpreter/Utils.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"

#include <iomanip>
#include <sstream>
#include <string>

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

InterpreterState::MemoryBlob::MemoryBlob(llvm::BumpPtrAllocator &allocator,
                                         int64_t baseAddr, size_t size,
                                         size_t align, unsigned addressSpace,
                                         MemoryHandleAttr hdl, size_t refCount)
    : baseAddr(baseAddr), size(size), align(align), addressSpace(addressSpace),
      memory(hdl ? MemoryT(hdl) : MemoryT(allocator.Allocate(size, align))),
      refCount(refCount) {}

/// Do one of the following:
///  (1) Mark a region as a Pointer
///  (2) Clear a Pointer Region by passing in None to a blob with regions
///  (3) Mark a region as a Symbol
ErrorOrSuccess InterpreterState::MemoryBlob::setMarkedRegion(
    int64_t offset, int64_t regionSize, int64_t pointerSize,
    RegionMark markedRegion) {
  std::optional<llvm::BitVector> *regionFieldPtr = nullptr;
  bool write = false;
  switch (markedRegion) {
  case RegionMark::None:
    if (pointerRegions) {
      regionFieldPtr = &pointerRegions;
      break;
    }
    return success();
  case RegionMark::Pointer:
    write = true;
    regionFieldPtr = &pointerRegions;
    break;
  case RegionMark::Symbol:
    write = true;
    regionFieldPtr = &symbolRegions;
    break;
  }
  std::optional<llvm::BitVector> &regionField = *regionFieldPtr;
  if (!regionField) {
    regionField.emplace(size);
    regionField->set(offset);
    return success();
  }
  // The write clobbers a pointer region if a bit is set between
  // `(offset - pointerSize, offset)` or between
  // `(offset + size - pointerSize, offset + size)`, indicating partial
  // overwrite of a pointer region.
  if (regionSize >= 0 &&
      (regionField->find_first_in(
           std::max<int64_t>(0, offset - pointerSize + 1), offset) != -1 ||
       regionField->find_first_in(
           std::max<int64_t>(0, offset + regionSize - pointerSize + 1),
           offset + regionSize) != -1))
    return Error("write clobbers a pointer region");

  if (!write)
    regionField->reset(offset, offset + regionSize);
  else
    regionField->set(offset);
  return success();
}

ErrorOr<InterpreterState::MemoryBlob &> InterpreterState::MemoryTable::addBlob(
    llvm::BumpPtrAllocator &allocator, size_t size, size_t align,
    unsigned addressSpace, MemoryHandleAttr hdl, bool resetRefCount) {
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
  blobs.emplace_back(allocator, baseAddr, size, align, addressSpace, hdl,
                     (resetRefCount ? 0 : 1));
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
  notifyAllocationOnFrame();

  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Stack).addBlob(allocator, size, align);
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
      getTable(MemoryKind::Heap).addBlob(allocator, size, align);
  if (blob.isError())
    return blob.takeError();
  // Zero-initialize the memory.
  memset(blob->getOwned(), 0, size);
  return blob->baseAddr;
}

ErrorOr<int64_t>
InterpreterState::allocatePersistentMemory(size_t size, size_t align,
                                           unsigned addressSpace) {
  ErrorOr<MemoryBlob &> blob =
      getTable(MemoryKind::Persistent)
          .addBlob(allocator, size, align, addressSpace);
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
  ErrorOr<MemoryBlob &> blob = table.addBlob(
      allocator, hdl.getSize(), hdl.getAlign(), /*addressSpace=*/0, hdl);
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
                                                    RegionMark regionMark) {
  ErrorOr<std::pair<MemoryBlob &, int64_t>> memref = getMemory(addr, size);
  if (memref.isError())
    return memref.takeError();
  auto [blob, offset] = memref.takeValue();
  if (!blob.isOwned())
    return Error("cannot write to constant global memory");

  // If the access is a pointer write, then mark the region as a pointer. The
  // pointer write size must be equal to the target pointer size.
  size_t pointerSize = target.getDataLayout().getPointerSize();
  if ((regionMark != RegionMark::None) && size != pointerSize)
    return Error("pointer write size is not equal to pointer bitwidth");
  if (ErrorOrSuccess err =
          blob.setMarkedRegion(offset, size, pointerSize, regionMark);
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
    ErrorOr<void *> mem = getWritableMemory(addr, size, RegionMark::None);
    if (mem.isError())
      return mem.takeError();

    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      APInt value = intAttr.getValue();
      if (isa<IndexType>(intAttr.getType()))
        value = value.sextOrTrunc(target.resolveIndexBitWidth());
      llvm::StoreIntToMemory(intAttr.getValue(),
                             reinterpret_cast<uint8_t *>(*mem), size);
      return success();
    }

    llvm::StoreIntToMemory(cast<FloatAttr>(value).getValue().bitcastToAPInt(),
                           reinterpret_cast<uint8_t *>(*mem), size);
    return success();
  }

  // Ignore UninitMemAttr: implementations of MemoryableTypeInterface may
  // not be compatible with it, and storing a noop is a noop.
  if (isa<UninitMemAttr>(value))
    return success();

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
    // The bitwidth is tied to the IndexType. If the target bitwidth is used an
    // IntegerType must be used instead of IndexType, resulting in type errors
    // if the result of the interpretation is materialized. To mitigate this
    // issue, we perform all calculations in the target bit width then extend
    // back to index width so the type doesn't change.
    int64_t size = *DataLayoutInterface::getTypeStoreSize(target, type);
    ErrorOr<const void *> mem = getReadableMemory(addr, size);
    if (mem.isError())
      return mem.takeError();

    if (auto fpType = dyn_cast<FloatType>(type)) {
      APInt intVal(fpType.getWidth(), 0);
      llvm::LoadIntFromMemory(intVal, (const uint8_t *)*mem, size);
      APFloat value(fpType.getFloatSemantics(), intVal);
      return FloatAttr::get(fpType, value);
    }
    bool isIndex = isa<IndexType>(type);
    unsigned bitWidth = isIndex ? target.resolveIndexBitWidth()
                                : cast<IntegerType>(type).getWidth();
    APInt value(bitWidth, 0);
    llvm::LoadIntFromMemory(value, (const uint8_t *)*mem, size);
    if (isIndex)
      value = value.sextOrTrunc(IndexType::kInternalStorageBitWidth);
    return IntegerAttr::get(type, value);
  }

  if (auto itf = dyn_cast<MemoryableTypeInterface>(type))
    return itf.readFrom(addr, *this);

  return Error(mlir::debugString(type) +
               " does not implement MemoryableTypeInterface");
}

uint64_t InterpreterState::addSymbolToSymbolTable(TypedAttr symbol) {
  uint64_t result = symbols.size();
  for (auto [index, existing] : llvm::enumerate(symbols)) {
    if (symbol == existing)
      return index;
  }
  symbols.push_back(symbol);
  return result;
}

ErrorOr<TypedAttr &> InterpreterState::getSymbol(uint64_t slot) {
  if (slot >= symbols.size())
    return Error("symbolic memory slot is out-of-bounds: " + Twine(slot));
  return symbols[slot];
}

ErrorOr<Attribute>
InterpreterState::readAttributeFromPointer(Attribute pointer,
                                           Type elementType) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(pointer))
    return readAttributeFromMemory(ptr.getAddr(), elementType);
  return Error("not a pointer constant");
}

static void
handleMarkedRegions(function_ref<void(int)> indexHandler,
                    const std::optional<llvm::BitVector> &bitVectorMaybe) {
  if (bitVectorMaybe) {
    int index = -1;
    // Iterate over the symbol regions indices set in the bitvector.
    while ((index = bitVectorMaybe->find_next(index)) != -1)
      indexHandler(index);
  }
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
        auto addPointerRegionAtIndex = [&](int index) {
          APInt addrInt(target.getDataLayout().getPointerBitWidth(), 0);
          llvm::LoadIntFromMemory(addrInt, (uint8_t *)blob.getOwned() + index,
                                  target.getDataLayout().getPointerSize());
          ErrorOr<std::pair<MemoryBlob &, int64_t>> mem =
              getMemory(addrInt.getSExtValue(), 0);
          // If the address is garbage, just ignore it and let it live.
          if (mem.isError())
            return;
          auto [memBlob, offset] = mem.takeValue();
          pointerRegions.push_back(
              PointerRegion{index, blobIndices.at(&memBlob), offset});
        };
        handleMarkedRegions(addPointerRegionAtIndex, blob.pointerRegions);

        SmallVector<int64_t> symbolRegions;
        handleMarkedRegions([&](int index) { symbolRegions.push_back(index); },
                            blob.symbolRegions);

        // Add the new blob value to the blob manager if one with the same value
        // does not already exist.
        MemoryHandleAttr hdl;
        if (blob.isOwned()) {
          ArrayRef<char> dataRef((const char *)blob.getOwned(), blob.size);
          hdl = MemoryHandleAttr::get(ctx, blob.align, dataRef);
        } else {
          hdl = blob.getHandle();
        }
        blobs.push_back(MemoryBlobAttr::get(hdl, table.kind, pointerRegions,
                                            symbolRegions, blob.addressSpace));
      }
    }
    interpreterMemorySpace = MemorySpaceAttr::get(ctx, blobs);
    return interpreterMemorySpace;
  };

  // Externalize the symbols by replacing pointer values with coordinates into
  // memory space.
  mlir::AttrTypeReplacer symbolReplacer;
  symbolReplacer.addReplacement([&](PointerAttr ptr) -> Attribute {
    ErrorOr<std::pair<MemoryBlob &, int64_t>> mem = getMemory(ptr.getAddr(), 0);
    if (mem.isError())
      return ptr;
    auto [blob, offset] = mem.takeValue();
    getOrMaterializeMemory();
    uint64_t indexOfReferencedBlob = blobIndices.at(&blob);
    return CoordinateAttr::get(ptr.getContext(), offset, indexOfReferencedBlob,
                               ptr.getType());
  });
  for (unsigned index = 0, e = symbols.size(); index < e; index++)
    symbols[index] = ::cast<TypedAttr>(symbolReplacer.replace(symbols[index]));

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
    return MemRefAttr::get(
        ptr.getContext(),
        MemoryModelAttr::get(ptr.getContext(), space,
                             SymbolArrayAttr::get(ptr.getContext(), symbols)),
        blobIndices.at(&blob), offset, ptr.getType());
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
      [&](MemoryModelAttr model) -> ErrorOr<InternedMemorySpace &> {
    MemorySpaceAttr space = model.getMemory();
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
          table.addBlob(allocator, asmBlob.getSize(), asmBlob.getAlign(),
                        blob.getAddressSpace(), hdl, /*resetRefCount=*/true);
      if (mem.isError())
        return mem.takeError();
      if (!hdl)
        memcpy(mem->getOwned(), asmBlob.getData(), asmBlob.getSize());
    }
    DenseMap<uint64_t, uint64_t> symbolMap;
    // Replace coordinates into MemorySpace with PointerAttr to interpreter
    // address.
    mlir::AttrTypeReplacer symReplacer;
    symReplacer.addReplacement([&](CoordinateAttr coord) {
      int64_t index = coord.getRefSlot();
      int64_t offset = coord.getOffset();
      auto [ptrTab, ptrBlobIdx] = map.blobs[index];
      int64_t addr = ptrTab->blobs[index].baseAddr + offset;
      return PointerAttr::get(addr, coord.getType());
    });
    for (unsigned index = 0, numSymbols = model.getSymbols().size();
         index < numSymbols; index++) {
      Attribute newSymbol = symReplacer.replace(model.getSymbols()[index]);
      symbolMap[index] = addSymbolToSymbolTable(::cast<TypedAttr>(newSymbol));
    }

    // Now that all the blobs have been processed, map any pointer values.
    for (auto [blob, tabIdx] : llvm::zip(space.getValue(), map.blobs)) {
      auto [tab, blobIdx] = tabIdx;
      MemoryBlob *interned = &tab->blobs[blobIdx];
      for (const PointerRegion &ptr : blob.getPointerRegions()) {
        assert(interned->isOwned() && "const memory cannot have pointers");
        // Map the pointer to an interpreter address.
        auto [ptrTab, ptrBlobIdx] = map.blobs[ptr.blobIndex];
        int64_t addr = ptrTab->blobs[ptrBlobIdx].baseAddr + ptr.blobOffset;
        // Write the address in memory.
        APInt addrInt(target.getDataLayout().getPointerBitWidth(), addr);
        llvm::StoreIntToMemory(addrInt,
                               (uint8_t *)interned->getOwned() + ptr.offset,
                               target.getDataLayout().getPointerSize());

        // Mark ptrRegion for the internalized blob.
        (void)interned->setMarkedRegion(ptr.offset, -1,
                                        target.getDataLayout().getPointerSize(),
                                        RegionMark::Pointer);
      }

      // Update the symbol indices in the blobs so they are with respect to
      // interpreter memory.
      for (int64_t symbolRegion : blob.getSymbolRegions()) {
        uint8_t *owned = (uint8_t *)interned->getOwned() + symbolRegion;
        APInt index(target.getDataLayout().getPointerBitWidth(), 0);
        llvm::LoadIntFromMemory(index, owned,
                                target.getDataLayout().getPointerSize());
        APInt newIndex(target.getDataLayout().getPointerBitWidth(),
                       symbolMap[index.getZExtValue()]);
        llvm::StoreIntToMemory(newIndex, owned,
                               target.getDataLayout().getPointerSize());
      }
    }
    return interned.try_emplace(space, std::move(map)).first->second;
  };

  // Intern each resource once into the interpreter.
  auto getOrIntern = [&](MemRefAttr ref) -> ErrorOr<int64_t> {
    ErrorOr<InternedMemorySpace &> space = getOrInternSpace(ref.getModel());
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
        // Recursively lift the stored value first.
        TypedAttr liftedValue =
            cast<TypedAttr>(liftStore.replace(store.getValue()));
        Type ptrType = liftStore.replace(store.getType());

        Type valueType = liftedValue.getType();
        if (!getTarget()) {
          err = Error("store to memory requires a target model");
          return {store, WalkResult::interrupt()};
        }

        ErrorOr<PointerAttr> ptr = allocateInternalStackFor(valueType, ptrType);
        if (ptr.isError()) {
          err = ptr.takeError();
          return {store, WalkResult::interrupt()};
        }
        if (ErrorOrSuccess err =
                writeAttributeToMemory(ptr->getAddr(), liftedValue);
            err.isError()) {
          err = err.takeError();
          return {store, WalkResult::interrupt()};
        }
        return {ptr.takeValue(), WalkResult::skip()};
      });
  addCustomReplacementsToLiftStore(liftStore);

  mlir::AttrTypeWalker walker;
  std::function<ErrorOrSuccess(MemorySpaceAttr memSpace,
                               InternedMemorySpace & internedSpace,
                               size_t blobIdx)>
      refCountBlob = [&](MemorySpaceAttr memSpace,
                         InternedMemorySpace &internedSpace,
                         size_t blobIdx) -> ErrorOrSuccess {
    MemoryBlobAttr blob = memSpace[blobIdx];
    if (isGlobalBlob(blob))
      return success();

    // ref count heap memory handles
    if (blob.getKind() == MemoryKind::Heap) {
      auto [table, idx] = internedSpace.blobs[blobIdx];
      MemoryBlob &memBlob = table->blobs[idx];
      memBlob.refCount++;
    }

    // Recurse to ptrRegions for indirect memory references.
    auto ptrItr = blob.getPointerRegions().begin();
    auto ptrEnd = blob.getPointerRegions().end();
    for (; ptrItr != ptrEnd; ++ptrItr) {
      ErrorOrSuccess result =
          refCountBlob(memSpace, internedSpace, ptrItr->blobIndex);
      if (result.isError())
        return result.takeError();
    }
    return success();
  };

  walker.addWalk([&](MemRefAttr ref) {
    ErrorOr<InternedMemorySpace &> space = getOrInternSpace(ref.getModel());
    if (space.isError()) {
      err = space.takeError();
      return WalkResult::interrupt();
    }

    ErrorOrSuccess result =
        refCountBlob(ref.getModel().getMemory(), *space, ref.getIndex());
    if (result.isError()) {
      err = result.takeError();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  for (Attribute &arg : args) {
    walker.walk(arg);
    if (err)
      return std::move(*err);
  }

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
  ErrorOr<TypedAttr> attrOr =
      readAttributeFromMemory(cast<PointerAttr>(attr).getAddr(), type);
  if (attrOr.isError())
    return attrOr.takeError();
  SmallVector<Attribute> results;
  results.push_back(attrOr.takeValue());
  if (ErrorOrSuccess errorMaybe = externalizeMemory(results))
    return errorMaybe.takeError();
  Attribute result = results.front();
  return ::cast<TypedAttr>(result);
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
      getTable(MemoryKind::Stack).addBlob(allocator, *size, *align);
  if (blob.isError())
    return blob.takeError();
  // Zero-initialize the memory.
  memset(blob->getOwned(), 0, *size);
  return PointerAttr::get(blob->baseAddr, ptrType);
}

//===----------------------------------------------------------------------===//
// Interpreter Implementation

ErrorTreeOr<SmallVector<Attribute>>
InterpreterState::executeRegion(Region &region, ArrayRef<Attribute> arguments) {
  // Internalize memory inside function arguments.
  SmallVector<Attribute> args = llvm::to_vector(arguments);
  if (ErrorOrSuccess err = internalizeMemory(args))
    return ErrorTree(region.getLoc(), err.takeError());

  // Reset the interpret to a clean state.
  auto resetState = llvm::make_scope_exit([&] { reset(); });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result = interpretFunction(region, args);
  if (result)
    return addStackTrace(result.takeError());

  SmallVector<Attribute> results = result.takeValue();
  // Externalize references to interpreter memory.
  if (ErrorOrSuccess err = externalizeMemory(results))
    return ErrorTree(region.getLoc(), err.takeError());
  return results;
}

/// Execute a region that has a ByRefResult argument.
ErrorTreeOr<TypedAttr> InterpreterState::executeRegionWithResultSlot(
    Region &region, ArrayRef<Attribute> arguments,
    SmartVariant<Type, TypedAttr> resultValue) {
  Location loc = region.getLoc();
  if (region.getArguments().empty())
    return ErrorTree(loc, "internal error: region has no arguments");
  if (!getTarget())
    return ErrorTree(loc, Error("call into memory requires a target model"));

  // Allocate the result slot.
  Type resultPtrType = region.getArguments().back().getType();

  ErrorOr<PointerAttr> resultSlotAttrOr =
      allocateInternalStackFor(cast<Type>(resultValue), resultPtrType);
  if (resultSlotAttrOr.isError())
    return ErrorTree(loc, resultSlotAttrOr.takeError());
  TypedAttr resultSlotAttr = resultSlotAttrOr.takeValue();

  SmallVector<Attribute> allArgs;
  llvm::append_range(allArgs, arguments);
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
  if (result)
    return addStackTrace(result.takeError());

  ErrorOr<TypedAttr> resultOr = readAttributeFromMemory(
      cast<PointerAttr>(resultSlotAttr).getAddr(), cast<Type>(resultValue));
  if (resultOr.isError())
    return ErrorTree(loc, resultOr.takeError());
  TypedAttr value = resultOr.takeValue();

  if (ErrorOrSuccess err = externalizeMemory(value); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());
  return value;
}

//===----------------------------------------------------------------------===//
// IRInterpreter
//===----------------------------------------------------------------------===//

ErrorTree IRInterpreter::addStackTrace(ErrorTree error) {
  return addStackTraceImpl(
      std::move(error), stack.getArrayRef(),
      [](const StackFrame &frame) { return frame.origin; });
}

ErrorTreeOrSuccess
IRInterpreter::interpretOpWithFolder(Operation *op,
                                     ArrayRef<Attribute> operands) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(operands, results)))
    return reportFoldError(op, operands, "failed to fold operation ");
  for (auto [result, output] : llvm::zip(results, op->getResults())) {
    if (auto value = llvm::dyn_cast<Attribute>(result))
      mapOrOverwrite(output, value);
    else
      mapOrOverwrite(output, lookupValue(cast<Value>(result)));
  }
  return success();
}

ErrorTreeOr<SmallVector<Attribute>>
IRInterpreter::interpretFunction(Region &body, ArrayRef<Attribute> arguments) {
  if (auto err = callFunctionBody(body, arguments))
    return err.takeError();

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
    Operation &op = *pc;
    if (auto interpItf = dyn_cast<BytecodeInterpreterOpInterface>(op)) {
      OpBytecodeGenerator gen = interpItf.getBytecodeGenerator();
      if (GenBytecodeHook genBytecode = gen.genBytecode) {
        payload.reserve(gen.payloadSize);
        if (auto err = genBytecode(&op, payload.data(), getTarget()))
          return ErrorTree(op.getLoc(), err.takeError());
      }
      ErrorTreeOrSuccess err = interpItf.getBytecodeGenerator().interpret(
          &op, operands, payload.data(), *this);
      if (err.isError())
        return reportFoldError(&*pc, operands, "failed to interpret operation ")
            .addCause(err.takeError());

      // Otherwise, try to use the operation folder.
    } else if (ErrorTreeOrSuccess result =
                   interpretOpWithFolder(&*pc, operands)) {
      return result.takeError();
    }
  }

  // The stack frame must be empty.
  if (LLVM_UNLIKELY(!stack.empty())) {
    llvm::report_fatal_error(
        "exiting interpreter with remaining stack frames " +
        Twine(stack.size()));
  }
  return std::move(exitValues);
}

Operation *IRInterpreter::getOrigin(size_t depth) {
  if (depth >= stack.size())
    return nullptr;
  return stack[stack.size() - 1 - depth].origin;
}

void InterpreterState::dump() {
  // index memory blobs for pointers can be mapped. TODO: Maybe just print
  // interpreter address to avoid enumerating at a snapshot in time?
  DenseMap<const MemoryBlob *, int64_t> blobIndices;
  int64_t blobIndex = 0;
  for (const MemoryTable &table : memory) {
    for (const MemoryBlob &blob : table.blobs) {
      // Don't extern freed blobs.
      if (blob.isFreed())
        continue;
      blobIndices.try_emplace(&blob, blobIndex++);
    }
  }

  auto memoryToString = [](const void *memory, size_t size) -> std::string {
    const char *data = static_cast<const char *>(memory);
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
      if (i > 0 && i % 16 == 0)
        ss << "\n     ";
      ss << std::setfill('0') << std::setw(2) << std::hex
         << static_cast<int>(static_cast<unsigned char>(data[i])) << " ";
    }
    return ss.str();
  };
  auto ptrRegionAtIndex = [&](MemoryBlob &blob, int index) -> PointerRegion {
    APInt addrInt(target.getDataLayout().getPointerBitWidth(), 0);
    llvm::LoadIntFromMemory(addrInt, (uint8_t *)blob.getOwned() + index,
                            target.getDataLayout().getPointerSize());
    ErrorOr<std::pair<MemoryBlob &, int64_t>> mem =
        getMemory(addrInt.getSExtValue(), 0);
    if (mem.isError())
      return PointerRegion{index, -1, -1};
    auto [memBlob, offset] = mem.takeValue();
    return PointerRegion{index, blobIndices.at(&memBlob), offset};
  };
  auto printBlob = [&](MemoryKind kind) {
    switch (kind) {
    case MemoryKind::Heap:
      llvm::dbgs() << "HEAP:\n";
      break;
    case MemoryKind::Stack:
      llvm::dbgs() << "STACK:\n";
      break;
    case MemoryKind::ConstGlobal:
      llvm::dbgs() << "CONST GLOBAL:\n";
      break;
    case MemoryKind::Persistent:
      llvm::dbgs() << "PERSISTENT:\n";
      break;
    }

    for (auto blob : getTable(kind).blobs) {
      std::string data = memoryToString(blob.getMemory(), blob.size);
      // TODO: Support reading other values. This just illustrates printing an
      // index if the first field in the blob is an index. If you want to pretty
      // print all the values in a blob you will need to know the types (and
      // therefore sizes) of all fields in a blob.
      ErrorOr<TypedAttr> valueOrError =
          readAttributeFromMemory(blob.baseAddr, IndexType::get(ctx));
      TypedAttr value;
      if (valueOrError.isError()) {
        value = IntegerAttr::get(IndexType::get(ctx), 0);
      } else {
        value = valueOrError.takeValue();
      }

      std::string str;
      llvm::raw_string_ostream os(str);
      os << value;
      llvm::dbgs() << "blob:\n";
      llvm::dbgs() << "\tbase address:" << blob.baseAddr
                   << "\n\tsize: " << blob.size << "\n\talign: " << blob.align
                   << "\n\trefCount: " << blob.refCount
                   << "\n\tcontents at physical address: " << data.c_str()
                   << "\n\tvalue as index (fixme): " << os.str().c_str()
                   << "\n";

      // TODO: Print interpreter addresses instead of PointerRegions to avoid
      // dependence on number of blobs per table?
      if (blob.pointerRegions) {
        llvm::dbgs() << "\tpointer regions: ";
        llvm::BitVector bits = *blob.pointerRegions;
        for (size_t i = 0; i < bits.size(); ++i) {
          if (bits[i] == 1) {
            PointerRegion ptrRegion = ptrRegionAtIndex(blob, i);
            llvm::dbgs() << "(" << ptrRegion.offset << ", "
                         << ptrRegion.blobIndex << ", " << ptrRegion.blobOffset
                         << "), ";
          }
        }
        llvm::dbgs() << "\n";
      }
      // TODO: map PointerAttrs to PointerRegions in SymbolAttr
      if (blob.symbolRegions) {
        llvm::dbgs() << "\tsymbol regions: ";
        llvm::BitVector bits = *blob.symbolRegions;
        for (size_t i = 0; i < bits.size(); ++i) {
          if (bits[i] == 1)
            llvm::errs() << symbols[i] << ", ";
        }
        llvm::dbgs() << "\n";
      }
    }
    llvm::dbgs() << "\n";
  };
  printBlob(MemoryKind::Heap);
  printBlob(MemoryKind::Stack);
  printBlob(MemoryKind::ConstGlobal);
  printBlob(MemoryKind::Persistent);
}
