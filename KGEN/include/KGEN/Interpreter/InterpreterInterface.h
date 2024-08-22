//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpDefinition.h"

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

namespace M {
class InterpreterState {
public:
  InterpreterState(MLIRContext *ctx, TargetInfoAttr target = nullptr);
  InterpreterState(TargetInfoAttr target);

  InterpreterState(const InterpreterState &other) = delete;
  InterpreterState(InterpreterState &&other) = default;

  virtual ~InterpreterState() = default;

  MLIRContext *getContext() const { return ctx; }

  //===--------------------------------------------------------------------===//
  // Interpreter Global State

  /// Get the interpreter target.
  TargetInfoAttr getTarget() const { return target; }

  /// Lookup the body of the referenced function. This method is made virtual so
  /// that implementors that don't have a monolithic module available can
  /// implement it differently than a symbol table lookup.
  virtual ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) = 0;

  /// Lookup a type definition. This is used by operations that require an
  /// out-of-line definition of a type to interpret.
  virtual Operation *lookupTypeDefinition(SymbolRefAttr symbol) { return {}; }

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Management

  /// Allocate stack memory of the request size and alignment on the current
  /// stack frame.
  ErrorOr<int64_t> allocateStackMemory(size_t size, size_t align);

  /// Allocate internal interpreter heap memory of a requested size and
  /// alignment.
  ErrorOr<int64_t> allocateHeapMemory(size_t size, size_t align);

  /// Allocate internal interpreter memory for a persistent object.
  ErrorOr<int64_t> allocatePersistentMemory(size_t size, size_t align);

  /// Free heap-allocated memory from the interpreter.
  ErrorOrSuccess freeHeapMemory(int64_t addr);

  /// Map a constant global memory handle into the interpreter if it hasn't been
  /// added yet and return the address to the start of the blob.
  ErrorOr<int64_t> mapConstGlobalMemory(MemoryHandleAttr hdl);

  /// Get writable memory for the given address to interpreter memory.
  ///
  /// The `writePointer` flag must be set if the memory is being written to as a
  /// pointer value. Pointers are special because they are the only fundamental
  /// type whose in-memory representation differs between compile-time and
  /// run-time. The interpreter implements special handling for pointer values
  /// to ensure they are mapped in to and out of the interpreter memory space
  /// when required.
  ///
  /// Clobbering a pointer region is invalid. This can either be partially
  /// overwriting a pointer region with a non-pointer value or a pointer value.
  ErrorOr<void *> getWritableMemory(int64_t addr, size_t size,
                                    bool writePointer = false);

  /// Get readable memory for the given address to interpreter memory.
  ErrorOr<const void *> getReadableMemory(int64_t addr, size_t size);

  /// Write an attribute value of a given type to the provided chunk of memory.
  ErrorOrSuccess writeAttributeToMemory(int64_t addr, TypedAttr value);

  /// Read an attribute value of the given type from the provided chunk of
  /// memory.
  ErrorOr<TypedAttr> readAttributeFromMemory(int64_t addr, Type type);

  /// Allocate a slot of symbolic memory.
  uint64_t allocateSymbolicMemory(TypedAttr init);

  /// Lookup a symbolic memory slot.
  ErrorOr<TypedAttr &> getSymbolicMemory(uint64_t slot);

  /// Read an attribute value of a given type from either a SymbolicPointerAttr
  /// or a PointerAttr.
  ErrorOr<Attribute> readAttributeFromPointer(Attribute pointer,
                                              Type elementType);

  /// Exchange memory references for interpreter memory references upon entering
  /// the interpreter.
  ErrorOrSuccess internalizeMemory(MutableArrayRef<Attribute> args);

  /// Exchange raw pointers to interpreter memory to dialect resource references
  /// upon exit from the interpreter.
  ErrorOrSuccess externalizeMemory(MutableArrayRef<Attribute> results);

  /// Load a single attribute from memory from a memref.
  ErrorOr<TypedAttr> loadAttributeFromMemRef(MemRefAttr memref, Type type);

  //===--------------------------------------------------------------------===//
  // Interpreter Control Flow

  /// Run the interpreter starting from the first operation in the entry block
  /// of the provided region given the constant values of the region arguments.
  ErrorTreeOr<SmallVector<Attribute>>
  executeRegion(Region &region, ArrayRef<Attribute> arguments);

  /// Run the interpreter starting from the provided region using a result slot
  /// calling convention. The result of the function will be the materialized
  /// memory for the result slot. The caller is required to provide the type of
  /// the result slot.
  ErrorTreeOr<TypedAttr>
  executeRegionWithResultSlot(Region &region, ArrayRef<Attribute> arguments,
                              bool isInitSelf,
                              SmartVariant<Type, TypedAttr> result);

  /// Transfer control flow to the given operation. If the operation is null,
  /// this is indicating that the interpreter should exit. Otherwise, the
  /// current return values are taken as the results of the target operation.
  void transferControlFlowTo(Operation *target, ArrayRef<Attribute> values) {
    if (target) {
      block = target->getBlock();
      pc = target->getIterator();
      mapResults(values);
    } else {
      block = nullptr;
      pc = Block::iterator();
      exitValues = llvm::to_vector(values);
    }
  }

  /// Transfer control flow to the beginning of the given block with the
  /// constant values of the block arguments.
  void transferControlFlowTo(Block *target, ArrayRef<Attribute> arguments) {
    for (auto [arg, value] : llvm::zip(target->getArguments(), arguments))
      mapOrOverwrite(arg, value);
    block = target;
    pc = Block::iterator();
  }

  //===--------------------------------------------------------------------===//
  // Interpreter Stack Management

  /// A call stack frame contains the call operation and the value map at the
  /// callsite. The entry frame has a null operation. Also keep the operation
  /// the stack frame is for so that if an error occurs, we can emit a nice
  /// stacktrace.
  struct StackFrame {
    StackFrame() {}
    /// The operation that created the frame and invoked the function.
    Operation *origin;
    /// The corresponding function to the frame.
    Operation *func;
    /// The number of memory blobs allocated on the stack. This many blobs
    /// are popped off stack memory when the function returns.
    size_t numStackAllocs;
    /// The number of symbolic slots allocated on the stack. This many slots are
    /// popped off when the function returns.
    size_t numSymbolicAllocs;
    /// The map of SSA values to constant values in the current frame.
    DenseMap<Value, Attribute> values;
  };

  /// This function executes a function call in the interpreter, where the
  /// current operation is treated as the calling operation that the interpreter
  /// should return to when the callee returns, `body` is the region of the
  /// callee, and `arguments` are the call argument values.
  void callFunctionBody(Region &body, ArrayRef<Attribute> arguments) {
    // Function regions are isolated from above, so push a new stack frame.
    // Then, transfer control flow to the beginning of the function body.
    pushFrame(pc.isValid() ? &*pc : nullptr, body.getParentOp());
    transferControlFlowTo(&body.front(), arguments);
  }

  /// Return from the current function back to the caller using `returnValues`
  /// as the return values of the function.
  void returnFromFunction(ArrayRef<Attribute> returnValues) {
    // Pop the current frame and transfer control flow back to the call
    // operation, using the operands of the return as the results of the call.
    Operation *call = popFrame();
    transferControlFlowTo(call, returnValues);
  }

  /// Return the origin operation of the frame at the given depth in the stack.
  /// If the stack is not deep enough, return null.
  Operation *getOrigin(size_t depth);

  /// Set the value of a named global.
  void setNamedGlobal(StringAttr name, Attribute value) {
    namedGlobals[name] = value;
  }

  /// Get the value of a named global.
  Attribute getNamedGlobal(StringAttr name) {
    return namedGlobals.lookup(name);
  }

  //===--------------------------------------------------------------------===//
  // Interpreter Value Management

  /// Map a value to a constant value, overwriting the previous value if there
  /// was one.
  void mapOrOverwrite(Value value, Attribute attr) {
    getCurrentFrame().values[value] = attr;
  }

  /// Map the results of the current operation.
  void mapResults(ArrayRef<Attribute> results) {
    assert(pc->getNumResults() == results.size());
    for (auto [result, value] : llvm::zip(pc->getResults(), results))
      mapOrOverwrite(result, value);
  }

  /// Lookup a constant value for the value.
  Attribute lookupValue(Value value) {
    Attribute attr = getCurrentFrame().values[value];
    assert(attr && "value was not mapped");
    return attr;
  }

private:
  /// The MLIR context.
  MLIRContext *ctx;

  /// The interpreter target configuration.
  TargetInfoAttr target;

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Model

  /// This struct represents a piece of memory in the interpreter.
  struct MemoryBlob {
    using OwnedMemory = std::unique_ptr<void, void (*)(void *)>;
    using MemoryT = SmartVariant<OwnedMemory, MemoryHandleAttr, std::nullptr_t>;

    /// Create a memory blob. If `hdl` is null, an owned blob will be created.
    explicit MemoryBlob(int64_t baseAddr, size_t size, size_t align,
                        MemoryHandleAttr hdl);

    /// Mark or unmark the given region of the blob as a pointer value.
    ErrorOrSuccess setPointerRegion(int64_t offset, int64_t regionSize,
                                    int64_t pointerSize, bool writePointer);

    /// Return true if the memory is owned by the interpreter.
    bool isOwned() const { return isa<OwnedMemory>(memory); }

    /// Get the handle to external memory.
    MemoryHandleAttr getHandle() const {
      return cast<MemoryHandleAttr>(memory);
    }

    /// Get the pointer to owned memory.
    void *getOwned() const { return cast<OwnedMemory>(memory).get(); }

    /// Get the pointer to the memory.
    void *getMemory() const {
      if (isOwned())
        return getOwned();
      return const_cast<void *>(
          reinterpret_cast<const void *>(getHandle().getData()));
    }

    /// Return true if the memory has been freed.
    bool isFreed() const { return isa<std::nullptr_t>(memory); }

    /// Free the owned memory.
    void free() { memory = nullptr; }

    /// The base address of the blob.
    int64_t baseAddr;
    /// The size of the blob.
    size_t size;
    /// The alignment of the blob.
    size_t align;
    /// The actual memory managed by the interpreter.
    MemoryT memory;
    /// A bit is set for each offset value where pointer regions begin. The
    /// vector is lazily-initialized to save memory.
    std::optional<llvm::BitVector> pointerRegions;
  };

  /// A memory table is just a vector of blobs organized by ascending address.
  struct MemoryTable {
    MemoryTable(MemoryKind kind, int64_t minAddr, int64_t maxAddr)
        : kind(kind), minAddr(minAddr), maxAddr(maxAddr) {}

    /// Get the memory blob corresponding to the address.
    ErrorOr<MemoryBlob &> getBlob(int64_t addr);

    /// Allocate a new memory blob .
    ErrorOr<MemoryBlob &> addBlob(size_t size, size_t align,
                                  MemoryHandleAttr hdl);

    /// Return true if the table contains the address.
    bool contains(int64_t addr) { return addr >= minAddr && addr < maxAddr; }

    /// Reset the table.
    void reset() { blobs.clear(); }

    /// The kind of contained memory.
    MemoryKind kind;
    /// The base address of the table (inclusive).
    int64_t minAddr;
    /// The maximum address of the table (exclusive).
    int64_t maxAddr;
    /// The memory blobs in the table.
    std::vector<MemoryBlob> blobs;
  };

  /// Get the memory blob for the given address. Check that the access is
  /// in-bounds and then compute the offset into the blob.
  ErrorOr<std::pair<MemoryBlob &, int64_t>> getMemory(int64_t addr,
                                                      size_t size);

  ErrorOr<PointerAttr> allocateInternalStackFor(Type type, Type ptrType);

  /// Get the memory table for the memory kind.
  MemoryTable &getTable(MemoryKind kind) {
    return memory[static_cast<unsigned>(kind)];
  }

  /// All interpreter memory tables, containing stack, heap, persistent, and
  /// constant global memory.
  MemoryTable memory[4];

  /// Symbolic memory allocated on the stack frame.
  SmallVector<TypedAttr, 0> symbolicMemory;

  //===--------------------------------------------------------------------===//
  // Interpreter Execution

  StackFrame &getCurrentFrame() {
    assert(!stack.empty() && "expected a stack frame");
    return stack[stackIdx - 1];
  }

  /// Run the interpreter on the function body until completion, returning the
  /// final results of the function.
  ErrorTreeOr<SmallVector<Attribute>>
  interpretFunction(Region &body, ArrayRef<Attribute> arguments);

  /// Push a new stack frame.
  void pushFrame(Operation *origin, Operation *func) {
    StackFrame &frame =
        stackIdx == stack.size() ? stack.emplace_back() : stack[stackIdx];
    // Initialize or re-initialize the frame. This avoids unnecessary memory
    // pressure from freeing and allocating the contained DenseMap.
    frame.origin = origin;
    frame.func = func;
    frame.numStackAllocs = 0;
    frame.numSymbolicAllocs = 0;
    ++stackIdx;
  }

  /// Pop the current stack frame, returning the origin operation.
  Operation *popFrame() {
    // Drop all stack memory on the current frame.
    MemoryTable &table = getTable(MemoryKind::Stack);
    StackFrame &frame = getCurrentFrame();
    auto popBackCount = [](auto &vec, unsigned num) {
      vec.erase(vec.end() - num, vec.end());
    };

    popBackCount(table.blobs, frame.numStackAllocs);
    popBackCount(symbolicMemory, frame.numSymbolicAllocs);

    Operation *origin = frame.origin;
    // Soft-remove the frame from the stack.
    --stackIdx;
    return origin;
  }

  /// Reset all interpreter state.
  void reset();

  /// When the interpreter hits an error, construct an error tree given the
  /// current stack frame.
  ErrorTree addStackTrace(ErrorTree error);

  /// The current block being interpreter. The interpreter exits when the block
  /// is null. in which case the required invariant is that the stack frame is
  /// empty.
  Block *block = nullptr;
  /// The operation within the current block being interpreter. If the iterator
  /// is not valid, then this refers to the beginning of the block.
  Block::iterator pc;

  /// A call stack. The values in the current frame are available to the
  /// operation being interpreted.
  SmallVector<StackFrame, 32> stack;
  size_t stackIdx = 0;

  /// The return values of the top frame of the interpreter. These are set when
  /// the interpreter is exiting from the entry function.
  SmallVector<Attribute> exitValues;

  /// This map implements named global values. Named global values represent a
  /// mechanism for storing SSA value captures at compile time.
  DenseMap<StringAttr, Attribute> namedGlobals;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterOpInterface.h.inc"
#include "KGEN/Interpreter/MemoryableTypeInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
