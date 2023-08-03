//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/ErrorTree.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/DebugStringHelper.h"

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

  //===--------------------------------------------------------------------===//
  // Interpreter Global State

  /// Get the interpreter target.
  TargetInfoAttr getTarget() const { return target; }

  /// Lookup the body of the referenced function. This method is made virtual so
  /// that implementors that don't have a monolithic module available can
  /// implement it differently than a symbol table lookup.
  virtual ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) = 0;

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Management

  /// Allocate stack memory of the request size and alignment on the current
  /// stack frame.
  ErrorOr<int64_t> allocateStackMemory(size_t size, size_t align);

  /// Allocate internal interpreter heap memory of a requested size and
  /// alignment.
  ErrorOr<int64_t> allocateHeapMemory(size_t size, size_t align);

  /// Free heap-allocated memory from the interpreter.
  ErrorOrSuccess freeHeapMemory(int64_t addr);

  /// Try to get a memory reference at the given address.
  ErrorOr<void *> getMemory(int64_t addr, size_t size);

  /// Write an attribute value of a given type to the provided chunk of memory.
  ErrorOrSuccess writeAttributeToMemory(int64_t addr, TypedAttr value);

  /// Read an attribute value of the given type from the provided chunk of
  /// memory.
  ErrorOr<TypedAttr> readAttributeFromMemory(int64_t addr, Type type);

  //===--------------------------------------------------------------------===//
  // Interpreter Control Flow

  /// Run the interpreter starting from the first operation in the entry block
  /// of the provided region given the constant values of the region arguments.
  ErrorTreeOr<SmallVector<Attribute>>
  startInterpreterAt(Region &region, ArrayRef<Attribute> arguments);

  /// Transfer control flow to the given operation. If the operation is null,
  /// this is indicating that the interpreter should exit. Otherwise, the
  /// current return values are taken as the results of the target operation.
  void transferControlFlowTo(Operation *target);

  /// Transfer control flow to the beginning of the given block with the
  /// constant values of the block arguments.
  void transferControlFlowTo(Block *target, ArrayRef<Attribute> arguments);

  //===--------------------------------------------------------------------===//
  // Interpreter Stack Management

  /// A call stack frame contains the call operation and the value map at the
  /// callsite. The entry frame has a null operation. Also keep the operation
  /// the stack frame is for so that if an error occurs, we can emit a nice
  /// stacktrace.
  struct StackFrame {
    StackFrame(Operation *origin, Operation *func)
        : origin(origin), func(func), numStackAllocs(0) {}

    /// The operation that created the frame and invoked the function.
    Operation *origin;
    /// The corresponding function to the frame.
    Operation *func;
    /// The number of memory blobs allocated on the stack. This many blobs
    /// are popped off stack memory when the function returns.
    size_t numStackAllocs;
    /// The map of SSA values to constant values in the current frame.
    DenseMap<Value, Attribute> values;
  };

  /// Push a new stack frame.
  void pushFrame(Operation *origin, Operation *func) {
    stack.emplace_back(origin, func);
  }

  /// Pop the current stack frame, returning the origin operation.
  Operation *popFrame();

  /// Set the return values.
  void setReturnValues(ArrayRef<Attribute> values) {
    assert(!returnValues && "already have return values");
    returnValues = llvm::to_vector(values);
  }

  /// Take the current return values.
  SmallVector<Attribute> takeReturnValues() {
    assert(returnValues && "no return values");
    SmallVector<Attribute> values = std::move(*returnValues);
    returnValues.reset();
    return values;
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
  /// Run the interpreter until completion, returning the final results of the
  /// operation.
  ErrorTreeOr<SmallVector<Attribute>> runInterpreter();

  /// The interpreter target configuration.
  TargetInfoAttr target;

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Model

  /// This struct represents a piece of memory in the interpreter.
  struct MemoryBlob {
    /// The base address of the blob.
    int64_t baseAddr;
    /// The size of the blob.
    size_t size;
    /// The alignment of the blob.
    size_t align;
    /// The actual memory managed by the interpreter.
    std::unique_ptr<void, void (*)(void *)> memory;
  };

  /// A memory table is just a vector of blobs organized by ascending address.
  struct MemoryTable {
    explicit MemoryTable(MemoryKind kind, int64_t minAddr, int64_t maxAddr)
        : kind(kind), minAddr(minAddr), maxAddr(maxAddr) {}

    /// Get the memory blob corresponding to the address.
    ErrorOr<MemoryBlob &> getBlob(int64_t addr);

    /// Allocate a new memory blob .
    ErrorOr<MemoryBlob &> addBlob(size_t size, size_t align);

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

  /// Exchange raw pointers to interpreter memory to dialect resource references
  /// upon exit from the interpreter.
  ErrorOrSuccess externalizeMemory(Region &entry,
                                   MutableArrayRef<Attribute> results);

  /// Exchange memory references for interpreter memory references upon entering
  /// the interpreter.
  ErrorOrSuccess internalizeMemory(MutableArrayRef<Attribute> args);

  /// The blob manager to materializing interpreter memory into the IR. Access
  /// to the blob manager is thread-safe.
  MBlobManagerInterface &blobMgr;

  /// An internal memory table for heap-allocated memory.
  MemoryTable heapMemory;
  /// A stack of stack-allocated blobs.
  MemoryTable stackMemory;

  //===--------------------------------------------------------------------===//
  // Interpreter Execution

  /// The current operation being interpreted. The interpreter exits when the
  /// operation is null, in which case the required invariant be that the stack
  /// frame is empty.
  Operation *pc = nullptr;

  StackFrame &getCurrentFrame() {
    assert(!stack.empty() && "expected a stack frame");
    return stack.back();
  }

  /// A call stack. The values in the current frame are available to the
  /// operation being interpreted.
  std::vector<StackFrame> stack;

  /// An optional list of return values.
  std::optional<SmallVector<Attribute>> returnValues;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.h.inc"
#include "Support/Interpreter/MemoryableTypeInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
