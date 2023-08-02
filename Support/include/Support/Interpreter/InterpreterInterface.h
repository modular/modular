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

  /// Allocate internal interpreter memory of a requested size.
  int64_t allocateMemory(size_t size, size_t align, MemoryKind kind);

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
        : origin(origin), func(func) {}

    Operation *origin;
    Operation *func;
    DenseMap<Value, Attribute> values;
  };

  /// Push a new stack frame.
  void pushFrame(Operation *origin, Operation *func) {
    stack.emplace_back(origin, func);
  }

  /// Pop the current stack frame, returning the origin operation.
  Operation *popFrame() {
    Operation *origin = stack.back().origin;
    stack.pop_back();
    return origin;
  }

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
    stack.back().values[value] = attr;
  }

  /// Map the results of the current operation.
  void mapResults(ArrayRef<Attribute> results) {
    assert(pc->getNumResults() == results.size());
    for (auto [result, value] : llvm::zip(pc->getResults(), results))
      mapOrOverwrite(result, value);
  }

  /// Lookup a constant value for the value.
  Attribute lookupValue(Value value) {
    Attribute attr = stack.back().values[value];
    assert(attr && "value was not mapped");
    return attr;
  }

private:
  /// Run the interpreter until completion, returning the final results of the
  /// operation.
  ErrorTreeOr<SmallVector<Attribute>> runInterpreter();

  /// The interpreter target configuration.
  TargetInfoAttr target;

  /// This struct represents a piece of memory in the interpreter.
  struct MemoryBlob {
    /// The kind of memory in the slot.
    MemoryKind kind;
    /// The base address of the blob.
    int64_t baseAddr;
    /// The size of the blob.
    size_t size;
    /// The alignment of the blob.
    size_t align;
    /// The actual memory managed by the interpreter.
    std::unique_ptr<void, void (*)(void *)> memory;
  };

  /// Get the memory blob corresponding to the address.
  ErrorOr<MemoryBlob &> getBlob(int64_t addr);

  /// Exchange raw pointers to interpreter memory to dialect resource references
  /// upon exit from the interpreter.
  ErrorOrSuccess exchangeInterpreterMemory(Region &entry,
                                           MutableArrayRef<Attribute> results);

  /// An internal memory table.
  /// TODO: Support different address spaces.
  std::vector<MemoryBlob> memory;

  /// The current operation being interpreted. The interpreter exits when the
  /// operation is null, in which case the required invariant be that the stack
  /// frame is empty.
  Operation *pc;

  /// A call stack. The values in the current frame are available to the
  /// operation being interpreted.
  std::vector<StackFrame> stack;

  /// An optional list of return values.
  std::optional<SmallVector<Attribute>> returnValues;

  /// The blob manager to materializing interpreter memory into the IR. Access
  /// to the blob manager is thread-safe.
  MBlobManagerInterface &blobMgr;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.h.inc"
#include "Support/Interpreter/MemoryableTypeInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
