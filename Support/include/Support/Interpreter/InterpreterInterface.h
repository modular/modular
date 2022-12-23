//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/SymbolTableAnalysis.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/DebugStringHelper.h"

namespace M {
class MemoryReference;

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

class InterpreterState {
public:
  InterpreterState(SymbolTableAnalysis &analysis, TargetInfoAttr target)
      : analysis(analysis), target(target) {}

  /// Get the top-level symbol table.
  SymbolTable &getSymbolTable() { return analysis.getTopLevelSymbolTable(); }

  /// Get the symbol table collection.
  SymbolTableCollection &getSymbolTables() {
    return analysis.getSymbolTables();
  }

  /// Get the interpreter target.
  TargetInfoAttr getTarget() const { return target; }

  /// Allocate internal interpreter memory of a requested size.
  /// TODO: Allow alignment as well.
  intptr_t allocateMemory(size_t size);

  /// Get a memory reference.
  ErrorOr<MemoryReference> getMemory(intptr_t addr);

private:
  /// Get a pointer to the underlying memory given a memory reference.
  ErrorOr<void *> materializeReference(intptr_t addr, size_t size);

  /// Allow memory references to materialize themselves.
  friend class MemoryReference;

  /// The cached symbol table.
  SymbolTableAnalysis &analysis;

  /// The interpreter targt configuration.
  TargetInfoAttr target;

  /// An internal memory table.
  std::vector<uint8_t> memory;
};

//===----------------------------------------------------------------------===//
// MemoryableTypeInterface
//===----------------------------------------------------------------------===//

/// This class encapsulates a "safe" reference to raw interpreter memory.
class MemoryReference {
public:
  /// Request a chunk of memory of a certain size.
  ErrorOr<void *> get(size_t size);

private:
  MemoryReference(InterpreterState &state, intptr_t addr)
      : state(state), addr(addr) {}

  /// Allow the interpreter state to create memory references.
  friend class InterpreterState;

  /// The underlying interpreter state.
  InterpreterState &state;

  /// The "address" of the memory in the interpreter.
  intptr_t addr;
};

/// Write an attribute value of a given type to the provided chunk of memory.
/// This method should be used over invoking the interface methods directly,
/// since it covers builtin attributes and types.
ErrorOrSuccess writeAttributeToMemory(TypedAttr value, MemoryReference ref);

/// Read an attribute value of the given type from the provided chunk of memory.
/// This method should be used over invoking the interface methods directly,
/// since it covers builtin attributes and types.
ErrorOr<TypedAttr> readAttributeFromMemory(Type type, MemoryReference ref);

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.h.inc"
#include "Support/Interpreter/MemoryableTypeInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
