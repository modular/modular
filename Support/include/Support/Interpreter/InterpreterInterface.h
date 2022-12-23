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

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

namespace M {
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

  /// Write an attribute value of a given type to the provided chunk of memory.
  ErrorOrSuccess writeAttributeToMemory(intptr_t addr, TypedAttr value);

  /// Read an attribute value of the given type from the provided chunk of
  /// memory.
  ErrorOr<TypedAttr> readAttributeFromMemory(intptr_t addr, Type type);

  /// Try to get a memory reference at the given address.
  ErrorOr<void *> getMemory(intptr_t addr, size_t size);

private:
  /// The cached symbol table.
  SymbolTableAnalysis &analysis;

  /// The interpreter targt configuration.
  TargetInfoAttr target;

  /// An internal memory table.
  std::vector<uint8_t> memory;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterOpInterface.h.inc"
#include "Support/Interpreter/MemoryableTypeInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
