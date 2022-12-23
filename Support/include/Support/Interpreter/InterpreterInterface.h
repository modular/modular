//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/SymbolTableAnalysis.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/DebugStringHelper.h"

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

namespace M {
class InterpreterState {
public:
  InterpreterState(SymbolTableAnalysis &analysis) : analysis(analysis) {}

  /// Get the top-level symbol table.
  SymbolTable &getSymbolTable() { return analysis.getTopLevelSymbolTable(); }

  /// Get the symbol table collection.
  SymbolTableCollection &getSymbolTables() {
    return analysis.getSymbolTables();
  }

  /// Allocate a certain amount of memory of a given type. Reads and writes to
  /// these memory locations of incorrect type will fail.
  size_t allocateMemory(unsigned numElements, Type type);

  /// Attempt to read the given memory location.
  ErrorOr<TypedAttr> readMemory(size_t addr, Type type) const;

  /// Attempt to write to the given memory location.
  ErrorOrSuccess writeMemory(size_t addr, TypedAttr value);

private:
  /// The cached symbol table.
  SymbolTableAnalysis &analysis;

  /// An internal memory table.
  std::vector<std::pair<TypedAttr, Type>> memory;
};
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/Interpreter/InterpreterInterface.h.inc"

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
