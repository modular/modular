//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_SYMBOLTABLEANALYSIS_H
#define SUPPORT_COMPILER_SYMBOLTABLEANALYSIS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/AnalysisManager.h"

namespace M {
/// This is a simple analysis that contains a symbol table collection and, for
/// simplicity, a reference to the top-level symbol table. This allows symbol
/// tables to be preserved across passes. Most often, symbol tables are
/// automatically kept up-to-date via the `insert` and `erase` functions.
class SymbolTableAnalysis {
public:
  /// Create the symbol table analysis at the provided top-level operation and
  /// instantiate the symbol table of the top-level operation.
  SymbolTableAnalysis(Operation *op)
      : topLevelSymbolTable(symbolTables.getSymbolTable(op)) {}

  /// Get the symbol table collection.
  SymbolTableCollection &getSymbolTables() { return symbolTables; }

  /// Get the top-level symbol table.
  SymbolTable &getTopLevelSymbolTable() { return topLevelSymbolTable; }

  /// Get the top-level operation as a module.
  ModuleOp getModule() { return cast<ModuleOp>(topLevelSymbolTable.getOp()); }

  /// Symbol tables are kept up-to-date by passes. Assume that the analysis
  /// remains valid.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }

private:
  /// The symbol table collection containing cached symbol tables for all nested
  /// symbol table operations.
  SymbolTableCollection symbolTables;
  /// The symbol table of the top-level operation.
  SymbolTable &topLevelSymbolTable;
};
} // namespace M

#endif // SUPPORT_COMPILER_SYMBOLTABLEANALYSIS_H
