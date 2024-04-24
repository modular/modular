//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENOPS_H
#define KGEN_KGENDIALECT_KGENOPS_H

#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M {

//===----------------------------------------------------------------------===//
// KGENModule
//===----------------------------------------------------------------------===//

/// A KGEN module wraps a `ModuleOp` and a `SymbolTableCollection` for
/// convenient nested symbol lookups across the module.
class KGENModule {
public:
  /// Create KGEN module with the provided module and symbol table collection.
  KGENModule(ModuleOp module, SymbolTableCollection &symbolTable)
      : module(module), symbolTable(symbolTable) {}

  /// Get the KGEN module from the provided operation.
  static KGENModule from(Operation *op, SymbolTableCollection &symbolTable) {
    return {op->getParentOfType<ModuleOp>(), symbolTable};
  }

  template <typename OpT>
  OpT lookup(SymbolRefAttr symbol) {
    return dyn_cast_or_null<OpT>(symbolTable.lookupSymbolIn(module, symbol));
  }

private:
  /// The top-level IR module.
  ModuleOp module;

  /// A collection of symbol tables.
  SymbolTableCollection &symbolTable;
};

namespace KGEN {
//===----------------------------------------------------------------------===//
// Source location utilities
//===----------------------------------------------------------------------===//

/// Extract the original source location from a call location, taking into
/// account debuginfo and other structure within locations.
FileLineColLoc extractSourceLoc(Location callLoc);
} // namespace KGEN

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
