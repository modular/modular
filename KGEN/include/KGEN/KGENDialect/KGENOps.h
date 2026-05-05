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
// Transparent conversion thunks
//===----------------------------------------------------------------------===//
//
// A "transparent" conversion thunk bridges calling conventions to a wrapped
// function while delegating its public identity (linkage name, LLVM
// metadata) to that function. The thunk carries a
// `kgen.transparent_thunk_callee_expr` attribute holding a parametric
// expression that resolves - once the thunk's paramDecls are substituted
// with the callsite's paramValues - to a fully-bound SymbolConstantAttr
// for the wrapped function.

/// Look through a transparent thunk to its wrapped function. Returns the
/// wrapped function's SymbolConstantAttr bound at `callsite`, or null if
/// `gen` is not a transparent thunk.
SymbolConstantAttr resolveTransparentThunkCallee(mlir::Operation *gen,
                                                 SymbolConstantAttr callsite);

} // namespace KGEN

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
