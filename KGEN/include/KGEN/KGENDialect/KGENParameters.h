//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains helper functions for working with KGEN parameter
// expressions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPARAMETERS_H
#define KGEN_KGENPARAMETERS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::KGEN {
/// This class holds descriptions about parameter definitions and uses in a
/// func or generator context.
class ParameterDeclsAndUses {
public:
  ParameterDeclsAndUses() = default;

  /// Collect information about the parameter definitions and uses in the
  /// specified operation.  This assumes the IR is in a valid state. Returns the
  /// declarations and uses for the top-level operation and those of any nested
  /// scopes.
  DenseMap<DeclInterface, ParameterDeclsAndUses> calculate(DeclInterface op);

  /// Check deep invariants for a func/generator decl body, used by the
  /// verifiers for these operations.  If a problem is detected, this emits an
  /// error and returns failure. Return the declarations and uses for the
  /// top-level operation.
  LogicalResult calculateAndVerify(DeclInterface op,
                                   SymbolTableCollection &symbolTables);

  /// This defines the operation and the ParamDeclAttr inside of it that defines
  /// a parameter of a specified name.
  SmallDenseMap<StringAttr, std::pair<Operation *, ParamDeclAttr>> decls;

  /// A single operation may define and use multiple parameter declarations,
  /// either directly or through types on attributes and SSA operands/results.
  ///
  /// This list keeps track of all of the operations that define and use
  /// parameter declarations.  It is ordered in a topological order "top down"
  /// in the parameter dependence graph.  Each operation will only have a single
  /// entry in this list.
  ///
  /// This provides a handy list of parameter uses that the operation refers to,
  /// which will be empty if the operation just defines parameters but doesn't
  /// use any.  You can get its parameter declarations directly from its
  /// attribute list.
  ///
  /// Note that operations that use parameter expressions but not a
  /// ParamDeclRefAttr will not appear in this list.
  SmallVector<std::pair<Operation *, SmallVector<ParamDeclRefAttr>>, 8>
      usersAndDeclarers;

  /// Keep track of the operations which contain parameter expressions but which
  /// do not use or declare parameters themselves. These expressions need to be
  /// evaluated during elaboration.
  SmallVector<Operation *> constExprOps;

  /// Keep track of any nested parameter scopes encountered.
  SmallVector<DeclInterface> nestedDecls;

  /// Keep track of uses of parameters that were defined in a higher scope.
  SmallPtrSet<ParamDeclRefAttr, 8> usesFromAbove;

private:
  FailureOr<DenseMap<DeclInterface, ParameterDeclsAndUses>>
  calculateAndPotentiallyVerify(DeclInterface op,
                                SymbolTableCollection *symbolTable);
};
} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
