//===- KGEN/KGENDialect/KGENParameters.h ----------------------------------===//
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

namespace M::KGEN {
class ParamDeclAttr;
class ParamDeclRefAttr;

/// This class holds descriptions about parameter definitions and uses in a
/// func or generator context.
class ParameterDeclsAndUses {
public:
  ParameterDeclsAndUses(ParameterDeclsAndUses &&other) = default;

  /// Collect information about the parameter definitions and uses in the
  /// specified operation.  This assumes the IR is in a valid state.
  static ParameterDeclsAndUses calculate(Operation *op) {
    auto result = calculateAndPotentiallyVerify(op, nullptr);
    assert(succeeded(result) && "IR should be legal here!");
    return std::move(result.value());
  }

  /// Check deep invariants for a func/generator decl body, used by the
  /// verifiers for these operations.  If a problem is detected, this emits an
  /// error and returns failure.
  static FailureOr<ParameterDeclsAndUses>
  calculateAndVerify(Operation *op, SymbolTableCollection &symbolTables) {
    return calculateAndPotentiallyVerify(op, &symbolTables);
  }

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

  /// Return a list containing just the operations that are using and defining
  /// parameters in the analyzed region.
  SmallVector<Operation *> getUsingAndDeclaringOps() const {
    SmallVector<Operation *> result;
    result.reserve(usersAndDeclarers.size());
    for (const auto &elt : usersAndDeclarers)
      result.push_back(elt.first);
    return result;
  }

private:
  static FailureOr<ParameterDeclsAndUses>
  calculateAndPotentiallyVerify(Operation *op,
                                SymbolTableCollection *symbolTables);

  ParameterDeclsAndUses() = default;
  ParameterDeclsAndUses(const ParameterDeclsAndUses &) = delete;
  void operator=(const ParameterDeclsAndUses &) = delete;
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
