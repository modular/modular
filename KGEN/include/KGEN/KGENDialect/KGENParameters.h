//===- KGENParameters.h ---------------------------------------------------===//
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

/// Return true if the attribute is a valid parameter expression.
bool isValidParameterExpr(Attribute value);

/// This class holds descriptions about parameter definitions and uses in a
/// kernel or kernel generator context.
class ParameterDeclsAndUses {
public:
  ParameterDeclsAndUses(ParameterDeclsAndUses &&other) = default;

  /// Collect information about the parameter definitions and uses in the
  /// specified operation.  This emits an error and returns failure on an IR
  /// verification error.
  static FailureOr<ParameterDeclsAndUses> calculate(Operation *op);

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

private:
  ParameterDeclsAndUses() = default;
  ParameterDeclsAndUses(const ParameterDeclsAndUses &) = delete;
  void operator=(const ParameterDeclsAndUses &) = delete;
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
