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

/// Given a kernel, generator, or generator interface operation, return an array
/// of `ParamDeclAttr`s for the inputs and the array of `ParamDeclAttr`s for the
/// result parameters.  A concrete kernel will always return empty arrays.
std::pair<ArrayRef<Attribute>, ArrayRef<Attribute>>
getDeclParameterInfo(Operation *decl);

/// This class holds descriptions about parameter definitions and uses in a
/// kernel or kernel generator context.
class ParameterDeclsAndUses {
public:
  ParameterDeclsAndUses(ParameterDeclsAndUses &&other) = default;

  /// Collect information about the parameter definitions and uses in the
  /// specified operation.  This emits an error and returns `None` on an IR
  /// verification error.
  static Optional<ParameterDeclsAndUses> calculate(Operation *op);

  /// This defines the operation and the ParamDeclAttr inside of it that defines
  /// a parameter of a specified name.
  SmallDenseMap<StringAttr, std::pair<Operation *, ParamDeclAttr>> decls;

  /// Parameter uses can occur in any attribute and even in in types.  We
  /// collect all the uses we see by their operation.  Remember that attributes
  /// are uniqued, so the same ParamDeclRefAttr can be used by multiple
  /// operations, or even multiple times in the same operation.
  SmallVector<std::pair<Operation *, ParamDeclRefAttr>, 8> uses;

private:
  ParameterDeclsAndUses() {}
  ParameterDeclsAndUses(const ParameterDeclsAndUses &) = delete;
  void operator=(const ParameterDeclsAndUses &) = delete;
};

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
