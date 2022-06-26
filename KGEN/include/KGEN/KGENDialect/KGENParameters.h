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

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {

/// Return true if the attribute is a valid parameter expression.
bool isValidParameterExpr(Attribute value);

/// Given a kernel, generator, or generator interface operation, return an array
/// of `ParamDeclAttr`s for the inputs and the array of `ParamDeclAttr`s for the
/// result parameters.  A concrete kernel will always return empty arrays.
std::pair<ArrayRef<Attribute>, ArrayRef<Attribute>>
getDeclParameterInfo(Operation *decl);

/// Scan the body of the specified operation checking invariants on parameters,
/// diagnosing errors and returning failure if so.  This is used by verifiers
/// for ops with bodies, like kgen.generator.
LogicalResult checkParametersInOpBody(Operation *op);

} // namespace M::KGEN

#endif // KGEN_KGENPARAMETERS_H
