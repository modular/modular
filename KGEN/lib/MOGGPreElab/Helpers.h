//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_HELPERS_H
#define KGEN_LIB_MOGGPREELAB_HELPERS_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"

namespace M::KGEN::MOGGPreElab {

/// We have a special mojo hook which show us what the canonical lambda
/// looks like and a call which tells us the resulting type with the lambda
/// applied.
struct LambdaTemplate {
  LambdaTemplate() = default;

  /// Scan the hook for the properties that we know exist.
  LambdaTemplate(GeneratorOp hook) : templateOp(hook) {
    for (auto region : hook.getOps<ParamDeclareRegionOp>())
      canonicalLambda = region;
    for (auto call : hook.getOps<CallOp>())
      callUsingLambda = call;
  }
  /// The op we are pulling this info from.
  GeneratorOp templateOp;

  /// This the the template lambda we will clone as the input or output lambda.
  ParamDeclareRegionOp canonicalLambda;

  // This call shows us how the lambda needs to be bound.
  CallOp callUsingLambda;
};

inline bool isXType(LIT::StructType maybeTensor, StringLiteral root,
                    StringLiteral className) {
  if (maybeTensor.getSymbol().getRootReference() != root)
    return false;
  return maybeTensor.getSymbol().getLeafReference() == className;
}

inline bool isExtensibilityTensor(LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "extensibility", "Tensor");
}

inline bool isDPSTensor(LIT::StructType maybeTensor) {
  return maybeTensor.getSymbol().getRootReference().strref().starts_with(
             "tensor_utils") &&
         maybeTensor.getSymbol().getLeafReference() == "ManagedTensorSlice";
}

inline bool isCustomType(LIT::StructType maybeCustom) {
  return !isExtensibilityTensor(maybeCustom);
}

/// Remove the decorators from the function. Return true if any function had the
/// kernel decorators.
bool stripDecorators(LIT::FnOp func);

void stripDecorators(LIT::StructDeclOp structDecl);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H
