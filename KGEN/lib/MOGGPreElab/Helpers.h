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
#include "KGEN/MOGGPreElab/MOGGTensorAccessor.h"

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

inline bool isTensor(Attribute maybeTensor) {
  if (auto symbol = dyn_cast<StringAttr>(maybeTensor))
    return "MOGGTensor::Tensor" == symbol;
  return false;
}

inline bool isXType(LIT::StructType maybeTensor, StringLiteral root,
                    StringLiteral className) {
  if (maybeTensor.getSymbol().getRootReference() != root)
    return false;
  return maybeTensor.getSymbol().getLeafReference() == className;
}

inline bool isMOGGTensor(LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "MOGGTensor", "Tensor");
}

inline bool isExtensibilityTensor(LIT::StructType maybeTensor) {
  return isXType(maybeTensor, "extensibility", "Tensor");
}

inline bool isDPSTensor(LIT::StructType maybeTensor) {
  return maybeTensor.getSymbol().getRootReference().strref().starts_with(
             "tensor_utils") &&
         maybeTensor.getSymbol().getLeafReference() == "UnsafeTensorSlice";
}

inline bool isCustomType(LIT::StructType maybeCustom) {
  return !isMOGGTensor(maybeCustom) && !isExtensibilityTensor(maybeCustom);
}

/// Returns true if there is at least one recognizable tensor on the signature.
inline bool hasAtLeastOneTensor(GeneratorOp generator) {
  ArrayAttr names =
      dyn_cast_or_null<ArrayAttr>(generator->getAttr(MOGG_ARG_TYPE_NAMES));
  if (!names)
    return false;
  for (Attribute attr : names.getValue()) {
    if (isTensor(attr))
      return true;
  }
  return false;
}

/// Given a mojo function pull the tensor parameter information off of it. I.E
/// which parameter corresponds to which parameter in a given input.
inline std::optional<MOGG::MOGGTensorParamAccessor>
getTensorRepFromFunctionInput(GeneratorOp generator, size_t index) {
  ArrayAttr names =
      dyn_cast_or_null<ArrayAttr>(generator->getAttr(MOGG_ARG_TYPE_NAMES));
  ArrayAttr types =
      dyn_cast_or_null<ArrayAttr>(generator->getAttr(MOGG_ARG_PARAMS));

  if (!names || !types || names.size() <= index || types.size() <= index)
    return std::nullopt;

  if (!isTensor(names.getValue()[index]))
    return std::nullopt;

  ArrayAttr params = dyn_cast<ArrayAttr>(types.getValue()[index]);
  if (!params)
    return std::nullopt;

  MOGG::MOGGTensorParamAccessor tensor;
  for (auto [paramIdx, param] : llvm::enumerate(params.getValue())) {
    if (auto typedAttr = dyn_cast<TypedAttr>(param))
      tensor.assignParam(typedAttr, paramIdx);
  }
  return tensor;
}

/// Remove the decorators from the function. Return true if any function had the
/// kernel decorators.
bool stripDecorators(LIT::FuncOp func);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H
