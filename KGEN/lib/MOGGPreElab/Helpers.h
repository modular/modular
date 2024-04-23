//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOGGPREELAB_HELPERS_H
#define KGEN_LIB_MOGGPREELAB_HELPERS_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGTensorAccessor.h"

namespace M::KGEN::MOGGPreElab {

namespace {

// We have a special mojo hook which show us what the canonical lambda
// looks like and a call which tells us the resulting type with the lambda
// applied.
struct LambdaTemplate {
  LambdaTemplate() = default;

  // Scan the hook for the properties that we know exist.
  LambdaTemplate(GeneratorOp hook) : templateOp(hook) {
    for (auto region : hook.getOps<KGEN::ParamDeclareRegionOp>())
      canonicalLambda = region;
    for (auto call : hook.getOps<KGEN::CallOp>())
      callUsingLambda = call;
  }
  // The op we are pulling this info from.
  KGEN::GeneratorOp templateOp;

  // This the the template lambda we will clone as the input or output lambda.
  KGEN::ParamDeclareRegionOp canonicalLambda;

  // This call shows us how the lambda needs to be bound.
  KGEN::CallOp callUsingLambda;
};

bool isTensor(KGEN::LIT::DeclRefType maybeTensor) {
  // Look at the top level symbol name, it is structured like
  // Folder::File::ClassName.
  ArrayRef<FlatSymbolRefAttr> attr =
      maybeTensor.getSymbol().getNestedReferences();
  if (attr.size() == 0)
    return false;

  if (maybeTensor.getSymbol().getRootReference() != "MOGGTensor")
    return false;

  StringRef className = attr[attr.size() - 1].getValue();
  if (className == "Tensor")
    return true;
  return false;
}

std::optional<LIT::LITSignatureType> getSourceSig(GeneratorOp gen) {
  std::optional<PreservedAttr> sig = gen.getSourceSignature();
  if (!sig.has_value())
    return std::nullopt;
  auto typeAttr = dyn_cast<TypeAttr>(sig.value().getValue());
  if (!typeAttr)
    return std::nullopt;
  auto litSig = dyn_cast<LIT::LITSignatureType>(typeAttr.getValue());
  if (!litSig)
    return std::nullopt;
  return litSig;
}

// Returns true if there is at least one recognizable tensor on the signature.
[[maybe_unused]] bool hasAtLeastOneTensor(GeneratorOp generator) {
  std::optional<LIT::LITSignatureType> litSig = getSourceSig(generator);
  if (!litSig.has_value())
    return false;

  for (Type metadata : litSig->getValues().getInputs()) {
    // Tensors are expected to be passed as references.
    auto asLitRef = dyn_cast<LIT::RefType>(metadata);
    if (!asLitRef)
      continue;

    auto asDeclRef =
        dyn_cast<KGEN::LIT::DeclRefType>(asLitRef.getElementType());
    if (!asDeclRef)
      continue;
    if (isTensor(asDeclRef))
      return true;
  }

  return false;
}

// Given a mojo function pull the tensor parameter information off of it. I.E
// which parameter corresponds to which parameter in a given input.
[[maybe_unused]] std::optional<MOGG::MOGGTensorParamAccessor>
getTensorRepFromFunctionInput(GeneratorOp generator, size_t index) {
  std::optional<PreservedAttr> sig = generator.getSourceSignature();
  if (!sig.has_value())
    return std::nullopt;
  auto typeAttr = cast<TypeAttr>(sig.value().getValue());
  auto litSig = cast<LIT::LITSignatureType>(typeAttr.getValue());

  Type metadata = litSig.getValues().getInputs()[index];

  // Tensors are expected to be passed as references.
  auto asLitRef = dyn_cast<LIT::RefType>(metadata);
  if (!asLitRef)
    return std::nullopt;

  auto asDeclRef = dyn_cast<KGEN::LIT::DeclRefType>(asLitRef.getElementType());
  if (!asDeclRef)
    return std::nullopt;
  if (!isTensor(asDeclRef))
    return std::nullopt;

  MOGG::MOGGTensorParamAccessor tensor;
  for (auto [paramIdx, param] : llvm::enumerate(asDeclRef.getParamValues()))
    tensor.assignParam(param, paramIdx);

  return tensor;
}

} // namespace

/// Remove the decorators from the function. Return true if any function had the
/// kernel decorators.
bool stripDecorators(GeneratorOp func);

} // namespace M::KGEN::MOGGPreElab

#endif // KGEN_LIB_MOGGPREELAB_HELPERS_H
