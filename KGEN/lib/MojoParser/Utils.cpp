//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "KGEN/POPDialect/POPTypes.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

POP::PackType LIT::getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVarArg(index)
             ? ::cast<POP::PackType>(sig.getValueInputs()[index])
             : nullptr;
}

bool LIT::canZeroCostConvertSignature(SignatureType from, SignatureType to) {
  if (from.getArgNames().size() != to.getValueInputConventions().size())
    return false;
  auto newMetadata = FnMetadataAttr::get(
      from.getContext(), from.getArgNames(), to.getValueInputConventions(),
      to.getDefaultArguments(), to.getFnEffects());
  auto newSig =
      SignatureType::get(to.getInputParamTypes(), to.getResultParamTypes(),
                         to.getValues(), newMetadata);
  return newSig == from;
}
