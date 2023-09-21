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

#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

POP::PackType LIT::getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVarArg(index)
             ? ::cast<POP::PackType>(sig.getValueInputs()[index])
             : nullptr;
}

bool LIT::canZeroCostConvertSignature(LITSignatureType from,
                                      LITSignatureType to) {
  if (from.getArgNames().size() != to.getInputConventions().size())
    return false;
  auto newMetadata =
      FnMetadataAttr::get(from.getContext(), from.getArgNames(),
                          to.getDefaultArguments(), to.getDefaultParameters());
  auto newSig = SignatureType::get(
      to.getValues(), to.getInputParamTypes(), to.getResultParamTypes(),
      to.getInputConventions(), to.getFnEffects(), newMetadata);
  return newSig == from;
}
