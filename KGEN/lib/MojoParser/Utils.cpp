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

#include "Support/Compiler/Diags.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

POP::PackType LIT::getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVarArg(index)
             ? ::cast<POP::PackType>(sig.getValueInputs()[index])
             : nullptr;
}

bool LIT::canZeroCostConvertSignature(Type fromType, Type toType) {
  auto from = dyn_cast<LITSignatureType>(fromType);
  auto to = dyn_cast<LITSignatureType>(toType);
  if (!from || !to)
    return false;

  if (from.getArgNames().size() != to.getInputConventions().size())
    return false;
  if (from.getParamNames().size() != to.getInputParamTypes().size())
    return false;

  // Pos-or-kw arguments can be passed positionally.
  for (auto [toKind, fromKind] :
       llvm::zip(to.getArgPassingKinds(), from.getArgPassingKinds())) {
    if (toKind != fromKind) {
      if (toKind == PassingKind::PosOnly && fromKind == PassingKind::PosOrKw)
        continue;
      return false;
    }
  }

  auto newMetadata =
      FnMetadataAttr::get(from.getContext(), from.getArgNames(),
                          from.getArgPassingKinds(), from.getParamNames(),
                          to.getDefaultArguments(), to.getDefaultParameters());
  auto newSig = LITSignatureType::get(
      to.getValues(), to.getInputParamTypes(), to.getResultParamTypes(),
      to.getInputConventions(), to.getFnEffects(), newMetadata);
  return newSig == from;
}

void LIT::emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
                                   size_t maxAllowed, size_t numActual,
                                   Twine argOrParam) {
  diag << " expects ";

  // Tailor the diagnostic if the exact number of expected args is known.
  if (minRequired == maxAllowed && numActual != minRequired) {
    diag << minRequired << " " << argOrParam << plural(minRequired);
  } else if (numActual < minRequired) {
    diag << "at least " << minRequired << " " << argOrParam
         << plural(minRequired);
  } else {
    assert(numActual > maxAllowed);
    diag << "at most " << maxAllowed << " " << argOrParam << plural(maxAllowed);
  }

  diag << ", but " << numActual << plural(numActual, " was", " were")
       << " specified";
}

void LIT::emitUnexpectedKeywords(InflightDiag &diag,
                                 SmallVectorImpl<StringRef> &&unknownKeywords,
                                 StringRef argOrParam) {
  size_t numUnknownKws = unknownKeywords.size();
  diag << "unexpected keyword " << argOrParam << plural(numUnknownKws) << ": ";

  // We need to sort the unknown keywords to have reproducible errors.
  llvm::sort(unknownKeywords);
  llvm::interleave(
      unknownKeywords, [&](StringRef str) { diag << "'" << str << "'"; },
      [&]() { diag << ", "; });
}
