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
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/POPDialect/POPTypes.h"

#include "Support/Compiler/Diags.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

PackType LIT::getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVarArg(index) ? ::cast<PackType>(sig.getValueInputs()[index])
                                 : nullptr;
}

bool LIT::canZeroCostConvert(SharedState &shared, ASTType fromType,
                             ASTType toType) {
  // Permit upcasting any `!lit.metatype` to `!kgen.anyregtype` or
  // `!kgen.anytype`.
  // FIXME(traits): Binding a Mojo type to an MLIR type is a hack. We should
  // forbid this when traits are fully operational.
  if (isa<MetaTypeType>(fromType) && isa<AnyTypeType, AnyRegTypeType>(toType))
    return true;

  auto fromRef = dyn_cast<DeclRefType>(fromType);
  auto toRef = dyn_cast<DeclRefType>(toType);
  // Permit casting between concrete `!kgen.declref` types with different
  // metatypes. This arises when binding concrete types to generic types.
  if (fromRef && toRef && fromRef.getSymbol() == toRef.getSymbol() &&
      fromRef.getParamValues() == toRef.getParamValues())
    return true;

  // Check for closure structs and dig out their underlying signature types to
  // check whether the conversion can occur.
  auto fromDecl = dyn_cast_or_null<StructDeclOp>(fromType.getDecl(shared));
  auto toDecl = dyn_cast_or_null<StructDeclOp>(toType.getDecl(shared));
  if (fromDecl && toDecl) {
    SignatureType fromSig = fromDecl.getClosureSignature().value_or(nullptr);
    SignatureType toSig = toDecl.getClosureSignature().value_or(nullptr);
    if (fromSig && toSig) {
      // Compare the specialized signatures.
      auto cannotError = []() -> InFlightDiagnostic {
        llvm_unreachable("expected a valid specialized signature");
      };
      fromSig = fromSig.getSpecializedSignature(fromRef.getParamValues(),
                                                cannotError);
      toSig =
          toSig.getSpecializedSignature(toRef.getParamValues(), cannotError);
      return canZeroCostConvert(shared, fromSig, toSig);
    }
    return false;
  }

  auto from = dyn_cast<LITSignatureType>(fromType);
  auto to = dyn_cast<LITSignatureType>(toType);
  if (!from || !to)
    return false;

  // Allow signature types to be converted for free if they differ only in
  // argument or parameter names.
  if (from.getArgNames().size() != to.getInputConventions().size())
    return false;
  if (from.getParamNames().size() != to.getNumInputParams())
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

  auto newMetadata = FnMetadataAttr::get(
      from.getContext(), from.getArgNames(), from.getArgPassingKinds(),
      from.getParamNames(), from.getParamPassingKinds(),
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

/// Emit a comma separated list of strings sorted alphabetically.
static void emitSortedNames(InflightDiag &diag,
                            SmallVectorImpl<StringRef> &&names) {
  llvm::sort(names);
  llvm::interleave(
      names, [&](StringRef str) { diag << "'" << str << "'"; },
      [&]() { diag << ", "; });
}

void LIT::emitUnexpectedKeywords(InflightDiag &diag,
                                 SmallVectorImpl<StringRef> &&unknownKeywords,
                                 StringRef argOrParam) {
  diag << "unexpected keyword " << argOrParam << plural(unknownKeywords.size())
       << ": ";
  emitSortedNames(diag, std::move(unknownKeywords));
}

void LIT::emitPosOnlyPassedByKw(InflightDiag &diag,
                                SmallVectorImpl<StringRef> &&names,
                                StringRef argOrParam) {
  size_t numNames = names.size();
  diag << "positional-only " << argOrParam << plural(numNames)
       << " passed as keyword " << argOrParam << plural(numNames) << ": ";
  emitSortedNames(diag, std::move(names));
}
