//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include "MojoUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
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
  if (isa<MetaTypeType, TraitType>(fromType) && isa<AnyRegTypeType>(toType))
    return true;
  // Discard types can be converted to anything.
  if (isa<DiscardType>(fromType))
    return true;

  // Two types can be converted to each other if their metatypes can be as well.
  if (isa<ParamRefType, DeclRefType>(fromType) &&
      isa<ParamRefType, DeclRefType>(toType)) {
    if (canZeroCostConvert(shared, fromType.getMetaType(),
                           toType.getMetaType()))
      return true;
    // If we hit this branch, we can allow downcasting to a trait type because
    // the conversion has already been checked and enabled.
    if (isa<MetaTypeType>(fromType.getMetaType()) &&
        isa<TraitType>(toType.getMetaType()))
      return true;
  }

  // Check for closure structs and dig out their underlying signature types to
  // check whether the conversion can occur.
  auto fromDecl = dyn_cast_or_null<StructDeclOp>(fromType.getDecl(shared));
  auto toDecl = dyn_cast_or_null<StructDeclOp>(toType.getDecl(shared));
  if (fromDecl && toDecl) {
    SignatureType fromSig = fromDecl.getClosureSignature().value_or(nullptr);
    SignatureType toSig = toDecl.getClosureSignature().value_or(nullptr);
    if (fromSig && toSig) {
      // Compare the specialized signatures.
      auto noEmitError = []() -> InFlightDiagnostic {
        assert(false && "getSpecializedSignature shouldn't error here");
        return {};
      };
      fromSig = fromSig.getSpecializedSignature(fromType.getParamBindings(),
                                                noEmitError);
      toSig =
          toSig.getSpecializedSignature(toType.getParamBindings(), noEmitError);
      return canZeroCostConvert(shared, fromSig, toSig);
    }
    return false;
  }

  // Check reference downcasting.
  if (auto fromRef = dyn_cast<RefType>(fromType))
    if (auto toRef = dyn_cast<RefType>(toType)) {
      // Element types have to be exactly equal.
      if (!ASTType(fromRef.getElementType())
               .isEqualCanon(toRef.getElementType()))
        return false;
      // We can convert from mutable to immutable, but not the other way.
      if (fromRef.getIsMutable() != toRef.getIsMutable() &&
          toRef.getIsMutable())
        return false;
      // We can convert lifetimes to a superset of lifetimes.
      auto toLifetime = toRef.getLifetime();
      auto lifetimeUnion = LifetimeUnionAttr::get(
          toRef.getContext(), {toLifetime, fromRef.getLifetime()});
      return toLifetime == lifetimeUnion;
    }

  auto from = dyn_cast<LITSignatureType>(fromType);
  auto to = dyn_cast<LITSignatureType>(toType);
  if (!from || !to)
    return false;

  // Allow signature types to be converted for free if they differ only in
  // argument names, parameter names, or implicit lifetimes.
  if (from.getArgNames().size() != to.getArgNames().size())
    return false;
  if (from.getNumInputParams() != to.getNumInputParams())
    return false;
  if (from.getInputConventions() != to.getInputConventions())
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

  // Result types, and input/result parameter types must match exactly.
  if (from.getValueResults() != to.getValueResults() ||
      from.getInputParamTypes() != to.getInputParamTypes() ||
      from.getResultParamTypes() != to.getResultParamTypes() ||
      from.getFnEffects() != to.getFnEffects())
    return false;

  // The input argument types may have different implicit lifetimes.
  for (auto [fromTy, toTy, conv] :
       llvm::zip(from.getValueInputs(), to.getValueInputs(),
                 from.getInputConventions())) {
    Type fromTyCmp = fromTy;
    Type toTyCmp = toTy;
    if (SignatureType::hasAddress(conv)) {
      fromTyCmp = ASTType(fromTyCmp).getReferenceElementType();
      toTyCmp = ASTType(toTyCmp).getReferenceElementType();
    }
    if (!ASTType(fromTyCmp).isEqualCanon(toTyCmp))
      return false;
  }

  // Otherwise, everything seems compatible.
  return true;
}

/// Returns a type if there is a shared supertype for the two specified types,
/// e.g. two derived classes may have the same base class even if neither is
/// convertible to the other.  This returns null if there is no common type.
///
/// This is the implementation logic of getZeroCostCommonType and shouldn't be
/// called directly.
static ASTType getZeroCostCommonTypeImpl(SharedState &shared, ASTType type1,
                                         ASTType type2) {
  // Check reference downcasting.
  if (auto type1Ref = dyn_cast<RefType>(type1))
    if (auto type2Ref = dyn_cast<RefType>(type2)) {
      // Element types have to be exactly equal.
      auto eltType = type1Ref.getElementType();
      if (!ASTType(eltType).isEqualCanon(type2Ref.getElementType()))
        return {};

      // If so, we can form a common type with a subset of their mutability and
      // a union of their lifetimes.
      auto lifetime = LifetimeUnionAttr::get(
          type1Ref.getContext(),
          {type1Ref.getLifetime(), type2Ref.getLifetime()});
      return RefType::get(type1Ref.getIsMutable() && type2Ref.getIsMutable(),
                          eltType, lifetime);
    }

  // No common type found.
  return {};
}

/// Returns a type if there is a shared supertype for the two specified types,
/// e.g. two derived classes may have the same base class even if neither is
/// convertible to the other.  This returns null if there is no common type.
ASTType LIT::getZeroCostCommonType(SharedState &shared, ASTType type1,
                                   ASTType type2) {
  if (auto result = getZeroCostCommonTypeImpl(shared, type1, type2)) {
    // Make sure we can always convert to the common type!
    assert(canZeroCostConvert(shared, type1, result) &&
           canZeroCostConvert(shared, type2, result) &&
           "cannot convert to common type?");
    return result;
  }
  return {};
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

bool LIT::canSynthesizeIfMissing(
    StringRef name, bool rpTrivial, bool regPassable,
    std::optional<std::reference_wrapper<SmallVectorImpl<SpecialFunctionKind>>>
        specialFns) {

  auto addSpecialFn = [&](SpecialFunctionKind kind) {
    if (!specialFns)
      return;
    specialFns->get().push_back(kind);
  };
  // Allow types that lack `__del__` to conform. A no-op destructor will be
  // synthesized for them.
  if (name == "__del__") {
    addSpecialFn(SpecialFunctionKind::kDel);
    return true;
  }
  // Trivial types are not allowed to have explicit `__copyinit__` methods, so
  // if the trait requires them, consider them automatically satisfied by
  // trivial types.
  if (rpTrivial && name == "__copyinit__") {
    addSpecialFn(SpecialFunctionKind::kCopyInit);
    return true;
  }
  // All register-passable types are not allowed to have move  constructors, so
  // permit them to conform.
  if (regPassable && name == "__moveinit__") {
    addSpecialFn(SpecialFunctionKind::kMoveInit);
    return true;
  }
  return false;
}

void LIT::markRegionUnreachable(Region *deadRegion, Location unreachableLoc) {
  Block &deadBlock = deadRegion->front();
  {
    Operation *op = &deadBlock.front();
    // Erase bottom up to avoid deleting an op while something uses its results.
    while (&deadBlock.back() != op)
      deadBlock.back().erase();
    op->erase();
  }
  OpBuilder::atBlockBegin(&deadBlock).create<UnreachableOp>(unreachableLoc);
}
