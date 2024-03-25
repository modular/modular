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

PackType LIT::getIfPackType(LITSignatureType sig, size_t index) {
  return sig.isPackVarArg(index) ? ::cast<PackType>(sig.getArguments()[index])
                                 : nullptr;
}

bool LIT::canZeroCostConvert(SharedState &shared, ASTType fromType,
                             ASTType toType) {
  // Permit upcasting any `!lit.anystruct` to `!kgen.type` or
  // `!kgen.anytype`.
  // FIXME(traits): Binding a Mojo type to an MLIR type is a hack. We should
  // forbid this when traits are fully operational.
  if (isa<AnyStructType, TraitType>(fromType) && isa<TypeType>(toType))
    return true;

  // Two types can be converted to each other if their metatypes can be as well.
  if (isa<ParamRefType, DeclRefType>(fromType) &&
      isa<ParamRefType, DeclRefType>(toType)) {
    if (canZeroCostConvert(shared, fromType.getMetaType(),
                           toType.getMetaType()))
      return true;
    // If we hit this branch, we can allow downcasting to a trait type because
    // the conversion has already been checked and enabled.
    if (isa<AnyStructType>(fromType.getMetaType()) &&
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
      fromSig = fromSig.getSpecializedSignature(fromType.getParamBindings());
      toSig = toSig.getSpecializedSignature(toType.getParamBindings());
      return canZeroCostConvert(shared, fromSig, toSig);
    }
    return false;
  }

  // Check lifetime downcasting.  We can convert from (mutable or parametric) to
  // immutable, but nothing else.
  if (auto fromLife = dyn_cast<LifetimeType>(fromType))
    if (auto toLife = dyn_cast<LifetimeType>(toType)) {
      return fromLife.isMutable() == toLife.isMutable() ||
             toLife.isMutableKnown(false);
    }

  // Check reference downcasting.  The only thing allowed to disagree is the
  // lifetime set / mutability.
  if (auto fromRef = dyn_cast<RefType>(fromType))
    if (auto toRef = dyn_cast<RefType>(toType)) {
      // Element types and address space have to be exactly equal.
      if (!ASTType(fromRef.getElementType())
               .isEqualCanon(toRef.getElementType()) ||
          fromRef.getAddressSpace() != toRef.getAddressSpace())
        return false;

      // Verify compatible LifetimeType(mutability).  This is checking the type
      // of the lifetime, which contains its mutability specifier.
      auto toLifetimeType = toRef.getLifetimeType();
      if (fromRef.getLifetimeType() != toLifetimeType &&
          !canZeroCostConvert(shared, fromRef.getLifetimeType(),
                              toLifetimeType))
        return false;

      // We can convert lifetime subset to a lifetimes superset.
      auto toLifetime = toRef.getLifetime();
      auto lifetimeUnion = LifetimeUnionAttr::get(
          {toLifetime,
           LifetimeMutCastAttr::get(fromRef.getLifetime(), toLifetimeType)},
          toLifetimeType);
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
  if (from.getNumParams() != to.getNumParams())
    return false;
  if (from.getArgConventions() != to.getArgConventions())
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
  if (from.getResults() != to.getResults() ||
      from.getParamTypes() != to.getParamTypes() ||
      from.getResultParamTypes() != to.getResultParamTypes() ||
      from.getFnEffects() != to.getFnEffects())
    return false;

  // The input argument types may have different implicit lifetimes.
  for (auto [fromTy, toTy, conv] : llvm::zip(
           from.getArguments(), to.getArguments(), from.getArgConventions())) {
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
      // Element types and addr spaces have to be exactly equal.
      auto eltType = type1Ref.getElementType();
      if (!ASTType(eltType).isEqualCanon(type2Ref.getElementType()) ||
          type1Ref.getAddressSpace() != type2Ref.getAddressSpace())
        return {};

      // If so, we can form a common type with a subset of their mutability and
      // a union of their lifetimes.
      auto isMutableAttr = ParamOperatorAttr::get(
          POC::And, type1Ref.isMutable(), type2Ref.isMutable());

      auto l1 = LifetimeMutCastAttr::get(type1Ref.getLifetime(), isMutableAttr);
      auto l2 = LifetimeMutCastAttr::get(type2Ref.getLifetime(), isMutableAttr);

      auto lifetime =
          LifetimeUnionAttr::get({l1, l2}, cast<LifetimeType>(l1.getType()));
      return RefType::get(eltType, lifetime, type1Ref.getAddressSpace());
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

bool LIT::canSynthesizeIfMissing(StringRef name, bool rpTrivial,
                                 bool regPassable) {
  // Allow types that lack `__del__` to conform. A no-op destructor will be
  // synthesized for them.
  if (name == "__del__")
    return true;
  // Trivial types are not allowed to have explicit `__copyinit__` methods, so
  // if the trait requires them, consider them automatically satisfied by
  // trivial types.
  if (rpTrivial && name == "__copyinit__")
    return true;
  // All register-passable types are not allowed to have move constructors, so
  // permit them to conform.
  if (regPassable && name == "__moveinit__")
    return true;
  return false;
}

void LIT::markRegionUnreachable(Region *deadRegion, Location unreachableLoc) {
  // Erase bottom up to avoid deleting an op while something uses its results.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(deadRegion->front()))) {
    // Avoid erasing ops that correspond to lazily resolved decls.
    if (isa<UnresolvedImportOp, UnresolvedWildcardImportOp>(op))
      continue;
    op.erase();
  }

  OpBuilder::atBlockEnd(&deadRegion->front())
      .create<UnreachableOp>(unreachableLoc);
}

Type LIT::getVariadicKwargsType(Type dictRefType) {
  Type dictType = cast<RefType>(dictRefType).getElementType();
  return cast<TypeConstantAttr>(ASTType(dictType).getParamBindings()[1])
      .getValue();
}

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

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

/// Emit a comma separated list of names, each in '...'.
static void emitNames(InflightDiag &diag, ArrayRef<StringAttr> names) {
  llvm::interleave(
      names, [&](StringAttr str) { diag << str; }, [&]() { diag << ", "; });
}

void LIT::emitUnknownKeywords(InflightDiag &diag,
                              ArrayRef<StringAttr> unknownKeywords,
                              StringRef argOrParam) {
  diag << "unknown keyword " << argOrParam << plural(unknownKeywords.size())
       << ": ";
  emitNames(diag, unknownKeywords);
}

void LIT::emitPosOnlyPassedByKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                                StringRef argOrParam) {
  size_t numNames = names.size();
  diag << "positional-only " << argOrParam << plural(numNames)
       << " passed as keyword operand" << plural(numNames) << ": ";
  emitNames(diag, names);
}

void LIT::emitMissing(InflightDiag &diag, ArrayRef<StringAttr> names,
                      const Twine &kindStr) {
  size_t numNames = names.size();
  diag << "missing " << numNames << " required " << kindStr << plural(numNames)
       << ": ";
  emitNames(diag, names);
}

void LIT::emitByPosAndKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                         const Twine &kindStr) {
  size_t numNames = names.size();
  diag << kindStr << plural(numNames)
       << " passed both as positional and keyword operand: ";
  emitNames(diag, names);
}

void LIT::emitTooManyPositional(InflightDiag &diag, size_t numMaxAllowed,
                                size_t numActual, const Twine &kindStr) {
  diag << "expected at most " << numMaxAllowed << " positional " << kindStr
       << plural(numMaxAllowed) << ", got " << numActual;
}

std::string LIT::nameForPosOnly(size_t idx, const Twine &argOrParam) {
  return ("positional-only " + argOrParam + " #" + Twine(idx)).str();
}

void LIT::emitModuleCallSubscriptDiag(InflightDiag &diag,
                                      AnyStructType metaType,
                                      const Twine &callOrSubscript, SMLoc loc,
                                      SharedState &shared) {
  StringAttr name = metaType.getSymbol().getLeafReference();
  diag << "module " << name << " is not " << callOrSubscript << "able";

  LookupResult lookupResult = shared.lookupAndResolveDecl(
      name, loc, metaType, /*searchParentScopes=*/false);
  if (ArrayRef<ASTDecl *> resDecls = lookupResult.getIfSuccess();
      !resDecls.empty()) {
    diag << "; did you mean to " << callOrSubscript << ' ' << name.strref()
         << '.' << name.strref() << '?';
  }
}
