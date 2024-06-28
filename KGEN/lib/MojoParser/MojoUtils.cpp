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

/// Returns if a value of the specified type can be coerced to the other type
/// with a rebind.  This means that values of the two types have exactly the
/// same representation post-elaboration.
bool LIT::canConvertWithRebind(ASTType fromType, ASTType toType,
                               SharedState &shared) {
  if (fromType.isEqualCanon(toType))
    return true; // No rebind needed!

  // Trait metatypes are allowed to upcast to trivial types.
  if (isa<TypeType>(toType)) {
    if (isa<AnyTraitType>(fromType))
      return true;
    if (auto structType = dyn_cast<AnyStructType>(fromType)) {
      return ASTType(structType).getRegisterPassability(SMLoc(), shared) ==
             TypeConvention::RegisterPassableTrivial;
    }
  }

  // Handle conversions of values that have parametric type.
  if (isa<ParamRefType>(fromType) || isa<ParamRefType>(toType)) {
    if (auto fromMT = fromType.getMetaType()) {
      if (auto toMT = toType.getMetaType()) {
        // A value of parametric type or a result of parametric type can be
        // rebind to each other if the have the same struct metatype, because
        // there is exactly one type that implements it - the parameter must
        // resolve to that struct.
        if (isa<AnyStructType>(fromMT) && ASTType(fromMT).isEqualCanon(toMT))
          return true;

        // Allow conversion from parametric value of AnyTrait[SomeTrait]
        // metatype to SomeTrait. We don't know what trait type the parametric
        // value will resolve to, but we know that it conforms to SomeTrait.
        //
        // Note that it is not safe to allow conversions *TO* parametric
        // types, even if they have the same AnyTrait type.  This is because
        // post-elaboration they will resolve to a concrete type, not an erased
        // type, and the types may disagree.  For example, this needs to be
        // invalid because 'T' and 'U' can elaborate to different types:
        //
        //   fn different_traits[T: Copyable, U: Copyable](x: T) -> U:
        //      return x   # Cannot convert from related types T to U.
        if (isa<TraitType>(toType) && ASTType(fromMT).isEqualCanon(toMT))
          return true;
      }
    }

    // If the "from" type is a rebind of another type, it is a downcast from the
    // actual type we care about.  Strip it off and try again.
    if (auto fromRebind = dyn_cast<ParamOperatorAttr>(PValue(fromType).get());
        fromRebind && fromRebind.getOpcode() == POC::Rebind)
      return canConvertWithRebind(ASTType(fromRebind.getOperand(0)), toType,
                                  shared);
    // Strip them off 'to' type also.
    if (auto toRebind = dyn_cast<ParamOperatorAttr>(PValue(toType).get());
        toRebind && toRebind.getOpcode() == POC::Rebind)
      return canConvertWithRebind(fromType, ASTType(toRebind.getOperand(0)),
                                  shared);
  }

  // We can convert from AnyTraitType[Derived] to AnyTraitType[Base] with a
  // rebind.
  if (auto toAnyTrait = dyn_cast<AnyTraitType>(toType)) {
    if (auto fromAnyTrait = dyn_cast<AnyTraitType>(fromType)) {
      auto *fromDecl = ASTType(fromAnyTrait.getTraitType()).getDecl(shared);
      if (!fromDecl)
        return false;

      std::optional<InflightDiag> diag;
      if (fromDecl->doesNominalTypeConformsTo(toAnyTrait.getTraitType(), diag,
                                              shared))
        return true;
      if (diag)
        diag->abandon();
      return false;
    }
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
      return canConvertWithRebind(fromSig, toSig, shared);
    }
    return false;
  }

  // Check lifetime downcasting.  The safe conversions are:
  //   Lifetimes with identical mutability will be uniqued and already handled.
  //   Conversion from any mutability to KNOWN immutable is fine.
  //   Conversion from KNOWN mutable to any mutability is fine.
  if (auto fromLife = dyn_cast<LifetimeType>(fromType))
    if (auto toLife = dyn_cast<LifetimeType>(toType))
      return toLife.isMutableKnown(false) || fromLife.isMutableKnown(true);

  // Check reference downcasting.  The only thing allowed to disagree is the
  // lifetime set / mutability.
  if (auto fromRef = dyn_cast<RefType>(fromType)) {
    if (auto toRef = dyn_cast<RefType>(toType)) {
      // Element types and address space have to be exactly equal.
      if (fromRef.getAddressSpace() != toRef.getAddressSpace())
        return false;

      // The element type needs to exactly match, but we allow rebinds to a
      // different metatype in the way.
      auto fromEltType = fromRef.getElementType();
      auto toEltType = toRef.getElementType();
      if (!ASTType(fromEltType).isEqualCanon(toEltType)) {
        // If these are both parametric types, they may have a rebind in the
        // way.  This rebind will be a downcast of a trait, e.g. from Copyable
        // to AnyType, which is needed because Mojo/MLIR doesn't have subtype
        // type compatibility of attributes.
        bool isJustRebind = false;
        if (isa<ParamRefType>(fromEltType) && isa<ParamRefType>(toEltType) &&
            canConvertWithRebind(fromEltType, toEltType, shared))
          isJustRebind = true;

        if (!isJustRebind)
          return false;
      }

      // Verify compatible LifetimeType(mutability).  This is checking the type
      // of the lifetime, which contains its mutability specifier.
      auto toLifetimeType = toRef.getLifetimeType();
      if (fromRef.getLifetimeType() != toLifetimeType &&
          !canConvertWithRebind(fromRef.getLifetimeType(), toLifetimeType,
                                shared))
        return false;

      // We can convert lifetime subset to a lifetimes superset.
      auto toLifetime = toRef.getLifetime();
      auto lifetimeUnion = LifetimeUnionAttr::get(
          {toLifetime,
           LifetimeMutCastAttr::get(fromRef.getLifetime(), toLifetimeType)},
          toLifetimeType);
      return toLifetime == lifetimeUnion;
    }
  }

  auto from = dyn_cast<LITSignatureType>(fromType);
  auto to = dyn_cast<LITSignatureType>(toType);
  if (!from || !to)
    return false;

  // Allow signature types to be converted for free if they differ only in
  // argument names, parameter names, passing kinds, or implicit lifetimes.
  size_t fromNumArgs = from.getNumArguments();
  if (fromNumArgs != to.getNumArguments())
    return false;
  if (from.getNumParams() != to.getNumParams())
    return false;
  if (from.getArgConventions() != to.getArgConventions())
    return false;

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
    if (SignatureType::hasImplicitLifetime(conv)) {
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
ASTType LIT::getZeroCostCommonType(ASTType type1, ASTType type2,
                                   SharedState &shared) {
  if (auto result = getZeroCostCommonTypeImpl(shared, type1, type2)) {
    // Make sure we can always convert to the common type!
    assert(canConvertWithRebind(type1, result, shared) &&
           canConvertWithRebind(type2, result, shared) &&
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

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

void LIT::emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
                                   size_t maxAllowed, size_t numActual,
                                   const Twine &argOrParam) {
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

void LIT::printNameOrIdx(StringAttr name, size_t idx, InflightDiag &diag) {
  if (!name.empty())
    diag << "'" << name.getValue() << "'";
  else
    diag << "#" << idx;
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
