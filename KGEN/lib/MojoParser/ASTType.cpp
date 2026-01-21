//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ASTType class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "ParserEvaluationContext.h"

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Mojo Diagnostics
//===----------------------------------------------------------------------===//

MojoInflightDiag MojoDiags::emitError(llvm::SMLoc loc, const Twine &message) {
  return MojoInflightDiag(Diags::emitError(loc, message), {});
}
MojoInflightDiag MojoDiags::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return MojoInflightDiag(Diags::emitWarning(loc, message), {});
}

MojoInflightDiag MojoInflightDiag::attachNote(llvm::SMLoc loc) && {
  if (!getDiags())
    return std::move(*this);
  return std::move(*this).attachNote(getDiags()->translateLocation(loc));
}
MojoInflightDiag &MojoInflightDiag::attachNote(llvm::SMLoc loc) & {
  InflightDiag::attachNote(loc);
  return *this;
}

void MojoInflightDiag::addEmittedParam(TypedAttr param,
                                       std::optional<Location> loc,
                                       ASTDecl *ctxDecl) {
  // Remember that we emitted this type.
  emittedParams.push_back({loc.value_or(getLastLoc()), param, ctxDecl});
}

void M::addToDiagnostic(TypedAttr paramValue, InflightDiag &diag) {
  SharedState *shared =
      static_cast<MojoInflightDiag &>(diag).getSharedIfActive();
  if (!shared)
    return; // Ignore discarded diagnostics.

  // Format it of course.
  diag << '\'' << ASTType::getParamAsString(paramValue, /*forDiag=*/shared)
       << '\'';

  // Remember the context decl for when this type was emitted - it could change
  // before the diagnostic is emitted.  This happens (e.g.) in overload
  // resolution where lots of diagnostics are producted (with different callees)
  // and then are emitted in a deferred way.
  ASTDecl *ctxDecl = shared->declResolver->getDeclCurrentlyProcessing();

  // Remember we emitted this parameter so we can post-process the diagnostic.
  auto &mdiag = static_cast<MojoInflightDiag &>(diag);
  mdiag.addEmittedParam(paramValue, {}, ctxDecl);
}

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.
  if (!type) {
    diag << "<<NULL TYPE>>";
    return;
  }
  addToDiagnostic(PValue(type), diag);
}

void M::addToDiagnostic(MojoInflightDiag &&otherDiag, InflightDiag &diag) {
  auto &mdiag = static_cast<MojoInflightDiag &>(diag);

  for (auto [loc, param, ctxDecl] : otherDiag.getEmittedParams())
    mdiag.addEmittedParam(param, loc, ctxDecl);

  diag.addDiag(std::move(otherDiag));
}

namespace {
/// This struct implements textual type+parameter "diffing" to help dig into
/// long type names and identify what parts of them differ.
///
/// Context: A common complaint about mojo is that advanced metaprogramming can
/// produce very long types, particularly when using LayoutTensor. Given this,
/// it can be very difficult to understand what is going on when the compiler
/// barfs out some extremely type type name that isn't compatible with something
/// else.
///
/// We address this through maintenance and selective unwrapping of type sugar,
/// but still some types will have over a dozen parameters.  This digs into the
/// type tree to print something like:
///  .field.size of left value is 'SomeType.size_int' but the right value is '4'
struct ParamDiffer {
  SharedState &shared;
  std::string accessPath;
  TypedAttr leftNested, rightNested;

  /// The diff functions analyze the two specified attrs/types and either decide
  /// they are either 1) atomically incompatible, or 2) there is some
  /// subcomponent that is different.  In the first case, this should set
  /// leftNested/rightNested with the current values.  In the second case, this
  /// adds path information to accessPath and recurses on the subcomponents that
  /// disagree.
  void diff(TypedAttr lhs, TypedAttr rhs) {
    assert(!isEqualCanon(lhs, rhs) && "Cannot diff equal attrs!");

    // Look through type<->attr conversions.
    if (auto lhsTypeParam = dyn_cast<TypeParamAttr>(lhs)) {
      if (auto rhsTypeParam = dyn_cast<TypeParamAttr>(rhs)) {
        // Normally, the type values are inequal, so diff them.
        if (!isEqualCanon(lhsTypeParam.getTypeValue(),
                          rhsTypeParam.getTypeValue()))
          return diff(lhsTypeParam.getTypeValue(), rhsTypeParam.getTypeValue());
        if (!isEqualCanon(lhs.getType(), rhs.getType())) {
          accessPath += ".metatype";
          return diff(lhs.getType(), rhs.getType());
        }
      }
    }

    // Look through sugar to find problems.
    if (auto sugarAttr = dyn_cast<SugarAttr>(lhs)) {
      // Sugar representation of memberalias is weird.
      if (sugarAttr.getKind() != SugarKind::MemberAlias)
        diff(sugarAttr.getSugared(), rhs);
      else
        diff(sugarAttr.getExpanded(), rhs);
      if (leftNested == sugarAttr.getSugared())
        leftNested = sugarAttr; // Preserve the sugar.
      return;
    }
    if (auto sugarAttr = dyn_cast<SugarAttr>(rhs)) {
      // Sugar representation of memberalias is weird.
      if (sugarAttr.getKind() != SugarKind::MemberAlias)
        diff(lhs, sugarAttr.getSugared());
      else
        diff(lhs, sugarAttr.getExpanded());
      if (rightNested == sugarAttr.getSugared())
        rightNested = sugarAttr; // Preserve the sugar.
      return;
    }

    leftNested = lhs;
    rightNested = rhs;
  }

  void diff(ASTType lhs, ASTType rhs) {
    assert(!lhs.isEqualCanon(rhs) && "Cannot diff equal types!");

    // Look through type<->attr conversions.
    if (auto lhsParam = dyn_cast<ParamType>(lhs)) {
      if (auto rhsParam = dyn_cast<ParamType>(rhs))
        return diff(lhsParam.getParam(), rhsParam.getParam());
    }
    if (auto lhsTypeValue = dyn_cast<TypeValueType>(lhs)) {
      if (auto rhsTypeValue = dyn_cast<TypeValueType>(rhs))
        return diff(lhsTypeValue.getTypeValue(), rhsTypeValue.getTypeValue());
    }

    // If these are two metatypes, just transparently look through them.
    if (auto lhsMeta = dyn_cast<StructMetaType>(lhs)) {
      if (auto rhsMeta = dyn_cast<StructMetaType>(rhs))
        return diff(lhsMeta.getType(), rhsMeta.getType());
    }

    // TODO: We should diff function types. They can also get very long.

    // Check to see if these are two structs or struct meta types with differing
    // parameters values.  If so, diagnose that difference.

    // Must have the same declarations to compare, we just say that Int vs
    // String are different, we don't "diff" them.
    auto lhsDecl = lhs.getDecl(shared);
    if (!lhsDecl || lhsDecl != rhs.getDecl(shared)) {
      leftNested = PValue(lhs);
      rightNested = PValue(rhs);
      return;
    }

    assert(lhs.getParamBindings().size() == rhs.getParamBindings().size() &&
           "Type with the same decl should have consistent number of params");

    for (auto [idx, lhsParam, rhsParam] :
         llvm::enumerate(lhs.getParamBindings(), rhs.getParamBindings())) {
      if (isEqualCanon(lhsParam, rhsParam))
        continue;

      // Ok, we found a difference, recursively diff the two parameters.
      auto structDecl = cast<LIT::StructDeclOp>(lhsDecl->getIfOperation());
      accessPath += "." + structDecl.getParams()[idx].getName().str();
      return diff(lhsParam, rhsParam);
    }

    // Couldn't determine the failure.
    llvm_unreachable("params matched but type didn't?");
  }
};
} // end anonymous namespace

/// On destruction, emit notes about any sugared values in the types we emitted.
/// There may be more than one type, in which case we're complaining about a X
/// != Y sort of event. We should only unwrap any given identical alias once.
MojoInflightDiag::~MojoInflightDiag() {
  SharedState *shared = getSharedIfActive();
  // If abandoned, don't do anything.
  if (!shared || emittedParams.empty())
    return;

  // Copy the attribute list so we don't get more entries as we emit notes.
  auto emitted = emittedParams;

  // If we have multiple types emitted, then we're comparing the types.  It is
  // possible we have two small types like Scalar[f32] and Scalar[f64], but it
  // is also possible we have ridiculously huge types like happens in kernel
  // programming.  In this case, we should dig into the type to understand what
  // is going on and explain it in a way that doesn't require too much squinting
  // at long type names.
  if (emittedParams.size() > 1 &&
      !isEqualCanon(emittedParams[0].value, emittedParams[1].value)) {
    ParamDiffer differ{*shared, "", {}, {}};
    differ.diff(emittedParams[0].value, emittedParams[1].value);
    if (!differ.accessPath.empty()) {
      assert(differ.leftNested && differ.rightNested && "differ broken");

      // Only do this for very long type names. Don't clutter things up for
      // SIMD types that disagree obviously.
      auto first = ASTType::getParamAsString(emittedParams[0].value, shared);
      if (first.size() > 30) {
        const char *kind =
            LIT::isTypeExpr(differ.leftNested) ? "type" : "value";
        attachNote(emittedParams[0].loc)
            << differ.accessPath << " of left " << kind << " is ";
        {
          DeclResolver::DeclScopeChanger x(emittedParams[0].ctxDecl);
          *this << differ.leftNested;
        }
        {
          DeclResolver::DeclScopeChanger x(emittedParams[1].ctxDecl);
          *this << " but the right " << kind << " is " << differ.rightNested;
        }
      }

      // Keep track of these as printed so we can unpack sugar if needed.
      emitted.push_back(
          {emittedParams[0].loc, differ.leftNested, emittedParams[0].ctxDecl});
      emitted.push_back(
          {emittedParams[1].loc, differ.rightNested, emittedParams[1].ctxDecl});

      // If the nested values differ as the result of a parameter operator, emit
      // a note suggesting a rebind.  It is plausible we cannot prove equality.
      if (sugarIsa<ParamOperatorAttr>(differ.leftNested) ||
          sugarIsa<ParamOperatorAttr>(differ.rightNested)) {
        attachNote(getPrimaryLoc()) << "types parameters include unfolded "
                                       "expression at parser time; try "
                                       "rebinding to a consistent type?";
      }
    }
  }

  // Don't unpack a single attribute more than once, even if printed multiple
  // times.
  SmallPtrSet<Attribute, 4> unpackedAttr;

  // Finally, take a look at any of the parameters we've printed to see if they
  // have top-level sugar.  If so, unpack them so the user has a better chance
  // of understanding what is going on.
  for (auto [loc, attrValue, ctxDecl] : emitted) {
    // See if anything has alias sugar on it, and if so, unpack it so the user
    // has a better chance of understanding what is going on.  We don't want to
    // look into the body of an always_inline("builtin") calls though!
    TypedAttr desugared = SugarAttr::strip(attrValue, /*keepApplies=*/true);
    if (desugared == attrValue || !unpackedAttr.insert(attrValue).second)
      continue;

    // Make sure to unpack this in the right context so any parameter references
    // are referring to the right declaration.
    DeclResolver::DeclScopeChanger x(ctxDecl);

    // Ensure the strings are textually different.
    auto attrString = ASTType::getParamAsString(attrValue, /*forDiag=*/shared);
    auto sugString = ASTType::getParamAsString(desugared, /*forDiag=*/shared);
    if (attrString != sugString)
      attachNote(loc) << "'" << attrString << "' is aka '" << sugString << "'";
  }
}

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

// Initialize an ASTType from a parameter expression of metatype type.
ASTType::ASTType(TypedAttr typeParamExpr) {
  if (!typeParamExpr) // Null attribute.
    return;

  // ParamType is the canonical way to turn a parameter expression into a type.
  // It handles stripping of metatype information, looks through upcasts etc.
  assert(LIT::isTypeExpr(typeParamExpr) &&
         "parameter expr must be a type expression");
  mlirType = ParamType::get(typeParamExpr);
}

Type ASTType::getMetaType() const {
  if (!mlirType)
    return {};

  auto type = SugarAttr::strip(mlirType);
  if (auto declRef = dyn_cast<StructType>(type))
    return StructMetaType::get(declRef);
  if (auto metaRef = dyn_cast<StructMetaType>(type))
    return StructMetaMetaType::get(metaRef);
  if (auto paramRef = dyn_cast<ParamType>(type))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(type))
    return traitRef.getMetaType();
  if (auto closureType = dyn_cast<ClosureType>(type))
    return TypeType::get(closureType.getContext());
  if (auto module = dyn_cast<ModuleType>(type))
    return module; // Module's are their own metatype.

  // This is some generic MLIR type.
  return {};
}

/// If this is a user declared type, return the declaration that this came
/// from.  If this is a raw MLIR type or a metatype, return null.
ASTDecl *ASTType::getDecl(SharedState &shared) const {
  // We get the declaration from the metatype of the type.  For example, if we
  // have a parametric type like "T" where "T: AnyType", we can know that T has
  // AnyType bound.
  Type type = SugarAttr::strip(getMetaType());
  if (!type)
    return nullptr;

  // If our metatype is itself parametric, for example, we have something like:
  //     !kgen.param<:!lit.anytrait<<@Movable>> elt_trait>
  // Then this type conforms to some parametric trait that is bound by at least
  // Movable.  Use Movable as the declaration we're working with.
  if (auto paramRef = dyn_cast<ParamType>(type)) {
    // AnyTrait is the only metatype of a metatype.
    type = sugarCast<AnyTraitType>(paramRef.getParam().getType());
  }

  if (auto anyStruct = dyn_cast<StructMetaType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(anyStruct.getSymbol());

  if (auto anyMeta = dyn_cast<StructMetaMetaType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(anyMeta.getSymbol());

  if (auto anyTrait = dyn_cast<AnyTraitType>(type))
    type = anyTrait.getTraitType();

  if (auto traitType = dyn_cast<TraitType>(type))
    return shared.declResolver->getTraitDecl(traitType);

  if (auto module = dyn_cast<ModuleType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(module.getSymbol());

  return nullptr;
}

ArrayRef<TypedAttr> ASTType::getParamBindings() const {
  Type metatype = SugarAttr::strip(getMetaType());
  if (auto metaType = dyn_cast_or_null<StructMetaType>(metatype))
    return metaType.getParamValues();
  if (auto mmType = dyn_cast_or_null<StructMetaMetaType>(metatype))
    return mmType.getParamValues();
  return {};
}

/// Return this type with any parameter bindings removed.
ASTType ASTType::getWithoutParameters(SharedState &shared) const {
  if (!mlirType)
    return {};

  Type type = SugarAttr::strip(mlirType);
  if (auto declRef = dyn_cast<StructType>(type))
    return cast<StructDeclOp>(getDecl(shared)->getIfOperation())
        .bindReference();
  if (auto metaType = dyn_cast_or_null<StructMetaType>(type))
    return MetaType::get(
        ASTType(metaType.getType()).getWithoutParameters(shared));
  if (auto mmType = dyn_cast_or_null<StructMetaMetaType>(type))
    return MetaType::get(
        ASTType(mmType.getType()).getWithoutParameters(shared));

  // Not parameterized.
  return *this;
}

bool ASTType::isEqualCanon(ASTType other) const {
  // We have no type sugar yet so we can just do pointer equality tests.
  if (mlirType == other.mlirType)
    return true;
  // Struct types with the same metatype are always equal. This is used to
  // detect when two type aliases refer to the same underlying type.
  if (auto meta = dyn_cast_or_null<StructMetaType>(getMetaType()))
    if (meta == other.getMetaType())
      return true;

  return getCanonicalType(*this) == getCanonicalType(other);
}

/// Return true if this is the same as another ASTType are the same, or if they
/// match when UnknownAttr parameters in the 'this' type are treated as
/// the same as the corresponding parameter in the second type.
///    Foo[1] != Foo[2]   but  Bar[?, 1] == Bar[7, 1]
bool ASTType::isEqualAllowingUnknownAttr(ASTType other,
                                         SharedState &shared) const {
  if (isEqualCanon(other))
    return true;

  // Must have the same struct declarations.
  if (getDecl(shared) != other.getDecl(shared))
    return false;

  ArrayRef<TypedAttr> lhsParams = getParamBindings();
  ArrayRef<TypedAttr> rhsParams = other.getParamBindings();
  assert(lhsParams.size() == rhsParams.size() &&
         "Type with the same decl should have consistent number of params");
  for (auto [lhsParam, rhsParam] : llvm::zip(lhsParams, rhsParams)) {
    if (!isa<UnboundAttr>(lhsParam) &&
        getCanonicalAttr(lhsParam) != getCanonicalAttr(rhsParam))
      return false;
  }
  return true;
}

/// Return true if this is a None type.
bool ASTType::isNoneType() const { return sugarIsa<KGEN::NoneType>(mlirType); }

/// Return true if this is a TypeCheckError type.
bool ASTType::isTypeCheckErrorType() const {
  return sugarIsa<TypeCheckErrorType>(mlirType);
}

/// Return the nonmaterializable decorator target for the type, or null if there
/// is none.
ASTType ASTType::getNonmaterializableTarget(SharedState &shared) const {
  if (auto structDecl = getDecl(shared))
    if (auto structOp =
            dyn_cast_or_null<StructDeclOp>(structDecl->getIfOperation()))
      if (TypeAttr targetMlirType = structOp.getNonmaterializableTargetAttr())
        return ASTType(targetMlirType.getValue());
  return {};
}

/// Return whether the specified type is known to be @register_passable; if
/// generic, this returns the 'genericsDefault' value.
static TypeConvention getRegisterPassability(ASTType type, llvm::SMLoc loc,
                                             SharedState &shared,
                                             TypeConvention genericDefault) {
  // Downcast preserves register passability, strip it before querying the
  // property.
  if (auto paramRefTy = sugarDynCast<ParamType>(type.mlirType))
    type = DowncastAttr::strip(paramRefTy.getParam());

  ASTDecl *decl = type.getDecl(shared);

  if (!decl || sugarIsa<StructMetaType>(type.mlirType)) {
    // If this is a generic type, use the default specification.
    if (auto paramRefTy = sugarDynCast<ParamType>(type.mlirType))
      if (sugarIsa<ParamType, AnyTraitType>(paramRefTy.getParam().getType()))
        return genericDefault;

    // MLIR types are assumed to be register-passable + Trivial.
    return TypeConvention::RegisterPassableTrivial;
  }

  // Make sure we know about the signature of the type.
  if (failed(shared.declResolver->resolveSignature(*decl, loc)))
    return TypeConvention::MemoryOnly;

  // We don't yet have a runtime representation for packages or modules, but
  // when we do, it will not be register-passable.
  if (isa_and_nonnull<FileModuleOp, PackageOp>(decl->getIfOperation()))
    return TypeConvention::MemoryOnly;

  // Trait values are generic and therefore use the default specification.
  if (auto trait = dyn_cast_or_null<TraitDeclOp>(decl->getIfOperation())) {
    if (type.isAnyTrivialRegType(decl->getLoc(), shared))
      return TypeConvention::RegisterPassableTrivial;

    TypeConvention convention = trait.getConvention();
    if (convention == TypeConvention::Unspecified)
      return genericDefault;
    return convention;
  }

  if (TraitType traitType =
          sugarDynCastIfPresent<TraitType>(decl->getIfTypeValue())) {
    // The register passability of a trait composition is the strictest of its
    // members.
    TypeConvention convention = TypeConvention::Unspecified;
    for (SymbolRefAttr symbol : traitType.getSymbols()) {
      TypeConvention singleTraitRP =
          ASTType(TraitType::get(symbol)).getRegisterPassability(loc, shared);
      if (singleTraitRP == TypeConvention::Unspecified)
        continue;
      if (convention == TypeConvention::Unspecified)
        convention = singleTraitRP;
      else
        convention = std::max(convention, singleTraitRP);
    }
    if (convention == TypeConvention::Unspecified)
      return genericDefault;
    return convention;
  }

  auto structOp = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation());
  assert(structOp && "only one user-defined type so far");
  if (type.isAnyTrivialRegType(decl->getLoc(), shared))
    return TypeConvention::RegisterPassableTrivial;

  return structOp.getConvention();
}

/// Return the StructDeclOp::RegisterPassable enum for this type.
TypeConvention ASTType::getRegisterPassability(llvm::SMLoc loc,
                                               SharedState &shared) const {
  // If this is a generic type, we treat it as memory only. If the metatype
  // is a parameter reference, then pessimistically assume it is memory-only.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::MemoryOnly);
}

/// Return true if this type is a 'trivial' type, that is one that can be
/// passed around by copying the bits, and whose destructor is a noop.
bool ASTType::isTrivial(llvm::SMLoc loc, SharedState &shared) const {
  return getRegisterPassability(loc, shared) ==
         TypeConvention::RegisterPassableTrivial;
}

/// Return true if this type is a register-passable type that can be passed
/// around and copied in SSA values instead of having to live in memory.
///
/// The location specifies the location of the reference in case the use is
/// invalid in this location.
bool ASTType::isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const {
  TypeConvention convention = getRegisterPassability(loc, shared);
  return convention == TypeConvention::RegisterPassable ||
         convention == TypeConvention::RegisterPassableTrivial;
}

/// Return true if this type is @register_passable or if it is a generic type
/// that could bind to a concrete @register_passable type.
bool ASTType::mightBeRegisterPassable(llvm::SMLoc loc,
                                      SharedState &shared) const {
  // If this is a generic type, we treat it as register passable conservatively.
  return ::getRegisterPassability(*this, loc, shared,
                                  TypeConvention::RegisterPassable) !=
         TypeConvention::MemoryOnly;
}

/// Return true if this type needs to be destroyed.  This is false for trivial
/// types like Int.  Note: this resolves the body of a struct type.
bool ASTType::hasNontrivialDestructor(llvm::SMLoc loc,
                                      SharedState &shared) const {
  ASTDecl *decl = getDecl(shared);
  if (!decl) // MLIR types are assumed to be register-passable + Trivial.
    return false;

  // Generic types are assumed to have a destructor unless they are trivial.
  if (sugarIsa<TraitType>(getMetaType())) {
    return getRegisterPassability(loc, shared) !=
           TypeConvention::RegisterPassableTrivial;
  }

  // Make sure we know about the signature of the type.
  if (failed(shared.declResolver->resolveBody(*decl, loc)))
    return false;

  auto structOp = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation());
  assert(structOp && "only one user-defined type so far");
  return structOp.getDestructorAttr() != TypedAttr();
}

bool ASTType::isCopyable(llvm::SMLoc loc, SharedState &shared,
                         bool isImplicit) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are copyable.

  // If the type is trivial, then it is copyable.
  if (isTrivial(loc, shared))
    return true;

  StringRef traitName = isImplicit ? "ImplicitlyCopyable" : "Copyable";

  // Check whether the type conforms to `ImplicitlyCopyable` trait.
  ASTDecl *traitDecl =
      shared.lookupBuiltinTrait(traitName, typeDecl, typeDecl->getLoc());
  if (!traitDecl)
    return false;
  auto trait = dyn_cast_or_null<TraitDeclOp>(traitDecl->getIfOperation());
  if (!trait)
    return false;
  return typeDecl->doesNominalTypeConformTo(trait.bindReference());
}

/// Return true if this type is implicitly copyable, either because it is
/// trivial or conforms to ImplicitlyCopyable trait. Note: this resolves the
/// body of a struct type.
bool ASTType::isImplicitlyCopyable(llvm::SMLoc loc, SharedState &shared) const {
  return isCopyable(loc, shared, /*isImplicit=*/true);
}

/// Return true if this type is explicitly copyable, either because it is
/// trivial or conforms to the Copyable trait. Note: this resolves the
/// body of a struct type.
bool ASTType::isExplicitlyCopyable(llvm::SMLoc loc, SharedState &shared) const {
  return isCopyable(loc, shared, /*isImplicit=*/false);
}

/// Return true if this type is movable from its own type, either because it
/// is trivial or has a move constructor from self. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovable(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR types are movable.

  // If the type is register-passable, it is trivially movable.
  if (isRegisterPassable(loc, shared))
    return true;

  // Look for a move constructor.
  // TODO: this should be changed to trait conformance check as well.
  return shared.typeHasMember(*typeDecl, "__moveinit__", loc);
}

/// Return true if this type is movable, either because it is trivial, a
/// register passable type, or has a move constructor. Note: this resolves the
/// body of a struct type.
bool ASTType::isMovableFrom(ASTExprAnd<CValue> value,
                            SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl) // MLIR Types are movable.
    return true;

  // If the type is register passable at all, then it is movable.
  if (isRegisterPassable(value.expr->getLoc(), shared))
    return true;

  // Check all the available candidate to see if we have one that cooperates
  // with this value kind.
  if (!value.ir.getIfRValue())
    return false;

  return isMovable(value.expr->getLoc(), shared);
}

bool ASTType::isAnyTrivialRegType(llvm::SMLoc loc, SharedState &shared) const {
  ASTDecl *typeDecl = getDecl(shared);
  if (!typeDecl)
    return true; // MLIR Types are trivial register passable.

  // TODO: probably no need to check this
  // once we deprecate @register_passable("trivial")
  if (auto structOp =
          dyn_cast_or_null<StructDeclOp>(typeDecl->getIfOperation())) {
    if (structOp.isRegisterPassableTrivial())
      return true;
  }

  // Check whether the type conforms to `AnyTrivialRegType` trait.
  ASTDecl *traitDecl = shared.lookupBuiltinTrait("AnyTrivialRegType", typeDecl,
                                                 typeDecl->getLoc());

  if (!traitDecl)
    return false;
  auto trait = dyn_cast_or_null<TraitDeclOp>(traitDecl->getIfOperation());
  if (!trait)
    return false;
  return typeDecl->doesNominalTypeConformTo(trait.bindReference());
}

/// Given a reference, return the element as an ASTType.  This aborts
/// if the current type isn't a reference.
///
ASTType ASTType::getReferenceElementType() const {
  return ASTType(sugarCast<RefType>(mlirType).getElementType());
}

/// Given a VariadicType, return the element as an ASTType.  This aborts if
/// the current type isn't a VariadicType.
ASTType ASTType::getVariadicElementType() const {
  return ASTType(sugarCast<VariadicType>(mlirType).getElementType());
}

/// Return the RefPackType that corresponds to the VariadicPack instance.
RefPackType ASTType::getVariadicPackInfo(SharedState &shared) const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  // NOTE: `bindings[0]` and `bindings[2]` are expected to be the Mojo `Bool`
  // type, and `bindings[1]` is an Origin.
  assert(bindings.size() == 5 &&
         sugarIsa<LIT::StructType>(bindings[0].getType()) &&
         sugarIsa<LIT::StructType>(bindings[1].getType()) &&
         sugarIsa<LIT::StructType>(bindings[2].getType()) &&
         sugarIsa<AnyTraitType>(bindings[3].getType()) &&
         sugarIsa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");

  TypedAttr origin = ASTType::extractOriginOf(SMLoc(), bindings[1], shared);
  assert(origin && "Origin is null");
  return RefPackType::get(
      /*variadicList*/ bindings[4], origin,
      /*addrSpace*/
      IntegerAttr::get(IndexType::get(shared.getContext()), 0));
}

/// Return the type list for the variadic argument in a VariadicPack.  This
/// will be a VariadicAttr when concrete (e.g. on the caller side) or a
/// parameter on the callee side.
TypedAttr ASTType::getVariadicPackTypeList() const {
  assert(!isa<RefType>(mlirType) && "looking at a RefType not a VariadicPack");
  auto bindings = getParamBindings();
  // NOTE: `bindings[0]` and `bindings[2]` are expected to be the Mojo `Bool`
  // type, and `bindings[1]` is an Origin.
  assert(bindings.size() == 5 &&
         sugarIsa<LIT::StructType>(bindings[0].getType()) &&
         sugarIsa<LIT::StructType>(bindings[1].getType()) &&
         sugarIsa<LIT::StructType>(bindings[2].getType()) &&
         sugarIsa<AnyTraitType>(bindings[3].getType()) &&
         sugarIsa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");
  return bindings[4];
}

ASTType ASTType::getKwargsDictValueType() const {
  return ASTType(getParamBindings()[0]);
}

ASTType ASTType::getKwargsDictRefValueType() const {
  return getReferenceElementType().getKwargsDictValueType();
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
ASTType ASTType::getSignatureUserResultType() const {
  auto sigGenType = cast<FnTypeGeneratorType>(mlirType);
  return LIT::getSignatureUserResultType(sigGenType, sigGenType.getArguments(),
                                         sigGenType.getResults().front());
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }

RefType ASTType::getRefForArgument(const Twine &argName, bool isMut) {
  auto ctx = mlirType.getContext();
  auto selfOrigin = ParamDeclRefAttr::get(StringAttr::get(ctx, argName + "`"),
                                          OriginType::get(ctx, isMut));
  return RefType::get(mlirType, selfOrigin, /*addressSpace=*/0);
}

/// If this type is parameterized, and if any of the parameters refer to a
/// ParamIndexRefAttr, replace it with an UnboundAttr so parameter inference
/// will infer it.
///
/// This makes parameter inference sensitive to what to propagate vs infer. For
/// example, if expectedType is known to be 'SIMD[uint8, 1]', then we can infer
/// which constructor to use when the input is an IntLiteral.
///
/// On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
/// argument is an Int8, then we need the implicit conversion to infer the
/// base element.  Our solution to this is to rip and replace parameters that
/// contain unbound parameters, replacing them with UnboundAttr so inference
/// can find them.
ASTType ASTType::getWithUnknownParametersReplaced(SharedState &shared) const {
  // If this is a struct type, try unbinding just the parameters that have
  // parameter references in it.
  if (auto drt = sugarDynCast<StructType>(mlirType)) {
    ParamIndexRefAttrFinder finder;

    // Otherwise, check each bound parameter to see if it is unknown.  If so,
    // replace it.
    SmallVector<TypedAttr> newParams;
    bool anyBound = false;
    for (auto curValue : getParamBindings()) {
      if (!finder.hasReferences(curValue)) {
        // Keep this value if it has no references.
        anyBound = true;
      } else {
        // Keep the argument type if it has no references.
        Type paramType = curValue.getType();
        if (finder.hasReferences(paramType))
          paramType =
              ASTType(UnboundAttr::get(TypeType::get(paramType.getContext())));
        curValue = UnboundAttr::get(paramType);
      }
      newParams.push_back(curValue);
    }

    if (anyBound)
      return cast<StructDeclOp>(getDecl(shared)->getIfOperation())
          .bindReference(newParams);
  }

  // Otherwise return it with all parameters replaced.
  if (Type nonParam = getWithoutParameters(shared))
    return nonParam;
  return *this;
}

/// Return true if this type contains any origins that are unmaterializable
/// from comptime to runtime. Consider some code like this:
///
///   alias ptr = String("foo"+"bar").unsafe_ptr()
///   alias elt1 = ptr[0] # Yields "f", which works fine.
///   # This can't work.
///   var runtime_ptr = ptr
///
bool ASTType::containsUnmaterializableOrigins(SharedState &shared) const {
  for (auto o :
       shared.cachedOriginFinder.findOriginsIn(getCanonicalType(mlirType))) {
    // Ignore field sensitivity.
    while (auto field = dyn_cast<OriginFieldAttr>(o))
      o = field.getBase();

    // Actually global memory /can/ be materialized, so that it totally fine. We
    // allow AnyOriginAttr because it is the general "disable checking" origin.
    // Banning it prevents many important patterns from working, e.g. default
    // arguments of null UnsafePointer.
    if (sugarIsa<StaticOriginAttr, AnyOriginAttr>(o))
      continue;
    // Otherwise, it is something we can't track.
    return true;
  }

  return false;
}
