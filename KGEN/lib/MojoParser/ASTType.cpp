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
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return StructMetaType::get(declRef);
  if (auto paramRef = dyn_cast<ParamType>(mlirType))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(mlirType))
    return traitRef.getMetaType();
  if (auto closureType = dyn_cast<ClosureType>(mlirType))
    return TypeType::get(closureType.getContext());
  if (auto module = dyn_cast<ModuleType>(mlirType))
    return module; // Module's are their own metatype.

  // Look through sugar.
  ASTType stripped = stripTopLevelSugar();
  if (stripped.mlirType != mlirType)
    return stripped.getMetaType();

  // This is some generic MLIR type.
  return {};
}

/// If this is a user declared type, return the declaration that this came
/// from.  If this is a raw MLIR type or a metatype, return null.
ASTDecl *ASTType::getDecl(SharedState &shared) const {
  // We get the declaration from the metatype of the type.  For example, if we
  // have a parametric type like "T" where "T: AnyType", we can know that T has
  // AnyType bound.
  Type type = ASTType(getMetaType()).stripTopLevelSugar();
  if (!type)
    return nullptr;

  // If our metatype is itself parametric, for example, we have something like:
  //     !kgen.param<:!lit.anytrait<<@Movable>> elt_trait>
  // Then this type conforms to some parametric trait that is bound by at least
  // Movable.  Use Movable as the declaration we're working with.
  if (auto paramRef = dyn_cast<ParamType>(type)) {
    // AnyTrait is the only metatype of a metatype.
    type = cast<AnyTraitType>(paramRef.getParam().getType());
  }

  if (auto anyStruct = dyn_cast<StructMetaType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(anyStruct.getSymbol());

  if (auto anyTrait = dyn_cast<AnyTraitType>(type))
    type = anyTrait.getTraitType();

  if (auto traitType = dyn_cast<TraitType>(type))
    return shared.declResolver->getTraitDecl(traitType);

  if (auto module = dyn_cast<ModuleType>(type))
    return &shared.declResolver->getDeclForTypeSymbol(module.getSymbol());

  return nullptr;
}

ArrayRef<TypedAttr> ASTType::getParamBindings() const {
  Type metatype = ASTType(getMetaType()).stripTopLevelSugar();
  if (StructMetaType metaType = dyn_cast_or_null<StructMetaType>(metatype))
    return metaType.getParamValues();
  return {};
}

/// Return this type with any parameter bindings removed.
ASTType ASTType::getWithoutParameters(SharedState &shared) const {
  if (!mlirType)
    return {};
  if (auto declRef = dyn_cast<StructType>(mlirType))
    return cast<StructDeclOp>(getDecl(shared)->getIfOperation())
        .bindReference();
  if (StructMetaType metaType = dyn_cast_or_null<StructMetaType>(mlirType))
    return MetaType::get(
        ASTType(metaType.getType()).getWithoutParameters(shared));

  // Look through sugar.
  ASTType stripped = stripTopLevelSugar();
  if (stripped.mlirType != mlirType)
    return stripped.getWithoutParameters(shared);

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

/// Remove any top-level sugar nodes from this type, but don't fully
/// canonicalize it.
ASTType ASTType::stripTopLevelSugar() const {
  if (auto paramRef = dyn_cast_or_null<ParamType>(mlirType))
    return ASTType(SugarAttr::strip(paramRef.getParam()));
  return *this;
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
  ASTDecl *decl = type.getDecl(shared);

  if (!decl) {
    // If this is a generic type, use the default specification.
    if (auto paramRefTy = dyn_cast<ParamType>(type.mlirType))
      if (isa<ParamType, AnyTraitType>(paramRefTy.getParam().getType()))
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
    TypeConvention convention = trait.getConvention();
    if (convention == TypeConvention::Unspecified)
      return genericDefault;
    return convention;
  }

  if (TraitType traitType =
          dyn_cast_or_null<TraitType>(decl->getIfTypeValue())) {
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
  // NOTE: `bindings[0]` and `bindings[1]` are expected to be the Mojo `Bool`
  // type, and `bindings[2]` is an Origin.
  assert(bindings.size() == 5 && isa<LIT::StructType>(bindings[0].getType()) &&
         isa<LIT::StructType>(bindings[1].getType()) &&
         isa<LIT::StructType>(bindings[2].getType()) &&
         isa<AnyTraitType>(bindings[3].getType()) &&
         isa<VariadicType>(bindings[4].getType()) &&
         "Not a VariadicPack struct?");

  TypedAttr origin = ASTType::extractOriginOf(SMLoc(), bindings[2], shared);
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
  // NOTE: `bindings[0]` and `bindings[1]` are expected to be the Mojo `Bool`
  // type, and `bindings[2]` is an Origin.
  assert(bindings.size() == 5 && isa<LIT::StructType>(bindings[0].getType()) &&
         isa<LIT::StructType>(bindings[1].getType()) &&
         isa<LIT::StructType>(bindings[2].getType()) &&
         isa<AnyTraitType>(bindings[3].getType()) &&
         isa<VariadicType>(bindings[4].getType()) &&
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

void M::addToDiagnostic(ASTType type, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.

  auto *shared = static_cast<SharedState *>(diag.getDiags()->extraContext);
  diag << '\'' << type.getAsString(/*forDiag=*/shared) << '\'';
}

void M::addToDiagnostic(TypedAttr paramValue, InflightDiag &diag) {
  if (!diag.getDiags())
    return; // Ignore discarded diagnostics.

  diag << '\'';
  auto *shared = static_cast<SharedState *>(diag.getDiags()->extraContext);
  diag << ASTType::getParamAsString(paramValue, /*forDiag=*/shared);
  diag << '\'';
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
    if (isa<StaticOriginAttr, AnyOriginAttr>(o))
      continue;
    // Otherwise, it is something we can't track.
    return true;
  }

  return false;
}
