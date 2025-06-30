//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the trait conformance checking
// and special function synthesis logic.
//
//===----------------------------------------------------------------------===//

#include "Traits.h"
#include "CallEmission.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"
#include "StructEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// When dealing with a trait method that mentions an associated alias T, like
///
///     trait MyTrait[_Self: AnyType]: # Implicit _Self param-decl shown
///         alias T: AnyType
///         fn foo(self: _Self, x: Self.T) -> Int:
///             ...
///     struct MyStruct(MyTrait):
///         alias T = Bool
///         fn foo(self, x: Bool) -> Int:
///             ...
///
/// At some point verifyConformance will verify that MyStruct conforms to
/// MyTrait. During that, it'll look at the trait's
/// `fn foo(self: _Self, x: Self.T) -> Int` signature, to see if the same thing
/// exists in the struct.
///
/// To do that, we first need to substitute _Self and Self, to make
/// `fn foo(self: MyStruct, x: MyStruct.T)`
/// but we'll still need to simplify the `MyStruct.T` to get our desired
/// signature:
///
/// `fn foo(self: MyStruct, x: Bool)`
///
/// This last step is accomplished by this function here.
static FnTypeGeneratorType
simplifyGetVTableEntryOnSelf(PValue selfType,
                             const DenseMap<StringAttr, TypedAttr> *aliasValues,
                             FnTypeGeneratorType signature) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
    if (paramOp.getOpcode() == POC::GetVTableEntry &&
        paramOp.getOperand(0) == PValue(selfType).get()) {
      auto aliasName = cast<StringAttr>(paramOp.getOperand(1));
      // The vtable entries have type !kgen.string, but the entries from the
      // trait decl have a StringAttr with no type.  Reunique them to look up.
      aliasName = StringAttr::get(aliasName.getContext(), aliasName.strref());
      auto iter = aliasValues->find(aliasName);
      if (iter != aliasValues->end())
        return iter->second;
    }
    return paramOp;
  });
  return cast<FnTypeGeneratorType>(replacer.replace(signature));
}

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<FnTypeGeneratorType, ParamBindings>
getTraitFunctionSignature(IREmitter &emitter, FnOp traitFn,
                          ASTType structSelfType, SymbolRefAttr traitSymbol,
                          const ExprNode *expr,
                          const DenseMap<StringAttr, TypedAttr> &aliasValues,
                          ParserParameterEvaluator &traitAliasReplacer) {
  TraitType trait = TraitType::get(traitSymbol);
  FnTypeGeneratorType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = signature.getInputParamTypes();

  // Add trait's _Self param replacement.
  params.push_back(TypeParamAttr::get(structSelfType, trait));
  auto bindings = ParamBindings::getForDeclaredType(emitter.getDeclScope(),
                                                    structSelfType, expr);
  // Leave the rest alone.
  for (Type type : paramTypes.drop_front()) {
    params.push_back(UnboundAttr::get(type));
    bindings.addPrechecked(expr, params.back());
  }

  FnTypeGeneratorType newSignature = signature.getSpecializedGenerator(
      params, /*emitErrorFn=*/{},
      &emitter.getDeclScope().getShared().getEvaluationContext());

  auto selfStructAsTrait = TypeParamAttr::get(structSelfType, trait);

  newSignature = simplifyGetVTableEntryOnSelf(selfStructAsTrait, &aliasValues,
                                              newSignature);
  newSignature = traitAliasReplacer.replace(newSignature);
  return {newSignature, bindings};
}

LogicalResult LIT::verifyConformance(ASTDecl &structDecl, SymbolRefAttr parent,
                                     std::optional<InflightDiag> &diag,
                                     WitnessTable &witnessTable) {
  auto &shared = structDecl.getShared();
  auto structDeclOp = cast<StructDeclOp>(structDecl);

  bool hadErrors = false;
  SyntheticNode node(structDecl.getLoc());
  IREmitter emitter(structDecl, EC_Trait);
  ASTType selfType = structDecl.getTypeDeclSelf();

  ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(parent);
  TraitDeclOp traitDeclOp = cast<TraitDeclOp>(traitDecl);

  // Make sure to fully resolve the trait first.
  if (failed(shared.declResolver->resolveBody(traitDecl, structDecl.getLoc())))
    return failure();

  if (traitDeclOp.isRegisterPassable() && !structDeclOp.isRegisterPassable()) {
    diag = shared.emitError(structDecl.getLoc(),
                            "a struct must be register passable in order to "
                            "inherit from a register passable trait");
    return failure();
  }

  // In the below loops, we'll be checking that each trait method ("needle") is
  // present in the struct too.
  // For example:
  //
  //     trait MyTrait:
  //         fn zork(self) -> Int:
  //             ...
  //     struct MyStruct:
  //         fn zork(self) -> Int:
  //             ...
  //
  // We simply look to see if the trait's `fn zork(self) -> Int` (the needle)
  // exists in the struct too, which it does.
  //
  // Things get trickier when aliases are involved though, like:
  //
  //     trait MyTrait:
  //         alias X: AnyType
  //         fn zork(self) -> X:
  //             ...
  //     struct MyStruct:
  //         alias X: AnyType = Int
  //         fn zork(self) -> Int
  //
  // We can't just check if `fn zork(self) -> X` exists in the struct, because
  // it doesn't.
  // Instead, we need to substitute all the struct alias values (like `Int`)
  // into the needle.
  // Substituting the struct's `Int` in for `X`, the
  // `fn zork(self) -> X` needle becomes the correct
  // `fn zork(self) -> Int` needle.
  // We then check if that exists in the struct and it does, excellent.
  //
  // These traitAliasReplacer and aliasValues maps help us do the needle
  // substitutions later.
  //
  // TODO(MOCO-1993): Consolidate docs on this.
  ParserParameterEvaluator traitAliasReplacer(shared);
  DenseMap<StringAttr, TypedAttr> aliasValues;

  // Prepare an error. It will be abandoned if the check succeeds.
  diag = shared.emitError(structDecl.getLoc())
         << selfType << " does not implement all requirements for "
         << ASTType(TraitType::get(parent));

  // Returns failure() to stop the verifyConformance loop.
  auto checkMethod = [&](StringAttr name, ASTDecl *traitFnDecl,
                         FnOp traitFn) -> LogicalResult {
    // Skip inherited methods, they're checked at a different time.
    if (traitFn.getInheritedFrom())
      return success();

    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    if (decls.empty() || !isa<FnOp>(decls.front())) {
      diag->attachNote(traitFnDecl->getLoc())
          << "required function '" + name.str() + "' is not implemented";
      return failure(); // Stop the outer loop.
    }

    // Signature resolve the found decls first, so they can be checked.
    for (ASTDecl *decl : decls) {
      if (failed(shared.declResolver->resolveSignature(*decl,
                                                       structDecl.getLoc()))) {
        hadErrors = true;
        return success();
      }
    }

    SyntheticNode syntheticNode(structDecl.getLoc());
    auto [traitSignature, bindings] = getTraitFunctionSignature(
        emitter, traitFn, selfType, parent, syntheticNode, aliasValues,
        traitAliasReplacer);

    // Match against the transformed calling convention if the struct is
    // register-passable.
    OverloadSet ov(name, decls, std::move(bindings), node,
                   CallSyntax::kMethodCallSynthetic);
    PValue result = ov.filterOverloadSetForValueType(
        traitSignature, emitter.getDeclScope(),
        function_ref<InflightDiag &(SMLoc)>([&](SMLoc loc) -> InflightDiag & {
          return diag->attachNote(traitFnDecl->getLoc());
        }));
    if (result)
      witnessTable.emplace_back(name, result.get());
    else {
      return failure();
    }
    return success();
  };

  auto checkAlias = [&](StringAttr name, ASTDecl *traitAliasDecl,
                        AliasDeclOp traitAlias) -> LogicalResult {
    if (traitAlias.getInheritedFrom())
      return success();

    if (failed(shared.declResolver->resolveSignature(*traitAliasDecl,
                                                     structDecl.getLoc()))) {
      hadErrors = true;
      return success();
    }

    assert(!traitAlias.getValueAttr() && "trait alias shouldn't have a value");
    Type traitAliasType = traitAlias.getType();

    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    if (decls.empty() || !isa<LIT::AliasDeclOp>(decls.front())) {
      diag->attachNote(traitAlias->getLoc())
          << "required alias '" << name.str() << "' is not specified";
      return failure(); // Stop the outer loop.
    }
    ASTDecl *structAliasDecl = decls.front();
    AliasDeclOp structAliasDeclOp = cast<LIT::AliasDeclOp>(structAliasDecl);
    if (failed(shared.declResolver->resolveSignature(*structAliasDecl,
                                                     structDecl.getLoc()))) {
      hadErrors = true;
      return success();
    }
    Type structAliasType = structAliasDeclOp.getType();

    TypedAttr initializerExpr = structAliasDeclOp.getValueAttr();
    assert(initializerExpr && "Struct's alias should have initializer");

    // We don't yet put initializerExpr into the traitAliasReplacer or
    // aliasValues because they need to be converted first to the trait's
    // alias's type (see SAVMBCTATBS).
    SyntheticNode synthNode(structAliasDecl->getLoc());
    if (!IREmitter::canImplicitlyConvertToType({initializerExpr, synthNode},
                                               traitAliasType,
                                               emitter.getDeclScope())) {
      diag->attachNote(traitAliasDecl->getLoc())
          << "alias '" + name.str() + "' type " << structAliasType
          << " doesn't conform to trait's alias '" << name.str() << "' type "
          << traitAliasType;
      return failure();
    }

    ValueDest dest(traitAliasType, EC_AliasValue);
    CValue convertedValue = emitter.emitImplicitConversionToType(
        {initializerExpr, synthNode}, traitAliasType, dest);
    witnessTable.emplace_back(name, convertedValue.getIfPValue().get());

    // Struct Alias Values Must Be Converted To Trait Alias's Type Before
    // Substitution (SAVMBCTATBS):
    //
    // Things get a little trickier when we're using an alias as an input
    // parameter to something else, like here:
    //
    //     struct Container[T: AnyType]:
    //         ...
    //     trait MyTrait:
    //         alias X: AnyType
    //         fn zork(self) -> Container[X]:
    //             ...
    //     struct MyStruct:
    //         alias X: Copyable = Int       <-- "Int as Copyable" TypeParamAttr
    //         fn zork(self) -> Container[Int]
    //
    // Notice the X: Copyable.
    // That means that the struct's `X` has a value that's a TypeParamAttr,
    // basically an `Int` that's masquerading as a `Copyable`.
    // Unfortunately, when we do our substitution into our
    // `fn zork(self) -> Container[X]` needle, it becomes a
    // `fn zork(self) -> Container[Int as Copyable] needle, which is both wrong
    // and also doesn't exist in the struct, because the struct contains:
    // `fn zork(self) -> Container[Int as AnyType].
    // I say it's wrong because an "Int as Copyable" parameter-value cannot be
    // given to a parameter-decl that expects an AnyType. The vtables don't line
    // up.
    //
    // So, to fix that, when we substitute into the needle, we first convert the
    // struct alias's value (Int as Copyable) to the trait alias's type
    // (AnyType).
    // Here, we do it ahead of time, just before putting it into the replacer.
    //
    // TODO(MOCO-1993): Make sure this is consistently followed other places we
    // do trait substitution, and maybe centralize this arcana to somewhere.
    traitAliasReplacer.setParameterValue(traitAlias.getParamDecl(),
                                         convertedValue.getIfPValue());
    aliasValues[name] = convertedValue.getIfPValue();

    return success();
  };

  // TODO(MOCO-1143): this loop needs a ParserParameterEvaluator that is
  // populated with the mappings of trait alias requirements to their matched
  // values on the implementing struct, then you call getReboundType/Attribute
  // when checking both the function and future alias requirements
  // ```
  // trait Foo:
  //     alias N: Int
  //     # lit.fn @foo(%self: !kgen.param<Self>,
  //     #               %x: SIMD[float32, #kgen.param.decl.ref<"N">])
  //     fn foo(self, x: SIMD[DType.float32, N]):
  //         ...
  // struct Impl(Foo):
  //     alias N: Int = 4
  //     # lit.fn @foo(%self: !kgen.param<Self>, %x: SIMD[float32,  4])
  //     fn foo(self, x: SIMD[DType.float32, 4]):
  //         pass
  // ```
  bool allMatchFound = true;
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      // Skip any children that aren't methods or aliases.
      if (auto traitFn = dyn_cast<FnOp>(*decl)) {
        if (failed(checkMethod(name, decl, traitFn))) {
          allMatchFound = false;
          break;
        }
      }
      if (AliasDeclOp traitAlias = dyn_cast<LIT::AliasDeclOp>(*decl)) {
        if (failed(checkAlias(name, decl, traitAlias))) {
          allMatchFound = false;
          break;
        }
      }
    }

    // If we had signature resolution errors, don't try to check the
    // conformance.
    if (hadErrors) {
      diag->abandon();
      diag.reset();
      return failure();
    }
  }

  // If everything looks good, succeed without emitting an error.
  if (allMatchFound) {
    diag->abandon();
    diag.reset();
    return success();
  }

  // Otherwise, emit the set of requirements that are missing.
  diag->attachNote(traitDecl.getLoc())
      << "trait " << ASTType(TraitType::get(parent)) << " declared here";
  if (auto *inheritedFrom = structDecl.getTraitConformanceLineage()) {
    if (auto it = inheritedFrom->find(parent);
        it != inheritedFrom->end() && it->second.first != parent) {
      ASTDecl &parentDecl =
          emitter.getDeclResolver().getDeclForTypeSymbol(it->second.first);
      diag->attachNote(parentDecl.getLoc())
          << "inherited through '" << *parentDecl.getNameIfOperation()
          << "' here";
    }
  }
  return failure();
}

/// Given a decl for a struct or trait type, return true if this type conforms
/// to the specified trait type.  On failure, this may set 'diag' to an
/// inflight diagnostic that explains why this doesn't conform.  It can be
/// reported or abandoned based on the client's needs.
bool ASTDecl::doesNominalTypeConformTo(TraitType trait,
                                       std::optional<InflightDiag> &diag) {
  if (failed(shared.declResolver->resolveBody(*this, getLoc())))
    return false; // Error emitted.

  // Collect all the symbols that the type explicitly provides.
  TraitType providedCanonTrait;
  if (auto structOp = dyn_cast<StructDeclOp>(*this)) {
    providedCanonTrait = structOp.getCanonicalTrait();
  } else if (auto traitOp = dyn_cast<TraitDeclOp>(*this)) {
    providedCanonTrait = traitOp.getCanonicalTrait();
    if (providedCanonTrait == trait)
      return true;
  } else if (TraitType canonTraitType =
                 dyn_cast_or_null<TraitType>(getIfTypeValue())) {
    providedCanonTrait = canonTraitType;
    if (providedCanonTrait == trait)
      return true;
  } else {
    llvm_unreachable("Invalid decl kind");
  }

  ArrayRef<SymbolRefAttr> providedSymbols = providedCanonTrait.getSymbols();

  // Check the provided symbols against the required symbols by the target
  // trait. There's no need to canonicalize the required symbols as long as
  // the provided symbols list is canonical.
  DenseSet<SymbolRefAttr> requiredSymbols;
  requiredSymbols.insert(trait.getSymbols().begin(), trait.getSymbols().end());
  for (SymbolRefAttr symbol : providedSymbols)
    requiredSymbols.erase(symbol);

  if (requiredSymbols.empty()) {
    // If this is a struct decl, we need to verify explicit conformances by
    // fully resolving each conformance decl (see CALROC for more).
    bool conforms = true;
    if (auto structOp = dyn_cast<StructDeclOp>(*this)) {
      SmallVector<SymbolRefAttr> fullRequiredSymbols(trait.getSymbols());
      canonicalizeTraitCompositionSymbols(shared, fullRequiredSymbols);
      for (SymbolRefAttr symbol : fullRequiredSymbols) {
        ArrayRef<ASTDecl *> witnessTables =
            lookupInCurrentScope(getFlattenedSymbolName(symbol));
        if (witnessTables.empty()) {
          // There is only one way this can happen: this conformance check
          // occurred while parsing the body of the struct (a self-referential
          // type-value). Treat this as a success, because eventually all
          // conformances on this struct will be resolved & checked.
          continue;
        }
        assert(witnessTables.size() == 1);
        conforms &= succeeded(
            shared.declResolver->resolveBody(*witnessTables.front(), getLoc()));
      }
    }
    return conforms;
  }
  return false;
}

/// Helper for clients that don't care about the diagnostic.
bool ASTDecl::doesNominalTypeConformTo(TraitType trait) {
  std::optional<InflightDiag> diag;
  bool result = doesNominalTypeConformTo(trait, diag);
  if (diag)
    diag->abandon();
  return result;
}

void LIT::sortAndDeduplicateSymbols(SmallVectorImpl<SymbolRefAttr> &symbols) {
  llvm::sort(symbols, [&](SymbolRefAttr a, SymbolRefAttr b) {
    if (a.getRootReference() != b.getRootReference())
      return a.getRootReference().getValue() < b.getRootReference().getValue();
    // Compare each segment of the symbols in dictionary order.
    ArrayRef<FlatSymbolRefAttr> aSegments = a.getNestedReferences();
    ArrayRef<FlatSymbolRefAttr> bSegments = b.getNestedReferences();
    for (auto [aSeg, bSeg] : llvm::zip(aSegments, bSegments)) {
      if (aSeg != bSeg)
        return aSeg.getValue() < bSeg.getValue();
    }
    return aSegments.size() < bSegments.size();
  });
  symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());
}

void LIT::canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols) {
  // Pull in the entire ancestor chain.
  DenseSet<SymbolRefAttr> seen;
  for (SymbolRefAttr symbol : symbols) {
    if (!seen.insert(symbol).second)
      continue;
    ASTDecl &memberDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
    auto traitOp = cast<TraitDeclOp>(memberDecl);
    // Only one level of parent lookup is needed because parentTypes always
    // include their entire ancestor chain.
    ArrayRef<SymbolRefAttr> parentSymbols =
        traitOp.getCanonicalTrait().getSymbols();
    seen.insert(parentSymbols.begin(), parentSymbols.end());
  }
  symbols.assign(seen.begin(), seen.end());
  sortAndDeduplicateSymbols(symbols);
}

//===----------------------------------------------------------------------===//
// IREmitter::emitMetaTypeToTraitConversion
//===----------------------------------------------------------------------===//

namespace {
/// The signature for a trait requirement will have a Self parameter first whose
/// type is a TraitType for the trait it was found in.  We want to force
/// substitute a new parameter for the Self references even though it has a
/// different metatype.  This doesn't remove the parameter, that will be done
/// later.
struct TraitSelfBinder : public IndexParameterReplacer<TraitSelfBinder> {
  TypedAttr selfValue;

  TraitSelfBinder(TypedAttr selfValue) : selfValue(selfValue) {}

  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth) {
    // Replace a reference to $(0,0) with the new selfValue.
    auto paramRef = dyn_cast<ParamIndexRefAttr>(attr);
    if (!paramRef || paramRef.getIndex() != 0 ||
        // Check to see if this is a param ref referring to our Self or some
        // other Self (perhaps in a signature parameter-value that declares its
        // own self or something), see PSTIAIRAID.
        paramRef.getDepth() + 1 != depth)
      return {};
    return selfValue;
  }
  Type tryReplace(Type type, size_t depth) { return {}; }
};
} // namespace

/// Given a method from a trait like 'Movable.__del__', rebind the method to
/// have a different self for a conforming type, e.g.
/// 'RefinedMovableTrait.__del__' or 'Int.__del__'.  'newSelfType' is the
/// struct or trait type to bind.  For example, AnyType.__del__'s signature
/// looks like:
///    !lit.generator<<trait<@AnyType>>[1]("self":
///        !lit.ref<:trait<@AnyType> *(0,0), mut *[0,0]> owned_in_mem) -> none>
/// When binding this down to some MTT conforming to Movable, this will give us
/// something like:
///    !lit.generator<[1]>("self":
///        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
/// Resolving the *(0,0) into the Movable type, as well as the first param type.
static FnTypeGeneratorType
createRequirementSignature(FnOp traitFn, ASTType newSelfType,
                           ParserParameterEvaluator *traitAliasReplacer,
                           const DenseMap<StringAttr, TypedAttr> *aliasValues,
                           DeclResolver &declResolver) {
  // Get the selfType as a TypedAttr since we'll be using it as a parameter
  // value below.
  TypedAttr newSelfValue = PValue(newSelfType).get();

  // Start with the full signature for the trait requirement.
  FnTypeGeneratorType signature = traitFn.getFullSignature();

  if (auto paramType = dyn_cast<ParamType>(newSelfType.getMetaType())) {
    auto simpleTraitType =
        cast<AnyTraitType>(paramType.getParam().getType()).getTraitType();
    // Upcast from a parametric type of trait metatype value (e.g. "some
    // type that conforms to Movable) to the simple trait type (Movable)
    // so we can substitute the value into the signature.
    newSelfValue =
        UpcastAttr::get(simpleTraitType, PValue(newSelfType),
                        VTableAttr::get(simpleTraitType.getContext(), {}));
  }

  // The requirement will have a Self parameter whose type will be of the
  // current trait.  In order to get types to line up, we need to force it
  // to the implementation type.  This changes the parameter value, but also
  // changes the metatype of the value.  To support this, we use a custom
  // replacer.
  TraitSelfBinder selfBinder(newSelfValue);
  signature = selfBinder.replace(signature);

  // At this point, the first parameter is gone:
  //    !lit.generator<[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>

  // Next we'll replace trait aliases that appear in the trait methods, such
  // as:
  //
  //     trait MyTrait:
  //         alias T: ATrait
  //         fn bork(self) -> Something[T]: ...
  //         fn zork(self) -> Something[Self.T]: ...
  //
  // We'll replace them with the struct's trait value, like the int in:
  //
  //     struct MyStruct(MyTrait):
  //         alias T: ATrait = int
  //         fn bork(self) -> SIMD[int]: ...

  // bork's `T` is a regular paramRef, we use traitAliasReplacer to replace it.
  if (traitAliasReplacer)
    signature = traitAliasReplacer->replace(signature);

  // However, zork's `Self.T` is different, like: get_vtable_entry(Self, "T").
  // And after the first step, that Self is actually the struct, so the
  // requirementFn is really more like: get_vtable_entry(MyStruct, "T").
  // We'll manually replace those entire get_vtable_entry calls.
  if (aliasValues) {
    signature = simplifyGetVTableEntryOnSelf(PValue(newSelfType), aliasValues,
                                             signature);
  }

  // At this point, signature's `self` argument's type is the struct or
  // trait.  For example when binding Self down to some "MTT: Movable", we have:
  //    !lit.generator<<trait<@AnyType>>[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
  // Now we need to drop the "<trait<@AnyType>" parameter, which we do by
  // specializing it away.  We know all references to it are already gone.

  // NOTE: This is an UnknownAttr (which is an arbitrary attr that is never
  // used) not an UnboundAttr which remains an unbound parameter.
  ParserParameterEvaluator evaluator(declResolver.shared);
  evaluator.addInputValue(UnknownAttr::get(signature.getInputParamTypes()[0]));
  // Use UnboundAttr for any other parameters so they remain in the result.
  for (Type type : signature.getInputParamTypes().drop_front())
    evaluator.addInputValue(UnboundAttr::get(evaluator.getReboundType(type)));
  signature = signature.getSpecializedGenerator(
      evaluator.getInputParams(),
      /*emitErrorFn=*/{}, &declResolver.shared.getEvaluationContext());

  return signature;
}

FnTypeGeneratorType LIT::specializeSignature(FnOp traitFn, ASTType newSelfType,
                                             DeclResolver &declResolver) {
  return createRequirementSignature(traitFn, newSelfType, nullptr, nullptr,
                                    declResolver);
}

FailureOr<GetWitnessAttr>
LIT::getUniqueWitnessForTypeIfConforms(SharedState &shared, ASTType type,
                                       TraitType trait, StringRef entryName,
                                       SMLoc errorLoc) {
  // Get the decl for the type.
  ASTDecl *typeDecl = type.getDecl(shared);
  if (!typeDecl) {
    [[maybe_unused]] Type metaType = type.getMetaType();
    assert(!metaType || isa<TypeType>(metaType));
    // This is a MLIR type, so we need to bind it to the builtin stub.
    // Use a special wrapper decl in the builtins as stubs.
    typeDecl = shared.getBuiltinStubsMLIRType(errorLoc).getDecl(shared);
    if (!typeDecl || !isa<StructDeclOp>(typeDecl)) {
      shared.emitError(errorLoc, "malformed builtin._stubs.__MLIRType");
      return {};
    }
    // Need to update the type itself to the wrapper type.
    type = cast<StructDeclOp>(typeDecl).bindReference({PValue(type)});
  }

  if (!typeDecl->doesNominalTypeConformTo(trait)) {
    // Does not conform. This is the only non-error case where we return an
    // empty attr.
    return GetWitnessAttr();
  }

  // Make sure the trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(shared.declResolver->resolveBody(*traitDecl, errorLoc)))
    return failure();

  // Locate the entry in the trait.
  ArrayRef<ASTDecl *> entries = traitDecl->lookupInCurrentScope(entryName);
  if (entries.empty()) {
    shared.emitError(errorLoc, "trait ")
        << trait << " has no entry named " << entryName;
    return failure();
  }

  // If there are multiple entries, emit an error.
  if (entries.size() > 1) {
    shared.emitError(errorLoc, "trait ")
        << trait << " has multiple entries named " << entryName;
    return failure();
  }

  ASTDecl &entry = *entries.front();
  Type resultType;
  // TODO(BillyZ): Fix trait alias replacement here once #60811 lands. Currently
  // this function does not properly replace alias mentions with the struct
  // type, but that does not impact any use case since this is only ever called
  // on AnyType and Movable right now.
  if (auto aliasDecl = dyn_cast<AliasDeclOp>(entry)) {
    resultType = aliasDecl.getType();
  } else if (auto fnDecl = dyn_cast<FnOp>(entry)) {
    resultType = createRequirementSignature(fnDecl, type, nullptr, nullptr,
                                            *shared.declResolver);
  } else {
    llvm_unreachable("expected an alias or a function");
  }

  ASTDecl *parentTraitDecl = entry.getParentDecl();
  MLIRContext *ctx = parentTraitDecl->getContext();
  return GetWitnessAttr::get(
      PValue(type),
      StringAttr::get(ctx,
                      getFlattenedSymbolName(parentTraitDecl->getSymbolRef())),
      StringAttr::get(ctx, entryName), resultType);
}

/// Emit a metatype conversion to a trait type by materializing the meta type
/// of the specified CValue into a witness table for the trait.  For example,
/// if 'value' has struct type, and the trait is Movable, then this forms a
/// TypeParamAttr PValue with a vtable containing the __del__ and
/// __moveinit__ methods from the struct.
///
/// If the input value has a derived trait type and the required type is a
/// base trait, then this remaps each of the requirements into the expected
/// format of the result vtable, e.g.:
///   fn take_any_type[ATT: AnyType](x: ATT): pass
///   fn pass_movable[MTT: Movable](x: MTT): take_any_type(x)
///
/// Yields something like:
///     #kgen.type<!kgen.param<:trait<@Movable> MTT>, {
///        "__del__" : !lit.generator<[1](
///                    "self": !lit.ref<:trait<@Movable> MTT, ...)>
///          = get_vtable_entry(:trait<@Movable> MTT, "__del__")
///     }> : !lit.trait<@AnyType>
///
/// This maps from the Movable trait metatype into the AnyType trait metatype.
PValue IREmitter::emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                                TraitType trait) {
  // Only static vtables are supported right now.
  PValue typePValue = value.ir.getIfPValue();
  if (!typePValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }

  // Get the StructMetaType or the TraitType of the value that we're checking
  // for conversion to the trait type.  This can also bind empty variadic
  // parameter lists and default parameters.
  ASTType type = emitType({typePValue, value.expr}, /*allowUnbound*/ false);
  if (!type)
    return {};
  value.ir = PValue(type); // update value.ir if the type was rebound.

  // Check that the struct or super trait implements the trait.
  ASTDecl *metaTypeDecl = type.getDecl(shared);
  if (!metaTypeDecl) {
    emitError(value.expr->getLoc(), "cannot get metatype of ")
        << type << value.expr->getRange();
    return {};
  }

  std::optional<InflightDiag> checkDiag;
  if (!metaTypeDecl->doesNominalTypeConformTo(trait, checkDiag)) {
    InflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                        << type << " to trait " << ASTType(trait)
                        << value.expr->getRange();
    if (checkDiag)
      diag.attachNote(metaTypeDecl->getLoc()) << std::move(*checkDiag);
    return {};
  }

  // Synthesize the vtable required for the trait from the struct. Make sure the
  // trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(getDeclResolver().resolveBody(*traitDecl, value.expr->getLoc())))
    return {};

  // Determine if the conforming value is trivial or register passable.  If so,
  // this will affect the methods we can synthesize in conformance. Values of
  // trait type will already have been erased to a memory type.
  ArrayRef<ParamDeclAttr> structParamDecls;
  if (auto structDeclOp = dyn_cast<StructDeclOp>(metaTypeDecl))
    structParamDecls = structDeclOp.getParams();

  // When we're looking for a trait's method in a certain struct, like:
  //     trait TraitWithAliasMethod:
  //         alias T: ATrait
  //         fn bork(self) -> Something[T]: ...
  // and a struct overrides it:
  //     struct ExplicitStructWithAliasMethod(TraitWithAliasMethod):
  //         alias T: ATrait = int
  //         fn bork(self) -> SIMD[int]: ...
  // we don't want to look for a `fn bork(self) -> Something[T]` in the struct,
  // we want to look for a `fn bork(self) -> SIMD[int]`. This helps us do that.
  ParserParameterEvaluator traitAliasReplacer(shared);
  DenseMap<StringAttr, TypedAttr> aliasValues;

  // If the struct (e.g. List[T]) has an alias that uses an input parameter,
  // (e.g. `alias element_type = T`), then this will help us interpret that
  // alias value while filling the above traitAliasReplacer.
  // FIXME: We need to reject accessing aliases of a partially bound type, until
  // ParameterizedType is a thing!
  ParserParameterEvaluator implGenericsReplacer(shared, structParamDecls,
                                                type.getParamBindings());

  // Bind each trait requirement into vtable entries.
  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, requirementDecls] : traitDecl->getDeclsInScope()) {
    // Each entry can have multiple overloads in 'decls'.
    if (requirementDecls.empty())
      continue;

    // Find candidates in the implementing type (either a struct or trait) which
    // also may have multiple overloads.
    LookupResult result =
        shared.lookupAndResolveDecl(name, value.expr->getLoc(), *metaTypeDecl,
                                    /*searchParentScopes=*/false);
    ArrayRef<ASTDecl *> impls = result.getIfSuccess();

    if (auto traitAliasDecl = dyn_cast<AliasDeclOp>(requirementDecls.front())) {
      // These asserts should be safe because we already know it correctly
      // conforms because we called `doesNominalTypeConformTo` above.
      assert(impls.size() == 1);
      auto implAlias = cast<AliasDeclOp>(impls.front());

      TypedAttr newValue = implAlias.getValueAttr();
      if (newValue) {
        newValue = implGenericsReplacer.replace(newValue);
        // If a decl has a parameter "T : Trait" where Trait defines an
        // associated type "U : Trait2", then when we emit vtable for T, we must
        // also emit vtable for T.U.  We perform this by implicitly converting
        // to the alias' declared type.
        newValue = emitPValue({newValue, value.expr}, EC_Trait,
                              traitAliasDecl.getType());
      } else {
        // Must come from a child trait. Pull the alias value out of the vtable
        // of the child trait alias' type.
        newValue = ParamOperatorAttr::get(
            POC::GetVTableEntry,
            {PValue(type),
             StringAttr::get(name.getValue(), StringType::get(getContext()))},
            implAlias.getType());

        if (implAlias.getType() != traitAliasDecl.getType()) {
          // If the overriding alias's type is more specific than the
          // super-trait's type (see STATCBMS), we'll need to cast it.
          newValue =
              UpcastAttr::get(traitAliasDecl.getType(), newValue,
                              // This vtable can be empty because the vtable
                              // should come from the newValue.
                              VTableAttr::get(traitAliasDecl.getContext(), {}));
        }
      }

      if (!newValue)
        return {};

      vtable.push_back(VTableEntryAttr::get(name, newValue));
      traitAliasReplacer.setParameterValue(traitAliasDecl.getParamDecl(),
                                           newValue);
      aliasValues[name] = newValue;
      continue;
    }

    // Traits shouldn't have var decls or other things.
    if (!isa<FnOp>(requirementDecls.front()))
      continue;

    // Each requirement may be overloaded, resolve each individually.
    for (ASTDecl *expected : requirementDecls) {
      auto traitFn = dyn_cast<FnOp>(expected);
      assert(traitFn && "trait has an alias and a fn with the same name!");

      // For any given requirement, the implementing type may have multiple
      // overloads.  Resolve which one we're using by forming an overload set
      // and filtering it.  Start by finding a set of param bindings for the
      // implementing function that get bound, including:
      //
      //  * The self type if the conforming type is a trait.
      //  * The conforming struct's values for the trait's aliases.

      // The requirement will have a Self parameter whose type will be of the
      // current trait.  In order to get types to line up, we need to force it
      // to the implementation type.  This changes the parameter value, but also
      // changes the metatype of the value.  To support this, we use a custom
      // replacer.
      // TODO(MOCO-1789): This complicated logic will be removed once we have
      // symbolized witness tables.
      FnTypeGeneratorType requirementSig = createRequirementSignature(
          traitFn, type, &traitAliasReplacer, &aliasValues, getDeclResolver());

      // Form a set of bindings to plow into the impl signature by binding Self
      // to the appropriate Struct or derived Trait type.
      // We need to upcast the self type to the parent trait type, so that it
      // can be marked prechecked in the bindings of trait functions that have
      // parameters in their signature, e.g.:
      // trait Writable:
      //     fn write_to[W: Writer](self, mut writer: W): pass
      auto parentTraitType = cast<TraitType>(
          expected->getParentDecl()->getTypeDeclSelf().getMetaType());
      auto implBindings = ParamBindings::getForDeclaredType(
          getDeclScope(), type, value.expr, parentTraitType);

      // Leave the rest of the the parameters Unbound.
      ParserParameterEvaluator evaluator(shared);
      for (Type type : requirementSig.getInputParamTypes()) {
        auto unbound = UnboundAttr::get(evaluator.getReboundType(type));
        evaluator.addInputValue(unbound);
        implBindings.addPrechecked(value.expr, unbound);
      }

      // If the input type is a trait, no need to look through its methods since
      // trait inheritance is always explicit.
      if (isa<TraitType>(metaTypeDecl->getIfTypeValue())) {
        TypedAttr result = ParamOperatorAttr::get(
            POC::GetVTableEntry,
            {PValue(type),
             StringAttr::get(name.getValue(), StringType::get(getContext()))},
            requirementSig);
        vtable.push_back(VTableEntryAttr::get(name, result));
        continue;
      }

      // Grab the matching function.  If the input type is a StructType, this
      // will directly bind the method in question.  If the input type is
      // something like "T: Movable" and we're binding __del__ then this will
      // end up with `get_vtable_entry(T, "__del__")`.
      OverloadSet ov(name, impls, std::move(implBindings), value.expr,
                     CallSyntax::kMethodCallSynthetic);
      auto result = ov.filterOverloadSetForValueType(
          requirementSig, getDeclScope(), /*emitDiagnosticOnFailure=*/true);
      if (!result)
        return {};
      assert(result.getType().mlirType == requirementSig &&
             "didn't form a fn with signature of expected type");
      vtable.push_back(VTableEntryAttr::get(name, result));
    }
  }

  // Create the new type value with the vtable and the trait metatype.
  return TypeParamAttr::get(type, trait, VTableAttr::get(getContext(), vtable));
}
