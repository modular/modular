//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the trait conformance checking
// and special function synthesis logic.
//
//===----------------------------------------------------------------------===//

#include "Traits.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "OverloadSet.h"

#include "MojoUtils.h"
#include "ParserEvaluationContext.h"
#include "StabilityMarkers.h"
#include "StructEmitter.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetOperations.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Collect all declarations with a given name from both the struct and
/// relevant extensions. This is a type-agnostic multi-scope lookup helper;
/// the caller is responsible for filtering/checking declaration types.
/// TODO(MOCO-522): See if we can consolidate logic in an ASTDecl method, or
/// consolidate with the collectTypeAndExtensionDecls usage in ExprNodes.cpp.
static llvm::SmallVector<ASTDecl *>
collectMethodsWithNameFromDecls(ASTDecl &structDecl, StringAttr methodName,
                                ArrayRef<ASTDecl *> relevantExtensions) {
  // Look for declarations in the struct first
  ArrayRef<ASTDecl *> structDecls = structDecl.lookupInCurrentScope(methodName);
  llvm::SmallVector<ASTDecl *> allDecls(structDecls.begin(), structDecls.end());

  // Also look for declarations in relevant extensions
  for (ASTDecl *extDecl : relevantExtensions) {
    ArrayRef<ASTDecl *> extDecls = extDecl->lookupInCurrentScope(methodName);
    allDecls.insert(allDecls.end(), extDecls.begin(), extDecls.end());
  }

  return allDecls;
}

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type.
static FnTypeGeneratorType
getTraitFunctionSignature(ASTDecl &declScope, FnTypeGeneratorType fullSig,
                          ASTType structSelfType, TraitSymbolAttr traitSymbol) {
  ASTDecl &traitDecl = declScope.getShared().declResolver->getDeclForTypeSymbol(
      traitSymbol.getSymbol());
  LogicalResult result = declScope.getShared().declResolver->resolveSignature(
      traitDecl, llvm::SMLoc());
  assert(result.succeeded() && "failed to resolve signature");

  TraitDeclOp traitDeclOp = cast<TraitASTDecl>(traitDecl).getOp();
  auto trait = traitDeclOp.getCanonicalTrait();
  if (auto eva = populateTraitBindingEvaluator(traitSymbol, traitDeclOp))
    trait = eva->replace(trait);

  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = fullSig.getInputParamTypes();

  // Add trait's _Self param replacement.
  params.push_back(TypeParamAttr::get(structSelfType, trait));

  // Leave the rest alone.
  for (Type type : paramTypes.drop_front())
    params.push_back(UnboundAttr::get(type));

  FnTypeGeneratorType newSignature = fullSig.getSpecializedGenerator(
      params, &declScope.getShared().getEvaluationContext());
  return newSignature;
}

/// Used to determine if this particular child def op of a struct.decl op is one
/// corresponding to an inherited trait method. Checking just the inheritedFrom
/// attribute is insufficient as fnOp may actually be the fnOp in the parent
/// trait.
static bool isInheritedFnOp(FnOp fnOp) {
  return fnOp.getInheritedFrom().has_value() || fnOp.isDefaultedTraitFn();
}

// Signature resolves any methods in 'structDecl' with 'name' that were
// inherited from 'traitDecl'. This will also catch any errors where multiple
// parent traits define a function of the same signature/name with no override
// was provided in the child struct.
//
// At the start of this function ASTDecls corresponding to inherited trait
// methods still point to the def ops in the actual trait. This function takes
// care of actually creating the def op in the struct.decl with appropriate
// signature.
//
// Why constraint checking is needed here (not just in later witness selection):
// This function decides whether to discard the trait's default method stub or
// create a wrapper for it. If a struct method has a matching signature but
// incompatible constraints, we must NOT discard the stub, because:
//
// Example 1 - Struct method with incompatible constraints:
//   trait Copyable:
//     def __copyinit__(out self, existing: Self, /):
//       # default implementation
//
//   struct Wrapper[T: Movable](Copyable where conforms_to(T, Copyable)):
//     def __copyinit__(out self, existing: Self, /)
//         where not conforms_to(T, Copyable):  # <-- Incompatible!
//       ...
//
// Without constraint checking: We'd see a signature match, discard the stub,
// and lose the default implementation. Later constraint filtering would reject
// the struct's method, leaving no valid implementation.
//
// With constraint checking: We correctly see the constraints are incompatible,
// so we keep the stub and create a wrapper for the trait's default.
//
// Example 2 - Struct method with compatible constraints:
//   struct Wrapper[T: Movable](Copyable where conforms_to(T, Copyable)):
//     def __copyinit__(out self, existing: Self, /)
//         where conforms_to(T, Copyable):  # <-- Compatible!
//       ...
//
// The conformance constraint implies the method constraint, so this method
// is a valid override - we discard the stub.
static LogicalResult signatureResolveDefaultTraitFnStubs(
    StructASTDecl &structDecl, ASTDecl &traitDecl, StringAttr name,
    ArrayRef<ASTDecl *> candidates, ConstraintAttr conformanceConstraint) {

  auto &shared = structDecl.getShared();

  SyntheticNode node(structDecl.getLoc());
  IREmitter emitter(structDecl, EC_Trait);
  auto structSelfType = structDecl.getSelfType();

  /// Will attempt to create a wrapper fn op in structFnDecl that will call into
  /// the fn op in traitFnDecl (taking things like aliases that show up in
  /// signature into account). This will also report if another trait has
  /// already provided a default implementation and the struct itself does not
  /// provide an override.
  auto signatureResolveStub = [&](DefASTDecl *traitFnDecl,
                                  DefASTDecl *structFnDecl) -> LogicalResult {
    FnOp traitFn = traitFnDecl->getOp();

    SyntheticNode syntheticNode(structDecl.getLoc());
    auto wrapperSignature = getTraitFunctionSignature(
        structDecl, traitFn.getFullSignature(), structSelfType,
        TraitSymbolAttr::get(traitDecl.getSymbolRef()));

    // If a function with matching signature is defined in the same trait we're
    // golden since existing machinery takes care of reporting there is a
    // conflict. However in the case of conflicts from two different traits the
    // earliest we can catch them is right here (and incidentally also the
    // latest as things get wonky with overload resolution if two decls with
    // matching signatures exist in a struct).
    auto possibleOverloads = structDecl.lookupInCurrentScope(name);

    SmallVector<ASTDecl *> structDefinesMethods;
    for (ASTDecl *decl : possibleOverloads) {
      auto *implDecl = llvm::dyn_cast_if_present<DefASTDecl>(decl);
      if (!implDecl || isInheritedFnOp(implDecl->getOp()))
        continue;

      structDefinesMethods.push_back(decl);
    }

    // We can't just directly compare signatures because we may be in a
    // scenario like: trait Foo:
    //   alias X: AnyType
    //   def foo(self) -> Self.x
    //  ...
    //
    // struct Bar(Foo):
    //   alias X: AnyType = Int
    //   def foo(self) -> Self.X:
    //     ...
    //
    // Foo::foo's signature will be something like:
    // (self: !lit.ref<Foo>, __result__: !lit.ref<AnyType>)
    //
    // while Bar::foo's will be something like:
    //
    // (self: !lit.ref<Bar>) -> !lit.struct<Int>
    //
    // Defer to using filterOverloadSetForValueType since it already handles
    // such differences.
    // Only bind the `Self` through the contextual type, leave the rest
    // unbound.
    ParamBindings bindings(structDecl, &syntheticNode);
    bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);
    OverloadSet ov(name, structDefinesMethods, std::move(bindings),
                   CallSyntax::kMethodCallSynthetic,
                   /*contextualType=*/structSelfType);
    // This runs in the struct's scope, not the conformance's, so the
    // conformance `where` clause is not among the scope's own assumptions and
    // has to be handed over explicitly.
    if (conformanceConstraint &&
        !isTriviallyTrueConstraint(conformanceConstraint))
      ov.additionalAssumptions.push_back(conformanceConstraint);

    auto wrapperResult =
        ov.filterOverloadSetForValueType(wrapperSignature, nullptr);
    ASTDecl *decl =
        wrapperResult.isYes() ? wrapperResult.getYes().decl : nullptr;
    if (decl) {
      assert(structFnDecl->resolvedness <= DeclResolvedness::signature &&
             "synthesizeMethodInStruct is only valid on non-body resolved Fn "
             "ASTDecls");
      structFnDecl->resolvedness = DeclResolvedness::body;
      shared.declResolver->eraseDecl(*structFnDecl);
      return success();
    }

    // The struct doesn't provide an override, see if the wrapper def we're
    // about to create has a matching signature to an existing wrapper function
    // in the struct.
    for (ASTDecl *decl : possibleOverloads) {
      auto fnOp = cast<DefASTDecl>(decl)->getOp();

      auto existingSignature = fnOp.getFullSignature();
      // now we need to compare the full signature to the trait signature
      if (isEqualCanon(existingSignature, wrapperSignature)) {
        // Produce an informative diagnostic citing the conflicting traits
        // **and the struct name**.
        StringAttr currentTraitName =
            traitDecl.getSymbolRef().getLeafReference();

        ASTDecl *otherTraitFn = shared.resolveAndGetFuncDecl(
            fnOp.getDefaultFnRefAttr(), structDecl.getLoc());
        assert(otherTraitFn && "Couldn't find trait fn decl");

        StringAttr otherTraitName =
            otherTraitFn->getParentDecl()->getSymbolRef().getLeafReference();

        auto diag = shared.emitError(structDecl.getLoc())
                    << "trait method requirement " << traitFn.getDeclName()
                    << " has conflicting default implementations in "
                    << otherTraitName << " and " << currentTraitName
                    << "; you must implement it manually";

        diag.attachNote(*otherTraitFn)
            << "original default implementation from trait " << otherTraitName
            << " here";

        diag.attachNote(*traitFnDecl)
            << "conflicting implementation from trait " << currentTraitName
            << " here";

        // Set decl as erroneous here. To answer why consider a case like:
        //
        // trait Foo1:
        //     def foo(self) -> Int:
        //         return 1
        //
        // trait Foo2(Foo1):
        //     def foo(self) -> Int:
        //         return 2
        //
        // @fieldwise_init
        // struct Foo(Foo2):#), Foo3):
        //     pass
        //
        // In such a case if we're on this codepath decl would correspond to
        // Foo1.foo -- we set it erroneous here to prevent further processing
        // (namely body resolution) and additional spurious errors that would
        // cause.
        decl->setErroneous();
        return failure();
      }
    }

    StructEmitter structEmitter(structDecl);

    // Create builder positioned before the first ConformanceOp.
    auto structDeclOp = structDecl.getOp();
    Block &fieldsBlock = structDeclOp.getFields().front();

    // Position builder before the first ConformanceOp (there will always be
    // one).
    ConformanceOp firstConformanceOp =
        *fieldsBlock.getOps<ConformanceOp>().begin();
    ImplicitLocOpBuilder builder(structDeclOp.getLoc(),
                                 firstConformanceOp.getOperation());

    PogListAttr traitPogs = traitFn.getFuncTypeGenerator().getParamListAttrs();
    for (auto traitPog : traitPogs.getPogs()) {
      if (!structDecl.getParamByName(traitPog.getName()))
        continue;

      auto diag = shared.emitError(structDecl.getLoc())
                  << "name conflict between parameter " << traitPog.getName()
                  << " in the default trait method and a parameter in the "
                     "struct";
      diag.attachNote(*traitFnDecl) << "trait method declared here";
      return failure();
    }

    FnOp newFn = structEmitter.synthesizeDefaultTraitMethodWrapper(
        *structFnDecl, name.str(), wrapperSignature, traitFn, traitFnDecl,
        builder, traitDecl.getSymbolRef().getLeafReference().strref(),
        conformanceConstraint);

    // If newFn is null something went very wrong -- assert
    assert(newFn && "Couldn't synthesize default trait wrapper in body");
    (void)newFn;
    return success();
  };

  for (ASTDecl *decl : candidates) {
    auto *fnDecl = dyn_cast_if_present<DefASTDecl>(decl);
    if (!fnDecl)
      continue;
    FnOp fnOp = fnDecl->getOp();

    // Indicates we've already signature resolved this decl, should actually
    // be able to assume this given the decl is signature resolved, but
    // double check this.
    if (!fnOp.isDefaultedTraitFn())
      continue;

    assert(fnOp->getParentOfType<TraitDeclOp>() &&
           "Expected to have a parent trait decl");

    // It's theoretically possible to have gotten to this point and have
    // fnOp's decl not be signature resolved yet, guard against that.
    if (fnOp->getAttr("sym_namex"))
      continue;

    auto traitFnSymbolRef = getFullyResolvedSymbolRef(fnOp);

    auto traitFnDecl =
        shared.resolveAndGetFuncDecl(traitFnSymbolRef, structDecl.getLoc());
    assert(traitFnDecl && "Couldn't find trait fn decl");

    auto parentTraitRef = traitFnDecl->getParentDecl()->getSymbolRef();

    // resolve corresponds to the trait we're currently working on in
    // verifyConformance.
    if (parentTraitRef != traitDecl.getSymbolRef())
      continue;

    // Grab the actual decl corresponding to the trait function we'll be
    // wrapping.
    if (failed(signatureResolveStub(cast<DefASTDecl>(traitFnDecl),
                                    cast<DefASTDecl>(decl))))
      return failure();
  }
  return success();
}

LogicalResult
LIT::verifyAndBuildConformance(ConformanceASTDecl &conformanceDecl,
                               std::optional<MojoInflightDiag> &diag) {
  StructASTDecl &structDecl = conformanceDecl.getStructDecl();
  ConformanceOp op = conformanceDecl.getOp();
  TraitSymbolAttr parent = op.getTraitSymbolAttr();
  assert(parent && "conformance bodies are resolved before `lower-lit` erases "
                   "the trait reference");
  // If the conformance table already has witnesses, it was pre-built (e.g., for
  // closure wrappers). Skip verification since conformance is already complete.
  if (!op.getBody().front().empty())
    return success();

  // Set up builder to insert witness entry. We install witness op in the
  // conformance op as we verifying such that get_witness will be folded
  // correctly by the evaluation context. This allows us to fold dependent decls
  // in side the trait definition. E.g.,
  //
  // trait A:
  //    alias a : Int
  //    alias b : S[Self.a]
  //
  // Note that this is not only an optimization but also necessary to verify
  // dependent trait entry. Otherwise, for the following example:
  //
  // struct S(A):
  //    alias a : Int = 1
  //    alias b : S[1]
  //
  // `S[1]` is not the same type as `S[Self.a]` without a folding Self.a to `1`
  // (or we would need to insert a rebind).
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(op.getLoc(), &op.getBody().front());

  auto &shared = structDecl.getShared();
  auto structDeclOp = structDecl.getOp();

  // Find extensions that target this struct and implement this trait.
  // This implements a form of the orphan rule: extensions can be defined either
  // in the same file as the struct or in the same file as the trait. This
  // ensures that extensions/conformances are local to either the struct or the
  // trait, preventing conflicting implementations across different files.
  // TODO(MOCO-522): Turn this into arcana docs!
  llvm::SmallPtrSet<ASTDecl *, 4> uniqueExtensions;

  // Search for extensions in the struct's file scope
  if (FileModuleASTDecl *structFileScope =
          structDecl.getNearestDeclOfType<FileModuleASTDecl>()) {
    structFileScope->findExtensionsInScopeForStruct(structDecl.getSymbolRef(),
                                                    uniqueExtensions, parent);
  }

  // Search for extensions in the trait's file scope
  auto &traitDecl = cast<TraitASTDecl>(
      shared.declResolver->getDeclForTypeSymbol(parent.getSymbol()));
  if (FileModuleASTDecl *traitFileScope =
          traitDecl.getNearestDeclOfType<FileModuleASTDecl>()) {
    traitFileScope->findExtensionsInScopeForStruct(structDecl.getSymbolRef(),
                                                   uniqueExtensions, parent);
  }

  llvm::SmallVector<ASTDecl *> relevantExtensions(uniqueExtensions.begin(),
                                                  uniqueExtensions.end());

  bool hadErrors = false;
  IREmitter emitter(structDecl, EC_Trait);
  ASTType selfType = structDecl.getSelfType();
  TraitDeclOp traitDeclOp = traitDecl.getOp();

  // Make sure to fully resolve the trait first.
  if (failed(shared.declResolver->resolveBody(traitDecl, structDecl.getLoc())))
    return failure();

  // Make sure we've to at least signature resolve all the decls in the trait.
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (auto &decl : decls) {
      if (failed(
              shared.getDeclResolver().resolveSignature(*decl, decl->getLoc())))
        return failure();
    }
  }

  if (traitDeclOp.isRegisterPassable() && !structDeclOp.isRegisterPassable()) {
    // Non-RP structs may conditionally conform to RP traits only when the
    // conformance constraint implies the struct's RegisterPassable constraint.
    // This guarantees that whenever the conformance is active, the struct is
    // actually register-passable. In practice the ancestor implication check
    // in DeclResolution.cpp fires first, but this is defense-in-depth.
    ConstraintAttr rpConstraint =
        structDeclOp.getRegisterPassableConstraintAttr();
    bool conformanceImpliesRP =
        rpConstraint && op.getConstraintAttr() &&
        isImplicationProven(rpConstraint.getProposition(),
                            op.getConstraintAttr().getProposition());
    if (!conformanceImpliesRP) {
      diag = shared.emitError(structDecl.getLoc(),
                              "a struct must be register passable in order to "
                              "inherit from a register passable trait");
      return failure();
    }
  }

  // In the below loops, we'll be checking that each trait method ("needle") is
  // present in the struct too.
  // For example:
  //
  //     trait MyTrait:
  //         def zork(self) -> Int:
  //             ...
  //     struct MyStruct:
  //         def zork(self) -> Int:
  //             ...
  //
  // We simply look to see if the trait's `def zork(self) -> Int` (the needle)
  // exists in the struct too, which it does.
  //
  // Trait aliases in the needle are `get_witness<_Self, MyTrait, "X">`, so
  // specializing `_Self` to the struct resolves them to the struct's values.
  //
  // Alias requirements are not specialized that way: their types and
  // defaulted values still mention `_Self` by name, so rebind it here.
  //
  // TODO(MOCO-1993): Consolidate docs on this.
  auto traitSelfDecl =
      cast<ParamDeclRefAttr>(PValue(traitDecl.getSelfType()).get());
  ParameterEvaluator traitSelfBinder = shared.getParameterEvaluator();
  traitSelfBinder.setDeclBinding(traitSelfDecl.getName(),
                                 PValue(structDecl.getSelfType()));

  // Prepare an error. It will be abandoned if the check succeeds.
  diag = shared.emitError(structDecl.getLoc())
         << selfType << " does not implement all requirements for "
         << shared.declResolver->getCanonicalTrait(parent);

  // Returns failure() to stop the verifyConformance loop.
  auto checkMethod = [&](StringAttr name, ASTDecl *traitFnDecl,
                         StringAttr fnSymName,
                         FnTypeGeneratorType fullSig) -> LogicalResult {
    auto reportNotImplemented = [&]() -> LogicalResult {
      diag->attachNote(*traitFnDecl)
          << "required function '" + name.str() + "' is not implemented";
      return failure(); // Stop the outer loop.
    };

    // Collect all method declarations from struct and extensions
    llvm::SmallVector<ASTDecl *> decls =
        collectMethodsWithNameFromDecls(structDecl, name, relevantExtensions);

    if (decls.empty())
      return reportNotImplemented();

    // Signature resolve any decls that don't correspond to inherited trait
    // methods first. This helps us avoid any infinite loops as signature
    // resolution for inherited methods also calls verifyConformance.
    for (ASTDecl *decl : decls) {
      if (auto *fnDecl = dyn_cast<DefASTDecl>(decl))
        if (isInheritedFnOp(fnDecl->getOp()))
          continue;
      if (failed(shared.declResolver->resolveSignature(*decl,
                                                       structDecl.getLoc()))) {
        hadErrors = true;
        return success();
      }
    }

    // Signature resolve any stubs corresponding to defaulted methods for
    // the current trait here. We have to do this before the upcoming signature
    // resolution of all decls with a matching name to avoid cycles between
    // verifyConformance and signature resolution for FnOps.
    if (failed(signatureResolveDefaultTraitFnStubs(
            structDecl, traitDecl, name, decls, op.getConstraintAttr()))) {
      diag->abandon();
      return failure();
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
    auto traitSignature =
        getTraitFunctionSignature(conformanceDecl, fullSig, selfType, parent);

    // Every candidate goes to the overload set, which decides each one against
    // the conformance constraint: a candidate whose `where` clause contradicts
    // it is rejected, one that can be neither proven nor disproven is
    // inconclusive and blocks selection entirely (we cannot rule it out as the
    // witness), and what remains must name a single winner.
    // The bindings are scoped to the conformance decl, which carries the
    // conformance's `where` clause as a known assumption, so candidate
    // filtering picks it up from the scope without being handed it here.
    ParamBindings bindings(conformanceDecl, &syntheticNode);
    bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);
    OverloadSet ov(name, decls, std::move(bindings),
                   CallSyntax::kMethodCallSynthetic,
                   /*contextualType=*/selfType);
    auto emitError = [&](SMLoc loc) -> MojoInflightDiag & {
      // `attachNote(decl)` also appends the requirement's synthesized
      // signature, redundant with the "no candidates have type" note below;
      // `attachNote(Location)` does not. The location lives on the decl's
      // operation, so use it when present; otherwise fall back to the decl
      // overload to locate the note.
      if (Operation *op = traitFnDecl->getIfOperation())
        return diag->attachNote(op->getLoc());
      return diag->attachNote(*traitFnDecl);
    };
    auto calleeResult =
        ov.filterOverloadSetForValueType(traitSignature, emitError);

    if (!calleeResult.isYes()) {
      // When a candidate was undecidable rather than simply mismatched, point
      // at the requirement: the reader has to weigh the candidate's `where`
      // clause against what the conformance demands, and needs both in view.
      // A plain signature mismatch is already self-explanatory, and the note
      // would drag the requirement's synthesized signature along with it.
      if (calleeResult.isUnknown())
        diag->attachNote(*traitFnDecl) << "required by trait method here";
      return failure();
    }
    auto [result, selectedStructMethod] = calleeResult.getYes();

    // Check for API author error: stable struct implementing stable trait
    // must use stable methods for stable trait methods.
    if (selectedStructMethod) {
      checkStableTraitMemberImplementation(
          structDecl, traitDecl, *selectedStructMethod, *traitFnDecl, shared);
    }

    WitnessOp::create(b, fnSymName, /*sym_visibility=*/nullptr, result.get());
    return success();
  };

  auto checkAlias = [&](StringAttr name, ASTDecl *traitAliasDecl,
                        AliasDeclOp traitAlias) -> LogicalResult {
    if (failed(shared.declResolver->resolveSignature(*traitAliasDecl,
                                                     structDecl.getLoc()))) {
      hadErrors = true;
      return success();
    }

    Type traitAliasType = traitSelfBinder.getReboundType(traitAlias.getType());

    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    // If there is no user defined alias nor defaulted alias, raise an error.
    // When there are multiple decls, it can not be an AliasDeclOp either,
    // otherwise we should have already reported "redefinition" error.
    if (decls.empty() || !isa<AliasASTDecl>(decls.front())) {
      diag->attachNote(*traitAliasDecl)
          << "required member '" << name.str() << "' is not specified";
      return failure(); // Stop the outer loop.
    }
    auto *structAliasDecl = cast<AliasASTDecl>(decls.front());
    AliasDeclOp structAliasDeclOp = structAliasDecl->getOp();

    PValue aliasValue;
    if (!structAliasDeclOp.isDefaultedAssociatedAlias()) {
      if (failed(shared.declResolver->resolveSignature(*structAliasDecl,
                                                       structDecl.getLoc()))) {
        hadErrors = true;
        return success();
      }
      Type structAliasType = structAliasDeclOp.getType();
      TypedAttr initializerExpr = structAliasDeclOp.getValueAttr();
      assert(initializerExpr && "Struct's alias should have initializer");

      // Pass the conformance constraint as an assumption.
      SmallVector<ConstraintAttr, 1> conformanceAssumptions;
      if (ConstraintAttr conformanceConstraint = op.getConstraintAttr())
        conformanceAssumptions.push_back(conformanceConstraint);

      // The witness must hold the value converted to the trait alias's type
      // (see SAVMBCTATBS), so first check that the conversion is possible.
      SyntheticNode synthNode(structAliasDecl->getLoc());
      if (!IREmitter::canImplicitlyConvertToType(
              {initializerExpr, &synthNode}, traitAliasType,
              emitter.getDeclScope(), conformanceAssumptions)) {
        diag->attachNote(*traitAliasDecl)
            << "comptime member "
            << "'" + name.str() + "'" << " type " << ASTType(structAliasType)
            << " does not conform to trait's required type "
            << ASTType(traitAliasType);
        return failure();
      }

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
      //         def zork(self) -> Container[X]:
      //             ...
      //     struct MyStruct:
      //         alias X: Copyable = Int     <-- "Int as Copyable" TypeParamAttr
      //         def zork(self) -> Container[Int]
      //
      // Notice the X: Copyable.
      // That means that the struct's `X` has a value that's a TypeParamAttr,
      // basically an `Int` that's masquerading as a `Copyable`.
      // The needle `def zork(self) -> Container[X]` refers to `X` through
      // `get_witness<MyStruct, MyTrait, "X">`, which resolves to the witness
      // value. If that were `Int as Copyable`, the needle would become
      // `Container[Int as Copyable]`, which is both wrong and doesn't exist in
      // the struct, because the struct contains `Container[Int as AnyType]`.
      // An "Int as Copyable" parameter-value cannot be given to a
      // parameter-decl that expects an AnyType. The vtables don't line up.
      //
      // So the witness holds the struct alias's value converted to the trait
      // alias's type (AnyType).
      //
      // TODO(MOCO-1993): Make sure this is consistently followed other places
      // we do trait substitution, and maybe centralize this arcana to
      // somewhere.
      ExprDest dest(traitAliasType, EC_AliasValue);
      CValue convertedValue = emitter.emitImplicitConversionToType(
          {initializerExpr, &synthNode}, traitAliasType, dest,
          conformanceAssumptions);

      aliasValue = convertedValue.getIfPValue();
    } else {
      // Handle cases where this is a default value.
      //
      // `structAliasDeclOp` already points at the defaulted alias that
      // satisfies this requirement (resolved as part of the trait body
      // it lives in). When the struct doesn't declare the alias
      // directly, `insertDefaultDecl` inserts the most-refined defaulted
      // version from the conformance chain. For example, with
      // `trait B(A)` where `B` provides `T = Int` and `struct Foo(B)`
      // doesn't, the struct's `T` decl wraps B's defaulted `T`.
      //
      // Clone from this struct-side defaulted op directly rather than
      // re-looking up on `traitDecl` and resolving from there: when
      // `traitDecl` is a less-refined parent (e.g., A) whose own `T` is
      // abstract, the lookup would produce a non-defaulted, valueless
      // alias and break subsequent conformance checks against the
      // refining trait.
      //
      // Skip the clone if a prior conformance check (e.g. Foo→A) already
      // materialized this default into the struct body — otherwise the
      // second pass (Foo→B) would duplicate the decl and trigger a
      // "redeclaration" error during elaboration.
      AliasDeclOp clonedDefault;
      if (structAliasDecl->resolvedness == DeclResolvedness::body) {
        clonedDefault = structAliasDeclOp;
      } else {
        AliasDeclOp defaultAliasOp = structAliasDeclOp;
        // Position builder before the first ConformanceOp (there will always
        // be one since the struct should at least conform the trait that we
        // are cloning the defaulted value from).
        ConformanceOp firstConformanceOp =
            *structDeclOp.getFields().front().getOps<ConformanceOp>().begin();
        ImplicitLocOpBuilder builder(structDeclOp.getLoc(),
                                     firstConformanceOp.getOperation());
        clonedDefault = cast<AliasDeclOp>(builder.clone(*defaultAliasOp));
        Attribute newValAttr =
            traitSelfBinder.replace(clonedDefault->getAttrDictionary());
        clonedDefault->setAttrs(cast<DictionaryAttr>(newValAttr));
        structAliasDecl->setOp(clonedDefault);
        structAliasDecl->resolvedness = DeclResolvedness::body;
      }

      aliasValue = PValue(clonedDefault.getValueAttr());
      // Since we cloned the defaulted op from the trait, there is no need for
      // us to convert the type as they are guaranteed to be matched.
    }

    // Check for API author error: stable struct implementing stable trait
    // must use stable aliases for stable trait aliases.
    checkStableTraitMemberImplementation(
        structDecl, traitDecl, *structAliasDecl, *traitAliasDecl, shared);

    WitnessOp::create(b, name, /*sym_visibility=*/nullptr, aliasValue.get());

    return success();
  };

  bool allMatchFound = true;
  if (shared.isUniversalParametricClosureTrait(&traitDecl)) {
    assert(traitDecl.getDeclsInScope().size() == 1);
    assert(traitDecl.getDeclsInScope().front().first == "__call__");
    assert(traitDecl.getDeclsInScope().front().second.size() == 1);

    ASTDecl *decl = traitDecl.getDeclsInScope().front().second.front();
    AliasDeclOp callAlias = cast<AliasASTDecl>(decl)->getOp();
    auto evaluator = populateTraitBindingEvaluator(parent, shared);
    auto fullSig =
        cast<FnTypeGeneratorType>(evaluator->replace(callAlias.getType()));

    auto name = StringAttr::get(shared.getContext(), "__call__");
    allMatchFound = succeeded(checkMethod(name, &structDecl, name, fullSig));
  } else {
    for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        // Skip any children that aren't methods or aliases.
        if (auto *traitFnDecl = dyn_cast<DefASTDecl>(decl); traitFnDecl) {
          auto traitFn = traitFnDecl->getOp();
          if (failed(checkMethod(name, decl, traitFn.getSymNameAttr(),
                                 traitFn.getFullSignature()))) {
            allMatchFound = false;
            break;
          }
        }
        if (auto *traitAliasDecl = dyn_cast<AliasASTDecl>(decl)) {
          if (failed(checkAlias(name, decl, traitAliasDecl->getOp()))) {
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
  }

  // If everything looks good, succeed without emitting an error.
  if (allMatchFound) {
    diag->abandon();
    diag.reset();
    return success();
  }

  bool isSynthetic = shared.isUniversalParametricClosureTrait(&traitDecl);
  // We don't have a location for synthetic decl for the attached note.
  if (!isSynthetic) {
    // Otherwise, emit the set of requirements that are missing.
    diag->attachNote(traitDecl)
        << "trait " << ASTType(shared.declResolver->getCanonicalTrait(parent))
        << " declared here";
    if (auto *inheritedFrom =
            cast<StructASTDecl>(structDecl).getTraitConformanceLineage()) {
      if (auto it = inheritedFrom->find(parent);
          it != inheritedFrom->end() && it->second.first != parent) {
        ASTDecl &parentDecl = emitter.getDeclResolver().getDeclForTypeSymbol(
            it->second.first.getSymbol());
        diag->attachNote(parentDecl)
            << "inherited through '" << *parentDecl.getUserNameIfOperation()
            << "' here";
      }
    }
  }
  return failure();
}

static TraitType getDeclProvidedTrait(ASTDecl *decl) {
  // Collect all the symbols that the type explicitly provides.
  TraitType providedCanonTrait;

  if (auto *structDecl = dyn_cast<StructASTDecl>(decl)) {
    providedCanonTrait = structDecl->getOp().getCanonicalTrait();
  } else if (auto *traitDecl = dyn_cast<TraitASTDecl>(decl)) {
    providedCanonTrait = traitDecl->getOp().getCanonicalTrait();
  } else if (TraitType canonTraitType =
                 dyn_cast_or_null<TraitType>(decl->getIfTypeValue())) {
    providedCanonTrait = canonTraitType;
  } else if (isa<PackageASTDecl, FileModuleASTDecl, ImportASTDecl>(decl)) {
    // A package, module, or import reference provides no trait (and is not a
    // copyable runtime value - using one as a value is diagnosed elsewhere).
    providedCanonTrait = TraitType::get(decl->getContext(), {});
  } else {
    llvm_unreachable("Invalid decl kind");
  }

  return providedCanonTrait;
}

/// Uncached implementation of ASTDecl::doesNominalTypeConformTo. Only called by
/// that wrapper. If `sawErroneousExtension` is non-null, it is set to true when
/// a conformance-contributing extension of `self` is erroneous, signaling the
/// caller that the result is unstable and must not be cached.
///
/// When `details` is non-null, it is populated with the
/// conditional-conformance constraints behind the verdict: those refuted, or
/// if none was, those left unproven.
static TriBool doesNominalTypeConformToUncached(
    ASTDecl *self, TraitType trait, ASTType concreteType,
    ArrayRef<ConstraintAttr> callerAssumptions,
    bool *sawErroneousExtension = nullptr,
    SmallVectorImpl<ConstraintAttr> *details = nullptr);

/// Given a decl for a struct or trait type, check if this type conforms to the
/// specified trait type. If concreteType is provided, it is used to extract
/// parameter bindings for evaluating conditional trait conformances.
///
/// Returns:
/// - `yes` if the type definitely conforms
/// - `no` if the type definitely does not conform
/// - `unknown` if conformance depends on constraints that cannot be evaluated
///   statically
static TriBool
doesNominalTypeConformToCached(ASTDecl *self, TraitType trait,
                               ASTType concreteType,
                               ArrayRef<ConstraintAttr> callerAssumptions,
                               SmallVectorImpl<ConstraintAttr> *details) {
  TriBool result = TriBool::no();
  if (!callerAssumptions.empty()) {
    // Only the assumption-free queries are context-independent enough to
    // memoize; where-clause assumptions make the result caller-dependent, so
    // those bypass the cache entirely.
    result = doesNominalTypeConformToUncached(
        self, trait, concreteType, callerAssumptions,
        /*sawErroneousExtension=*/nullptr, details);
  } else {
    // Never consult or populate the cache for an erroneous decl: its
    // conformance answer is unstable -- an optimistic declared `yes` (e.g. a
    // struct that declares RegisterPassable but has a non-conforming member)
    // becomes `no` once the invalidating error is diagnosed -- and it is
    // irrelevant since compilation is already failing. Also skip
    // not-yet-signature-resolved decls: resolvedness only moves forward and
    // stores only happen at >= signature, so they can have no entry yet and the
    // lookup would always miss.
    const bool mayBeCached =
        self->resolvedness >= DeclResolvedness::signature &&
        !self->isErroneous();
    // If user requested failure details, we use the cached only if the verdict
    // was true.
    if (mayBeCached) {
      TriBool conforms = self->getShared().getCachedNominalConformance(
          self, trait, concreteType);
      if (conforms.isDefinite() && (!details || conforms.isTrue())) {
        if (details)
          details->clear();
        return conforms;
      }
    }

    // Caching a definitive answer for this (decl, trait, concreteType) assumes
    // the answer is stable once the decl is signature-resolved and
    // non-erroneous. That relies on the conformance-contributing decls being
    // stable too: extension conformances are currently unconditional (never
    // retracted after signature resolution), and `sawErroneousExtension`
    // catches the one exception -- an erroneous contributing extension whose
    // contribution may still change.
    bool sawErroneousExtension = false;
    result = doesNominalTypeConformToUncached(self, trait, concreteType,
                                              callerAssumptions,
                                              &sawErroneousExtension, details);

    // Cache only stable, definitive answers: skip `unknown` (phase-dependent,
    // may become yes/no as more of the program is type checked), erroneous
    // decls (see above), results that depend on an erroneous extension, and
    // results produced before the signature resolved (the error path returns
    // `no` and must re-run so its diagnostics re-emit on each query).
    // NB: re-read resolvedness/isErroneous here rather than reuse `mayBeCached`
    // -- the uncached call above resolves the signature, so a first query can
    // still populate the cache even though `mayBeCached` was false.
    if (result.isDefinite() &&
        self->resolvedness >= DeclResolvedness::signature &&
        !self->isErroneous() && !sawErroneousExtension)
      self->getShared().cacheNominalConformance(self, trait, concreteType,
                                                result.isTrue());
  }

  return result;
}

TriBool
ASTDecl::doesNominalTypeConformTo(TraitType trait, ASTType concreteType,
                                  ArrayRef<ConstraintAttr> callerAssumptions) {
  return doesNominalTypeConformToCached(this, trait, concreteType,
                                        callerAssumptions, /*details=*/nullptr);
}

ConstraintResult ASTDecl::doesNominalTypeConformToWithDetails(
    TraitType trait, ASTType concreteType,
    ArrayRef<ConstraintAttr> callerAssumptions) {
  SmallVector<ConstraintAttr, 2> details;
  TriBool result = doesNominalTypeConformToCached(this, trait, concreteType,
                                                  callerAssumptions, &details);
  return makeConstraintResult(result, std::move(details));
}

static TriBool doesNominalTypeConformToUncached(
    ASTDecl *self, TraitType trait, ASTType concreteType,
    ArrayRef<ConstraintAttr> callerAssumptions, bool *sawErroneousExtension,
    SmallVectorImpl<ConstraintAttr> *details) {
  SharedState &shared = self->getShared();

  // Clear so an early return leaves no stale constraints behind.
  if (details)
    details->clear();

  // We only need trait symbol to verify trait conformance, not the resolved
  // witness table.
  if (failed(shared.declResolver->resolveSignature(*self, self->getLoc())))
    return TriBool::no(); // Error emitted.

  // `where` clauses with messages only live on struct conformance lists, so
  // only a struct can supply diagnostic witnesses.
  auto *selfStructDecl = dyn_cast<StructASTDecl>(self);
  const bool selfIsStruct = selfStructDecl != nullptr;

  // Set up parameter evaluator if we have parameter bindings for constraint
  // evaluation. Uses the parser's evaluation context needed for folding
  // TypeConformsToTraitAttr.
  ParameterEvaluator evaluator;
  ArrayRef<TypedAttr> paramBindings =
      concreteType ? concreteType.getParamBindings() : ArrayRef<TypedAttr>{};
  if (!paramBindings.empty() && selfStructDecl)
    evaluator = shared.getParameterEvaluator(
        selfStructDecl->getOp().getInputParams(), paramBindings);

  TraitType declProvidedTrait = getDeclProvidedTrait(self);
  // Collect all the symbols that the type explicitly provides, only re-evaluate
  // the symbol, otherwise we might discard the error message on where clause.
  //
  // TODO: should we turn this into a util? getCanonicalSymbols should always
  // going through the rebinding process for struct/trait symbols.
  auto providedSymbols = llvm::map_to_vector(
      declProvidedTrait.getSymbols(), [&](TraitSymbolAttr symbol) {
        return cast<TraitSymbolAttr>(evaluator.replace(symbol));
      });
  TraitType providedCanonTrait = TraitType::get(
      self->getContext(), providedSymbols, declProvidedTrait.getConstraints());
  if (providedCanonTrait == trait)
    return TriBool::yes();

  ArrayRef<TraitSymbolAttr> providedSymbolsArr =
      providedCanonTrait.getSymbols();
  ArrayRef<ConstraintAttr> constraints = providedCanonTrait.getConstraints();

  // Map each provided symbol to the condition under which it is provided. A
  // null or trivially-true entry means the symbol is provided unconditionally.
  llvm::SmallDenseMap<TraitSymbolAttr, TypedAttr> providedConditions;
  assert((constraints.empty() ||
          constraints.size() == providedSymbolsArr.size()) &&
         "trait constraints must be parallel to symbols");
  for (auto [i, symbol] : llvm::enumerate(providedSymbolsArr))
    providedConditions[symbol] =
        constraints.empty()
            ? TypedAttr()
            : evaluator.getReboundAttribute(constraints[i].getProposition());

  // Parallel to `providedConditions`, retain each provider constraint so a
  // diagnostic caller can point at the failing `where` clause (and its optional
  // user message). Extension conformances below are unconditional today.
  llvm::SmallDenseMap<TraitSymbolAttr, ConstraintAttr> providedConstraints;
  if (details && selfIsStruct && !constraints.empty())
    for (auto [i, symbol] : llvm::enumerate(providedSymbolsArr))
      providedConstraints[symbol] = constraints[i];

  if (selfStructDecl) {
    llvm::SmallPtrSet<ASTDecl *, 4> uniqueExtensions;
    // Search for extensions in the struct's parent scope.
    // TODO(MOCO-522): Arcana docs on our orphan rule.
    if (ASTDecl *structParent = self->getParentDecl()) {
      structParent->findExtensionsInScopeForStruct(self->getSymbolRef(),
                                                   uniqueExtensions);
    }
    // Search for extensions in the trait's parent scope(s).
    // TODO(MOCO-522): Arcana docs on our orphan rule.
    for (TraitSymbolAttr traitSymbol : trait.getSymbols()) {
      ASTDecl &traitDecl =
          shared.declResolver->getDeclForTypeSymbol(traitSymbol.getSymbol());
      if (ASTDecl *traitParent = traitDecl.getParentDecl()) {
        traitParent->findExtensionsInScopeForStruct(self->getSymbolRef(),
                                                    uniqueExtensions);
      }
    }
    llvm::SmallVector<ASTDecl *> allExtensions(uniqueExtensions.begin(),
                                               uniqueExtensions.end());
    for (ASTDecl *extDecl : allExtensions) {
      if (auto *extensionDecl = dyn_cast<ExtensionASTDecl>(extDecl)) {
        // Signature resolve the extension so we can access its canonicalTrait.
        // A failed resolution marks the extension erroneous, and its
        // (non-)contribution may change once the error is diagnosed, so tell
        // the caller not to cache this result.
        if (failed(shared.declResolver->resolveSignature(*extDecl,
                                                         extDecl->getLoc()))) {
          if (sawErroneousExtension)
            *sawErroneousExtension = true;
          continue;
        }
        // An erroneous contributing extension makes the result unstable (an
        // optimistic `yes` can be demoted once the error surfaces), so signal
        // the caller to skip caching.
        if (extDecl->isErroneous() && sawErroneousExtension)
          *sawErroneousExtension = true;
        // Extensions sometimes don't have canonical traits (like in isolated
        // tests).
        ExtensionDeclOp extOp = extensionDecl->getOp();
        if (!extOp.getCanonicalTrait().has_value())
          continue;
        TraitType extCanonicalTrait = extOp.getCanonicalTrait().value();
        for (TraitSymbolAttr symbol : extCanonicalTrait.getSymbols()) {
          // Extension conformances are currently unconditional.
          providedConditions[symbol] = TypedAttr();
        }
      }
    }
  }

  ArrayRef<ConstraintAttr> requiredConstraints = trait.getConstraints();
  SmallVector<TypedAttr> callerAssumptionProps =
      llvm::map_to_vector(callerAssumptions, [](ConstraintAttr constraint) {
        return constraint.getProposition();
      });
  // Fold the caller conjunction on first use, not up front: most requirements
  // are unconditionally provided and never consult the assumptions, and a
  // scalar-bool AND now pays a full `insertClause` fold to build.
  std::optional<TypedAttr> callerCanonicalCache;
  auto callerCanonical = [&] {
    if (!callerCanonicalCache) {
      if (callerAssumptionProps.empty())
        callerCanonicalCache =
            SIMDAttr::getScalarBool(self->getContext(), true);
      else if (callerAssumptionProps.size() == 1)
        callerCanonicalCache = callerAssumptionProps.front();
      else
        callerCanonicalCache =
            ParamOperatorAttr::get(POC::And, callerAssumptionProps);
    }
    return *callerCanonicalCache;
  };

  // Check the provided symbols against the required symbols by the target
  // trait: a definitively-missing required symbol keeps the whole thing `no`,
  // an unproven one makes it `unknown`, and all-present makes it `yes`.

  // With `details`, collect every provider constraint behind the verdict and
  // dedupe on (loc, proposition) so derived/ancestor copies of the same
  // `where` clause emit once. Cold path.
  const bool collectAll = details != nullptr;
  DenseSet<std::pair<LocationAttr, Attribute>> seenConstraints;
  TriBool result = TriBool::yes();
  auto recordFailure = [&](TraitSymbolAttr required, TriBool kind) {
    if (!collectAll)
      return;
    assert(kind.isFalse() || kind.isUnknown());
    ConstraintAttr constraint = providedConstraints.lookup(required);
    // No provider constraint means the symbol was never declared (even
    // conditionally); the primary "does not conform" diagnostic covers that.
    if (!constraint || isTriviallyTrueConstraint(constraint))
      return;
    // Update the details based on the new failure kind.
    if (kind.isFalse() && !result.isFalse()) {
      // If the result is not false, we're seeing false for the first time.
      // Clear the details since it currently contains unproven constraints,
      // which no longer matter.
      details->clear();
      seenConstraints.clear();
    } else if (result.isFalse() && kind.isUnknown()) {
      // If the result is already false and we're only seeing unknown, ignore.
      return;
    }
    if (!seenConstraints
             .insert({constraint.getLoc(), constraint.getProposition()})
             .second)
      return;
    details->push_back(constraint);
  };

  for (auto [i, required] : llvm::enumerate(trait.getSymbols())) {
    // Assume each requirement's own condition while checking it. Remember that
    // an empty constraints array means every requirement is unconditional.
    std::optional<TypedAttr> refined;
    auto assumption = [&] { return refined ? *refined : callerCanonical(); };
    if (!requiredConstraints.empty()) {
      TypedAttr requiredCond = requiredConstraints[i].getProposition();
      if (isPropositionImplied(requiredCond, callerCanonical()).isFalse())
        continue;
      refined =
          ParamOperatorAttr::get(POC::And, {callerCanonical(), requiredCond});
    }

    auto it = providedConditions.find(required);
    TriBool provided = it == providedConditions.end() ? TriBool::no()
                       : (!it->second || isTriviallyTrueProposition(it->second))
                           ? TriBool::yes()
                           : isPropositionImplied(it->second, assumption());

    if (provided.isTrue())
      continue; // Symbol is definitely provided.

    if (provided.isUnknown()) {
      // Symbol is conditionally provided but its constraint is unproven.
      recordFailure(required, TriBool::unknown());
      result &= TriBool::unknown();
      continue;
    }

    // Symbol is not provided (absent, or its condition is disproven). If the
    // type's concrete identity is unknown, i.e. its metatype is a trait bound
    // rather than a concrete struct metatype, then a missing trait is not
    // definitively absent: a `where conforms_to(...)` assumption can supply it.
    // Consult those assumptions here — prove the required symbol and treat it
    // as provided; otherwise stay at `unknown` rather than answering `no`.
    if (concreteType) {
      Type meta = ASTType(getCanonicalType(concreteType)).extractMetaType();
      if (sugarIsa<AnyTraitType, TraitType>(meta)) {
        auto singleTrait = TraitType::get(self->getContext(), {required});
        if (isPropositionImplied(
                TypeConformsToTraitAttr::get(PValue(concreteType).get(),
                                             singleTrait.getPValue()),
                assumption())
                .isTrue())
          continue;
        recordFailure(required, TriBool::unknown());
        result &= TriBool::unknown();
        // Not a pure short-circuit: a later refuted symbol can still fold this
        // to `no`. Bailing here trades that refinement away for the early exit,
        // which is safe only because `unknown` is the conservative answer.
        // TODO: fold both paths the same way and delete this exit.
        if (!collectAll)
          return TriBool::unknown();
        continue;
      }
    }
    recordFailure(required, TriBool::no());
    result &= TriBool::no();
    // `no` is a definitive answer, so early exiting here is safe.
    // Collecting details keeps going only to name every failing symbol.
    if (!collectAll)
      return TriBool::no();
  }

  // All required symbols are present (proven `yes`), or `result` already folded
  // in the `unknown`/`no` outcomes of any that were not.
  return result;
}

void LIT::canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<TraitSymbolAttr> &symbols) {
  canonicalizeTraitCompositionSymbols(
      symbols, [&](SymbolRefAttr symbol) -> TraitDeclOp {
        ASTDecl &memberDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
        return cast<TraitASTDecl>(memberDecl).getOp();
      });

  sortAndDeduplicateTraitSymbols(symbols);
}

SmallVector<ConstraintAttr> LIT::canonicalizeTraitSymbolsAndConstraints(
    SharedState &shared, SmallVectorImpl<TraitSymbolAttr> &symbols,
    const DenseMap<TraitSymbolAttr, ConstraintAttr> &constraintMap) {

  // Canonicalize the symbols first.
  canonicalizeTraitCompositionSymbols(shared, symbols);
  if (constraintMap.empty())
    return {};

  // fill in the missing constraints if we have an non-empty map.
  SmallVector<ConstraintAttr> constraints;
  constraints.reserve(symbols.size());
  for (TraitSymbolAttr symbol : symbols) {
    if (auto it = constraintMap.find(symbol); it != constraintMap.end()) {
      constraints.push_back(it->second);
    } else {
      // FIXME(MOCO-4250): `True` is wrong, should be inherited from parents.
      constraints.push_back(getUnconditionalConstraint(shared.getContext()));
    }
  }
  return constraints;
}

static std::pair<TraitType, StructDeclOp> extractTraitBound(SharedState &shared,
                                                            ASTType type) {
  if (auto anyTrait = dyn_cast<AnyTraitType>(type.extractMetaType()))
    return {anyTrait.getTraitType(), nullptr};
  if (isa<KGEN::NonStructTypeType>(type))
    return extractTraitBound(shared, shared.getBuiltinStubsMLIRType(SMLoc()));

  ASTDecl *decl = type.getDecl(shared);
  StructDeclOp structDecl = cast<StructASTDecl>(decl)->getOp();
  return {structDecl.getCanonicalTrait(), structDecl};
}

ConstraintAttr LIT::fuseConstraints(SharedState &shared,
                                    ArrayRef<ConstraintAttr> constraints) {
  assert(!constraints.empty() && "cannot fuse zero constraints");
  // A single constraint keeps its proposition, location, and user message.
  if (constraints.size() == 1)
    return constraints.front();

  // Conjoining constraints has no single correct `where` message, so the
  // message is dropped. Unobservable today: authored messages only live on
  // struct conformance lists, and conformance diagnostics read them off the
  // source struct, never a fused meta-type bound. Revisit if traits ever gain
  // `where` clauses with user messages (see the "Failure messages" section of
  // Mojo/proposals/where_clauses.md).
  SmallVector<TypedAttr> props;
  SmallVector<Location> locs;
  props.reserve(constraints.size());
  locs.reserve(constraints.size());
  for (ConstraintAttr c : constraints) {
    props.push_back(c.getProposition());
    locs.push_back(c.getLoc());
  }
  return ConstraintAttr::get(ParamOperatorAttr::get(POC::And, props),
                             FusedLoc::get(shared.getContext(), locs),
                             /*message=*/StringAttr());
}

Type LIT::mergeTwoMetaTypeBounds(SharedState &shared, ASTType typeA,
                                 ASTType typeB) {
  if (typeA.isEqualCanon(typeB))
    return typeA;

  auto [traitA, structA] = extractTraitBound(shared, typeA);
  auto [traitB, structB] = extractTraitBound(shared, typeB);
  // Constraints are scope-dependent, even when constraints expr appears the
  // same, they might be evaluated to different value depending on the scope.
  if (traitA == traitB && traitA.getConstraints().empty())
    return traitA;

  auto populateReplacer = [&](StructDeclOp structDecl,
                              StructMetaType structType) -> ParameterEvaluator {
    ParameterEvaluator replacer = shared.getParameterEvaluator();
    ArrayRef<ParamDeclAttr> paramDecls = structDecl.getInputParams();
    for (auto [decl, param] :
         llvm::zip_equal(paramDecls, structType.getParamValues())) {
      replacer.setDeclBinding(decl, param);
    }
    return replacer;
  };

  std::optional<ParameterEvaluator> replacerA, replacerB;
  if (auto metaA = sugarDynCast<StructMetaType>(typeA); metaA && structA)
    replacerA = populateReplacer(structA, metaA);
  if (auto metaB = sugarDynCast<StructMetaType>(typeB); metaB && structB)
    replacerB = populateReplacer(structB, metaB);

  llvm::SmallDenseSet<TraitSymbolAttr> symbolsA(traitA.getSymbols().begin(),
                                                traitA.getSymbols().end());
  llvm::SmallDenseSet<TraitSymbolAttr> symbolsB(traitB.getSymbols().begin(),
                                                traitB.getSymbols().end());
  llvm::set_intersect(symbolsA, symbolsB);

  DenseMap<TraitSymbolAttr, ConstraintAttr> constraints;
  if (!traitA.getConstraints().empty() || !traitB.getConstraints().empty()) {
    auto findConstraint = [](TraitType trait, TraitSymbolAttr symbol) {
      auto it = llvm::find(trait.getSymbols(), symbol);
      size_t idx = std::distance(trait.getSymbols().begin(), it);
      return trait.getConstraints()[idx];
    };

    for (TraitSymbolAttr commonTrait : symbolsA) {
      // The original constraints for the common trait.
      SmallVector<ConstraintAttr, 2> origCons;
      if (!traitA.getConstraints().empty()) {
        ConstraintAttr consA = findConstraint(traitA, commonTrait);
        if (replacerA)
          consA = cast<ConstraintAttr>(replacerA->getReboundAttribute(consA));
        origCons.push_back(consA);
      }
      if (!traitB.getConstraints().empty()) {
        ConstraintAttr consB = findConstraint(traitB, commonTrait);
        if (replacerB)
          consB = cast<ConstraintAttr>(replacerB->getReboundAttribute(consB));
        origCons.push_back(consB);
      }

      assert(!origCons.empty());
      constraints[commonTrait] = fuseConstraints(shared, origCons);
    }
  }

  SmallVector<TraitSymbolAttr> symbols(symbolsA.begin(), symbolsA.end());
  SmallVector<ConstraintAttr> mergedConstraints =
      canonicalizeTraitSymbolsAndConstraints(shared, symbols, constraints);

  return TraitType::get(shared.getContext(), symbols, mergedConstraints);
}

SmallVector<TraitSymbolAttr>
LIT::reduceTraitCompositionSymbols(SharedState &shared,
                                   ArrayRef<TraitSymbolAttr> symbols) {
  DenseSet<TraitSymbolAttr> impliedSymbols;
  for (TraitSymbolAttr symbol : symbols) {
    ASTDecl &traitDecl =
        shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
    TraitDeclOp traitOp = cast<TraitASTDecl>(traitDecl).getOp();
    for (TraitSymbolAttr ancestor : traitOp.getCanonicalTrait().getSymbols()) {
      if (ancestor != symbol)
        impliedSymbols.insert(ancestor);
    }
  }

  // Keep only the symbols that are not already implied by another symbol in the
  // composition. `{Movable, AnyType}` will be reduced to `{Movable}`
  SmallVector<TraitSymbolAttr> reduced;
  for (TraitSymbolAttr symbol : symbols) {
    if (!impliedSymbols.contains(symbol))
      reduced.push_back(symbol);
  }

  sortAndDeduplicateTraitSymbols(reduced);
  return reduced;
}

TraitSymbolAttr LIT::extractClosureSymbol(SharedState &shared,
                                          TraitType trait) {
  if (!trait)
    return nullptr;

  for (TraitSymbolAttr symbol : trait.getTraitSymbols())
    if (shared.isUniversalParametricClosureTrait(symbol))
      return symbol;
  return nullptr;
}

// TODO(MOCO-4627): lining up several closure signatures on one type needs a
// proper design; until then a composition may name at most one.
LogicalResult LIT::checkAtMostOneClosureTrait(SharedState &shared,
                                              ArrayRef<TraitSymbolAttr> symbols,
                                              SMLoc loc, SourceRange range) {
  SmallVector<TraitSymbolAttr> closures;
  for (TraitSymbolAttr symbol : symbols) {
    if (shared.isUniversalParametricClosureTrait(symbol))
      closures.push_back(symbol);
    if (closures.size() > 1)
      break;
  }
  if (closures.size() < 2)
    return success();

  // Print through the canonical trait: a bare single-symbol TraitType is not
  // canonical, and rendering one as a value trips the canonicalization
  // invariant in `DeclResolver::getTraitDecl`.
  DeclResolver &resolver = *shared.declResolver;
  auto diag = shared.emitError(loc)
              << "trait composition must not contain more than one closure "
                 "type; remove "
              << ASTType(resolver.getCanonicalTrait(closures[0])) << " or "
              << ASTType(resolver.getCanonicalTrait(closures[1]));
  if (range.isValid())
    diag << range;
  return failure();
}

TraitType
LIT::getTraitBoundFromAssumptions(TypedAttr typeAttr, SharedState &shared,
                                  ArrayRef<ConstraintAttr> assumptions) {
  return getTraitBoundFromAssumptions(
      typeAttr, assumptions, [&](SymbolRefAttr symbol) -> TraitDeclOp {
        ASTDecl &memberDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
        return cast<TraitASTDecl>(memberDecl).getOp();
      });
}

//===----------------------------------------------------------------------===//
// IREmitter::emitMetaTypeToTraitConversion
//===----------------------------------------------------------------------===//

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
createRequirementSignature(FnTypeGeneratorType signature, ASTType newSelfType,
                           DeclResolver &declResolver) {
  // Get the selfType as a TypedAttr since we'll be using it as a parameter
  // value below.
  TypedAttr newSelfValue = PValue(newSelfType).get();

  if (auto paramType = sugarDynCast<ParamType>(newSelfType.extractMetaType())) {
    auto simpleTraitType =
        sugarCast<AnyTraitType>(paramType.getParam().getType()).getTraitType();
    // Upcast from a parametric type of trait metatype value (e.g. "some
    // type that conforms to Movable) to the simple trait type (Movable)
    // so we can substitute the value into the signature.
    newSelfValue = UpcastAttr::get(simpleTraitType, PValue(newSelfType));
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

  // At this point, signature's `self` argument's type is the struct or
  // trait.  For example when binding Self down to some "MTT: Movable", we have:
  //    !lit.generator<<trait<@AnyType>>[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
  // Now we need to drop the "<trait<@AnyType>" parameter, which we do by
  // specializing it away.  We know all references to it are already gone.

  // NOTE: This is an UnknownAttr (which is an arbitrary attr that is never
  // used) not an UnboundAttr which remains an unbound parameter.
  // NOTE: we need to adjust the depth by -1 since we are pulling out the `Self`
  // from a parameter scope.
  ParameterEvaluator evaluator = declResolver.shared.getParameterEvaluator();
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);
  evaluator.appendIndexBinding(UnknownAttr::get(
      minusOneAdjuster.replace(signature.getInputParamTypes()[0])));

  // Use UnboundAttr for any other parameters so they remain in the result.
  for (Type type : signature.getInputParamTypes().drop_front())
    evaluator.appendIndexBinding(
        UnboundAttr::get(evaluator.getReboundType(getCanonicalType(type))));
  signature = signature.getSpecializedGenerator(
      evaluator.getIndexBindings(),
      &declResolver.shared.getEvaluationContext());

  return signature;
}

FnTypeGeneratorType
LIT::specializeSignature(FnTypeGeneratorType traitFnSignature,
                         ASTType newSelfType, DeclResolver &declResolver) {
  return createRequirementSignature(traitFnSignature, newSelfType,
                                    declResolver);
}

FailureOr<TypedAttr> LIT::getUniqueWitnessForTypeIfConforms(
    SharedState &shared, ASTType type, TraitType trait, StringRef entryName,
    ArrayRef<ConstraintAttr> callerAssumptions, SMLoc errorLoc) {
  // Get the decl for the type.
  ASTDecl *typeDecl = type.getDecl(shared);
  if (!typeDecl) {
    [[maybe_unused]] Type metaType = type.extractMetaType();
    assert(sugarIsa<NonStructTypeType>(metaType) ||
           sugarIsa<FnLiteralTypeGeneratorMetaType>(metaType));

    // This is a MLIR type, so we need to bind it to the builtin stub.
    // Use a special wrapper decl in the builtins as stubs.
    typeDecl = shared.getBuiltinStubsMLIRType(errorLoc).getDecl(shared);
    auto *structDecl = dyn_cast_if_present<StructASTDecl>(typeDecl);
    if (!structDecl) {
      shared.emitError(errorLoc, "malformed builtin._stubs.__MLIRType");
      return {};
    }

    auto typeValue =
        TypeParamAttr::get(type, NonStructTypeType::get(shared.getContext()));

    // Need to update the type itself to the wrapper type.
    type = structDecl->getOp().bindReference({typeValue});
  }

  if (type.doesConformTo(trait, shared, callerAssumptions).isFalse()) {
    // Does not conform. This is the only non-error case where we return an
    // empty attr.
    return TypedAttr();
  }

  // Make sure the trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(shared.declResolver->resolveBody(*traitDecl, errorLoc)))
    return failure();

  // Locate the entry in the trait.
  ArrayRef<ASTDecl *> entries = traitDecl->lookupInCurrentScope(entryName);
  if (entries.empty()) {
    shared.emitError(errorLoc, "trait ")
        << ASTType(trait) << " has no entry named " << entryName;
    return failure();
  }

  // If there are multiple entries, emit an error.
  if (entries.size() > 1) {
    shared.emitError(errorLoc, "trait ")
        << ASTType(trait) << " has multiple entries named " << entryName;
    return failure();
  }

  ASTDecl &entry = *entries.front();
  Type resultType;
  StringRef witnessName = entryName;
  // TODO(BillyZ): Fix trait alias replacement here once #60811 lands. Currently
  // this function does not properly replace alias mentions with the struct
  // type, but that does not impact any use case since this is only ever called
  // on AnyType and Movable right now.
  if (auto *aliasDecl = dyn_cast<AliasASTDecl>(&entry)) {
    resultType = aliasDecl->getOp().getType();
  } else if (auto *fnDecl = dyn_cast<DefASTDecl>(&entry); fnDecl) {
    // Ensure the function is signature resolved so we can access the mangled
    // name.
    if (failed(shared.declResolver->resolveSignature(entry, errorLoc)))
      return failure();
    resultType = createRequirementSignature(fnDecl->getOp().getFullSignature(),
                                            type, *shared.declResolver);
    // Use the mangled name from the trait declaration for function witnesses.
    witnessName = *fnDecl->getOp().getSymName();
  } else {
    llvm_unreachable("expected an alias or a function");
  }

  ASTDecl *parentTraitDecl = entry.getParentDecl();
  MLIRContext *ctx = parentTraitDecl->getContext();
  return shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
      PValue(type), TraitSymbolAttr::get(parentTraitDecl->getSymbolRef()),
      StringAttr::get(ctx, witnessName), resultType);
}

/// Emit a metatype conversion to a trait type by materializing the meta type
/// of the specified CValue into a witness table for the trait.  For example,
/// if 'value' has struct type, and the trait is Movable, then this forms a
/// TypeParamAttr PValue with a reference to the witness tables for this
/// struct's conformance to the trait.
///
/// If the input value has a derived trait type and the required type is a
/// base trait, then this simply upcasts the type value, e.g.:
///   def take_any_type[ATT: AnyType](x: ATT): pass
///   def pass_movable[MTT: Movable](x: MTT): take_any_type(x)
///
/// Yields something like:
///     #kgen.type<!kgen.param<:trait<@Movable> MTT>> : !lit.trait<@AnyType>
///
/// This maps from the Movable trait metatype into the AnyType trait metatype.
PValue IREmitter::emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                                TraitType trait) {
  // Only parameter-domain type-values are supported right now.
  PValue typePValue = value.ir.getIfPValue();
  if (!typePValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }

  // Get the StructMetaType or the TraitType of the value that we're checking
  // for conversion to the trait type.  This can also bind empty variadic
  // parameter lists and default parameters.
  ASTType type = emitType({typePValue, value.expr});
  if (!type)
    return {};

  if (sugarIsa<NonStructTypeType, FnLiteralTypeGeneratorMetaType>(
          type.extractMetaType())) {
    // Create the new type value with the trait metatype.
    return this->bindNonStructTypeToTrait({type, value.expr}, trait);
  }

  value.ir = PValue(type); // update value.ir if the type was rebound.

  // Check that the struct or super trait implements the trait.
  // Assumptions needed: e.g. `where AllWritable[*Ts]` proves
  // Tuple[*Ts]: Writable.
  auto conformance = type.doesConformToWithDetails(
      trait, shared, ASTDecl::getAssumptionsFromScope(&getDeclScope()));
  if (conformance.isNo()) {
    MojoInflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                            << type << " to trait " << ASTType(trait)
                            << value.expr->getRange();
    attachConstraintNotes(diag, conformance);
    return {};
  }

  // If conformance is unprovable (but not contradicted) and a deferral context
  // is installed, record the conformance obligation as deferred, and emit a
  // downcast into the target trait type.
  if (conformance.isUnknown() && deferredTypingContext) {
    // A parameter's trait bound is always unconditional.
    assert(!trait.hasConstraints() &&
           "deferred conformance bound should always bean unconditional trait");
    TypedAttr conformsTo =
        shared.getEvaluationContext().getAndFold<TypeConformsToTraitAttr>(
            typePValue, trait.getPValue());
    deferredTypingContext->deferredConstraints.push_back(
        {ConstraintAttr::get(
             conformsTo, shared.diags.translateLocation(value.expr->getLoc()),
             /*message=*/StringAttr()),
         value.expr->getLoc()});
    return DowncastAttr::get(trait, typePValue);
  }

  // Create the new type value with the trait metatype.
  return UpcastAttr::get(trait, typePValue);
}

std::optional<ParameterEvaluator>
LIT::populateTraitBindingEvaluator(TraitSymbolAttr traitSymbol,
                                   SharedState &shared) {
  TraitDeclOp traitDecl =
      cast<TraitASTDecl>(
          shared.declResolver->getDeclForTypeSymbol(traitSymbol.getSymbol()))
          .getOp();
  return populateTraitBindingEvaluator(traitSymbol, traitDecl);
}
