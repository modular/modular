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
#include "StabilityMarkers.h"
#include "StructEmitter.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

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
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<FnTypeGeneratorType, ParamBindings>
getTraitFunctionSignature(IREmitter &emitter, FnOp traitFn,
                          ASTType structSelfType, SymbolRefAttr traitSymbol,
                          const ExprNode *expr,
                          ParameterEvaluator &traitAliasReplacer) {
  TraitType trait = TraitType::get(traitSymbol);
  FnTypeGeneratorType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = signature.getInputParamTypes();

  // Add trait's _Self param replacement.
  params.push_back(TypeParamAttr::get(structSelfType, trait));
  auto bindings = ParamBindings::getForDeclaredType(emitter.getDeclScope(),
                                                    structSelfType, expr);

  // Only bind the `Self`, leave the rest unbound.
  bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);

  // Leave the rest alone.
  for (Type type : paramTypes.drop_front())
    params.push_back(UnboundAttr::get(type));

  FnTypeGeneratorType newSignature = signature.getSpecializedGenerator(
      params, &emitter.getDeclScope().getShared().getEvaluationContext());

  newSignature = traitAliasReplacer.replace(newSignature);
  return {newSignature, std::move(bindings)};
}

/// Used to determine if this particular child def op of a struct.decl op is one
/// corresponding to an inherited trait method. Checking just the inheritedFrom
/// attribute is insufficient as fnOp may actually be the fnOp in the parent
/// trait.
static bool isInheritedFnOp(FnOp fnOp) {
  return fnOp.getInheritedFrom().has_value() || fnOp.isDefaultedTraitFn();
}

/// Check method constraints against a conformance constraint for witness table
/// selection. Following overload selection rules, all candidates must be
/// definitively satisfied or violated - unprovable constraints trigger errors.
///
/// Returns:
///   - Satisfied: conformance implies all method constraints
///   - Violated: method constraints contradict the conformance
///   - Unprovable: constraints cannot be proven or disproven - error case
static ConstraintResult
checkMethodConstraintStatus(FnOp method, ConstraintAttr conformanceConstraint,
                            ASTDecl &structDecl) {
  ArrayRef<ConstraintAttr> methodConstraints =
      method.getFuncTypeGenerator().getBody().getMetadata().getConstraints();
  if (methodConstraints.empty())
    return ConstraintResult::Satisfied;

  TypedAttr confProp = getCanonicalAttr(conformanceConstraint.getProposition());

  bool hasUnprovable = false;
  for (ConstraintAttr methodConstraint : methodConstraints) {
    TypedAttr methodProp = getCanonicalAttr(methodConstraint.getProposition());

    // Skip trivially true method constraints.
    if (isTriviallyTrueProposition(methodProp))
      continue;

    // Trivially false method constraint is always violated.
    if (auto intProp = dyn_cast<IntegerAttr>(methodProp))
      if (intProp.getValue().isZero())
        return ConstraintResult::Violated;

    // Unconditional conformance can't prove non-trivial constraints, but
    // keep scanning for violations before returning Unprovable.
    if (isTriviallyTrueProposition(confProp)) {
      hasUnprovable = true;
      continue;
    }

    // Check contradiction first (more specific), then implication.
    if (constraintsContradict(confProp, methodProp))
      return ConstraintResult::Violated;
    if (constraintImplies(confProp, methodProp))
      continue;

    hasUnprovable = true;
  }

  return hasUnprovable ? ConstraintResult::Unprovable
                       : ConstraintResult::Satisfied;
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
// This function decides whether to disable the trait's default method stub or
// create a wrapper for it. If a struct method has a matching signature but
// incompatible constraints, we must NOT disable the stub, because:
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
// Without constraint checking: We'd see a signature match, disable the stub,
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
// is a valid override - we disable the stub.
static LogicalResult signatureResolveDefaultTraitFnStubs(
    ASTDecl &structDecl, ASTDecl &traitDecl, StringAttr name,
    ArrayRef<ASTDecl *> candidates, ParameterEvaluator &traitAliasReplacer,
    ConstraintAttr conformanceConstraint) {

  auto &shared = structDecl.getShared();

  SyntheticNode node(structDecl.getLoc());
  IREmitter emitter(structDecl, EC_Trait);
  auto structSelfType = structDecl.getTypeDeclSelf();

  /// Will attempt to create a wrapper fn op in structFnDecl that will call into
  /// the fn op in traitFnDecl (taking things like aliases that show up in
  /// signature into account). This will also report if another trait has
  /// already provided a default implementation and the struct itself does not
  /// provide an override.
  auto signatureResolveStub = [&](ASTDecl *traitFnDecl,
                                  ASTDecl *structFnDecl) -> LogicalResult {
    FnOp traitFn = cast<FnOp>(traitFnDecl->getIfOperation());

    SyntheticNode syntheticNode(structDecl.getLoc());
    auto [wrapperSignature, bindings] = getTraitFunctionSignature(
        emitter, traitFn, structSelfType, traitDecl.getSymbolRef(),
        syntheticNode, traitAliasReplacer);

    // If a function with matching signature is defined in the same trait we're
    // golden since existing machinery takes care of reporting there is a
    // conflict. However in the case of conflicts from two different traits the
    // earliest we can catch them is right here (and incidentally also the
    // latest as things get wonky with overload resolution if two decls with
    // matching signatures exist in a struct).
    auto possibleOverloads = structDecl.lookupInCurrentScope(name);

    SmallVector<ASTDecl *> structDefinesMethods;
    for (ASTDecl *decl : possibleOverloads) {
      auto impl = llvm::dyn_cast_if_present<FnOp>(decl->getIfOperation());
      if (!impl || isInheritedFnOp(impl))
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
    OverloadSet ov(name, structDefinesMethods, std::move(bindings),
                   CallSyntax::kMethodCallSynthetic);

    auto [_, decl] = ov.filterOverloadSetForValueType(
        wrapperSignature, emitter.getDeclScope(), nullptr);
    if (decl) {
      // Check if this method's constraints are valid for the conformance.
      // Following overload selection rules, we require constraints to be
      // definitively provable or disproved - unprovable constraints are errors.
      ConstraintResult status =
          checkMethodConstraintStatus(cast<FnOp>(decl->getIfOperation()),
                                      conformanceConstraint, structDecl);
      switch (status) {
      case ConstraintResult::Satisfied:
        // Since we are not using the default implementation, set the ASTDecl
        // which were inserted for referencing default method to be fully
        // resolved.
        assert(structFnDecl->resolvedness <= DeclResolvedness::signature &&
               "synthesizeMethodInStruct is only valid on non-body resolved Fn "
               "ASTDecls");
        // This was pointed to the trait default implementation, now that we
        // know this decl is useless, simply disable it. We mark the ASTDecl as
        // disabled instead of creating a fake FnOp and mark the FnOp to be
        // disabled.
        structFnDecl->markDisabled();
        return success();
      case ConstraintResult::Unprovable:
        // Cannot prove or disprove - error per overload selection rules.
        shared.emitError(decl->getLoc())
            << "method '" << name.str()
            << "' has constraints that cannot be proven or disproven from "
               "conformance constraint; all candidates must have provable "
               "or contradicted constraints";
        return failure();
      case ConstraintResult::Violated:
        // Method constraints contradict conformance - not a valid override.
        break;
      }
    }

    // The struct doesn't provide an override, see if the wrapper def we're
    // about to create has a matching signature to an existing wrapper function
    // in the struct.
    for (ASTDecl *decl : possibleOverloads) {
      // Skip any decls currently pointing to the parent trait method
      if (decl->isDisabled())
        continue;

      auto fnOp = cast<FnOp>(decl->getIfOperation());

      auto existingSignature = fnOp.getFullSignature();
      // now we need to compare the full signature to the trait signature
      if (isEqualCanon(existingSignature, wrapperSignature)) {
        // Produce an informative diagnostic citing the conflicting traits
        // **and the struct name**.
        StringAttr currentTraitName =
            traitDecl.getSymbolRef().getLeafReference();

        auto otherTraitFn = shared.getDeclResolver().getDeclForFuncSymbol(
            fnOp.getInheritedFromAttr());

        auto otherTraitName =
            otherTraitFn->getParentDecl()->getSymbolRef().getLeafReference();

        auto diag = shared.emitError(structDecl.getLoc())
                    << "trait method requirement " << traitFn.getDeclName()
                    << " has conflicting default implementations in "
                    << otherTraitName << " and " << currentTraitName
                    << " you must implement it manually";

        diag.attachNote(decl->getLoc())
            << "original default implementation from trait " << otherTraitName
            << " here";

        diag.attachNote(traitFn.getLoc())
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
    auto structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
    Block &fieldsBlock = structDeclOp.getFields().front();

    // Position builder before the first ConformanceOp (there will always be
    // one).
    ConformanceOp firstConformanceOp =
        *fieldsBlock.getOps<ConformanceOp>().begin();
    ImplicitLocOpBuilder builder(structDeclOp.getLoc(),
                                 firstConformanceOp.getOperation());

    PogListAttr traitPogs = traitFn.getFuncTypeGenerator().getParamListAttrs();
    for (auto traitPog : traitPogs.getPogs()) {
      ArrayRef<ASTDecl *> ref =
          structDecl.lookupInCurrentScope(traitPog.getName());

      // There could at most be one parameter with the same name in the struct.
      if (ref.empty() || ref.size() != 1)
        continue;

      if (isa_and_nonnull<ParamDeclRefAttr>(
              ref.front()->getIfIRValue().getIfPValue().get())) {
        auto diag =
            shared.emitError(structDecl.getLoc())
            << "name conflict between parameter " << traitPog.getName()
            << " in the default trait method and a parameter in the struct";

        diag.attachNote(traitFn.getLoc()) << "trait method declared here";

        return failure();
      }
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
    FnOp fnOp = dyn_cast_if_present<FnOp>(decl->getIfOperation());
    if (!fnOp)
      continue;

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
        shared.declResolver->getDeclForFuncSymbol(traitFnSymbolRef);
    auto parentTraitRef = traitFnDecl->getParentDecl()->getSymbolRef();

    // resolve corresponds to the trait we're currently working on in
    // verifyConformance.
    if (parentTraitRef != traitDecl.getSymbolRef())
      continue;

    // Grab the actual decl corresponding to the trait function we'll be
    // wrapping.
    assert(traitFnDecl && "Couldn't find trait fn decl");
    if (failed(signatureResolveStub(traitFnDecl, decl)))
      return failure();
  }
  return success();
}

LogicalResult
LIT::verifyAndBuildConformance(ASTDecl &structDecl, SymbolRefAttr parent,
                               std::optional<MojoInflightDiag> &diag,
                               ConformanceOp op) {
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
  auto structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());

  // Find extensions that target this struct and implement this trait.
  // This implements a form of the orphan rule: extensions can be defined either
  // in the same file as the struct or in the same file as the trait. This
  // ensures that extensions/conformances are local to either the struct or the
  // trait, preventing conflicting implementations across different files.
  // TODO(MOCO-522): Turn this into arcana docs!
  llvm::SmallPtrSet<ASTDecl *, 4> uniqueExtensions;

  // Search for extensions in the struct's file scope
  if (ASTDecl *structFileScope =
          structDecl.getNearestDeclOfType<FileModuleOp>()) {
    structFileScope->findExtensionsInScopeForStruct(structDecl.getSymbolRef(),
                                                    uniqueExtensions, parent);
  }

  // Search for extensions in the trait's file scope
  ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(parent);
  if (ASTDecl *traitFileScope =
          traitDecl.getNearestDeclOfType<FileModuleOp>()) {
    traitFileScope->findExtensionsInScopeForStruct(structDecl.getSymbolRef(),
                                                   uniqueExtensions, parent);
  }

  llvm::SmallVector<ASTDecl *> relevantExtensions(uniqueExtensions.begin(),
                                                  uniqueExtensions.end());

  bool hadErrors = false;
  IREmitter emitter(structDecl, EC_Trait);
  ASTType selfType = structDecl.getTypeDeclSelf();
  TraitDeclOp traitDeclOp = cast<TraitDeclOp>(*traitDecl.getIfOperation());

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
        constraintImplies(op.getConstraintAttr().getProposition(),
                          rpConstraint.getProposition());
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
  // Things get trickier when aliases are involved though, like:
  //
  //     trait MyTrait:
  //         alias X: AnyType
  //         def zork(self) -> X:
  //             ...
  //     struct MyStruct:
  //         alias X: AnyType = Int
  //         def zork(self) -> Int
  //
  // We can't just check if `def zork(self) -> X` exists in the struct, because
  // it doesn't.
  // Instead, we need to substitute all the struct alias values (like `Int`)
  // into the needle.
  // Substituting the struct's `Int` in for `X`, the
  // `def zork(self) -> X` needle becomes the correct
  // `def zork(self) -> Int` needle.
  // We then check if that exists in the struct and it does, excellent.
  //
  // These traitAliasReplacer help us do the needle substitutions later.
  //
  // TODO(MOCO-1993): Consolidate docs on this.
  auto structSelf = PValue(structDecl.getTypeDeclSelf());
  auto traitSelfDecl =
      cast<ParamDeclRefAttr>(PValue(traitDecl.getTypeDeclSelf()).get());
  ParameterEvaluator traitAliasReplacer = shared.getParameterEvaluator();
  traitAliasReplacer.setDeclBinding(traitSelfDecl.getName(), structSelf);

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

    // Collect all method declarations from struct and extensions
    llvm::SmallVector<ASTDecl *> decls =
        collectMethodsWithNameFromDecls(structDecl, name, relevantExtensions);

    if (decls.empty()) {
      diag->attachNote(traitFnDecl->getLoc())
          << "required function '" + name.str() + "' is not implemented";
      return failure(); // Stop the outer loop.
    }

    // Signature resolve any decls that don't correspond to inherited trait
    // methods first. This helps us avoid any infinite loops as signature
    // resolution for inherited methods also calls verifyConformance.
    for (ASTDecl *decl : decls) {
      if (Operation *op = decl->getIfOperation()) {
        if (auto fnOp = dyn_cast<FnOp>(op)) {
          if (isInheritedFnOp(fnOp))
            continue;
        }
      }
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
    if (failed(signatureResolveDefaultTraitFnStubs(structDecl, traitDecl, name,
                                                   decls, traitAliasReplacer,
                                                   op.getConstraintAttr()))) {
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
    auto [traitSignature, bindings] = getTraitFunctionSignature(
        emitter, traitFn, selfType, parent, syntheticNode, traitAliasReplacer);

    // Get the conformance constraint for checking method constraints.
    ConstraintAttr conformanceConstraint = op.getConstraintAttr();

    // Check each candidate's constraint status. Candidates with provable
    // constraints are valid, those with disproved constraints are rejected,
    // and those with unprovable constraints cause an error if any match the
    // trait signature (since we can't definitively select a witness).
    SmallVector<ASTDecl *> provableDecls;
    SmallVector<ASTDecl *> unprovableDecls;
    for (ASTDecl *decl : decls) {
      auto fnOp = dyn_cast_or_null<FnOp>(decl->getIfOperation());
      if (!fnOp) {
        provableDecls.push_back(decl);
        continue;
      }

      ConstraintResult status =
          checkMethodConstraintStatus(fnOp, conformanceConstraint, structDecl);
      switch (status) {
      case ConstraintResult::Satisfied:
        provableDecls.push_back(decl);
        break;
      case ConstraintResult::Violated:
        // Method constraints contradict conformance - not a valid candidate.
        break;
      case ConstraintResult::Unprovable:
        // Track unprovable candidates - if any match the trait signature,
        // we must error since we can't definitively select a witness.
        unprovableDecls.push_back(decl);
        break;
      }
    }

    // Check if there are unprovable candidates whose signature matches the
    // trait requirement. Following overload selection rules, if ANY candidate
    // has unprovable constraints and could match the signature, we must error
    // because we can't rule it out as a potential witness table entry.
    if (!unprovableDecls.empty()) {
      ParamBindings unprovableBindings = ParamBindings::getForDeclaredType(
          emitter.getDeclScope(), selfType, &syntheticNode);
      OverloadSet unprovableOv(name, unprovableDecls,
                               std::move(unprovableBindings),
                               CallSyntax::kMethodCallSynthetic);
      auto [unprovableResult, _] = unprovableOv.filterOverloadSetForValueType(
          traitSignature, emitter.getDeclScope(), nullptr);
      if (unprovableResult) {
        // An unprovable candidate matches the signature - error.
        // This follows overload selection rules: we can't prove or disprove
        // this candidate, so we can't definitively select a witness.
        for (ASTDecl *unprovableDecl : unprovableDecls) {
          diag->attachNote(unprovableDecl->getLoc())
              << "method '" << name.str()
              << "' has constraints that cannot be proven or disproven from "
                 "conformance constraint";
        }
        diag->attachNote(traitFnDecl->getLoc())
            << "required by trait method here";
        return failure();
      }
    }

    // Now try to find a match among the provable candidates.
    OverloadSet ov(name, provableDecls, std::move(bindings),
                   CallSyntax::kMethodCallSynthetic);
    auto emitError = [&](SMLoc loc) -> MojoInflightDiag & {
      return diag->attachNote(traitFnDecl->getLoc());
    };
    auto [result, selectedStructMethod] = ov.filterOverloadSetForValueType(
        traitSignature, emitter.getDeclScope(), emitError);

    if (!result)
      return failure();

    // Check for API author error: stable struct implementing stable trait
    // must use stable methods for stable trait methods.
    if (selectedStructMethod) {
      checkStableTraitMemberImplementation(
          structDecl, traitDecl, *selectedStructMethod, *traitFnDecl, shared);
    }

    WitnessOp::create(b, traitFn.getSymNameAttr(), result.get());
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

    Type traitAliasType =
        traitAliasReplacer.getReboundType(traitAlias.getType());

    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    // If there is no user defined alias nor defaulted alias, raise an error.
    // When there are multiple decls, it can not be an AliasDeclOp either,
    // otherwise we should have already reported "redefinition" error.
    if (decls.empty() ||
        !isa_and_nonnull<AliasDeclOp>(decls.front()->getIfOperation())) {
      diag->attachNote(traitAlias->getLoc())
          << "required member '" << name.str() << "' is not specified";
      return failure(); // Stop the outer loop.
    }
    ASTDecl *structAliasDecl = decls.front();
    auto structAliasDeclOp = cast<AliasDeclOp>(decls.front()->getIfOperation());

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

      // We don't yet put initializerExpr into the traitAliasReplacer because
      // they need to be converted first to the trait's alias's type (see
      // SAVMBCTATBS).
      SyntheticNode synthNode(structAliasDecl->getLoc());
      if (!IREmitter::canImplicitlyConvertToType({initializerExpr, synthNode},
                                                 traitAliasType,
                                                 emitter.getDeclScope())) {
        diag->attachNote(traitAliasDecl->getLoc())
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
      // Unfortunately, when we do our substitution into our
      // `def zork(self) -> Container[X]` needle, it becomes a
      // `def zork(self) -> Container[Int as Copyable] needle, which is both
      // wrong and also doesn't exist in the struct, because the struct
      // contains: `def zork(self) -> Container[Int as AnyType]. I say it's
      // wrong because an "Int as Copyable" parameter-value cannot be given to a
      // parameter-decl that expects an AnyType. The vtables don't line up.
      //
      // So, to fix that, when we substitute into the needle, we first convert
      // the struct alias's value (Int as Copyable) to the trait alias's type
      // (AnyType).
      // Here, we do it ahead of time, just before putting it into the replacer.
      //
      // TODO(MOCO-1993): Make sure this is consistently followed other places
      // we do trait substitution, and maybe centralize this arcana to
      // somewhere.
      ValueDest dest(traitAliasType, EC_AliasValue);
      CValue convertedValue = emitter.emitImplicitConversionToType(
          {initializerExpr, synthNode}, traitAliasType, dest);

      aliasValue = convertedValue.getIfPValue();
    } else {
      // Handle cases where this is a default value.

      // Found the decl in the trait that provided the default value, body
      // resolve the aliasDeclOp to get the concrete value.
      ArrayRef<ASTDecl *> decls = traitDecl.lookupInCurrentScope(name);
      assert(decls.size() == 1 &&
             isa_and_nonnull<AliasDeclOp>(decls.front()->getIfOperation()) &&
             "expected one AliasDeclOp in the trait");
      // NOTE: Strictly speaking, we only need the alias type here to verify
      // conformance, but AliasDeclOp resolves it body (i.e., value) as well in
      // at signature resolution phase. The extra work is a source of parsing
      // cycle.
      if (failed(shared.declResolver->resolveSignature(*decls.front(),
                                                       traitDecl.getLoc()))) {
        hadErrors = true;
        return success();
      }

      auto defaultAliasOp = cast<AliasDeclOp>(decls.front()->getIfOperation());
      // Position builder before the first ConformanceOp (there will always be
      // one since the struct should at least conform the trait that we are
      // cloning the defaulted value from).

      ConformanceOp firstConformanceOp =
          *structDeclOp.getFields().front().getOps<ConformanceOp>().begin();
      ImplicitLocOpBuilder builder(structDeclOp.getLoc(),
                                   firstConformanceOp.getOperation());
      auto clonedDefault = cast<AliasDeclOp>(builder.clone(*defaultAliasOp));
      Attribute newValAttr =
          traitAliasReplacer.replace(clonedDefault->getAttrDictionary());
      clonedDefault->setAttrs(cast<DictionaryAttr>(newValAttr));
      structAliasDecl->setIRValue(clonedDefault);
      structAliasDecl->resolvedness = DeclResolvedness::body;

      aliasValue = PValue(clonedDefault.getValueAttr());
      // Since we cloned the defaulted op from the trait, there is no need for
      // us to convert the type as they are guaranteed to be matched.
    }

    // Check for API author error: stable struct implementing stable trait
    // must use stable aliases for stable trait aliases.
    checkStableTraitMemberImplementation(
        structDecl, traitDecl, *structAliasDecl, *traitAliasDecl, shared);

    WitnessOp::create(b, name, aliasValue.get());
    traitAliasReplacer.setDeclBinding(traitAlias.getParamDecl(), aliasValue);

    return success();
  };

  // TODO(MOCO-1143): this loop needs a ParameterEvaluator that is
  // populated with the mappings of trait alias requirements to their matched
  // values on the implementing struct, then you call getReboundType/Attribute
  // when checking both the function and future alias requirements
  // ```
  // trait Foo:
  //     alias N: Int
  //     # lit.fn @foo(%self: !kgen.param<Self>,
  //     #               %x: SIMD[float32, #kgen.param.decl.ref<"N">])
  //     def foo(self, x: SIMD[DType.float32, N]):
  //         ...
  // struct Impl(Foo):
  //     alias N: Int = 4
  //     # lit.fn @foo(%self: !kgen.param<Self>, %x: SIMD[float32,  4])
  //     def foo(self, x: SIMD[DType.float32, 4]):
  //         pass
  // ```
  bool allMatchFound = true;
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      // Skip any children that aren't methods or aliases.
      if (auto traitFn = dyn_cast_or_null<FnOp>(decl->getIfOperation())) {
        if (failed(checkMethod(name, decl, traitFn))) {
          allMatchFound = false;
          break;
        }
      }
      if (AliasDeclOp traitAlias =
              dyn_cast_or_null<LIT::AliasDeclOp>(decl->getIfOperation())) {
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
          << "inherited through '" << *parentDecl.getUserNameIfOperation()
          << "' here";
    }
  }
  return failure();
}

static TraitType getDeclProvidedTrait(ASTDecl *decl) {
  // Collect all the symbols that the type explicitly provides.
  TraitType providedCanonTrait;

  auto declOp = decl->getIfOperation();
  if (auto structOp = dyn_cast_or_null<StructDeclOp>(declOp)) {
    providedCanonTrait = structOp.getCanonicalTrait();
  } else if (auto traitOp = dyn_cast_or_null<TraitDeclOp>(declOp)) {
    providedCanonTrait = traitOp.getCanonicalTrait();
  } else if (TraitType canonTraitType =
                 dyn_cast_or_null<TraitType>(decl->getIfTypeValue())) {
    providedCanonTrait = canonTraitType;
  } else if (isa<PackageOp>(declOp) || isa<FileModuleOp>(declOp)) {
    providedCanonTrait = TraitType::get(decl->getContext(), {});
  } else {
    llvm_unreachable("Invalid decl kind");
  }

  return providedCanonTrait;
}

/// Given a decl for a struct or trait type, check if this type conforms to the
/// specified trait type. If concreteType is provided, it is used to extract
/// parameter bindings for evaluating conditional trait conformances.
///
/// Returns:
/// - ConformanceResult::Yes if the type definitely conforms
/// - ConformanceResult::No if the type definitely does not conform
/// - ConformanceResult::NeedsEvidence if conformance depends on constraints
///   that cannot be evaluated statically
ConformanceResult
ASTDecl::doesNominalTypeConformTo(TraitType trait, ASTType concreteType,
                                  ArrayRef<ConstraintAttr> callerAssumptions) {
  // We only need trait symbol to verify trait conformance, not the resolved
  // witness table.
  if (failed(shared.declResolver->resolveSignature(*this, getLoc())))
    return ConformanceResult::No; // Error emitted.

  // Collect all the symbols that the type explicitly provides.
  TraitType providedCanonTrait = getDeclProvidedTrait(this);
  if (providedCanonTrait == trait)
    return ConformanceResult::Yes;

  // Set up parameter evaluator if we have parameter bindings for constraint
  // evaluation. Uses the parser's evaluation context needed for folding
  // TypeConformsToTraitAttr.
  std::optional<ParameterEvaluator> evaluator;
  ArrayRef<TypedAttr> paramBindings =
      concreteType ? concreteType.getParamBindings() : ArrayRef<TypedAttr>{};
  if (!paramBindings.empty()) {
    if (auto structOp = dyn_cast_or_null<StructDeclOp>(this->getIfOperation()))
      evaluator.emplace(shared.getParameterEvaluator(structOp.getInputParams(),
                                                     paramBindings));
  }

  ArrayRef<SymbolRefAttr> providedSymbolsArr = providedCanonTrait.getSymbols();
  ArrayRef<ConstraintAttr> constraints = providedCanonTrait.getConstraints();

  // Track symbols that are definitely provided (proven) vs conditionally
  // provided (unprovable constraint).
  DenseSet<SymbolRefAttr> provenSymbols;
  DenseSet<SymbolRefAttr> unprovenSymbols;

  // Collect all provided symbols, starting with the struct's own conformances.
  // For conditional conformances, track whether constraints are proven or not.
  if (constraints.empty()) {
    // No constraints array means all traits are unconditionally provided.
    provenSymbols.insert(providedSymbolsArr.begin(), providedSymbolsArr.end());
  } else {
    for (auto [i, symbol] : llvm::enumerate(providedSymbolsArr)) {
      ConstraintAttr constraint = constraints[i];

      // If constraint is trivially true, the trait is unconditionally provided.
      if (isTriviallyTrueConstraint(constraint)) {
        provenSymbols.insert(symbol);
        continue;
      }

      // If we have no evaluator (no param bindings), we can't evaluate
      // constraints. Mark as unproven.
      if (!evaluator) {
        unprovenSymbols.insert(symbol);
        continue;
      }

      switch (evaluateConstraint(*evaluator, constraint, callerAssumptions)) {
      case ConformanceResult::Yes:
        provenSymbols.insert(symbol);
        break;
      case ConformanceResult::NeedsEvidence:
        unprovenSymbols.insert(symbol);
        break;
      case ConformanceResult::No:
        break;
      }
    }
  }

  if (auto structOp = dyn_cast_or_null<StructDeclOp>(this->getIfOperation())) {
    llvm::SmallPtrSet<ASTDecl *, 4> uniqueExtensions;
    // Search for extensions in the struct's parent scope.
    // TODO(MOCO-522): Arcana docs on our orphan rule.
    if (ASTDecl *structParent = this->getParentDecl()) {
      structParent->findExtensionsInScopeForStruct(this->getSymbolRef(),
                                                   uniqueExtensions);
    }
    // Search for extensions in the trait's parent scope(s).
    // TODO(MOCO-522): Arcana docs on our orphan rule.
    for (SymbolRefAttr traitSymbol : trait.getSymbols()) {
      ASTDecl *traitDecl =
          shared.declResolver->getTraitDecl(TraitType::get(traitSymbol));
      assert(traitDecl && "couldn't find trait decl for trait symbol");
      if (ASTDecl *traitParent = traitDecl->getParentDecl()) {
        traitParent->findExtensionsInScopeForStruct(this->getSymbolRef(),
                                                    uniqueExtensions);
      }
    }
    llvm::SmallVector<ASTDecl *> allExtensions(uniqueExtensions.begin(),
                                               uniqueExtensions.end());
    for (ASTDecl *extDecl : allExtensions) {
      if (auto extOp =
              dyn_cast_or_null<ExtensionDeclOp>(extDecl->getIfOperation())) {
        // Signature resolve the extension so we can access its canonicalTrait.
        if (failed(shared.declResolver->resolveSignature(*extDecl,
                                                         extDecl->getLoc())))
          continue;
        // Extensions sometimes don't have canonical traits (like in isolated
        // tests).
        if (!extOp.getCanonicalTrait().has_value())
          continue;
        TraitType extCanonicalTrait = extOp.getCanonicalTrait().value();
        for (SymbolRefAttr symbol : extCanonicalTrait.getSymbols()) {
          // Extension conformances are currently unconditional.
          provenSymbols.insert(symbol);
        }
      }
    }
  }

  // Check the provided symbols against the required symbols by the target
  // trait. Track whether any required symbol relies on unproven constraints.
  bool hasUnprovenRequired = false;
  for (SymbolRefAttr required : trait.getSymbols()) {
    if (provenSymbols.contains(required)) {
      // Symbol is definitely provided.
      continue;
    }
    if (unprovenSymbols.contains(required)) {
      // Symbol is conditionally provided but constraint is unproven.
      hasUnprovenRequired = true;
      continue;
    }
    // Symbol is not provided at all.
    return ConformanceResult::No;
  }

  // All required symbols are present (either proven or unproven).
  if (hasUnprovenRequired)
    return ConformanceResult::NeedsEvidence;
  return ConformanceResult::Yes;
}

void LIT::canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols) {
  canonicalizeTraitCompositionSymbols(
      symbols, [&](SymbolRefAttr symbol) -> TraitDeclOp {
        ASTDecl &memberDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
        return cast<TraitDeclOp>(memberDecl.getIfOperation());
      });

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
                           ParameterEvaluator *traitAliasReplacer,
                           DeclResolver &declResolver) {
  // Get the selfType as a TypedAttr since we'll be using it as a parameter
  // value below.
  TypedAttr newSelfValue = PValue(newSelfType).get();

  // Start with the full signature for the trait requirement.
  FnTypeGeneratorType signature = traitFn.getFullSignature();

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

  // Next we'll replace trait aliases that appear in the trait methods, such
  // as:
  //
  //     trait MyTrait:
  //         alias T: ATrait
  //         def bork(self) -> Something[T]: ...
  //         def zork(self) -> Something[Self.T]: ...
  //
  // We'll replace them with the struct's trait value, like the int in:
  //
  //     struct MyStruct(MyTrait):
  //         alias T: ATrait = int
  //         def bork(self) -> SIMD[int]: ...

  // bork's `T` is a regular paramRef, we use traitAliasReplacer to replace it.
  if (traitAliasReplacer)
    signature = traitAliasReplacer->replace(signature);

  // At this point, signature's `self` argument's type is the struct or
  // trait.  For example when binding Self down to some "MTT: Movable", we have:
  //    !lit.generator<<trait<@AnyType>>[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
  // Now we need to drop the "<trait<@AnyType>" parameter, which we do by
  // specializing it away.  We know all references to it are already gone.

  // NOTE: This is an UnknownAttr (which is an arbitrary attr that is never
  // used) not an UnboundAttr which remains an unbound parameter.
  ParameterEvaluator evaluator = declResolver.shared.getParameterEvaluator();
  evaluator.appendIndexBinding(
      UnknownAttr::get(signature.getInputParamTypes()[0]));
  // Use UnboundAttr for any other parameters so they remain in the result.
  for (Type type : signature.getInputParamTypes().drop_front())
    evaluator.appendIndexBinding(
        UnboundAttr::get(evaluator.getReboundType(type)));
  signature = signature.getSpecializedGenerator(
      evaluator.getIndexBindings(),
      &declResolver.shared.getEvaluationContext());

  return signature;
}

FnTypeGeneratorType LIT::specializeSignature(FnOp traitFn, ASTType newSelfType,
                                             DeclResolver &declResolver) {
  return createRequirementSignature(traitFn, newSelfType, nullptr,
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
    if (!typeDecl ||
        !isa_and_nonnull<StructDeclOp>(typeDecl->getIfOperation())) {
      shared.emitError(errorLoc, "malformed builtin._stubs.__MLIRType");
      return {};
    }

    auto typeValue =
        TypeParamAttr::get(type, NonStructTypeType::get(shared.getContext()));

    // Need to update the type itself to the wrapper type.
    type = cast<StructDeclOp>(typeDecl->getIfOperation())
               .bindReference({typeValue});
  }

  if (type.checkConformance(trait, shared, callerAssumptions) ==
      ConformanceResult::No) {
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
  if (auto aliasDecl = dyn_cast_or_null<AliasDeclOp>(entry.getIfOperation())) {
    resultType = aliasDecl.getType();
  } else if (auto fnDecl = dyn_cast_or_null<FnOp>(entry.getIfOperation())) {
    // Ensure the function is signature resolved so we can access the mangled
    // name.
    if (failed(shared.declResolver->resolveSignature(entry, errorLoc)))
      return failure();
    resultType =
        createRequirementSignature(fnDecl, type, nullptr, *shared.declResolver);
    // Use the mangled name from the trait declaration for function witnesses.
    witnessName = *fnDecl.getSymName();
  } else {
    llvm_unreachable("expected an alias or a function");
  }

  ASTDecl *parentTraitDecl = entry.getParentDecl();
  MLIRContext *ctx = parentTraitDecl->getContext();
  return shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
      PValue(type),
      StringAttr::get(ctx,
                      getFlattenedSymbolName(parentTraitDecl->getSymbolRef())),
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
  if (type.checkConformance(
          trait, shared, ASTDecl::getAssumptionsFromScope(&getDeclScope())) ==
      ConformanceResult::No) {
    MojoInflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                            << type << " to trait " << ASTType(trait)
                            << value.expr->getRange();
    return {};
  }

  // Create the new type value with the trait metatype.
  return UpcastAttr::get(trait, typePValue);
}
