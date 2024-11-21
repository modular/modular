//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Traits.h"
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "StructEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/STLExtras.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<LITSignatureType, ParamBindings>
getTraitFunctionSignature(ExprEmitter &emitter, LIT::FuncOp traitFn,
                          ASTType structSelfType, TraitType trait,
                          const ExprNode *expr) {
  LITSignatureType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = signature.getParamTypes();

  // Add trait's _Self param replacement.
  params.push_back(TypeConstantAttr::get(structSelfType, trait));
  ParserParamEvaluator evaluator(emitter.getDeclResolver(), params);
  auto bindings = ParamBindings::getForDeclaredType(emitter.getDeclScope(),
                                                    structSelfType, expr);
  // Leave the rest alone.
  for (Type type : paramTypes.drop_front()) {
    params.push_back(UnboundAttr::get(type));
    evaluator.addInputValue(params.back());
    bindings.addPrechecked(expr, params.back());
  }

  return {signature.getSpecializedSignature(params), std::move(bindings)};
}

/// Allow synthesizing default implementations of certain special functions.
static void synthesizeSpecialFunction(ASTDecl &structDecl,
                                      SpecialFunctionKind kind) {
  auto &shared = structDecl.getShared();
  StructEmitter gen(structDecl.getShared());
  auto selfRefType =
      structDecl.getTypeDeclSelf().getRefForArgument("self", /*isMut=*/true);
  MLIRContext *ctx = structDecl.getContext();
  auto empty = StringAttr::get(ctx);

  // Synthesize the required special method. Importantly, don't mark the struct
  // as actually having this method so that destructors et al. are not
  // needlessly emitted.
  LIT::FuncOp func;
  if (kind == SpecialFunctionKind::kDel) {
    // Synthesize an empty destructor. Don't do anything special, because we
    // want check origins to insert a call to the real destructor here, if it
    // has one.
    auto [dtor, _] = gen.synthesizeMethodInStruct(
        "__del__", selfRefType, ArgConvention::OwnedInMem,
        PogListAttr::get(ctx, {empty}, {PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, structDecl.getLoc(), kind,
        FnEffects(), "_thunk");
    if (!dtor)
      return;
    func = dtor;
  } else {
    // Determine the name and argument conventions of the function.
    ArgConvention existingConv;
    switch (kind) {
    case SpecialFunctionKind::kCopyInit:
      existingConv = ArgConvention::BorrowedInMem;
      break;
    case SpecialFunctionKind::kMoveInit:
      existingConv = ArgConvention::OwnedInMem;
      break;
    default:
      llvm_unreachable("unexpected special function kind to synthesize");
    }
    StringRef name = SpecialFunctionInfo::get(kind).name;
    Type existingType;
    bool isMut = existingConv == ArgConvention::OwnedInMem;
    existingType =
        structDecl.getTypeDeclSelf().getRefForArgument("existing", isMut);
    auto [ctor, _] = gen.synthesizeMethodInStruct(
        name, {selfRefType, existingType},
        {ArgConvention::InitSelf, existingConv},
        PogListAttr::get(ctx, {empty, empty},
                         {PassingKind::PosOnly, PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, structDecl.getLoc(), kind,
        FnEffects(), "_thunk");
    if (!ctor)
      return;
    func = ctor;
    // In every case, the implementation is a load+store.
    auto b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());
    Value value;
    if (kind == SpecialFunctionKind::kMoveInit)
      value = b.create<LIT::LoadConsumeOp>(func.getArgument(1));
    else
      value = b.create<RefLoadOp>(func.getArgument(1));
    b.create<RefStoreOp>(value, func.getArgument(0));
  }
  func.setInlineLevel(InlineLevel::AlwaysNoDebug);
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), func.getBody());
  b.create<KGEN::ReturnOp>(
      Value(b.create<ParamConstantOp>(b.getAttr<NoneAttr>())));
}

LogicalResult LIT::verifyConformance(ASTDecl &structDecl,
                                     TypeLineageAttr parent,
                                     std::optional<InflightDiag> &diag) {
  auto trait = dyn_cast<TraitType>(parent.getType());
  if (!trait)
    return success();

  auto &shared = structDecl.getShared();
  auto structDeclOp = cast<StructDeclOp>(structDecl);
  bool rpTrivial = structDeclOp.isRegisterPassableTrivial();
  bool regPassable = structDeclOp.isRegisterPassable();
  bool hadErrors = false;
  SyntheticNode node(structDecl.getLoc());
  ExprEmitter emitter(structDecl, EC_Trait);
  ASTType selfType = structDecl.getTypeDeclSelf();

  // These are the special methods that need to be synthesized.
  SmallVector<SpecialFunctionKind> specialFns;

  ASTDecl &traitDecl =
      emitter.getDeclResolver().getDeclForTypeSymbol(trait.getSymbol());

  // Make sure to fully resolve the trait first.
  if (failed(shared.declResolver->resolveFully(traitDecl, structDecl.getLoc())))
    return failure();

  ParserParamEvaluator traitAliasReplacer(*shared.declResolver);

  bool allMatchFound = true;
  // Prepare an error. It will be abandoned if the check succeeds.
  StringRef traitName = cast<TraitDeclOp>(traitDecl).getSymName();
  diag = shared.emitError(structDecl.getLoc(), "struct ")
         << selfType << " does not implement all requirements for '"
         << traitName << "'";

  // Returns failure() to stop the verifyConformance loop.
  auto checkMethod = [&](const mlir::StringAttr &name, ASTDecl *traitFnDecl,
                         LIT::FuncOp traitFn) -> LogicalResult {
    if (traitFn.getIsInherited()) {
      // Skip inherited methods, they're checked at a different time.
      return success();
    }
    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    if (decls.empty() || !isa<LIT::FuncOp>(decls.front())) {
      if (canSynthesizeIfMissing(name, rpTrivial, regPassable)) {
        specialFns.push_back(SpecialFunctionInfo::getKind(name));
        return success();
      }
      diag->attachNote(traitFnDecl->getLoc())
          << "required function '" + name.str() + "' is not implemented";
      allMatchFound = false;
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
    auto [newSignature, bindings] = getTraitFunctionSignature(
        emitter, traitFn, selfType, trait, syntheticNode);
    // Match against the transformed calling convention if the struct is
    // register-passable.
    LITSignatureType traitSignature = traitAliasReplacer.replace(newSignature);

    // Omit errors for certain special functions where the parser will
    // specifically verify their signatures if present.
    bool emitError = !llvm::is_contained({SpecialFunctionKind::kMoveInit,
                                          SpecialFunctionKind::kCopyInit,
                                          SpecialFunctionKind::kDel},
                                         SpecialFunctionInfo::getKind(name));

    OverloadSet ov(name, decls, std::move(bindings), node,
                   CallSyntax::kMethodCallSynthetic);
    PValue result = ov.filterOverloadSetForValueType(
        traitSignature, emitter.getDeclScope(),
        emitError ? function_ref<InflightDiag &(SMLoc)>(
                        [&](SMLoc loc) -> InflightDiag & {
                          return diag->attachNote(traitFnDecl->getLoc());
                        })
                  : nullptr);
    if (!result && emitError)
      allMatchFound = false;

    return success();
  };

  auto checkAlias = [&](const mlir::StringAttr &name, ASTDecl *traitAliasDecl,
                        AliasDeclOp traitAlias) -> LogicalResult {
    // TODO(MOCO-1140): check traitAlias.getIsInherited(); implement inheritance
    // of alias decls.
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
      allMatchFound = false;
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

    traitAliasReplacer.setParameterValue(traitAlias.getParamDecl(),
                                         initializerExpr);

    SyntheticNode synthNode(structAliasDecl->getLoc());
    if (!ExprEmitter::canImplicitlyConvertToType({initializerExpr, synthNode},
                                                 traitAliasType,
                                                 emitter.getDeclScope())) {
      diag->attachNote(traitAliasDecl->getLoc())
          << "alias '" + name.str() + "' type " << structAliasType
          << " doesn't conform to trait's alias '" << name.str() << "' type "
          << traitAliasType;
      allMatchFound = false;
      return success();
    }

    return success();
  };

  // TODO(MOCO-1143): this loop needs a ParameterEvaluator that is populated
  // with the mappings of trait alias requirements to their matched values on
  // the implementing struct, then you call getReboundType/Attribute when
  // checking both the function and future alias requirements
  // ```
  // trait Foo:
  //     alias N: Int
  //     # lit.func @foo(%self: !kgen.paramref<Self>,
  //     #               %x: SIMD[float32, #kgen.param.decl.ref<"N">])
  //     fn foo(self, x: SIMD[DType.float32, N]):
  //         ...
  // struct Impl(Foo):
  //     alias N: Int = 4
  //     # lit.func @foo(%self: !kgen.paramref<Self>, %x: SIMD[float32,  4])
  //     fn foo(self, x: SIMD[DType.float32, 4]):
  //         pass
  // ```
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      // Skip any children that aren't methods or aliases.
      if (auto traitFn = dyn_cast<LIT::FuncOp>(*decl)) {
        if (failed(checkMethod(name, decl, traitFn)))
          break;
      }
      if (AliasDeclOp traitAlias = dyn_cast<LIT::AliasDeclOp>(*decl)) {
        if (failed(checkAlias(name, decl, traitAlias)))
          break;
      }
    }
  }

  if (allMatchFound) {
    diag->abandon();
    diag.reset();
  } else {
    diag->attachNote(traitDecl.getLoc())
        << "trait '" << traitName << "' declared here";
    if (!parent.getInheritedFrom().empty()) {
      ASTDecl &parentDecl = emitter.getDeclResolver().getDeclForTypeSymbol(
          cast<TraitType>(parent.getInheritedFrom().front()).getSymbol());
      diag->attachNote(parentDecl.getLoc())
          << "inherited through '" << *parentDecl.getNameIfOperation()
          << "' here";
    }
    hadErrors = true;
  }

  if (hadErrors)
    return failure();
  for (SpecialFunctionKind kind : specialFns)
    synthesizeSpecialFunction(structDecl, kind);
  return success();
}

/// Given a decl for a struct or trait type, return true if this type conforms
/// to the specified trait type.  On failure, this may set 'diag' to an inflight
/// diagnostic that explains why this doesn't conform.  It can be reported or
/// abandoned based on the client's needs.
bool ASTDecl::doesNominalTypeConformsTo(TraitType trait,
                                        std::optional<InflightDiag> &diag) {
  assert((::isa<StructDeclOp, TraitDeclOp>(*this)) && "Invalid decl kind");

  if (failed(shared.declResolver->resolveFully(*this, getLoc())))
    return false; // Error emitted.

  ArrayRef<TypeLineageAttr> parentTypes;
  auto structOp = dyn_cast<StructDeclOp>(*this);
  if (structOp)
    parentTypes = structOp.getParentTypes();
  else
    parentTypes = cast<TraitDeclOp>(*this).getParentTypes();

  // Check if the type explicitly conforms to the trait.
  if (contains_if(parentTypes, [trait](TypeLineageAttr type) {
        return type.getType() == trait;
      }))
    return true;

  // Check to see if this is already literally this trait.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (!traitDecl)
    return false; // Erroneous.

  // Self conformance.
  if (traitDecl == this)
    return true;

  // Only structs can implicitly conform to traits.
  if (!structOp)
    return false;

  // Check if the type *implicitly* conforms to the trait.
  SmallVector<TypeLineageAttr> newParentTypes =
      llvm::to_vector(structOp.getParentTypes());
  unsigned curNumParents = newParentTypes.size();
  StructEmitter::appendTraits(newParentTypes, traitDecl);
  for (TypeLineageAttr newParent :
       llvm::drop_begin(newParentTypes, curNumParents))
    if (failed(verifyConformance(*this, newParent, diag)))
      return false;

  // If we succeeded, remember this so we don't check again.
  structOp.setParentTypes(newParentTypes);
  return true;
}
