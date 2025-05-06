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
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<FnTypeGeneratorType, ParamBindings>
getTraitFunctionSignature(ExprEmitter &emitter, FnOp traitFn,
                          ASTType structSelfType, SymbolRefAttr traitSymbol,
                          const ExprNode *expr,
                          const DenseMap<StringAttr, TypedAttr> &aliasValues,
                          ParameterEvaluator &traitAliasReplacer) {
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

  FnTypeGeneratorType newSignature = signature.getSpecializedGenerator(params);

  auto selfStructAsTrait = TypeParamAttr::get(structSelfType, trait);

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
    if (paramOp.getOpcode() == POC::GetVTableEntry &&
        paramOp.getOperand(0) == selfStructAsTrait) {
      auto aliasName = cast<StringAttr>(paramOp.getOperand(1));
      // The vtable entries have type !kgen.string, but the entries from the
      // trait decl have a StringAttr with no type.  Reunique them to look up.
      aliasName = StringAttr::get(aliasName.getContext(), aliasName.strref());
      auto iter = aliasValues.find(aliasName);
      if (iter != aliasValues.end())
        return iter->second;
    }
    return paramOp;
  });
  newSignature =
      cast<KGEN::FuncTypeGeneratorType>(replacer.replace(newSignature));
  newSignature = traitAliasReplacer.replace(newSignature);
  return {newSignature, bindings};
}

/// Allow synthesizing default implementations of certain special functions.
static ASTDecl *synthesizeSpecialFunction(ASTDecl &structDecl,
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
  FnOp func;
  ASTDecl *decl = nullptr;
  if (kind == SpecialFunctionKind::kDel) {
    // Synthesize an empty destructor. Don't do anything special, because we
    // want check origins to insert a call to the real destructor here, if it
    // has one.
    std::tie(func, decl) = gen.synthesizeMethodInStruct(
        "__del__", selfRefType, ArgConvention::OwnedMem,
        PogListAttr::get(ctx, {empty}, {PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, structDecl.getLoc(), kind,
        FnEffects(), "_thunk");
    if (!func)
      return nullptr;
  } else {
    // Determine the name and argument conventions of the function.
    ArgConvention existingConv;
    switch (kind) {
    case SpecialFunctionKind::kCopyInit:
      existingConv = ArgConvention::ReadMem;
      break;
    case SpecialFunctionKind::kMoveInit:
      existingConv = ArgConvention::OwnedMem;
      break;
    default:
      llvm_unreachable("unexpected special function kind to synthesize");
    }
    StringRef name = SpecialFunctionInfo::get(kind).name;
    Type existingType;
    bool isMut = existingConv == ArgConvention::OwnedMem;
    existingType =
        structDecl.getTypeDeclSelf().getRefForArgument("existing", isMut);
    std::tie(func, decl) = gen.synthesizeMethodInStruct(
        name, {existingType, selfRefType},
        {existingConv, ArgConvention::ByRefResult},
        PogListAttr::get(ctx, {empty, empty},
                         {PassingKind::PosOnly, PassingKind::Implicit}),
        shared.getNoneType(), structDecl, structDecl.getLoc(), kind,
        FnEffects(), "_thunk");
    if (!func)
      return nullptr;
    // In every case, the implementation is a load+store.
    auto b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());
    Value value;
    if (kind == SpecialFunctionKind::kMoveInit)
      value = b.create<LIT::LoadConsumeOp>(func.getArgument(0));
    else
      value = b.create<RefLoadOp>(func.getArgument(0));
    b.create<RefStoreOp>(value, func.getArgument(1));
  }
  func.setInlineLevel(InlineLevel::AlwaysNoDebug);
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), func.getBody());
  b.create<KGEN::ReturnOp>(
      Value(b.create<ParamConstantOp>(b.getAttr<NoneAttr>())));
  return decl;
}

LogicalResult LIT::verifyConformance(ASTDecl &structDecl, SymbolRefAttr parent,
                                     std::optional<InflightDiag> &diag,
                                     WitnessTable &witnessTable) {
  auto &shared = structDecl.getShared();
  MLIRContext *ctx = structDecl.getContext();
  auto structDeclOp = cast<StructDeclOp>(structDecl);

  // TODO(MOCO-1468): Pull out into a helper method.
  bool implicitlyDestructible = false;
  for (auto parentSymbol : structDeclOp.getCanonicalTrait().getSymbols()) {
    ASTDecl &parentDecl =
        shared.declResolver->getDeclForTypeSymbol(parentSymbol);
    if (auto parentTrait = dyn_cast<TraitDeclOp>(parentDecl)) {
      if (parentTrait.getSymName() == "AnyType") {
        implicitlyDestructible = true;
        break;
      }
    }
  }

  bool rpTrivial = structDeclOp.isRegisterPassableTrivial();
  bool regPassable = structDeclOp.isRegisterPassable();
  bool hadErrors = false;
  SyntheticNode node(structDecl.getLoc());
  ExprEmitter emitter(structDecl, EC_Trait);
  ASTType selfType = structDecl.getTypeDeclSelf();

  // These are the special methods that need to be synthesized.
  SmallVector<SpecialFunctionKind> specialFns;

  ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(parent);
  TraitDeclOp traitDeclOp = cast<TraitDeclOp>(traitDecl);

  // Make sure to fully resolve the trait first.
  if (failed(shared.declResolver->resolveBody(traitDecl, structDecl.getLoc())))
    return failure();

  if (traitDeclOp.isRegisterPassable() && !regPassable) {
    diag = shared.emitError(structDecl.getLoc(),
                            "a struct must be register passable in order to "
                            "inherit from a register passable trait");
    return failure();
  }

  ParameterEvaluator traitAliasReplacer;
  DenseMap<StringAttr, TypedAttr> aliasValues;

  bool allMatchFound = true;
  // Prepare an error. It will be abandoned if the check succeeds.
  diag = shared.emitError(structDecl.getLoc(), "struct ")
         << selfType << " does not implement all requirements for "
         << ASTType(TraitType::get(parent));

  // Returns failure() to stop the verifyConformance loop.
  auto checkMethod = [&](const mlir::StringAttr &name, ASTDecl *traitFnDecl,
                         FnOp traitFn) -> LogicalResult {
    if (traitFn.getInheritedFrom()) {
      // Skip inherited methods, they're checked at a different time.
      return success();
    }
    ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
    if (decls.empty() || !isa<FnOp>(decls.front())) {
      if (canSynthesizeIfMissing(name, rpTrivial, regPassable,
                                 implicitlyDestructible)) {
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
    auto [traitSignature, bindings] = getTraitFunctionSignature(
        emitter, traitFn, selfType, parent, syntheticNode, aliasValues,
        traitAliasReplacer);

    // Match against the transformed calling convention if the struct is
    // register-passable.

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
    if (result)
      witnessTable.emplace_back(name, result.get());
    return success();
  };

  auto checkAlias = [&](const mlir::StringAttr &name, ASTDecl *traitAliasDecl,
                        AliasDeclOp traitAlias) -> LogicalResult {
    // TODO(MOCO-1140): check traitAlias.getInheritedFrom(); implement
    // inheritance of alias decls.
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
    aliasValues[name] = initializerExpr;

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

    ValueDest dest(traitAliasType, EC_AliasValue);
    CValue convertedValue = emitter.emitImplicitConversionToType(
        {initializerExpr, synthNode}, traitAliasType, dest);
    witnessTable.emplace_back(name, convertedValue.getIfPValue().get());
    return success();
  };

  // TODO(MOCO-1143): this loop needs a ParameterEvaluator that is populated
  // with the mappings of trait alias requirements to their matched values on
  // the implementing struct, then you call getReboundType/Attribute when
  // checking both the function and future alias requirements
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
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      // Skip any children that aren't methods or aliases.
      if (auto traitFn = dyn_cast<FnOp>(*decl)) {
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
  } else if (traitDecl.getIfOperation()) {
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
    hadErrors = true;
  }

  if (hadErrors)
    return failure();
  for (SpecialFunctionKind kind : specialFns) {
    if (ASTDecl *decl = synthesizeSpecialFunction(structDecl, kind)) {
      ASTType selfType = structDecl.getTypeDeclSelf();
      SmallVector<TypedAttr> bindings(selfType.getParamBindings());
      FnOp func = cast<FnOp>(decl);
      for (auto param : func.getInputParams())
        bindings.push_back(ParamDeclRefAttr::get(param));

      witnessTable.emplace_back(
          StringAttr::get(ctx, SpecialFunctionInfo::get(kind).name),
          func.getBoundSymbolRef(ParameterExprArrayAttr::get(ctx, bindings)));
    }
  }

  return success();
}

/// Given a decl for a struct or trait type, return true if this type conforms
/// to the specified trait type.  On failure, this may set 'diag' to an inflight
/// diagnostic that explains why this doesn't conform.  It can be reported or
/// abandoned based on the client's needs.
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
        if (failed(shared.declResolver->resolveBody(*witnessTables.front(),
                                                    getLoc())))
          return false;
      }
    }

    return true;
  }

  // Only structs can implicitly conform to traits.
  auto structOp = dyn_cast<StructDeclOp>(*this);
  if (!structOp)
    return false;

  // TODO(MOCO-1788): Deprecate the following logic for implicit conformance.
  // Check if the type *implicitly* conforms to the trait.
  DenseSet<SymbolRefAttr> newSymbols;
  newSymbols.insert(providedSymbols.begin(), providedSymbols.end());

  SmallVector<SymbolRefAttr> fullRequiredSymbols(requiredSymbols.begin(),
                                                 requiredSymbols.end());
  canonicalizeTraitCompositionSymbols(shared, fullRequiredSymbols);

  // Check each conformance manually, instead of going through ConformanceOp
  // ASTDecl resolution. This way a conformance failure is not an error.
  DenseMap<SymbolRefAttr, WitnessTable> witnessTableCollection;
  for (SymbolRefAttr symbol : fullRequiredSymbols) {
    if (newSymbols.contains(symbol)) {
      // This is an already provided symbol. Check explicit conformance.
      ArrayRef<ASTDecl *> witnessTables =
          lookupInCurrentScope(getFlattenedSymbolName(symbol));
      if (witnessTables.empty())
        continue;
      assert(witnessTables.size() == 1);
      if (failed(shared.declResolver->resolveBody(*witnessTables.front(),
                                                  getLoc())))
        return false;
      continue;
    }

    if (failed(verifyConformance(*this, symbol, diag,
                                 witnessTableCollection[symbol])))
      return false;
    newSymbols.insert(symbol);
  }

  // Warn against implicit conformance. Only emit on success.
  shared.emitWarning(getLoc(), "struct '")
      << *getNameIfOperation() << "' utilizes conformance to trait "
      << ASTType(trait)
      << " but does not explicitly declare it (implicit conformance is "
         "deprecated)";

  // If we succeeded, build the fully-populated conformance tables.
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  for (auto &[symbol, witnesses] : witnessTableCollection) {
    StringAttr name = b.getStringAttr(getFlattenedSymbolName(symbol));
    ConformanceOp witnessTable = b.create<ConformanceOp>(name, symbol);
    witnessTable.getBody().push_back(new Block());

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&witnessTable.getBody().front());
    for (auto &[name, value] : witnesses)
      b.create<WitnessOp>(name, value);

    shared.declResolver->addFullyResolvedDecl(witnessTable, name, getLoc(),
                                              this);
  }

  // Update canonical traits so we never check these again.
  SmallVector<SymbolRefAttr> newSymbolsVec(newSymbols.begin(),
                                           newSymbols.end());
  // No need to pull in ancestors again as `newSymbols` and
  // `fullRequiredSymbols` both contain the full ancestor chain already, so
  // their merged set also contains the full list.
  sortAndDeduplicateSymbols(newSymbolsVec);
  structOp.setCanonicalTrait(
      TraitType::get(structOp.getContext(), newSymbolsVec));
  return true;
}

/// Helper for clients that don't care about the diagnostic.
bool ASTDecl::doesNominalTypeConformTo(TraitType trait) {
  std::optional<InflightDiag> diag;
  auto result = doesNominalTypeConformTo(trait, diag);
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
