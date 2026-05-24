//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "IREmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "SpecializeInf.h"
#include "Traits.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/NameMangling.h"
#include "Support/Compiler/OperationUtils.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// File-local
namespace {
static constexpr char kToDeviceType[] = "_to_device_type";
static constexpr char kIsDeviceTypeConvertible[] =
    "_is_convertible_to_device_type";
static constexpr char kDeviceType[] = "device_type";

static bool usesClosurePipeline(FnOp fn) {
  return fn->getParentOfType<FnOp>() && !fn.isOptionalSymbol() &&
         !fn.getFuncTypeGenerator().isCapturing();
}
} // namespace

static FnOp getFnOpNamed(TraitDeclOp traitDecl, StringRef name) {
  for (FnOp candidate : traitDecl.getFields().getOps<FnOp>()) {
    if (candidate.getInheritedFrom())
      continue;
    StringRef sourceName = *candidate.getSourceName();
    if (sourceName.contains(name))
      return candidate;
  }
  return {};
}

static FnOp getInit(StructDeclOp structDeclOp) {
  FnOp init;
  for (auto fn : structDeclOp.getFields().getOps<FnOp>()) {
    if (fn.getSpecialFunctionKind() == SpecialFunctionKind::kInit) {
      assert(!init && "Wrapper has exactly one normal ctor");
      init = fn;
    }
  }
  assert(init && "Wrapper has exactly one constructor but could not find it");
  return init;
}

static LogicalResult emitForwardingCall(ImplicitLocOpBuilder &builder,
                                        ASTDecl &declScope, TypedAttr callee,
                                        FnTypeGeneratorType calleeSig,
                                        Type resultType,
                                        ArrayRef<Value> arguments) {
  IREmitter emitter(declScope, builder);
  // We are forwarding the call in a synthetic function, pushing the debug
  // scope with the synthetic function scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (declScope.getShared().diBuilder) {
    auto fnOp = cast<FnOp>(builder.getInsertionBlock()->getParentOp());
    diScopeGuard =
        declScope.getShared().diBuilder->pushScopeGuard(fnOp.getLocScope());
  }

  ExprDest dest(EC_ReturnValue);
  if (!calleeSig.isAsync() && calleeSig.hasMemoryOnlyResult())
    dest = ExprDest(MLValue(arguments.back()), EC_ReturnValue);

  SyntheticNode syntheticExpr(declScope.getLoc());
  CallOperands callOperands(CallSyntax::kMethodCall, &syntheticExpr,
                            std::move(dest));
  for (auto [bbArg, convention, pog] :
       llvm::zip_equal(arguments, calleeSig.getArgConventions(),
                       calleeSig.getArgListAttrs().getPogs())) {
    if (convention == ArgConvention::ByRefResult ||
        convention == ArgConvention::ByRefError)
      continue;

    AnyValue argValue = [&]() -> AnyValue {
      if (convention == ArgConvention::ReadReg)
        return SRValue(bbArg);
      // Forward the moved argument.
      if (convention == ArgConvention::OwnedMem ||
          convention == ArgConvention::DeinitMem)
        return MRValue(bbArg);
      return CValue::getMValueForRef(bbArg);
    }();

    if (pog.getPassingKind() == PassingKind::KwOnly)
      callOperands.addKeyword(pog.getName(), {argValue, &syntheticExpr});
    else if (pog.isPosVarArg() || pog.isPack())
      callOperands.addUnpackedPositional({argValue, &syntheticExpr});
    else
      callOperands.add({argValue, &syntheticExpr});
  }

  CValue callResult =
      emitter.emitCallUnchecked(callee, std::move(callOperands));
  assert(callResult && "call should have succeeded");
  if (!calleeSig.isAsync()) {
    auto regRet = callResult.getIfSRValue();
    if (regRet && resultType != regRet.getType())
      regRet = RebindOp::create(builder, resultType, regRet);

    IREmitter::emitNormalReturn(builder, regRet);
    return success();
  }

  // Handle async calls.
  ExprDest awaitDest(MLValue(arguments.back()), EC_SynthesizedMethod);
  if (!emitter.emitNamedMethodCall(
          "__await__",
          CallOperands(CallSyntax::kMethodCallSynthetic, &syntheticExpr,
                       std::move(awaitDest), {{callResult, &syntheticExpr}})))
    return failure();

  IREmitter::emitNormalReturn(builder);
  return success();
}

static void addConformanceTable(
    ASTDecl &structDecl, ClosureEmitter::ClosureParent closureParent,
    ArrayRef<std::pair<StringRef, TypedAttr>> witnesses, ASTDecl &fileModule) {
  // Insert the new witness into the conformance table.
  MLIRContext *ctx = structDecl.getContext();
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp.getContext());
  b.setInsertionPointToEnd(&structDeclOp.getBodyRegion().front());
  TraitDeclOp traitDeclOp = closureParent.getTrait(fileModule);
  SymbolRefArrayAttr immediateParents = traitDeclOp.getImmediateParentsAttr();
  SymbolRefAttr parentSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation()));
  StringAttr parentName = b.getStringAttr(getFlattenedSymbolName(parentSymbol));
  ConformanceOp witnessTable =
      ConformanceOp::create(b, parentName, parentSymbol, immediateParents);
  Block &block = witnessTable.getBody().emplaceBlock();
  b.setInsertionPointToStart(&block);
  for (auto [name, newWitness] : witnesses)
    WitnessOp::create(b, StringAttr::get(ctx, name), newWitness);

  // Register the conformance with the ASTDecl so lookupInCurrentScope can find
  // it during constraint checking.
  ASTDecl &conformDecl = structDecl.getShared().getDeclResolver().addDecl(
      witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
  conformDecl.resolvedness = DeclResolvedness::signature;

  // Update the types of the struct wrapper.
  SymbolRefAttr symbol = closureParent.getSymbolRef(fileModule);
  TraitType oldTraitType = structDeclOp.getCanonicalTrait();
  SmallVector<SymbolRefAttr> symbols;
  llvm::append_range(symbols, oldTraitType.getSymbols());
  symbols.push_back(symbol);
  TraitType traitType = TraitType::get(ctx, symbols);
  structDeclOp.setCanonicalTrait(traitType);
}

ClosureEmitter::ClosureEmitter(SharedState &shared)
    : FunctionEmitter(shared), ctx(shared.getContext()),
      selfName(StringAttr::get(ctx, "self")),
      copyName(StringAttr::get(ctx, "copy")),
      anyParent("AnyType", "", ClosureMethod::NONE),
      moveParent("Movable", "__init__", ClosureMethod::MOVE),
      implicitlyDestructibleParent("ImplicitlyDestructible", "__del__",
                                   ClosureMethod::DEL),
      registerPassableParent("RegisterPassable", "", ClosureMethod::NONE),
      trivialRegisterTypeParent("TrivialRegisterPassable", "",
                                ClosureMethod::NONE),
      copyParent("Copyable", "__init__", ClosureMethod::COPY),
      implicitlyCopyableParent("ImplicitlyCopyable", "", ClosureMethod::NONE) {}

TraitDeclOp ClosureEmitter::ClosureParent::getTrait(ASTDecl &moduleDecl) {
  if (trait)
    return trait;
  SharedState &shared = moduleDecl.getShared();
  auto traitDeclParent =
      shared.lookupBuiltinTrait(traitName, moduleDecl.getLoc());
  if (traitDeclParent->resolvedness < DeclResolvedness::body) {
    [[maybe_unused]] bool outcome = succeeded(shared.declResolver->resolveBody(
        *traitDeclParent, traitDeclParent->getLoc()));
    assert(outcome && "builtins should not fail body resolution.");
  }

  for (auto [_, decls] : traitDeclParent->getDeclsInScope()) {
    for ([[maybe_unused]] auto decl : decls) {
      assert(succeeded(shared.declResolver->resolveSignature(*decl,
                                                             decl->getLoc())) &&
             "builtin trait nested decls should not fail signature resolution");
    }
  }
  trait = dyn_cast_or_null<TraitDeclOp>(traitDeclParent->getIfOperation());
  // If the trait does not define any methods, do not try and resolve anything.
  if (traitFnName.empty())
    return trait;
  definingFn = getFnOpNamed(trait, traitFnName);
  assert(definingFn && "missing function in builtin trait");
  return trait;
}

FnOp ClosureEmitter::ClosureParent::getDefiningOp(ASTDecl &moduleDecl) {
  if (definingFn)
    return definingFn;
  getTrait(moduleDecl);
  return definingFn;
}

SymbolRefAttr ClosureEmitter::ClosureParent::getSymbolRef(ASTDecl &moduleDecl) {
  if (sym)
    return sym;
  sym = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(getTrait(moduleDecl).getOperation()));
  return sym;
}

StringAttr
ClosureEmitter::ClosureParent::getFullSymbolName(ASTDecl &moduleDecl) {
  if (fullSymbolName)
    return fullSymbolName;
  SymbolRefAttr parentSymbol = getSymbolRef(moduleDecl);
  fullSymbolName = StringAttr::get(parentSymbol.getContext(),
                                   getFlattenedSymbolName(parentSymbol));
  return fullSymbolName;
}

static StructFieldOp addFieldOpAndDecl(StringAttr name, Type type,
                                       StructDeclOp structOp,
                                       ASTDecl &structDecl, OpBuilder &b,
                                       DeclResolver &declResolver) {
  auto field = StructFieldOp::create(b, structOp.getLoc(), name, type);
  declResolver.addFullyResolvedDecl(&*field, field.getNameAttr(),
                                    structDecl.getLoc(), &structDecl);
  return field;
}

static void addFieldsToStruct(StructDeclOp structOp, ASTDecl &structDecl,
                              ArrayRef<Type> fields,
                              DeclResolver &declResolver) {
  OpBuilder b(structOp.getRegion());
  b.setInsertionPointToStart(&structOp.getFields().front());
  for (auto [i, type] : llvm::enumerate(fields)) {
    addFieldOpAndDecl(StringAttr::get(b.getContext(), "field" + Twine(i)), type,
                      structOp, structDecl, b, declResolver);
  }
}

static std::pair<ASTDecl &, StructDeclOp>
createStruct(SharedState &shared, ASTDecl &moduleDecl, StringAttr name,
             ArrayRef<ParamDeclAttr> params, SMLoc loc) {
  auto module = cast_or_null<FileModuleOp>(moduleDecl.getIfOperation());
  OpBuilder b(module.getRegion());
  SmallVector<StringAttr> paramNames;
#ifndef NDEBUG // Only used for assertion checks below.
  SmallPtrSet<StringAttr, 16> paramNamesSet;
#endif
  for (ParamDeclAttr param : params) {
    // The parameter for a synthesized closure are captured variable name, do
    // not demangle the capture parameter name here, as they can never be
    // referenced by user.
    paramNames.push_back(param.getName());
    assert(paramNamesSet.insert(param.getName()).second &&
           "duplicate parameter name");
  }
  // TODO: The type may contain decl references that need to be remapped.
  SmallVector<PassingKind> passingKinds(params.size(), PassingKind::PosOnly);
  auto paramListAttr =
      PogListAttr::get(b.getContext(), paramNames, passingKinds);

  StructDeclOp declOp =
      StructDeclOp::create(b, shared.diags.translateLocation(loc), name);
  declOp.setSynthetic(true);

  // Set attributes in bulk.
  NamedAttrList attrs = declOp->getAttrDictionary();
  attrs.set(declOp.getParamsAttrName(), b.getAttr<ParamDeclArrayAttr>(params));
  auto sig = TypeSignatureType::remapToSignature(
      [&]() -> InFlightDiagnostic {
        llvm_unreachable("unexpected invalid signature");
      },
      ParamDeclArrayAttr::get(b.getContext(), params), paramListAttr);
  attrs.set(declOp.getSignatureAttrName(), TypeAttr::get(sig));
  declOp->setAttrs(attrs.getDictionary(module.getContext()));

  ASTDecl &structDecl = shared.declResolver->addFullyResolvedDecl(
      &*declOp, name, loc, &moduleDecl);

  structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(declOp));
  return {structDecl, declOp};
}

/// Given a signature of a function, create a FuncType by inserting a closure
/// argument at index 0 with the given convention.
static FnTypeGeneratorType
addClosureSelfArgToFunctionSignature(Type closureType, ArgConvention convention,
                                     FnTypeGeneratorType sig) {
  MLIRContext *ctx = sig.getContext();

  unsigned newArgCount = sig.getNumArguments() + 1;
  SmallVector<Type> signatureInputs;
  signatureInputs.reserve(newArgCount);
  SmallVector<ArgConvention> argConventions;
  argConventions.reserve(newArgCount);
  SmallVector<PogMetadataAttr> argPogs;
  argPogs.reserve(newArgCount);

  // Add self.
  signatureInputs.push_back(closureType);
  argConventions.push_back(convention);
  argPogs.emplace_back(
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  // Add the rest of the arguments.
  FnMetadataAttr oldFnMetadata = sig.getFnMetadata();
  PogListAttr argListAttr = sig.getArgListAttrs();
  llvm::append_range(signatureInputs, sig.getArguments());
  llvm::append_range(argConventions, sig.getArgConventions());
  // For a fully-populated source `argListAttr`, append its pogs to keep
  // `argPogs.size() == argConventions.size()`. For an empty source (a 0-arg
  // closure with no source-level metadata), the prepended `self` pog is the
  // only pog the closure trait method has — fill the rest with anonymous
  // positional-only pogs so the synthetic trait method is fully shaped.
  llvm::append_range(argPogs, argListAttr.getPogs());
  while (argPogs.size() < argConventions.size())
    argPogs.emplace_back(
        PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  assert(argPogs.size() == argConventions.size());

  // Closure storage is carried by the inserted self argument, not by FnEffects.
  auto newArgListAttr = argListAttr.cloneWith(argPogs);
  auto metadata = FnMetadataAttr::get(
      ctx, oldFnMetadata.getNumImplicitOriginDecls(),
      oldFnMetadata.getCaptureOrigins(),
      oldFnMetadata.getIsNestedOriginExclusivityCheckingDisabled());
  return FuncTypeGeneratorType::get(
      sig.getInputParamTypes(),
      FunctionType::get(ctx, signatureInputs, sig.getResults()), argConventions,
      sig.getFnEffects(), metadata, sig.getMetadata(), newArgListAttr);
}

std::pair<TraitDeclOp, ASTDecl *> ClosureEmitter::createTraitOp(
    ASTDecl &moduleDecl, StringAttr name,
    SmallVector<ClosureParent> &closureParents,
    SMLoc nestedFunctionOrTypeLocation,
    llvm::function_ref<
        void(ASTDecl &traitDecl,
             DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
        populateTrait) {
  OpBuilder b(shared.getTopLevelDecl().getIfOperation());
  b.setInsertionPointToStart(
      &cast<ModuleOp>(shared.getTopLevelDecl().getIfOperation())
           .getBodyRegion()
           .front());
  MLIRContext *ctx = b.getContext();
  Location location =
      shared.diags.translateLocation(nestedFunctionOrTypeLocation);
  StringRef originalName = name.getValue();
  auto closureTrait =
      TraitDeclOp::create(b, location, StringAttr::get(ctx, originalName));
  ASTDecl &traitDecl = shared.declResolver->addFullyResolvedDecl(
      &*closureTrait, name, moduleDecl.getLoc(), &shared.getTopLevelDecl());

  closureTrait.setDefinesClosure(true);
  // Populate the trait with parent and self methods.
  SmallVector<SymbolRefAttr> parents;
  DenseSet<SymbolRefAttr> immediateParents;
  for (ClosureParent &p : closureParents) {
    SymbolRefAttr sym = p.getSymbolRef(moduleDecl);
    immediateParents.insert(sym);
    parents.push_back(sym);
  }
  (void)shared.declResolver->addSelfTypeToTrait(closureTrait, traitDecl,
                                                parents, immediateParents);
  DenseSet<std::pair<StringAttr, StringAttr>> existingFns;
  populateTrait(traitDecl, existingFns);
  shared.declResolver->addParentDeclsToTrait(closureTrait, traitDecl);
  /// Force synthesis of the anytype and movable members in the closure trait.
  for (const ClosureParent &p : closureParents)
    shared.lookupAndResolveDecl(p.getDefiningOpName(), traitDecl.getLoc(),
                                traitDecl, /*searchParentScopes=*/false);
  return std::pair<TraitDeclOp, ASTDecl *>(closureTrait, &traitDecl);
}

/// Converts function type generator parameters to ParamDeclAttr instances.
///
/// The function type generator stores parameters as (name, metadata) pairs and
/// types, where types can reference earlier parameters by index. This function
/// converts these to ParamDeclAttr instances with canonical types that use
/// named references.
///
/// @param sig The function type generator type
/// @return Vector of ParamDeclAttr instances with canonical types
static SmallVector<ParamDeclAttr>
populateParametersFromFnGeneratorType(FnTypeGeneratorType sig) {
  auto pogAttrs = sig.getParamListAttrs().getPogs();
  SmallVector<StringAttr> pogNames = llvm::map_to_vector(
      pogAttrs, [&](PogMetadataAttr pog) { return pog.getName(); });
  ParamRefRemapper replacer(pogNames);
  SmallVector<ParamDeclAttr> parameters;
  parameters.reserve(pogAttrs.size());

  for (auto [pog, type] : llvm::zip(pogAttrs, sig.getInputParamTypes())) {
    Type canonicalType = replacer.replace(type);
    parameters.push_back(ParamDeclAttr::get(pog.getName(), canonicalType));
  }

  return parameters;
}

/// Given a wrapper function, the wrapper type, and the wrapped field, populate
/// the operands and implicit origins necessary to bind the arguments of the
/// wrapped function.
static void getUnwrappedOperands(
    ImplicitLocOpBuilder &b, FnOp op, Type wrapperType,
    StructFieldOp wrappedField,
    llvm::SmallDenseSet<StringRef> const &explicitParameters,
    SmallVector<Value> &operands,
    std::optional<std::function<Value(Value)>> transform = {}) {
  for (Value arg : op.getBodyRegion().front().getArguments()) {
    // replace wrapper type with impl type
    RefType refType = dyn_cast<RefType>(arg.getType());
    if (!refType) {
      if (transform.has_value())
        operands.push_back((*transform)(arg));
      else
        operands.push_back(arg);
      continue;
    }

    if (refType.getElementType() == wrapperType) {
      operands.push_back(
          RefStructGEROp::create(b, arg, wrappedField)->getResults().front());
    } else {
      if (transform.has_value())
        operands.push_back((*transform)(arg));
      else
        operands.push_back(arg);
    }
  }
}

static TraitType
getTraitType(SmallVector<ClosureEmitter::ClosureParent> &closureParents,
             ASTDecl &moduleDecl) {
  SmallVector<SymbolRefAttr> symbols;
  llvm::append_range(
      symbols, llvm::map_to_vector(closureParents,
                                   [&](ClosureEmitter::ClosureParent &parent) {
                                     return parent.getSymbolRef(moduleDecl);
                                   }));
  return TraitType::get(moduleDecl.getContext(), symbols);
}

/// Replace GetWitnessAttr lookups on a specific type with lookups on the impl
/// parameter. Used to redirect trait Self lookups to the wrapper struct's impl.
static FnTypeGeneratorType replaceTraitWitnessLookupsWithParamWitnessLookups(
    FnTypeGeneratorType sig, Type replaceMeType, ParamDeclAttr implType) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
    if (getWitness.getTypeValue().getType() != replaceMeType)
      return getWitness;
    return GetWitnessAttr::get(
        ParamDeclRefAttr::get(implType), getWitness.getTraitName(),
        getWitness.getWitnessName(), getWitness.getType());
  });
  return cast<FnTypeGeneratorType>(replacer.replace(sig));
}

static std::pair<TypedAttr, SmallVector<TypedAttr>>
selfContainedSymbolAndCaptures(PValue fnPValue,
                               FnTypeGeneratorType wrapperImplType,
                               SharedState &shared, Location loc);

static SymbolConstantAttr
buildSymbol(FnOp impl, ArrayRef<ParamDeclAttr> structLevelParams) {
  MLIRContext *ctx = impl.getContext();
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));
  // Build symbol by binding struct level parameters and explicit parameters.
  FuncTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  SmallVector<TypedAttr> params;
  llvm::append_range(
      params, llvm::map_range(structLevelParams, [](ParamDeclAttr param) {
        return ParamDeclRefAttr::get(param);
      }));
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr reference) -> TypedAttr {
    return UnboundAttr::get(reference.getType());
  });
  for (auto param : impl.getInputParams().drop_back(
           impl.getFuncTypeGenerator().getNumImplicitOriginDecls()))
    params.push_back(
        cast<TypedAttr>(replacer.replace(ParamDeclRefAttr::get(param))));
  SymbolConstantAttr symbolConstant =
      SymbolConstantAttr::get(ctx, implSymbol, params, baseSigGen);
  return symbolConstant;
}

static SymbolConstantAttr
buildSymbol(FnOp impl, ParamDeclAttr implType,
            std::optional<ParamDeclAttr> originSetParam) {
  SmallVector<ParamDeclAttr> structLevelParams{implType};
  if (originSetParam)
    structLevelParams.push_back(*originSetParam);
  return buildSymbol(impl, structLevelParams);
}

static SymbolConstantAttr
buildSymbolWithBindings(FnOp impl, ArrayRef<ParamDeclAttr> structLevelParams,
                        ArrayRef<TypedAttr> fnLevelBindings) {
  MLIRContext *ctx = impl.getContext();
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));

  // Build symbol by binding struct level parameters.
  SmallVector<TypedAttr> params;
  llvm::append_range(
      params, llvm::map_range(structLevelParams, [](ParamDeclAttr param) {
        return ParamDeclRefAttr::get(param);
      }));
  llvm::append_range(params, fnLevelBindings);
  FuncTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  FuncTypeGeneratorType specializedSigGen = baseSigGen.getSpecializedGenerator(
      fnLevelBindings, /*evaluationContext=*/nullptr, impl.getLoc());

  SymbolConstantAttr symbolConstant =
      SymbolConstantAttr::get(ctx, implSymbol, params, specializedSigGen);
  return symbolConstant;
}

std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
ClosureEmitter::pushBackTraitFunctionImpl(FnOp traitFnOp, ASTDecl &structDecl,
                                          bool synthetic,
                                          StringAttr customName) {
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp.getLoc(), structDeclOp);
  b.setInsertionPointToEnd(&structDeclOp.getFields().front());
  SharedState &shared = structDecl.getShared();
  // Wrapper signature is the signature of the method on the wrapper struct.
  // We create it by specializing the trait method by binding the struct type
  // to the self parameter.
  FnTypeGeneratorType wrapperSignature = specializeSignature(
      traitFnOp, structDecl.getTypeDeclSelf(), *shared.declResolver);

  wrapperSignature = replaceTraitWitnessLookupsWithParamWitnessLookups(
      wrapperSignature, structDecl.getTypeDeclSelf().extractMetaType(),
      structDeclOp.getParams().front());

  // Calculate the argument types and result types in terms of the named
  // parameters. Since the name of the parameters have not changed from the
  // trait definition, we can avoid another remap of the indexed types in
  // parameters and instead reuse the trait function's input parameters.
  size_t traitParamCount = traitFnOp.getInputParams().size();
  size_t implicitOrigins = wrapperSignature.getNumImplicitOriginDecls();
  assert(implicitOrigins <= traitParamCount &&
         "implicit origins cannot exceed total param count");
  size_t explicitParamCount = traitParamCount - implicitOrigins;
  ArrayRef<ParamDeclAttr> parameters =
      ArrayRef<ParamDeclAttr>(traitFnOp.getInputParams())
          .take_front(explicitParamCount);
  ParamRefRemapper replacer(parameters);
  SmallVector<Type> argumentTypes;
  llvm::append_range(
      argumentTypes,
      llvm::map_range(wrapperSignature.getArguments(), [&](Type original) {
        return replacer.replace(original);
      }));
  Type result = replacer.replace(wrapperSignature.getResults().front());
  StringAttr funcName = customName ? customName : traitFnOp.getSourceNameAttr();
  auto [op, decl] = synthesizeFunction(
      structDecl, funcName, parameters, wrapperSignature.getParamListAttrs(),
      argumentTypes, wrapperSignature.getArgConventions(),
      wrapperSignature.getArgListAttrs(), result,
      traitFnOp.getSpecialFunctionKind(), structDecl.getLoc(), b,
      wrapperSignature.getFnEffects(), "", synthetic,
      traitFnOp.getInlineLevel());
  return {op, parameters, result};
}

static SymbolConstantAttr getSymbolNoParamValues(StructDeclOp declOp,
                                                 FnOp impl) {
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));
  FnTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  baseSigGen = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      declOp.getInputParams(),
      FunctionType::get(baseSigGen.getContext(),
                        baseSigGen.getBody().getArguments(),
                        baseSigGen.getResultType()),
      baseSigGen.getArgConventions(), baseSigGen.getFnEffects(),
      baseSigGen.getFnMetadata(), {});
  return SymbolConstantAttr::get(implSymbol, baseSigGen, {});
}

static ConformanceOp lookupConformanceTable(StructDeclOp op,
                                            SymbolRefAttr traitSymbol) {
  for (auto conformance : op.getFields().getOps<ConformanceOp>()) {
    if (conformance.getTraitRef() == traitSymbol) {
      return conformance;
    }
  }

  assert(false && "conformance table should be present");
  return {};
}

static void generateIsTrivialSpecialAlias(StringRef name, bool value,
                                          SharedState &shared,
                                          ASTDecl &structDecl,
                                          ClosureEmitter::ClosureParent &parent,
                                          ASTDecl &moduleDecl) {
  auto ctx = shared.getContext();
  auto declOp = dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  auto conformanceOp =
      lookupConformanceTable(declOp, parent.getSymbolRef(moduleDecl));

  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      declOp->getLoc(), &declOp.getBodyRegion().front());
  TypedAttr valueAttr = BoolAttr::get(ctx, value);
  ParamDeclAttr paramAttr =
      ParamDeclAttr::get(ctx, StringAttr::get(ctx, name), valueAttr.getType());
  AliasDeclOp aliasOp = LIT::AliasDeclOp::create(
      b, declOp.getBodyRegion().getLoc(), paramAttr, valueAttr);
  aliasOp.setInheritedFromAttr(parent.getSymbolRef(moduleDecl));
  shared.declResolver->addFullyResolvedDecl(aliasOp, StringAttr::get(ctx, name),
                                            structDecl.getLoc(), &structDecl);

  b.setInsertionPointToEnd(&conformanceOp.getBody().front());
  WitnessOp::create(b, StringAttr::get(ctx, name), valueAttr);
}

//===----------------------------------------------------------------------===//
// Closure Parameter Type Constraint Collection
//===----------------------------------------------------------------------===//

void ClosureEmitter::processClosureTraits(
    TraitType traitType, std::function<void(TraitDeclOp)> const &process) {
  for (SymbolRefAttr traitSymbol : traitType.getSymbols()) {
    ASTDecl *traitDecl =
        shared.getDeclResolver().getDeclForTypeSymbolIfExists(traitSymbol);
    if (!traitDecl)
      continue;
    auto closureTrait = dyn_cast<TraitDeclOp>(traitDecl->getIfOperation());
    if (!closureTrait || !closureTrait.getDefinesClosure())
      continue;
    process(closureTrait);
  }
}

bool ClosureEmitter::isClosureType(SharedState &shared, Type type) {
  auto definesClosure = [&](TraitType traitType) {
    for (auto sym : traitType.getSymbols()) {
      ASTDecl &decl = shared.getDeclResolver().getDeclForTypeSymbol(sym);
      if (auto traitOp =
              dyn_cast_if_present<TraitDeclOp>(decl.getIfOperation())) {
        if (traitOp.getDefinesClosure())
          return true;
      }
    }
    return false;
  };
  if (auto traitType = dyn_cast<TraitType>(type))
    return definesClosure(traitType);
  if (auto structType = dyn_cast<LIT::StructType>(type)) {
    ASTDecl &structDecl =
        shared.getDeclResolver().getDeclForTypeSymbol(structType.getSymbol());
    auto structDeclOp =
        dyn_cast_if_present<LIT::StructDeclOp>(structDecl.getIfOperation());
    if (!structDeclOp)
      return false;
    return definesClosure(structDeclOp.getCanonicalTrait());
  }
  if (auto refType = dyn_cast<RefType>(type))
    return isClosureType(shared, refType.getElementType());
  return false;
}

void ClosureEmitter::collectClosureExternalRefs(
    ParamDeclAttr closureParam, SmallVectorImpl<ClosureExternalRef> &refs) {

  auto traitType = sugarDynCast<TraitType>(closureParam.getType());
  if (!traitType)
    return;

  // Collect alias ops - these represent external parameter references.
  auto collectAliases = [&](TraitDeclOp closureTrait) {
    for (AliasDeclOp aliasOp : closureTrait.getOps<AliasDeclOp>()) {
      // Skip aliases that are inherited from a parent trait: Those are not
      // captured parameters by the closure.
      if (aliasOp.getInheritedFrom())
        continue;
      refs.push_back({closureParam, aliasOp});
    }
  };
  processClosureTraits(traitType, collectAliases);
}

/// Format a closure signature for diagnostics, omitting argument names.
/// E.g. "def(Int) -> Int".
static std::string formatClosureSignature(FnTypeGeneratorType sig,
                                          SharedState &shared,
                                          unsigned numPrependedCaptures = 0) {
  std::string result;
  llvm::raw_string_ostream os(result);

  // FIXME: This is incorrectly replicating function printing logic!
  // Switch to:
  //   ASTType(sig).print(os, &shared);
  //   return result;
  os << "def";
  SmallVector<ParamDeclAttr> parameters =
      populateParametersFromFnGeneratorType(sig);
  ParamRefRemapper replacer(parameters);

  if (!sig.getInputParamTypes().empty()) {
    os << '[';
    PogListAttr paramInfo = sig.getParamListAttrs();
    for (auto [idx, paramType] : llvm::enumerate(sig.getInputParamTypes())) {
      if (idx)
        os << ", ";
      if (paramInfo) {
        StringRef name = paramInfo.getName(idx).strref();
        if (!name.empty())
          os << name << ": ";
      }
      Type reboundType = cast<Type>(replacer.replace(paramType));
      os << ASTType(reboundType).getAsString(&shared);
      if (numPrependedCaptures && idx + 1 == numPrependedCaptures)
        os << ", #";
    }
    os << ']';
  }

  FnType body = sig.getBody();
  os << '(';
  auto args = llvm::enumerate(body.getArguments(), body.getArgConventions());
  llvm::interleaveComma(
      llvm::make_filter_range(args,
                              [](auto entry) {
                                auto [idx, argType, convention] = entry;
                                return !isResultSlot(convention);
                              }),
      os, [&](auto entry) {
        auto [idx, argType, convention] = entry;
        if (convention != ArgConvention::ReadReg &&
            convention != ArgConvention::ReadMem)
          os << getUserSyntax(convention) << ' ';

        StringAttr name = body.getArgName(idx);
        if (name && !name.empty())
          os << name.getValue() << ": ";
        Type stripped = RefType::stripRefConvention(argType, convention);
        Type reboundType = cast<Type>(replacer.replace(stripped));
        os << ASTType(reboundType).getAsString(&shared);
      });
  os << ')';

  if (sig.isThrows())
    os << " raises";
  if (sig.isAsync())
    os << " async";

  os << " -> ";
  Type resultType = body.getUserResultType();
  if (isa<KGEN::NoneType>(resultType))
    os << "None";
  else
    os << ASTType(cast<Type>(replacer.replace(resultType)))
              .getAsString(&shared);

  return result;
}

ASTDecl *ClosureEmitter::createStructWrapper(ASTDecl &moduleDecl,
                                             StringRef name, ASTDecl &traitDecl,
                                             SMLoc smLocation,
                                             TypeConvention typeConvention,
                                             bool isCopyable, bool isStateless,
                                             FnTypeGeneratorType sig) {
  StringRef implName = "impl";
  StringRef originSet = "origin_set";
  TraitDeclOp trait = cast<TraitDeclOp>(traitDecl.getIfOperation());

  auto module = cast<FileModuleOp>(moduleDecl.getIfOperation());
  Location location = shared.diags.translateLocation(smLocation);
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(location, module->getBlock());
  b.setInsertionPointAfter(trait);
  MLIRContext *ctx = b.getContext();

  SmallVector<ClosureParent> closureParents{
      ClosureParent(trait, getFnOpNamed(trait, "__call__"),
                    ClosureMethod::CALL),
      moveParent, implicitlyDestructibleParent, anyParent};
  if (isCopyable) {
    closureParents.push_back(copyParent);
    closureParents.push_back(implicitlyCopyableParent);
  }
  if (typeConvention == TypeConvention::RegisterPassableTrivial) {
    closureParents.push_back(trivialRegisterTypeParent);
    closureParents.push_back(registerPassableParent);
  } else if (typeConvention == TypeConvention::RegisterPassable)
    closureParents.push_back(registerPassableParent);

  TraitType traitType = getTraitType(closureParents, moduleDecl);

  // Give the struct a parameter "impl" of metatype trait.
  SmallVector<ParamDeclAttr> implParameters;
  ParamDeclAttr implType = ParamDeclAttr::get(implName, traitType);
  ParamDeclAttr originSetParam =
      ParamDeclAttr::get(originSet, OriginSetType::get(ctx));
  Type paramType = ParamType::get(ParamDeclRefAttr::get(implType));
  implParameters.push_back(implType);
  implParameters.push_back(originSetParam);
  ASTType selfType(paramType);

  // For each aliasOp of the trait, create a GetWitnessAttr lookup on the impl
  // parameter. The alias value is Self.impl.AliasName (lookup on wrapped type).
  llvm::MapVector<StringAttr, std::pair<Type, TypedAttr>> aliases;
  StringAttr traitName =
      b.getStringAttr(getFlattenedSymbolName(getFullyResolvedSymbolRef(
          cast<mlir::SymbolOpInterface>(trait.getOperation()))));
  // Only collect closure-specific aliases. Inherited AliasDeclOps (e.g.
  // `__del__is_trivial`) are cloned into the trait's fields by lazy body
  // resolution and are marked with `inheritedFrom`; skip them.
  for (auto alias : trait.getFields().getOps<AliasDeclOp>()) {
    if (alias.getInheritedFrom())
      continue;
    StringAttr aliasName = alias.getParamDecl().getName();
    Type aliasType = alias.getType();
    TypedAttr aliasValue = GetWitnessAttr::get(ParamDeclRefAttr::get(implType),
                                               traitName, aliasName, aliasType);
    aliases.insert({aliasName, {aliasType, aliasValue}});
  }

  // Create a struct with a single field of type "impl".
  std::pair<ASTDecl &, StructDeclOp> pair =
      createStruct(shared, moduleDecl, StringAttr::get(b.getContext(), name),
                   implParameters, smLocation);
  ASTDecl &structDecl = pair.first;
  StructDeclOp declOp = pair.second;
  declOp.setConvention(typeConvention);
  addFieldsToStruct(declOp, structDecl,
                    KGEN::ParamType::get(ParamDeclRefAttr::get(implType)),
                    *shared.declResolver);
  StructFieldOp wrappedField = *declOp.getFieldDecls().begin();

  b.setInsertionPointToEnd(&declOp.getFields().front());
  for (auto [name, value] : aliases)
    AliasDeclOp::create(b, ParamDeclAttr::get(name, value.first), value.second);

  // Populate the wrapper methods with a call to the result of a witness lookup.
  auto populateTraitFn = [&](ClosureParent &closureParent) -> FnOp {
    FnOp traitFnOp = closureParent.getDefiningOp(moduleDecl);
    b.setInsertionPointToEnd(&declOp.getFields().front());
    FnTypeGeneratorType implCallSig =
        specializeSignature(traitFnOp, selfType, *shared.declResolver);
    auto [op, parameters, result] =
        pushBackTraitFunctionImpl(traitFnOp, structDecl);

    // Generate the call op by collecting the operands and rebinding the
    // signature.
    b.setInsertionPointToEnd(&op.getBodyRegion().front());
    Value selfArgument = op.getBodyRegion().front().getArgument(0);
    SmallVector<Value> operands;
    operands.reserve(op.getNumArguments());
    Type wrapperType = cast<RefType>(selfArgument.getType()).getElementType();

    // Since this is a wrapper we know all the origins of the function must be
    // bound to the single call op in the body.
    SmallVector<TypedAttr> origins;
    llvm::SmallDenseSet<StringRef> explicitParameters;
    for (auto explicitParam : parameters)
      explicitParameters.insert(explicitParam.getName().getValue());

    // For __call__ methods with captured parameters (aliases), we need to
    // rebind non-self arguments from the wrapper's parameter types to the
    // impl's expected types (using GetWitnessAttr lookups).
    bool needsRebinding =
        closureParent.getClosureMethod() == ClosureMethod::CALL &&
        !aliases.empty();

    // Create rebinding transform to cast call operands.
    DenseMap<StringRef, TypedAttr> paramToAliasValue;
    for (auto [paramName, aliasPair] :
         llvm::zip(parameters.take_front(aliases.size()), aliases))
      paramToAliasValue.insert({paramName.getName(), aliasPair.second.second});
    mlir::AttrTypeReplacer aliasReplacer;
    aliasReplacer.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
      auto it = paramToAliasValue.find(paramRef.getName().getValue());
      if (it != paramToAliasValue.end())
        return it->second;
      return paramRef;
    });
    std::function<Value(Value)> rebindToSelfTypes =
        [&](Value valueOverSelf) -> Value {
      Type implArgType =
          cast<Type>(aliasReplacer.replace(valueOverSelf.getType()));
      if (implArgType != valueOverSelf.getType())
        return RebindOp::create(b, implArgType, valueOverSelf);
      return valueOverSelf;
    };
    SmallVector<TypedAttr> paramArgs;
    if (needsRebinding) {
      // Bind the alias parameters to the auxiliary parameters
      SmallVector<TypedAttr> auxiliary = llvm::to_vector(
          llvm::map_range(parameters, [&](ParamDeclAttr p) -> TypedAttr {
            auto ptr = paramToAliasValue.find(p.getName());
            if (ptr != paramToAliasValue.end())
              return ptr->getSecond();
            Type paramType = cast<Type>(aliasReplacer.replace(p.getType()));
            TypedAttr argument = ParamOperatorAttr::getRebind(
                ParamDeclRefAttr::get(p), paramType);
            paramArgs.push_back(argument);
            return UnboundAttr::get(ctx, argument.getType());
          }));
      // remove the auxiliary parameters from the impl call function type by
      // specializing on aliases.
      implCallSig = implCallSig.getSpecializedGenerator(
          auxiliary, &shared.getEvaluationContext(), op.getLoc());
    } else {
      llvm::append_range(
          paramArgs,
          llvm::map_range(parameters, [&](ParamDeclAttr p) -> TypedAttr {
            return ParamDeclRefAttr::get(p);
          }));
    }
    StringAttr parentName = closureParent.getFullSymbolName(moduleDecl);
    getUnwrappedOperands(b, op, wrapperType, wrappedField, explicitParameters,
                         operands, rebindToSelfTypes);

    TypedAttr symbol = GetWitnessAttr::get(
        ctx, ParamDeclRefAttr::get(implType.getName(), implType.getType()),
        parentName, traitFnOp.getSymNameAttr(), implCallSig);
    TypedAttr boundSymbol =
        BindParamsAttr::get(symbol, paramArgs, &shared.getEvaluationContext());
    // Mark `__call__` as a transparent thunk so its identity delegates to the
    // wrapped impl (only `__call__` needs this; the other forwarders don't).
    if (closureParent.getClosureMethod() == ClosureMethod::CALL)
      op->setAttr(kTransparentThunkCalleeExprAttr, boundSymbol);
    auto calleeSig = cast<FnTypeGeneratorType>(boundSymbol.getType());
    if (failed(emitForwardingCall(b, structDecl, boundSymbol, calleeSig, result,
                                  operands)))
      return {};
    return op;
  };
  DenseMap<StringRef, FnOp> nameToImpl;
  for (ClosureParent &closureParent : closureParents) {
    if (!closureParent.isEmpty()) {
      FnOp impl = populateTraitFn(closureParent);
      if (closureParent.getClosureMethod() == ClosureMethod::CALL)
        impl.setInlineLevel(InlineLevel::Always);
      switch (closureParent.getClosureMethod()) {
      case ClosureMethod::COPY:
        declOp.setCopyInitAttr(getSymbolNoParamValues(declOp, impl));
        break;
      case ClosureMethod::MOVE:
        declOp.setMoveInitAttr(getSymbolNoParamValues(declOp, impl));
        break;
      case ClosureMethod::DEL:
      default:
        break;
      }
      nameToImpl.insert(
          {*closureParent.getDefiningOp(moduleDecl).getSymName(), impl});
    }
  }

  // Emit conformance tables
  StringAttr moveParentStrAttr;
  auto addWitnessEntry = [&](TraitDeclOp traitParent, FnOp fnOp) {
    StringRef name = *fnOp.getSourceName();
    StringRef symName = *fnOp.getSymName();
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    SymbolRefArrayAttr immediateParents = traitParent.getImmediateParentsAttr();
    SymbolRefAttr parentSymbol = getFullyResolvedSymbolRef(
        cast<mlir::SymbolOpInterface>(traitParent.getOperation()));
    StringAttr parentName =
        b.getStringAttr(getFlattenedSymbolName(parentSymbol));
    if (fnOp.getSpecialFunctionKind() == SpecialFunctionKind::kMoveCtor)
      moveParentStrAttr = parentName;

    ConformanceOp witnessTable =
        ConformanceOp::create(b, parentName, parentSymbol, immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();
    b.setInsertionPointToStart(&block);
    assert(nameToImpl.contains(symName) &&
           "expected all trait ops to be implemented");
    FnOp impl = nameToImpl[symName];
    SymbolConstantAttr symbolConstant =
        buildSymbol(impl, implType, originSetParam);
    WitnessOp::create(b, fnOp.getSymNameAttr(), symbolConstant);
    if (name == "__call__") {
      for (auto [name, value] : aliases)
        WitnessOp::create(b, name, value.second);
    }
    ASTDecl &conformDecl = shared.getDeclResolver().addDecl(
        witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
    conformDecl.resolvedness = DeclResolvedness::signature;
  };

  for (ClosureParent &closureParent : closureParents) {
    if (!closureParent.isEmpty()) {
      addWitnessEntry(closureParent.getTrait(moduleDecl),
                      closureParent.getDefiningOp(moduleDecl));
    }
  }

  // AnyType has no methods, but TypeConformsToTraitAttr::simplify() needs a
  // ConformanceOp to verify conformance on concrete closure types.
  addConformanceTable(structDecl, anyParent, {}, moduleDecl);

  assert(moveParentStrAttr && "closures are expected to conform to Movable");
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ArgConvention> argConventions;

  initArgumentTypes.reserve(2);
  argNames.reserve(2);
  argPassingKinds.reserve(2);
  argConventions.reserve(2);

  // the constructor takes an instance of type "impl" and an instance of type
  // "self"
  Type refInitImplType = ASTType((paramType)).getRefForArgument(implName, true);
  argConventions.push_back(ArgConvention::OwnedMem);
  initArgumentTypes.push_back(refInitImplType);
  argNames.push_back(StringAttr::get(ctx, implName));
  argPassingKinds.push_back(PassingKind::PosOnly);

  RefType refSelfType = ASTType(structDecl.getTypeDeclSelf())
                            .getRefForArgument(selfName.getValue(), true);
  argConventions.push_back(ArgConvention::ByRefResult);
  initArgumentTypes.push_back(refSelfType);
  argNames.push_back(selfName);
  argPassingKinds.push_back(PassingKind::Implicit);
  b.setInsertionPointToEnd(&declOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions, PogListAttr::get(ctx, argNames, argPassingKinds),
      NoneType::get(ctx), SpecialFunctionKind::kInit, smLocation, b,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true,
      InlineLevel::Automatic);

  // Generate the body of the constructor, which should contain a call to the
  // move constructor.
  FnOp moveFn = moveParent.getDefiningOp(moduleDecl);
  FnTypeGeneratorType moveSignature =
      specializeSignature(moveFn, paramType, *shared.declResolver);
  b.setInsertionPointToStart(&initFnOp.getBodyRegion().front());

  TypedAttr moveSymbol = GetWitnessAttr::get(
      ctx, ParamDeclRefAttr::get(implType.getName(), implType.getType()),
      moveParentStrAttr, moveFn.getSymNameAttr(), moveSignature);
  SmallVector<Value> operands;
  llvm::SmallDenseSet<StringRef> explicitParameters;
  getUnwrappedOperands(b, initFnOp, refSelfType.getElementType(), wrappedField,
                       explicitParameters, operands);
  LogicalResult result =
      emitForwardingCall(b, structDecl, moveSymbol, moveSignature,
                         moveSignature.getResultType(), operands);
  assert(succeeded(result) && "move call should have succeeded");
  declOp.setCanonicalTrait(traitType);

  if (typeConvention == TypeConvention::RegisterPassableTrivial)
    addConformanceToDevicePassable(structDecl, wrappedField, implType,
                                   originSetParam);
  // Generate is-trivial special aliases
  bool trivialValue = typeConvention == TypeConvention::RegisterPassableTrivial;
  generateIsTrivialSpecialAlias("__del__is_trivial", trivialValue, shared,
                                structDecl, implicitlyDestructibleParent,
                                moduleDecl);
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", trivialValue, shared,
                                structDecl, moveParent, moduleDecl);
  if (isCopyable)
    generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", trivialValue,
                                  shared, structDecl, copyParent, moduleDecl);

  // Populate a readable source name for diagnostics.
  declOp.setDefinesClosure(true);
  if (sig) {
    std::string prettyName = formatClosureSignature(sig, shared);
    declOp.setSourceNameAttr(DebugInfo::SourceNameAttr::get(
        StringAttr::get(shared.getContext(), prettyName)));
  }

  return &structDecl;
}

ASTDecl *
ClosureEmitter::createFnStructWrapper(ASTDecl &moduleDecl, ASTDecl &traitDecl,
                                      FnTypeGeneratorType rawSignatureType,
                                      SMLoc smLocation) {
  FnTypeGeneratorType signatureType =
      cast<FnTypeGeneratorType>(getCanonicalType(rawSignatureType));
  auto [capturedRefs, selfContainedSignature] =
      DeclResolver::createSelfContainedSignature(signatureType);
  selfContainedSignature =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSignature));

  // The struct we're trying to create looks like this:
  // struct FnClosureWrapper[Impl: def() -> Int](`def() -> Int`):
  //   def __init__(self):
  //     pass
  //   def __call__(self) -> Int:
  //     return Impl()

  // The wrapper relies only on the function signature. Use that as the struct
  // name.
  SmallString<128> name(ASTType(selfContainedSignature).getAsString(&shared));
  name += "_PtrWrapper";
  TraitDeclOp trait = cast<TraitDeclOp>(traitDecl.getIfOperation());
  if (auto decls = moduleDecl.lookupInCurrentScope(name); !decls.empty()) {
    ASTDecl *existing = decls.front();
    // Two closure traits that share a canonical signature but
    // differ in implicit parameter name suffixes will hit the same cached
    // wrapper. If the traits are different then emit conformance.
    [[maybe_unused]] auto outcome = augmentWitnessTablesToConformTo(
        existing->getTypeDeclSelf(), &traitDecl);
    assert(succeeded(outcome) && "unexpected failure in lazy conformance");
    return existing;
  }

  StringRef implName = "Impl";

  auto module = cast<FileModuleOp>(moduleDecl.getIfOperation());
  Location location = shared.diags.translateLocation(smLocation);
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(location, module->getBlock());
  b.setInsertionPointAfter(trait);
  MLIRContext *ctx = b.getContext();

  // Give the struct a parameter "Impl" of the def pointer type.
  SmallVector<ParamDeclAttr> implParameters;
  SmallVector<ParamDeclAttr> captureParams;
  llvm::MapVector<StringAttr, std::pair<Type, TypedAttr>> aliases;
  {
    size_t aliasCount = 0;
    for (auto alias : trait.getFields().getOps<AliasDeclOp>()) {
      if (alias.getInheritedFrom())
        continue;
      aliasCount++;
      StringAttr aliasName = alias.getParamDecl().getName();
      StringAttr captureName =
          b.getStringAttr("__capture_" + aliasName.getValue());
      ParamDeclAttr captureParam =
          ParamDeclAttr::get(ctx, captureName, alias.getType());
      captureParams.push_back(captureParam);
      TypedAttr captureRef = ParamDeclRefAttr::get(captureParam);
      aliases.insert({aliasName, {alias.getType(), captureRef}});
    }
    assert(aliasCount == capturedRefs.size() &&
           "expected top-level wrapper captures to mirror trait aliases");
  }
  llvm::append_range(implParameters, captureParams);
  ParamDeclAttr implType = ParamDeclAttr::get(implName, selfContainedSignature);
  implParameters.push_back(implType);

  // Create a zero-size struct with the Impl parameter.
  std::pair<ASTDecl &, StructDeclOp> pair =
      createStruct(shared, moduleDecl, StringAttr::get(b.getContext(), name),
                   implParameters, smLocation);
  ASTDecl &structDecl = pair.first;
  StructDeclOp declOp = pair.second;
  declOp.setDefinesClosure(true);
  declOp.setConvention(TypeConvention::RegisterPassableTrivial);

  ClosureParent callParent{trait, getFnOpNamed(trait, "__call__"),
                           ClosureMethod::CALL};
  SmallVector<ClosureParent> parents{callParent,
                                     anyParent,
                                     moveParent,
                                     copyParent,
                                     implicitlyCopyableParent,
                                     implicitlyDestructibleParent,
                                     trivialRegisterTypeParent,
                                     registerPassableParent};
  TraitType traitType = getTraitType(parents, moduleDecl);
  declOp.setCanonicalTrait(traitType);
  b.setInsertionPointToEnd(&declOp.getFields().front());
  for (auto [aliasName, value] : aliases)
    AliasDeclOp::create(b, ParamDeclAttr::get(aliasName, value.first),
                        value.second);

  // Emit conformance tables
  auto addWitnessEntry = [&](ClosureParent &parent, FnOp impl) {
    auto traitParent = parent.getTrait(moduleDecl);
    auto fnOp = parent.getDefiningOp(moduleDecl);
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    SymbolRefArrayAttr immediateParents = traitParent.getImmediateParentsAttr();
    SymbolRefAttr parentSymbol = getFullyResolvedSymbolRef(
        cast<mlir::SymbolOpInterface>(traitParent.getOperation()));
    StringAttr parentName =
        b.getStringAttr(getFlattenedSymbolName(parentSymbol));

    ConformanceOp witnessTable =
        ConformanceOp::create(b, parentName, parentSymbol, immediateParents);
    ASTDecl &witnessDecl = shared.declResolver->addDecl(
        witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
    witnessDecl.resolvedness = DeclResolvedness::body;
    Block &block = witnessTable.getBody().emplaceBlock();
    b.setInsertionPointToStart(&block);
    SymbolConstantAttr symbolConstant = buildSymbol(impl, implParameters);
    WitnessOp::create(b, fnOp.getSymNameAttr(), symbolConstant);
    if (parent.getClosureMethod() == ClosureMethod::CALL) {
      for (auto [aliasName, value] : aliases)
        WitnessOp::create(b, aliasName, value.second);
    }

    return witnessTable;
  };

  // The constructor is a no-op.
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<ArgConvention> argConventions;

  RefType refSelfType = ASTType(structDecl.getTypeDeclSelf())
                            .getRefForArgument(selfName.getValue(), true);
  argConventions.push_back(ArgConvention::ByRefResult);
  initArgumentTypes.push_back(refSelfType);
  b.setInsertionPointToEnd(&declOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions,
      PogListAttr::get(ctx, {selfName}, {PassingKind::Implicit}),
      NoneType::get(ctx), SpecialFunctionKind::kInit, smLocation, b,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true, InlineLevel::Always);
  b.setInsertionPointToStart(&initFnOp.getBodyRegion().front());
  IREmitter::emitNormalReturn(b);
  initDecl->resolvedness = DeclResolvedness::body;

  StructEmitter structEmitter(structDecl);

  // Empty __del__
  auto delFnOp = structEmitter.synthesizeEmptyDtor();
  addWitnessEntry(implicitlyDestructibleParent, delFnOp);

  // Empty move ctor.
  auto moveFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(true);
  declOp.setMoveInitAttr(getSymbolNoParamValues(declOp, moveFnOp));
  addWitnessEntry(moveParent, moveFnOp);

  // Empty copy ctor
  auto copyFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(false);
  declOp.setCopyInitAttr(getSymbolNoParamValues(declOp, copyFnOp));
  addWitnessEntry(copyParent, copyFnOp);

  // All of these operations are trivial in all cases; the struct has no fields.
  generateIsTrivialSpecialAlias("__del__is_trivial", true, shared, structDecl,
                                implicitlyDestructibleParent, moduleDecl);
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", true, shared,
                                structDecl, moveParent, moduleDecl);
  generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", true, shared,
                                structDecl, copyParent, moduleDecl);

  // Generate the __call__ method based on the function signature.
  // The __call__ method is effectively the in-source body of the function.
  // Mark it as *not* synthetic so that debugging will step into the body.
  auto [callMethod, parameters, result] = pushBackTraitFunctionImpl(
      callParent.getDefiningOp(moduleDecl), structDecl,
      /*synthetic=*/false);
  callMethod.setInlineLevel(InlineLevel::Always);
  addWitnessEntry(callParent, callMethod);

  // AnyType has no methods, but TypeConformsToTraitAttr::simplify() needs a
  // ConformanceOp to verify conformance on concrete closure types.
  addConformanceTable(structDecl, anyParent, {}, moduleDecl);

  // Populate the body of ClosureWrapper::__call__.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(callMethod.getLocScope());
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        callMethod.getLoc(), callMethod.getBody());

    TypedAttr callee = ParamDeclRefAttr::get(implType);
    SmallVector<TypedAttr> paramArgs;
    ArrayRef<ParamDeclAttr> callParams = parameters;
    ArrayRef<ParamDeclAttr> auxiliaryParams;
    if (!captureParams.empty()) {
      assert(parameters.size() >= captureParams.size() &&
             "wrapper auxiliary parameters must correspond to captures");
      auxiliaryParams = parameters.take_front(captureParams.size());
      callParams = parameters.drop_front(captureParams.size());
    }
    llvm::append_range(
        paramArgs, llvm::map_range(auxiliaryParams, [](ParamDeclAttr param) {
          return TypedAttr(ParamDeclRefAttr::get(param));
        }));
    llvm::append_range(
        paramArgs,
        llvm::map_range(callParams, [](ParamDeclAttr p) -> TypedAttr {
          return ParamDeclRefAttr::get(p);
        }));
    if (!paramArgs.empty()) {
      callee = BindParamsAttr::get(callee, paramArgs,
                                   &shared.getEvaluationContext());
    }

    // Mark `__call__` as a transparent thunk so its identity delegates to the
    // wrapped function pointer's underlying generator.
    callMethod->setAttr(kTransparentThunkCalleeExprAttr, callee);

    SmallVector<Value> arguments;
    // Ignore the self field and pass the other arguments as-is.
    llvm::append_range(arguments,
                       callMethod.getBody()->getArguments().drop_front());
    auto calleeSig = cast<FnTypeGeneratorType>(callee.getType());
    if (failed(emitForwardingCall(builder, structDecl, callee, calleeSig,
                                  result, arguments)))
      return {};
  }

  return &structDecl;
}

Type ClosureEmitter::getConcreteClosureWrapperTypeForFnSymbol(
    ASTDecl &declScope, SMLoc loc, PValue fnPValue) {
  auto fnSig = cast<FnTypeGeneratorType>(fnPValue.getType());
  ASTDecl &moduleDecl = *declScope.getNearestDeclOfType<FileModuleOp>();
  auto rvClosureTrait = shared.getOrCreateClosureTrait(loc, moduleDecl, fnSig);
  ASTDecl *wrapper =
      createFnStructWrapper(moduleDecl, *rvClosureTrait, fnSig, loc);
  auto structDeclOp = cast<StructDeclOp>(wrapper->getIfOperation());

  auto [fnVal, captureBindings] = selfContainedSymbolAndCaptures(
      fnPValue,
      cast<FnTypeGeneratorType>(structDeclOp.getInputParams().back().getType()),
      shared, shared.diags.translateLocation(loc));
  SmallVector<TypedAttr> wrapperBindings;
  llvm::append_range(wrapperBindings, captureBindings);
  wrapperBindings.push_back(fnVal);
  return structDeclOp.bindReference(wrapperBindings);
}

ASTDecl *ClosureEmitter::getOrCreateClosureTrait(
    FnTypeGeneratorType key, llvm::function_ref<ASTDecl *()> creation) {
  auto ptr = closureTraitCache.find(key);
  ASTDecl *traitDecl;
  if (ptr != closureTraitCache.end()) {
    traitDecl = ptr->getSecond();
  } else {
    traitDecl = creation();
    closureTraitCache.insert({key, traitDecl});
  }
  return traitDecl;
}

static std::pair<TypedAttr, SmallVector<TypedAttr>>
selfContainedSymbolAndCaptures(PValue fnPValue,
                               FnTypeGeneratorType wrapperImplType,
                               SharedState &shared, Location loc) {
  // Rebuild the symbol with captures materialized as leading parameters.
  auto fnSig = cast<FnTypeGeneratorType>(fnPValue.getType());
  auto [captures, selfContainedSig] =
      DeclResolver::createSelfContainedSignature(fnSig);
  selfContainedSig =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSig));

  // Remove captures in signature from symbol.
  DenseSet<StringAttr> captureNames;
  for (auto capture : captures)
    captureNames.insert(capture.getName());
  auto symbol = cast<SymbolConstantAttr>(fnPValue.get());
  mlir::AttrTypeReplacer captureRewriter;
  captureRewriter.addReplacement([&](ParamDeclRefAttr reference) -> TypedAttr {
    if (!captureNames.contains(reference.getName()))
      return reference;
    return UnboundAttr::get(reference.getType());
  });
  SmallVector<TypedAttr> bindings;
  for (auto binding : symbol.getParamValues()) {
    auto bind = cast<TypedAttr>(captureRewriter.replace(binding));
    bindings.push_back(bind);
  }

  TypedAttr fnVal = SymbolConstantAttr::get(
      shared.getContext(), symbol.getSymbol(), bindings, selfContainedSig);
  assert(
      ClosureEmitter::isTypeRebindableTo(selfContainedSig, wrapperImplType) &&
      "self-contained promoted signature must match wrapper Impl canonically");
  if (fnVal.getType() != wrapperImplType)
    fnVal = ParamOperatorAttr::getRebind(fnVal, wrapperImplType);

  ParameterEvaluator evaluator(populateParametersFromFnGeneratorType(fnSig),
                               symbol.getParamValues());
  SmallVector<TypedAttr> captureBindings;
  captureBindings.reserve(captures.size());
  for (ParamDeclRefAttr capture : captures)
    captureBindings.push_back(
        cast<TypedAttr>(evaluator.getReboundAttribute(capture)));
  return {fnVal, captureBindings};
}

// Find all extern parameter references in the sig. For each reference, create
// an alias. Replace the original extern parameter reference by calling the
// custom replacer
static std::pair<FnTypeGeneratorType, llvm::MapVector<StringRef, Type>>
extractParameterReferencesIntoAliasRef(
    FnTypeGeneratorType dependentSignatureType, StringRef selfName,
    llvm::function_ref<TypedAttr(ParamDeclRefAttr)>
        externParameterRefReplacer) {
  DenseSet<StringRef> callParams;
  for (PogMetadataAttr pog :
       dependentSignatureType.getParamListAttrs().getPogs())
    callParams.insert(pog.getName());
  callParams.insert(selfName);
  llvm::MapVector<StringRef, Type> aliasMembers;
  mlir::AttrTypeReplacer externRefReplacer;
  FnTypeGeneratorType canonicalType =
      cast<FnTypeGeneratorType>(getCanonicalType(dependentSignatureType));
  externRefReplacer.addReplacement(
      [&](ParamDeclRefAttr reference) -> TypedAttr {
        if (!callParams.contains(reference.getName().getValue())) {
          auto ptr = aliasMembers.find(reference.getName());
          if (ptr == aliasMembers.end())
            aliasMembers.insert({reference.getName(), reference.getType()});
          return externParameterRefReplacer(reference);
        }
        return reference;
      });
  auto newSignature =
      cast<FnTypeGeneratorType>(externRefReplacer.replace(canonicalType));
  return {newSignature, aliasMembers};
}

static std::pair<FnTypeGeneratorType, llvm::MapVector<StringRef, Type>>
extractParameterReferencesIntoAliasRef(
    ASTDecl &decl, FnTypeGeneratorType dependentSignatureType) {
  TraitDeclOp closureTrait = cast<TraitDeclOp>(decl.getIfOperation());
  SharedState &shared = decl.getShared();
  MLIRContext *ctx = shared.getContext();
  ASTType selfType = decl.getTypeDeclSelf();
  auto declRef = dyn_cast<ParamType>(selfType.mlirType);
  auto ref = dyn_cast_if_present<ParamDeclRefAttr>(declRef.getParam());
  assert(ref && "expected the self type of a trait to be a parameter");
  StringAttr traitName = StringAttr::get(
      ctx, getFlattenedSymbolName(getFullyResolvedSymbolRef(closureTrait)));
  StringRef selfName = ref.getName().getValue();
  auto externParamReplacer = [&](ParamDeclRefAttr reference) -> TypedAttr {
    return GetWitnessAttr::get(PValue(selfType), traitName, reference.getName(),
                               reference.getType());
  };
  return extractParameterReferencesIntoAliasRef(dependentSignatureType,
                                                selfName, externParamReplacer);
}

std::pair<FnTypeGeneratorType, unsigned>
ClosureEmitter::getClosureTraitKey(FnTypeGeneratorType rawSignature) {
  auto [capturedRefs, selfContainedSig] =
      DeclResolver::createSelfContainedSignature(rawSignature);
  auto canonicalSig =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSig));
  FnTypeGeneratorType key = FnTypeGeneratorType::get(
      canonicalSig.getInputParamTypes(), canonicalSig.getValues(),
      canonicalSig.getArgConventions(),
      canonicalSig.getFnEffects().setCapturing(false),
      canonicalSig.getFnMetadata(), canonicalSig.getMetadata(),
      canonicalSig.getArgListAttrs());
  return {key, capturedRefs.size()};
}

ASTDecl *ClosureEmitter::createClosureTrait(
    ASTDecl &moduleDecl, FnTypeGeneratorType dependentSignatureType,
    FnTypeGeneratorType key, unsigned numPrependedCaptures,
    SMLoc nestedFunctionOrTypeLocation) {
  // Generate the movable, destructable closure trait, populating the trait
  // definition with the single characteristic "__call__" method.
  SmallVector<ClosureParent> parents{moveParent, implicitlyDestructibleParent};
  auto populate = [&](ASTDecl &decl,
                      DenseSet<std::pair<StringAttr, StringAttr>> &functions) {
    TraitDeclOp closureTrait = cast<TraitDeclOp>(decl.getIfOperation());
    auto [signatureNoSelf, aliasMembers] =
        extractParameterReferencesIntoAliasRef(decl, dependentSignatureType);
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
        closureTrait.getLoc(), &closureTrait.getFields().front());
    for (auto [aliasName, aliasType] : aliasMembers) {
      shared.declResolver->addFullyResolvedDecl(
          AliasDeclOp::create(
              builder, ParamDeclAttr::get(ctx, builder.getStringAttr(aliasName),
                                          aliasType)),
          aliasName, decl.getLoc(), &decl);
    }

    RefType refType = decl.getTypeDeclSelf().getRefForArgument("self", true);
    FnTypeGeneratorType sig = addClosureSelfArgToFunctionSignature(
        refType, ArgConvention::ReadMem, signatureNoSelf);
    // Augment the call function with auxiliary parameters. These auxiliary
    // parameters enable rebinding argument types in terms of external
    // parameters (e.g. "T") in terms of the alias members of closure type C
    SmallVector<ParamDeclAttr> sigParams(
        populateParametersFromFnGeneratorType(sig));
    SmallVector<PogMetadataAttr> extendedPogs;
    DenseMap<StringRef, ParamDeclAttr> aliasNameToParam;
    SmallVector<ParamDeclAttr> parameters;
    for (auto [aliasName, aliasType] : aliasMembers) {
      StringAttr nameAttr = builder.getStringAttr("_" + Twine(aliasName));
      ParamDeclAttr param = ParamDeclAttr::get(ctx, nameAttr, aliasType);
      parameters.push_back(param);
      aliasNameToParam[aliasName] = param;
      extendedPogs.push_back(
          PogMetadataAttr::get(nameAttr, PassingKind::Inferred));
    }
    llvm::append_range(extendedPogs, sig.getParamListAttrs().getPogs());
    PogListAttr extendedParamListAttrs = PogListAttr::get(
        ctx, extendedPogs, sig.getParamListAttrs().getBodyConstraints(),
        sig.getParamListAttrs().getOrigVariadicConvention());
    auto callName = StringAttr::get(ctx, "__call__");
    // Calculate the argument types and result types in terms of the named
    // parameters. Also replace GetWitnessAttr references to aliases with
    // references to auxiliary parameters.
    ParamRefRemapper replacer(sigParams);
    mlir::AttrTypeReplacer aliasReplacer;
    aliasReplacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
      StringRef witnessName = getWitness.getWitnessName().getValue();
      auto it = aliasNameToParam.find(witnessName);
      if (it != aliasNameToParam.end())
        return ParamDeclRefAttr::get(it->second);
      return getWitness;
    });
    llvm::append_range(parameters,
                       llvm::map_range(sigParams, [&](ParamDeclAttr p) {
                         return cast<ParamDeclAttr>(
                             aliasReplacer.replace(replacer.replace(p)));
                       }));
    SmallVector<Type> argumentTypes;
    llvm::append_range(
        argumentTypes, llvm::map_range(sig.getArguments(), [&](Type original) {
          return cast<Type>(aliasReplacer.replace(replacer.replace(original)));
        }));
    Type result = cast<Type>(
        aliasReplacer.replace(replacer.replace(sig.getResultType())));
    // TODO: remove capturing when legacy closures are removed.
    auto [fnOp, fnDecl] = synthesizeFunction(
        decl, callName, parameters, extendedParamListAttrs, argumentTypes,
        sig.getArgConventions(), sig.getArgListAttrs(), result,
        SpecialFunctionKind::kNormal, nestedFunctionOrTypeLocation, builder,
        sig.getFnEffects().setCapturing(true), "", true, InlineLevel::Always);
    builder.setInsertionPointToEnd(&fnOp.getBodyRegion().front());
    UnreachableOp::create(builder);
    functions.insert({callName, fnOp.getSymNameAttr()});
  };
  StringAttr name = StringAttr::get(
      shared.getContext(),
      formatClosureSignature(key, shared, numPrependedCaptures));
  auto createTraitFn = [&]() -> ASTDecl * {
    auto [closureTrait, traitDecl] = createTraitOp(
        moduleDecl, name, parents, nestedFunctionOrTypeLocation, populate);
    closureTrait.setClosureSignature(key);
    std::string prettyName =
        formatClosureSignature(dependentSignatureType, shared);
    closureTrait.setSourceNameAttr(DebugInfo::SourceNameAttr::get(
        StringAttr::get(shared.getContext(), prettyName)));
    return traitDecl;
  };
  return getOrCreateClosureTrait(key, createTraitFn);
}

static bool hasCapturingParameterType(SharedState &shared,
                                      ArrayRef<ParamDeclAttr> params) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  walker.addWalk([&](SymbolRefAttr symbol) {
    ASTDecl *traitDecl =
        shared.getDeclResolver().getDeclForTypeSymbolIfExists(symbol);
    if (!traitDecl)
      return WalkResult::advance();
    auto traitDeclOp =
        dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
    if (traitDeclOp && traitDeclOp.getDefinesClosure())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return llvm::any_of(params, [&](ParamDeclAttr param) {
    return walker.walk(param).wasInterrupted();
  });
}

ASTDecl *ClosureEmitter::promoteStatelessClosure(
    ASTDecl &nestedFnDecl, ArrayRef<ParamDeclRefAttr> paramCaptures) {
  assert(nestedFnDecl.resolvedness == DeclResolvedness::body &&
         "nested decl must be fully resolved to promote");
  MLIRContext *ctx = shared.getContext();
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  SMLoc loc = nestedFnDecl.getLoc();
  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();

  FnTypeGeneratorType nestedSignature = nestedFn.getFuncTypeGenerator();
  SmallVector<ParamDeclAttr> promotedParams;
  if (!paramCaptures.empty()) {
    promotedParams = llvm::map_to_vector(paramCaptures, [](auto capture) {
      return ParamDeclAttr::get(capture);
    });
    nestedSignature =
        FnTypeGeneratorType::prependParams(nestedSignature, promotedParams);
  }
  bool shouldBeCapturing = nestedSignature.isCapturing() ||
                           hasCapturingParameterType(shared, promotedParams);
  // The promoted signature will not have register_passable; capturing is set
  // from the promoted form.
  FnTypeGeneratorType promotedSignature = FnTypeGeneratorType::get(
      nestedSignature.getInputParamTypes(), nestedSignature.getValues(),
      nestedSignature.getArgConventions(),
      nestedSignature.getFnEffects().setCapturing(shouldBeCapturing),
      nestedSignature.getFnMetadata(), nestedSignature.getMetadata(),
      nestedSignature.getArgListAttrs());

  OpBuilder builder = moduleDecl->getDeclEndBuilder();
  nestedFn->moveBefore(builder.getInsertionBlock(),
                       builder.getInsertionPoint());
  FnOp promotedFn = nestedFn;
  // We need to mangle the symbol name because we're lifting these into the file
  // scope - if you have two closures with the same name in different functions,
  // that's fine, but when we lift them to the file scope they need to have
  // unique names.
  promotedFn.setSymName(
      moduleDecl->mangleParamName(nestedFn.getSymName()->str()));
  promotedFn.setFuncTypeGenerator(promotedSignature);
  if (!promotedParams.empty()) {
    SmallVector<ParamDeclAttr> allParams(promotedFn.getParams());
    allParams.insert(allParams.begin(), promotedParams.begin(),
                     promotedParams.end());
    promotedFn.setParamsAttr(ParamDeclArrayAttr::get(ctx, allParams));
  }
  // Transfer the linkage name to the promoted op: the mangled sym_name
  // above overwrites the original name, so preserve it so it survives
  // into elaboration.
  if (auto linkageName = nestedFn.getLinkageNameAttr())
    promotedFn.setLinkageNameAttr(linkageName);
  promotedFn.setNoDocRequired(true);
  auto &decl = shared.declResolver->addFullyResolvedDecl(
      promotedFn, nestedFn.getSourceNameAttr(), loc, moduleDecl);
  // Transfer child decls from the original to the promoted decl. Since the op
  // was moved (not cloned), all mlir::Value pointers are still valid.
  decl.takeDecls(nestedFnDecl);
  // Register the lifted function to the symbol table.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(promotedFn, decl);
  assert(!existing && "unexpected redefinition of promoted closure");
  if (promotedParams.empty()) {
    nestedFnDecl.setIRValue(promotedFn);
    return &decl;
  }

  ArrayRef<ParamDeclAttr> promotedFnParams = promotedFn.getParams();
  ArrayRef<ParamDeclAttr> captureParams =
      promotedFnParams.take_front(promotedParams.size());

  SmallVector<TypedAttr> bindings;
  bindings.reserve(promotedFnParams.size());
  size_t captureIndex = 0;
  for (auto [paramIndex, param] : llvm::enumerate(promotedFnParams)) {
    if (paramIndex < captureParams.size()) {
      bindings.push_back(ParamDeclRefAttr::get(captureParams[captureIndex++]));
      continue;
    }
    bindings.push_back(UnboundAttr::get(ctx, param.getType()));
  }
  assert(captureIndex == captureParams.size() &&
         "all capture params must be rebound");
  nestedFnDecl.setIRValue(PValue(promotedFn.getFuncLiteralGenerator(
      shared.getEvaluationContext(),
      ParameterExprArrayAttr::get(ctx, bindings))));

  return &decl;
}

template <typename T>
static SymbolRefAttr getFullyResolvedSymbolRefUpTo(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  Operation *current = op;
  while (current && !isa<T>(current)) {
    if (mlir::SymbolOpInterface next =
            dyn_cast<mlir::SymbolOpInterface>(current))
      symbols.push_back(FlatSymbolRefAttr::get(next.getNameAttr()));
    current = current->getParentOp();
  }
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

static void meetOriginMutability(DenseMap<StringAttr, bool> &originMutability,
                                 StringAttr name, bool isKnownImmutable) {
  auto [it, isNew] = originMutability.try_emplace(name, /*mutable=*/false);
  if (!isKnownImmutable)
    it->second = true;
}

// Given an attribute, update origin mutability information. Returns false to
// stop recursion for origin typed nodes so cast operands aren't counted
// separately.
static bool checkOriginMutableCast(DenseMap<StringAttr, bool> &originMutability,
                                   Attribute attr) {
  auto handleOriginValue = [&](TypedAttr originValue, bool isKnownImmutable) {
    if (auto ref = dyn_cast<ParamDeclRefAttr>(
            OriginType::stripMutCastAndRebind(originValue)))
      meetOriginMutability(originMutability, ref.getName(), isKnownImmutable);
  };

  if (auto typed = dyn_cast<TypedAttr>(attr);
      typed && isa<OriginType>(typed.getType())) {
    handleOriginValue(typed, OriginType::isMutableKnown(typed, false));
    return false;
  }
  return true;
}

// Given an attribute or type, determine if the references to an origin are a
// net mutable or net immutable.
template <typename AttrOrType>
static void checkMutableImpl(DenseMap<StringAttr, bool> &originMutability,
                             AttrOrType attrOrType) {

  if constexpr (std::is_convertible_v<AttrOrType, Attribute>) {
    if (!checkOriginMutableCast(originMutability, attrOrType))
      return;
  }
  attrOrType.walkImmediateSubElements(
      [&](Attribute attribute) {
        checkMutableImpl(originMutability, attribute);
      },
      [&](Type type) { checkMutableImpl(originMutability, type); });
}

template <typename AttrOrType>
static void checkMutable(DenseMap<StringAttr, bool> &originMutability,
                         AttrOrType attrOrType) {
  if constexpr (std::is_convertible_v<AttrOrType, Attribute>)
    attrOrType = cast<AttrOrType>(getCanonicalAttr(attrOrType));
  else
    attrOrType = cast<AttrOrType>(getCanonicalType(attrOrType));
  checkMutableImpl(originMutability, attrOrType);
}

static SmallPtrSet<StringAttr, 8>
collectPromotedOrigins(MLIRContext *ctx,
                       SmallVectorImpl<StructDefFieldAttr> &fieldDecls,
                       SmallVectorImpl<ParamDeclAttr> &structParams,
                       SmallVectorImpl<TypedAttr> &structBindings) {
  DenseMap<StringAttr, bool> originMutability;
  for (StructDefFieldAttr fieldDecl : fieldDecls)
    checkMutable(originMutability, fieldDecl.getTypeValue());

  SmallPtrSet<StringAttr, 8> promotedOriginNames;
  for (auto [index, param] : llvm::enumerate(structParams)) {
    StringAttr name = param.getName();
    auto originType = dyn_cast<OriginType>(param.getType());
    // If an origin is already immutable, no need to promote.
    if (!originType || originType.isMutableKnown(false))
      continue;
    if (auto it = originMutability.find(name);
        it != originMutability.end() && it->second)
      continue;
    structParams[index] = ParamDeclAttr::get(name, OriginType::get(ctx, false));
    structBindings[index] =
        OriginMutCastAttr::get(structBindings[index], false);
    promotedOriginNames.insert(name);
  }
  return promotedOriginNames;
}

static KGEN::StructType getMlirType(MLIRContext *ctx,
                                    StructInstanceType structInstType,
                                    ArrayRef<ParamDeclAttr> structParams,
                                    ArrayRef<TypedAttr> structBindings) {
  ParameterEvaluator structEvaluator(structParams, structBindings);
  SmallVector<Type> mlirFieldTypes;
  for (StructDefFieldAttr field : structInstType.getFields()) {
    TypedAttr concreteFieldAttr =
        structEvaluator.getReboundAttribute(field.getTypeValue());
    if (auto typeParam = dyn_cast<TypeParamAttr>(concreteFieldAttr))
      mlirFieldTypes.push_back(typeParam.getMlirType());
    else if (isa<ParamDeclRefAttr>(concreteFieldAttr))
      mlirFieldTypes.push_back(ParamType::get(concreteFieldAttr));
    else
      mlirFieldTypes.push_back(concreteFieldAttr.getType());
  }
  bool isMemOnly = cast<BoolAttr>(structInstType.getIsMemoryOnly()).getValue();
  return KGEN::StructType::get(ctx, mlirFieldTypes, isMemOnly);
}

TypedAttr ClosureEmitter::addWitnessTablesToClosure(
    ASTDecl &moduleDecl, SMLoc smLoc,
    SmallVector<ClosureParent> &closureParents, SymbolRefAttr parentSymbolRef,
    llvm::MapVector<StringRef, Type> const &aliases,
    SmallVector<StructDefFieldAttr> &&concreteFieldDecls,
    SmallVector<ParamDeclAttr> &&concreteStructParams,
    SmallVector<TypedAttr> &&concreteStructBindings, StringAttr name,
    bool isRegPassable) {
  Location location = shared.translateLocation(smLoc);
  MLIRContext *ctx = shared.getContext();
  SmallPtrSet<StringAttr, 8> promotedOriginNames = collectPromotedOrigins(
      ctx, concreteFieldDecls, concreteStructParams, concreteStructBindings);

  if (!promotedOriginNames.empty()) {
    mlir::AttrTypeReplacer promoteOriginRefs;
    promoteOriginRefs.addReplacement(
        [&](TypedAttr attr) -> std::optional<TypedAttr> {
          if (!isa<OriginType>(attr.getType()))
            return std::nullopt;
          auto originRef = dyn_cast<ParamDeclRefAttr>(
              OriginType::stripMutCastAndRebind(attr));
          if (!originRef || !promotedOriginNames.contains(originRef.getName()))
            return std::nullopt;
          return ParamDeclRefAttr::get(originRef.getName(),
                                       OriginType::get(ctx, false));
        });
    for (StructDefFieldAttr &fieldDecl : concreteFieldDecls) {
      auto promotedTypeValue =
          cast<TypedAttr>(promoteOriginRefs.replace(fieldDecl.getTypeValue()));
      fieldDecl =
          StructDefFieldAttr::get(fieldDecl.getName(), promotedTypeValue);
    }
  }

  SmallVector<TypedAttr> selfRefParamValues = llvm::map_to_vector(
      concreteStructParams, [](ParamDeclAttr declAttr) -> TypedAttr {
        return ParamDeclRefAttr::get(declAttr);
      });
  SmallVector<StringAttr> paramNames = llvm::map_to_vector(
      concreteStructParams,
      [](ParamDeclAttr declAttr) -> StringAttr { return declAttr.getName(); });
  StructInstanceType structInstType = StructInstanceType::get(
      StringAttr::get(ctx, Twine(getFlattenedSymbolName(parentSymbolRef))
                               .concat("::")
                               .concat(name.getValue())),
      paramNames, selfRefParamValues, concreteFieldDecls,
      BoolAttr::get(ctx, !isRegPassable));

  ParamDeclArrayAttr parameters =
      ParamDeclArrayAttr::get(ctx, concreteStructParams);
  ImplicitLocOpBuilder builder(location, ctx);
  builder.setInsertionPointToStart(
      &cast<FileModuleOp>(moduleDecl.getIfOperation()).getBodyRegion().front());
  TraitType traitType = getTraitType(closureParents, moduleDecl);
  auto structGen = StructGeneratorOp::create(
      builder, structInstType.getName(), parameters, structInstType, traitType);
  Block *structGenBody = builder.createBlock(&structGen.getRegion());

  // Register the struct generator in declForTypeSymbol so it can be looked up
  // when resolving conformance for closure types.
  SymbolRefAttr structGenSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(structGen.getOperation()));
  shared.declResolver->registerStructGeneratorDecl(structGen, structGenSymbol,
                                                   smLoc, moduleDecl);

  // Emit the conformance ops into the struct gen body by finding the closure
  // method and FnOp associated with the parent trait.
  auto addWitnessTable = [&](ClosureParent &closureParent) {
    TraitDeclOp traitParent = closureParent.getTrait(moduleDecl);
    builder.setInsertionPointToStart(structGenBody);
    SymbolRefArrayAttr immediateParents = traitParent.getImmediateParentsAttr();
    SymbolRefAttr parentSymbol = closureParent.getSymbolRef(moduleDecl);
    StringAttr parentName = closureParent.getFullSymbolName(moduleDecl);
    ConformanceOp witnessTable = ConformanceOp::create(
        builder, parentName, parentSymbol, immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();

    // Marker traits like AnyType have no methods -- empty ConformanceOp is
    // sufficient for TypeConformsToTraitAttr::simplify().
    if (closureParent.isEmpty())
      return;

    builder.setInsertionPointToStart(&block);
    ClosureMethod method = closureParent.getClosureMethod();
    FnOp fnOp = closureParent.getDefiningOp(moduleDecl);
    FnTypeGeneratorType sig =
        specializeSignature(fnOp, structInstType, *shared.declResolver);
    SmallVector<TypedAttr> paramValues;
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParamDeclRefAttr parameterReference) {
      return UnboundAttr::get(parameterReference.getType());
    });
    replacer.addReplacement([&](ParamIndexRefAttr parameterReference) {
      return UnboundAttr::get(parameterReference.getType());
    });
    for (Type paramType : sig.getInputParamTypes())
      paramValues.push_back(UnboundAttr::get(replacer.replace(paramType)));

    TypedAttr symbol = ClosureSymbolAttr::get(
        ctx, parentSymbolRef, name, ClosureMethodAttr::get(ctx, method),
        paramValues, sig);
    WitnessOp::create(builder, fnOp.getSymNameAttr(), symbol);

    // add the alias entries
    if (closureParent.getClosureMethod() == ClosureMethod::CALL) {
      for (auto [name, type] : aliases)
        WitnessOp::create(
            builder, name,
            ParamDeclRefAttr::get(StringAttr::get(ctx, name), type));
    }
  };

  for (ClosureParent &closureParent : closureParents)
    addWitnessTable(closureParent);

  // create a SymbolRefAttr from the StructGeneratorOp
  SymbolRefAttr structGenSymbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(structGen.getOperation()));
  // Type value contains the reference to the struct gen op with the witness
  // table.
  auto typeValue = KGEN::TypeValueType::get(
      ctx, TypeGeneratorRefAttr::get(ctx, structGenSymbolRef,
                                     concreteStructBindings, traitType));
  KGEN::StructType kgenStructType = getMlirType(
      ctx, structInstType, concreteStructParams, concreteStructBindings);

  auto typeParamAttr = TypeParamAttr::get(typeValue, kgenStructType, traitType);
  return typeParamAttr;
}

static unsigned conventionRank(TypeConvention convention) {
  if (convention == TypeConvention::Unspecified)
    return 0;
  return static_cast<unsigned>(convention);
}

static TypeConvention meetCaptureConvention(TypeConvention lhs,
                                            TypeConvention rhs) {
  return conventionRank(lhs) <= conventionRank(rhs) ? lhs : rhs;
}

MemSymbolTripleAttr ClosureEmitter::validateAndBuildTriple(
    TypedAttr copy, TypedAttr move, TypedAttr del, CaptureConvention convention,
    const Capture &capture, UnitAttr &isMove, ASTDecl &nestedFnDecl) {
  MLIRContext *ctx = shared.getContext();
  if (convention == CaptureConvention::kConventionCopy && !copy) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " by copy because it is not copyable.");
    return nullptr;
  }
  if (convention == CaptureConvention::kConventionMove) {
    if (!move) {
      shared.emitError(nestedFnDecl.getLoc(),
                       "cannot capture " + capture.getSpelling() +
                           " by move because it is not movable.");
      return nullptr;
    }
    isMove = UnitAttr::get(ctx);
  }
  if (!del) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because it is not destructable.");
  }
  return MemSymbolTripleAttr::get(ctx, copy, move, del, isMove);
}

std::pair<MemSymbolTripleAttr, TypeConvention>
ClosureEmitter::buildStructCaptureInfo(StructType structType,
                                       const Capture &capture,
                                       CaptureConvention convention,
                                       UnitAttr &isMove,
                                       ASTDecl &nestedFnDecl) {
  SymbolConstantAttr del, move, copy;
  ASTDecl &structDecl =
      shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  TypeConvention availableConvention =
      structDeclOp.isRegisterPassableTrivial()
          ? TypeConvention::RegisterPassableTrivial
      : structDeclOp.isRegisterPassable() ? TypeConvention::RegisterPassable
                                          : TypeConvention::MemoryOnly;
  ArrayRef<ASTDecl *> results = structDecl.lookupInCurrentScope("__del__");
  if (results.size() == 1) {
    FnOp destructor = dyn_cast<FnOp>(results.front()->getIfOperation());
    if (destructor)
      del = destructor.getBoundSymbolRef(shared.getEvaluationContext());
  }
  if (structDeclOp.getMoveInit().has_value())
    move = *structDeclOp.getMoveInit();
  if (structDeclOp.getCopyInit().has_value())
    copy = *structDeclOp.getCopyInit();

  auto paramValues = structType.getParamValues();
  auto paramArray =
      ParameterExprArrayAttr::get(shared.getContext(), paramValues);
  auto bind = [&](SymbolConstantAttr sym) -> SymbolConstantAttr {
    if (!sym)
      return sym;
    ASTDecl *fnDecl =
        shared.declResolver->getDeclForFuncSymbol(sym.getSymbol());
    if (!fnDecl) {
      // Demand-resolve the signature so the declForFuncSymbol map is populated
      // when the referenced fn lives in a bytecode-loaded module.
      if (failed(shared.resolveDeclReferencesIn(nestedFnDecl.getLoc(), sym)))
        return {};
      fnDecl = shared.declResolver->getDeclForFuncSymbol(sym.getSymbol());
    }
    if (!fnDecl)
      return {};
    auto fnOp = cast<FnOp>(fnDecl->getIfOperation());
    return fnOp.getBoundSymbolRef(shared.getEvaluationContext(), paramArray);
  };
  copy = bind(copy);
  move = bind(move);
  del = bind(del);

  auto triple = validateAndBuildTriple(copy, move, del, convention, capture,
                                       isMove, nestedFnDecl);
  return {triple, availableConvention};
}

std::pair<MemSymbolTripleAttr, TypeConvention>
ClosureEmitter::buildParamCaptureInfo(ParamType paramType,
                                      const Capture &capture,
                                      CaptureConvention convention,
                                      UnitAttr &isMove, ASTDecl &nestedFnDecl,
                                      ASTDecl &moduleDecl) {
  MLIRContext *ctx = shared.getContext();
  auto paramRef = dyn_cast<ParamDeclRefAttr>(paramType.getParam());
  if (!paramRef) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type is not a parameter reference.");
    return {nullptr, TypeConvention::Unspecified};
  }

  auto traitType = dyn_cast<TraitType>(paramRef.getType());
  if (!traitType) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type constraint is not a trait.");
    return {nullptr, TypeConvention::Unspecified};
  }

  TypedAttr typeValue =
      ParamDeclRefAttr::get(paramRef.getName(), paramRef.getType());
  ASTType selfType(paramType);
  auto assumptions = ASTDecl::getAssumptionsFromScope(&nestedFnDecl);
  TypeConvention availableConvention =
      ASTType(paramType).getRegisterPassability(nestedFnDecl.getLoc(), shared);
  auto makeWitness = [&](ClosureParent &parent) -> TypedAttr {
    FnOp fnOp = parent.getDefiningOp(moduleDecl);
    if (!fnOp)
      return nullptr;
    FnTypeGeneratorType sig =
        specializeSignature(fnOp, selfType, *shared.declResolver);
    StringAttr parentName = parent.getFullSymbolName(moduleDecl);
    return GetWitnessAttr::get(ctx, typeValue, parentName,
                               fnOp.getSymNameAttr(), sig);
  };

  auto conformsToBuiltinTrait = [&](StringRef traitName) {
    TraitType trait =
        shared.lookupBuiltinTraitType(traitName, nestedFnDecl.getLoc());
    return trait && selfType.checkConformance(trait, shared, assumptions) ==
                        ConformanceResult::Yes;
  };

  TypedAttr move =
      selfType.isMovable(nestedFnDecl.getLoc(), shared, &nestedFnDecl)
          ? makeWitness(moveParent)
          : nullptr;
  TypedAttr del = conformsToBuiltinTrait("ImplicitlyDestructible")
                      ? makeWitness(implicitlyDestructibleParent)
                      : nullptr;
  TypedAttr copy = selfType.isExplicitlyCopyable(nestedFnDecl.getLoc(), shared,
                                                 &nestedFnDecl)
                       ? makeWitness(copyParent)
                       : nullptr;

  return {validateAndBuildTriple(copy, move, del, convention, capture, isMove,
                                 nestedFnDecl),
          availableConvention};
}

Value ClosureEmitter::emitClosureOp(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                                    ArrayRef<Capture> captures,
                                    TraitDeclOp trait, Location location,
                                    bool isCopyable,
                                    FnTypeGeneratorType closureSig,
                                    ArrayRef<ParamDeclRefAttr> paramCaptures) {
  // (1) Create the closure instance.
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  FnOp parent = nestedFn->getParentOfType<FnOp>();
  assert(parent && "expected the function to be a nested function");
  ImplicitLocOpBuilder builder(location, shared.getContext());
  builder.setInsertionPoint(nestedFn);
  MLIRContext *ctx = builder.getContext();
  StringAttr fnName = nestedFn.getSourceNameAttr();
  ASTDecl *symbolParent = nestedFnDecl.getParentDecl();
  do {
    if (isa_and_nonnull<FnOp>(symbolParent->getIfOperation()))
      break;
    symbolParent = symbolParent->getParentDecl();
  } while (symbolParent);
  // The location of the closure init op should have its parent's subprogram
  // as a scope. We will also store an independent scope on the op to validate
  // the nested ops.
  Location fileOnlyLoc = DebugInfo::extractSourceLoc(location);
  Location opLoc = fileOnlyLoc;
  if (DebugInfo::DISubprogramAttr subprogram =
          cast<FnOp>(symbolParent->getIfOperation()).getSubprogramScope()) {
    opLoc = FusedLoc::get(
        ctx, fileOnlyLoc,
        cast<FnOp>(symbolParent->getIfOperation()).getSubprogramScope());
  }

  // TODO: remove name mangling and replace with abstraction (MOCO-2265)
  auto parentSymbolRef = SymbolRefAttr::get(
      ctx, getFlattenedSymbolName(getFullyResolvedSymbolRefUpTo<ModuleOp>(
               cast<mlir::SymbolOpInterface>(parent.getOperation()))));
  auto closureAttr = KGEN::ClosureAttr::get(
      ctx, ParamClosureType::get(ctx, parentSymbolRef, fnName));
  SmallVector<Attribute> captureInfo;
  SmallVector<Value> captureValues;
  SmallVector<Attribute> captureTypes;
  SmallVector<Attribute> captureNames;

  TraitType anyType =
      shared.lookupBuiltinTraitType("AnyType", nestedFnDecl.getLoc());
  IREmitter emitter(*nestedFnDecl.getParentDecl(), builder);

  TypeConvention highestCaptureConvention =
      TypeConvention::RegisterPassableTrivial;
  SmallVector<StructDefFieldAttr> fieldDecls;
  SmallVector<ParamDeclAttr> allStructParams;
  SmallVector<TypedAttr> structParamBindings;

  SmallPtrSet<StringAttr, 8> byValueCapturedOriginParamNames;
  auto updateCaptureConvention = [&](TypeConvention captureConventionMet,
                                     StringRef captureName) {
    highestCaptureConvention =
        meetCaptureConvention(highestCaptureConvention, captureConventionMet);
    (void)captureName;
  };
  for (const Capture &capture : captures) {
    Value value = capture.getValue().getMlirValue();
    captureValues.push_back(value);

    SyntheticNode synthNode(nestedFnDecl.getLoc());
    ExprDest dest(anyType, EC_Type);
    PValue captureTypeValue =
        emitter
            .emitImplicitConversionToType({value.getType(), synthNode}, anyType,
                                          dest)
            .getIfPValue();
    auto captureTypeAttr = cast<TypedAttr>(captureTypeValue.get());
    captureTypes.push_back(captureTypeAttr);
    auto captureName = StringAttr::get(ctx, capture.getSpelling());
    captureNames.push_back(captureName);
    UnitAttr isMove;
    auto captureConvention = capture.getCaptureConvention();
    switch (captureConvention) {
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionRef: {
      // Mutability casts should have been emitted during parse time.
      TypeConvention captureConventionMet = TypeConvention::MemoryOnly;
      if (auto refType = dyn_cast<LIT::RefType>(value.getType())) {
        // TODO: Pointers are register passable, so this demotion
        // should become unnecessary once downstream passes are fixed.
        captureConventionMet =
            ASTType(refType.getElementType())
                .getRegisterPassability(nestedFnDecl.getLoc(), shared);
        captureInfo.push_back(refType.getOrigin());
      } else {
        captureConventionMet =
            ASTType(value.getType())
                .getRegisterPassability(nestedFnDecl.getLoc(), shared);
        captureInfo.push_back(UnitAttr::get(ctx));
      }
      updateCaptureConvention(captureConventionMet, capture.getSpelling());
      break;
    }
    case CaptureConvention::kConventionTrivialCopy:
      captureInfo.push_back(UnitAttr::get(ctx));
      break;
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionMove: {
      Type mlirType;
      if (auto refType = dyn_cast<LIT::RefType>(value.getType())) {
        mlirType = refType.getElementType();
        if (auto captureOriginParam = dyn_cast<ParamDeclRefAttr>(
                OriginType::stripMutCastAndRebind(refType.getOrigin())))
          byValueCapturedOriginParamNames.insert(captureOriginParam.getName());
      } else {
        mlirType = value.getType();
      }
      // Copy/move captures materialize storage for the captured value itself,
      // not for a reference wrapper. Use the pointee as the field type.
      if (isa<LIT::RefType>(value.getType()))
        captureTypeAttr = TypeParamAttr::get(mlirType, anyType);

      if (auto structType = dyn_cast<StructType>(mlirType)) {
        auto [memTriple, captureConventionMet] = buildStructCaptureInfo(
            structType, capture, captureConvention, isMove, nestedFnDecl);
        if (!memTriple)
          return {};
        updateCaptureConvention(captureConventionMet, capture.getSpelling());
        captureInfo.push_back(memTriple);
      } else if (auto traitType = dyn_cast<TraitType>(mlirType)) {
        shared.emitError(nestedFnDecl.getLoc(),
                         "cannot capture a value of trait type yet because "
                         "existentials are not implemented.");
        return {};
      } else if (auto paramType = dyn_cast<ParamType>(mlirType)) {
        auto [memTriple, captureConventionMet] =
            buildParamCaptureInfo(paramType, capture, captureConvention, isMove,
                                  nestedFnDecl, moduleDecl);
        if (!memTriple)
          return {};
        updateCaptureConvention(captureConventionMet, capture.getSpelling());
        captureInfo.push_back(memTriple);
      } else {
        captureInfo.push_back(UnitAttr::get(ctx));
      }
      break;
    }
    }
    fieldDecls.push_back(StructDefFieldAttr::get(captureName, captureTypeAttr));
  }
  bool isRegPassable = highestCaptureConvention != TypeConvention::MemoryOnly;
  KGEN::ClosureType closureType =
      ClosureType::get(ctx, closureAttr,
                       isRegPassable ? ClosureMemoryKind::REGISTER_PASSABLE
                                     : ClosureMemoryKind::NONESCAPING);
  FnTypeGeneratorType wrapperSig = FnTypeGeneratorType::get(
      closureSig.getInputParamTypes(), closureSig.getValues(),
      closureSig.getArgConventions(), closureSig.getFnEffects(),
      closureSig.getFnMetadata(), closureSig.getMetadata(),
      closureSig.getArgListAttrs());
  trait = cast<TraitDeclOp>(shared
                                .getOrCreateClosureTrait(nestedFnDecl.getLoc(),
                                                         moduleDecl, wrapperSig)
                                ->getIfOperation());
  ASTDecl *closureWrapperDecl = shared.getOrCreateClosureWrapper(
      nestedFnDecl.getLoc(), wrapperSig, &moduleDecl, isCopyable,
      highestCaptureConvention, captures.empty());
  StructDeclOp wrapper =
      cast<StructDeclOp>(closureWrapperDecl->getIfOperation());

  StringAttr originAttr =
      nestedFnDecl.getParentDecl()->mangleParamName(fnName.getValue());
  SmallVector<ClosureParent> closureParents{
      ClosureParent(trait, getFnOpNamed(trait, "__call__"),
                    ClosureMethod::CALL),
      moveParent, implicitlyDestructibleParent, anyParent};
  if (isCopyable) {
    closureParents.push_back(copyParent);
    closureParents.push_back(implicitlyCopyableParent);
  }
  if (highestCaptureConvention == TypeConvention::RegisterPassableTrivial) {
    closureParents.push_back(trivialRegisterTypeParent);
    closureParents.push_back(registerPassableParent);
  } else if (highestCaptureConvention == TypeConvention::RegisterPassable)
    closureParents.push_back(registerPassableParent);

  ParamDeclAttr origin =
      ParamDeclAttr::get(originAttr, OriginType::get(ctx, true));
  auto refType = RefType::get(closureType, ParamDeclRefAttr::get(origin));
  FnTypeGeneratorType original = nestedFn.getFuncTypeGenerator();
  // TODO: Remove capturing when legacy closures are removed
  FnTypeGeneratorType closureBodySignature = FnTypeGeneratorType::get(
      original.getInputParamTypes(), original.getValues(),
      original.getArgConventions(), original.getFnEffects().setCapturing(true),
      original.getFnMetadata(), original.getMetadata(),
      original.getArgListAttrs());
  auto [capturedRefs, _] =
      DeclResolver::createSelfContainedSignature(closureBodySignature);
  llvm::MapVector<StringRef, Type> aliases;
  for (ParamDeclRefAttr reference : capturedRefs) {
    auto [_, inserted] =
        aliases.insert({reference.getName().getValue(), reference.getType()});
    (void)inserted;
  }

  SmallPtrSet<StringAttr, 8> seen;
  for (ParamDeclAttr param : allStructParams)
    seen.insert(param.getName());
  for (auto capturedParam : paramCaptures) {
    if (byValueCapturedOriginParamNames.contains(capturedParam.getName()))
      continue;
    if (!seen.insert(capturedParam.getName()).second)
      continue;
    allStructParams.push_back(
        ParamDeclAttr::get(capturedParam.getName(), capturedParam.getType()));
    structParamBindings.push_back(capturedParam);
  }

  TypedAttr witnessTable = addWitnessTablesToClosure(
      moduleDecl, nestedFnDecl.getLoc(), closureParents,
      SymbolRefAttr::get(
          ctx,
          getFlattenedSymbolName(getFullyResolvedSymbolRefUpTo<FileModuleOp>(
              cast<mlir::SymbolOpInterface>(parent.getOperation())))),
      aliases, std::move(fieldDecls), std::move(allStructParams),
      std::move(structParamBindings), closureType.getName(), isRegPassable);

  // The nested function's DISubroutineType only reflects user-visible
  // parameters. Add the  closure self argument.
  DebugInfo::DISubprogramAttr originalSubprogram =
      nestedFn.getSubprogramScope();
  DebugInfo::DISubprogramAttr closureSubprogram = originalSubprogram;
  if (closureSubprogram) {
    auto subroutineType =
        cast<DebugInfo::DISubroutineType>(closureSubprogram.getType());
    SmallVector<DebugInfo::DIType> updatedArgTypes;
    updatedArgTypes.push_back(
        DebugInfo::DIUnresolvedMLIRType::get(closureType));
    llvm::append_range(updatedArgTypes, subroutineType.getArgumentTypes());
    auto newSubroutineType = DebugInfo::DISubroutineType::get(
        ctx, subroutineType.getCallingConvention(), updatedArgTypes,
        subroutineType.getResultTypes());
    closureSubprogram = DebugInfo::DISubprogramAttr::get(
        closureSubprogram.getCompileUnit(), closureSubprogram.getScope(),
        closureSubprogram.getSourceName(), closureSubprogram.getLinkageName(),
        closureSubprogram.getFile(), closureSubprogram.getLine(),
        closureSubprogram.getScopeLine(),
        closureSubprogram.getSubprogramFlags(),
        cast<DebugInfo::DISubroutineType>(newSubroutineType));
  }

  auto closure = LIT::ClosureInitOp::create(
      builder, opLoc, refType, closureBodySignature, nestedFn.getFunctionType(),
      ValueRange(captureValues), ArrayAttr::get(ctx, captureInfo),
      nestedFn.getInputParams(), nestedFn.getInlineLevel(), origin,
      witnessTable, ArrayAttr::get(ctx, captureTypes),
      ArrayAttr::get(ctx, captureNames));
  // TODO: remove closure type from op
  Type concreteStructType = witnessTable.getType();
  if (auto typeParam = dyn_cast<TypeParamAttr>(witnessTable))
    concreteStructType = typeParam.getMlirType();

  // ClosureInitOp result type still wraps ClosureType so that
  // OutlineClosuresNew can extract the ClosureType via getClosureType().  A
  // RebindOp inserted after the init bridges callers that expect the explicit
  // StructInstanceType.
  auto structRefType =
      RefType::get(concreteStructType, ParamDeclRefAttr::get(origin));
  auto rebind = RebindOp::create(builder, structRefType, closure.getResult());

  // Transfer optional attributes from the nested function to the closure
  // op.
  if (closureSubprogram)
    closure.setNestedFnScopeAttr(closureSubprogram);
  if (ArrayAttr metadata = nestedFn.getLLVMMetadataArray();
      metadata && !metadata.empty())
    closure.setLLVMMetadataArrayAttr(metadata);
  if (ArrayAttr argMetadata = nestedFn.getLLVMArgMetadataArray();
      argMetadata && !argMetadata.empty())
    closure.setLLVMArgMetadataArrayAttr(argMetadata);
  if (LinkageNameAttr linkageName = nestedFn.getLinkageNameAttr())
    closure.setLinkageNameAttr(linkageName);
  llvm::SmallSetVector<ParamDeclAttr, 8> hoistedDecls;
  for (ParamDeclRefAttr capturedRef : paramCaptures)
    hoistedDecls.insert(ParamDeclAttr::get(capturedRef));
  if (!hoistedDecls.empty())
    closure.setHoistedCapturesAttr(
        ParamDeclArrayAttr::get(ctx, hoistedDecls.getArrayRef()));

  closure.getBodyRegion().takeBody(nestedFn.getBodyRegion());

  // The body ops still reference the original subprogram in their locations.
  // Update them to reference the new subprogram with the closure self arg.
  if (closureSubprogram && closureSubprogram != originalSubprogram) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](DebugInfo::DISubprogramAttr sp) -> DebugInfo::DISubprogramAttr {
          if (sp == originalSubprogram)
            return closureSubprogram;
          return sp;
        });
    replacer.recursivelyReplaceElementsIn(closure, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true);
  }

  // (2) Create the wrapper instance and populate it with the closure init op
  // value.

  // The wrapper takes ownership of the closure.
  OwnershipUseOp::create(builder, location, closure);

  // Create the wrapper instance by emitting a call to the Wrapper
  // constructor. The wrapper struct only has impl and origin_set parameters;
  // alias values are derived via GetWitnessAttr lookups on impl.
  auto originSet = OriginSetAttr::get(ctx, ArrayRef<TypedAttr>{});
  SmallVector<TypedAttr> paramArgs({witnessTable, originSet});

  LIT::StructType closureWrapperType = wrapper.bindReference(paramArgs);
  VarDeclOp var = VarDeclOp::create(
      builder, location, closureWrapperType, fnName.getValue(),
      nestedFnDecl.getParentDecl()->mangleParamName(fnName.getValue()),
      VarDeclKind::Var);
  SmallVector<Value> operands({rebind.getResult(), var});
  SmallVector<TypedAttr> implicitOrigins(
      {ParamDeclRefAttr::get(origin), var.getType().getOrigin()});
  FnOp init = getInit(wrapper);
  SymbolRefAttr symbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(init.getOperation()));
  FnTypeGeneratorType fullSig =
      LIT::getFullSignature(wrapper, init.getFuncTypeGenerator());
  auto boundSig = fullSig.getSpecializedGenerator(
      paramArgs, /*evaluationContext=*/nullptr, location);
  TypedAttr symbol = SymbolConstantAttr::get(symbolRef, boundSig, paramArgs);
  LIT::CallOp::create(builder, location, boundSig.getBody().getResults(),
                      symbol, implicitOrigins, operands);
  return MLValue(var);
}

static CValue ASTDeclToCValue(ASTDecl *decl, OpBuilder &builder, Location loc) {
  if (!decl)
    return {};
  if (auto cv = decl->getIfIRValue()) {
    return cv;
  } else if (auto var = dyn_cast_or_null<VarDeclOp>(decl->getIfOperation())) {
    if (!sugarIsa<RefType>(var.getType()))
      return SRValue(var);
    Value value = var;
    if (var.getKind() == VarDeclKind::Ref)
      value = RefLoadOp::create(builder, loc, var);
    return CValue::getMValueForRef(value);
  }
  return {};
}

ASTDecl *ClosureEmitter::addCaptureValue(SharedState &shared, ASTDecl &closure,
                                         StringRef name, SMLoc location) {
  CaptureConvention capture = shared.defaultCaptureConventionInScope(closure);
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  IREmitter emitter(*closure.getParentDecl(), OpBuilder(funcOp));
  return ClosureEmitter::addCaptureValue(closure, location, name, capture,
                                         emitter);
}

// Lookup the decl in the named decls that have been collected thus far. This
// may be an incomplete list because we have not finished resolving the scope.
static FailureOr<ASTDecl *> partialLookup(StringAttr name, ASTDecl &scope,
                                          llvm::SMLoc loc) {
  for (auto [declName, list] : scope.getDeclsInScope()) {
    if (name == declName) {
      if (list.size() != 1) {
        scope.getShared().emitError(loc, "ambiguous captured value: ") << name;
        return failure();
      }
      return list.front();
    }
  }
  return nullptr;
}

// Search the scope and its parents for a decl with the name without resolving
// anything.
static FailureOr<ASTDecl *> findCapture(SharedState &shared, StringRef name,
                                        llvm::SMLoc loc, ASTDecl &scope) {
  auto nameAttr = StringAttr::get(shared.getContext(), name);
  ASTDecl *current = &scope;
  do {
    FailureOr<ASTDecl *> result = partialLookup(nameAttr, *current, loc);
    if (failed(result))
      return failure();
    if (result.value())
      return result.value();
  } while ((current = current->getParentDecl()));
  return nullptr;
}

ASTDecl *ClosureEmitter::addCaptureValue(ASTDecl &closure, SMLoc location,
                                         StringRef name,
                                         CaptureConvention parsedConvention,
                                         IREmitter &emitter,
                                         ASTDecl *signatureDecl) {
  // Check if already emitted.
  SharedState &shared = emitter.shared;
  if (shared.captureInstanceExistsInScope(closure, name)) {
    auto nameAttr = StringAttr::get(shared.getContext(), name);
    ArrayRef<ASTDecl *> existing = closure.lookupInCurrentScope(nameAttr);
    assert(existing.size() == 1 &&
           "if the capture instance exists in the scope then it should have "
           "been registered in the scope");
    return existing.front();
  }
  // If this is a nested closure, emit the parent capture first.
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  ASTDecl *fnParentDecl = closure.getParentDecl()->getNearestDeclOfType<FnOp>();
  auto parentFn = cast<FnOp>(fnParentDecl->getIfOperation());
  ASTDecl *result = nullptr;
  if (usesClosurePipeline(parentFn))
    result = addCaptureValue(shared, *fnParentDecl, name, location);

  if (!result) {
    auto hitMaybe = partialLookup(StringAttr::get(shared.getContext(), name),
                                  closure, location);
    if (failed(hitMaybe))
      return nullptr;
    // No need to emit a capture instance since this closure defines the
    // value.
    if (hitMaybe.value())
      return hitMaybe.value();

    // otherwise, this is a capture. Find the def.
    auto maybeResult =
        findCapture(shared, name, location, *closure.getParentDecl());
    if (failed(maybeResult))
      return nullptr;
    result = maybeResult.value();
    if (!result) {
      shared.emitError(location, "reference to an unknown value: ") << name;
      return nullptr;
    }
    if (auto pval = result->getIfIRValue().getIfPValue()) {
      shared.emitError(location, "value ")
          << name << " is a parameter and does not need a capture convention";
      return nullptr;
    }
  }

  CValue valueInParent =
      ASTDeclToCValue(result, *emitter.builder, funcOp->getLoc());
  emitter.builder->setInsertionPoint(closure.getIfOperation());

  CaptureConvention convention;
  /// The captureValue is a map of the valueInParent. For example, the
  /// valueInParent may be an immutable borrowed value. If this value is
  /// captured by copy the capturedValue in the body of the closure is a
  /// mutable owned value. Since the captured value does not exist until
  /// later, we have to create a temporary value to represent the change in
  /// the properties of the value in the body of the closure.
  CValue captureValue;
  // Switch the DI Scope to the enclosing function before emitting the
  // load so the debug information is accurate.
  DebugInfo::DIBuilder::ScopeGuard diGuard;
  if (shared.diBuilder)
    diGuard = shared.diBuilder->pushScopeGuard(parentFn.getLocScope());

  auto captureByRef = [&](CValue value,
                          std::optional<bool> mutability) -> CValue {
    // Ensure we are not capturing an immutable reference by mutable
    // reference.
    if (auto refType = dyn_cast<RefType>(value.getType().mlirType)) {
      // If the mutability is not specified or the reference type match the
      // specified mutability, return the original value.
      OriginType originType = refType.getOriginType();
      if (!mutability.has_value() || originType.isMutableKnown(*mutability))
        return value;

      if (originType.isMutableKnown(false)) {
        // mutable capture of an immutable reference, error.
        shared.emitError(location, "Cannot capture ")
            << name << " by mut because it could be immutable";
        return {};
      }

      if (originType.isMutableKnown(true)) {
        // convert a mut ref to immut ref
        auto refImmutOp = LIT::RefImmutOp::create(
            *emitter.builder, parentFn.getLoc(), valueInParent.getMlirValue());
        return MBValue(refImmutOp->getResult(0));
      }
    }

    // Not a reference capture, then it must be a read effect.
    if (mutability.has_value() && *mutability == false)
      return value;

    shared.emitError(location, "register passible value '")
        << name << "' can not be captured by "
        << (mutability.has_value() ? "'mut'" : "'ref'")
        << ". Do you mean 'read'?";
    return {};
  };

  switch (parsedConvention) {
  case CaptureConvention::kConventionMove: {
    Type type = valueInParent.getType().mlirType;
    if (auto ref = dyn_cast<RefType>(valueInParent.getType().mlirType))
      type = ref.getElementType();
    if (!ASTType(type).isMovable(closure.getLoc(), shared)) {
      shared.emitError(location, "Cannot capture ")
          << name << " by move because the type is not movable";
      return nullptr;
    }
    if (valueInParent.getIfBValue()) {
      shared.emitError(location, "Cannot capture")
          << name << " by move because the value is read only";
      return nullptr;
    }
    // If it was captured by move then there was a transfer operation.
    convention = parsedConvention;
    valueInParent = MRValue(valueInParent.getMlirValue());
    captureValue = valueInParent;
    [[fallthrough]];
  }
  case CaptureConvention::kConventionCopy: {
    ASTType originalType = valueInParent.getRValueType();
    if (originalType.isTrivial(closure.getLoc(), shared)) {
      // Remap to trivial copy convention to avoid storing symbols.
      convention = CaptureConvention::kConventionTrivialCopy;
      // if we are capturing by mutable copy and its trivial do not capture
      // the reference.
      if (isa<RefType>(valueInParent.getType())) {
        SyntheticNode node(result->getLoc());
        ExprDest dest(EC_Capture);
        captureValue = emitter.emitRValue(
            {CValue::getMValueForRef(valueInParent.getMlirValue()), node},
            dest);
      } else {
        captureValue = valueInParent;
      }
    } else {
      convention = parsedConvention;
      if (auto refType = dyn_cast<RefType>(valueInParent.getType().mlirType)) {
        OriginType originType = refType.getOriginType();
        if (originType.isMutableKnown(false)) {
          Location fusedLoc =
              FusedLoc::get(funcOp.getLoc().getContext(), funcOp.getLoc(),
                            parentFn.getSubprogramScope());
          auto refImmutOp = LIT::RefImmutOp::create(
              *emitter.builder, fusedLoc, valueInParent.getMlirValue());
          captureValue = MBValue(refImmutOp->getResult(0));
        }
      }
      ExprDest dest(EC_Capture);
      SyntheticNode node(result->getLoc());
      ASTExprAnd<CValue> valueInParentExpr{valueInParent, node};
      LValue copiedOrMovedValue =
          dest.getLValueForResult(valueInParentExpr.expr->getLoc(),
                                  valueInParentExpr.ir.getRValueType(),
                                  /*allowIncompatibleTypes=*/false,
                                  /*requireMLValue=*/false, emitter);
      emitter.emitStoreToLValue(valueInParentExpr, copiedOrMovedValue,
                                dest.getContext());
      captureValue = copiedOrMovedValue;
    }
    break;
  }
  case CaptureConvention::kConventionMut:
  case CaptureConvention::kConventionRead:
  case CaptureConvention::kConventionRef: {
    convention = parsedConvention;
    auto mutability = [convention]() -> std::optional<bool> {
      if (convention == CaptureConvention::kConventionRef)
        return std::nullopt;
      return convention == CaptureConvention::kConventionMut;
    }();
    captureValue = captureByRef(valueInParent, mutability);
    if (!captureValue)
      return nullptr;
    break;
  }
  default:
    llvm_unreachable("All capture conventions should be handled above");
    break;
  }
  assert(captureValue && "must set capture value");
  // Ensure the capture value we created is used when parsing the body of the
  // closure.
  ASTDecl &captureValueDecl = shared.getDeclResolver().addFullyResolvedDecl(
      captureValue, name, closure.getLoc(),
      signatureDecl ? signatureDecl : &closure);
  shared.addCaptureToScope(closure, result,
                           Capture(captureValue, convention, name));
  return &captureValueDecl;
}

/// If an alias is already bound, verify the new
/// value is consistent with the existing binding. Returns false if
/// inconsistent.
static bool tryRecordSubstitution(AliasSubstitutions &substitutions,
                                  StringAttr aliasName, TypedAttr newValue) {
  if (!newValue)
    return false;
  auto it = substitutions.find(aliasName);
  if (it != substitutions.end()) {
    Type existingType = it->second.getType();
    Type newType = newValue.getType();
    if (auto existingParam = dyn_cast<TypeParamAttr>(it->second))
      existingType = existingParam.getTypeValue();
    if (auto newParam = dyn_cast<TypeParamAttr>(newValue))
      newType = newParam.getTypeValue();
    return isEqualCanon(existingType, newType);
  }
  substitutions[aliasName] = newValue;
  return true;
}

namespace M::KGEN::LIT {

struct AuxiliaryParameters {
  size_t startingIndex;
  size_t numStructAuxiliaryParams;
  SmallVector<ParamDeclAttr> traitAuxiliaryParameters;
  SmallVector<TypedAttr> structAliases;
  SmallVector<StringAttr> traitAliases;

  TypedAttr getAliasRef(size_t index) {
    return get<TypedAttr>(index, structAliases);
  }

private:
  template <typename Result>
  Result get(size_t index, SmallVector<Result> &container) {
    if (index < startingIndex)
      return Result{};
    size_t auxIdx = index - startingIndex;
    if (auxIdx >= container.size())
      return Result{};
    return container[auxIdx];
  }
};

} // namespace M::KGEN::LIT

namespace {

static TypedAttr getUnderlyingParamRef(TypedAttr attr) {
  if (auto upcast = dyn_cast<UpcastAttr>(attr))
    return getUnderlyingParamRef(upcast.getInputTypeValue());

  if (auto typeParam = dyn_cast<TypeParamAttr>(attr)) {
    if (auto paramType = dyn_cast<ParamType>(typeParam.getTypeValue()))
      return getUnderlyingParamRef(paramType.getParam());
  }

  if (isa<ParamDeclRefAttr, ParamIndexRefAttr>(attr))
    return attr;

  return {};
}

// Given a type parameter that wraps a strong type than its type, convert it to
// an upcast, which explicitly communicates the relationship between the
// underlying parameter and the expected type.
static TypedAttr makeExplicitUpcastBinding(TypedAttr binding) {
  if (auto typeParam = dyn_cast<TypeParamAttr>(binding)) {
    if (auto paramType = dyn_cast<ParamType>(typeParam.getTypeValue())) {
      if (auto paramRef = dyn_cast<ParamDeclRefAttr>(paramType.getParam())) {
        if (!isEqualCanon(paramRef.getType(), typeParam.getType()))
          return UpcastAttr::get(typeParam.getType(), paramRef);
      }
    }
  }
  return binding;
}
struct ConformanceTableEntryMapper {
  struct Result {
    // the conformance table entry
    TypedAttr binding;
    // The name of the parameter that violates no escaping parameter rule. Used
    // for error messages.
    StringAttr escapedParamName;
  };

  // Maps bindings inferred against the actual struct method into the trait
  // conformance table's auxiliary parameter space.
  //
  // For example, suppose we have:
  //
  //   struct X:
  //     alias A: Coord
  //     def __call__[_A: Coord](self, x: Cartesian, z: _A):
  //         pass
  //
  // and:
  //
  //   trait Y:
  //     alias T: Coord
  //     alias R: Coord
  //     def __call__[_T: Coord, _R: Coord](self, y: _T, z: _R):
  //         ...
  //
  // Specialization inference produces bindings in the actual method's
  // parameter space, here {Cartesian, _A}. To populate the conformance table,
  // those bindings must be rewritten into the trait/struct alias space,
  // yielding {Cartesian, Self.A} for {T, R}.
  ConformanceTableEntryMapper(FnOp actualFn, AuxiliaryParameters &ctx) {
    StructDeclOp structDeclOp = actualFn->getParentOfType<StructDeclOp>();
    for (ParamDeclAttr structParam : structDeclOp.getInputParams())
      allowedConformanceScopeParams.insert(structParam.getName());

    FnTypeGeneratorType actualSig = actualFn.getFuncTypeGenerator();
    ArrayRef<ParamDeclAttr> actualParams = actualFn.getInputParams().drop_back(
        actualSig.getNumImplicitOriginDecls());
    assert(actualParams.size() >= ctx.numStructAuxiliaryParams &&
           "struct auxiliary params should be present in function signature");
    ArrayRef<ParamDeclAttr> actualAuxiliaryParams =
        actualParams.take_front(ctx.numStructAuxiliaryParams);
    for (auto [offset, auxiliaryParam] :
         llvm::enumerate(actualAuxiliaryParams)) {
      TypedAttr aliasValue = ctx.getAliasRef(ctx.startingIndex + offset);
      if (aliasValue)
        auxiliaryBindingsByName[auxiliaryParam.getName()] = aliasValue;
    }

    walker.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
      if (allowedConformanceScopeParams.contains(paramRef.getName()))
        return paramRef;

      auto it = auxiliaryBindingsByName.find(paramRef.getName());
      if (it == auxiliaryBindingsByName.end()) {
        pendingEscapedParamName = paramRef.getName();
        return paramRef;
      }
      return it->second;
    });
  }

  Result map(TypedAttr binding) {
    pendingEscapedParamName = {};
    TypedAttr mappedBinding = cast<TypedAttr>(walker.replace(binding));
    return {mappedBinding, pendingEscapedParamName};
  }
  // The only parameter references allowed to remain in a conformance table
  // entry are parameters from the enclosing closure struct itself. For closure
  // structs this is typically "impl" or "origin_set"; for top-level closure
  // structs it may also be "symbol".
  bool isAllowedConformanceScopeRef(TypedAttr attr) const {
    TypedAttr paramRef = getUnderlyingParamRef(attr);
    auto declRef = dyn_cast_if_present<ParamDeclRefAttr>(paramRef);
    return declRef && allowedConformanceScopeParams.contains(declRef.getName());
  }

private:
  DenseMap<StringAttr, TypedAttr> auxiliaryBindingsByName;
  DenseSet<StringAttr> allowedConformanceScopeParams;
  mlir::AttrTypeReplacer walker;
  StringAttr pendingEscapedParamName;
};

static bool canFunctionSignatureMatchTraitParamInf(FnOp actualFn,
                                                   FnTypeGeneratorType target,
                                                   AuxiliaryParameters &ctx,
                                                   SharedState &shared,
                                                   AdapteeParts &adapteeParts) {
  FnTypeGeneratorType actualSig = actualFn.getFuncTypeGenerator();
  if (!actualSig.hasMemoryOnlyResult() && target.hasMemoryOnlyResult())
    adapteeParts.needsResultConversion = true;
  else if (actualSig.hasMemoryOnlyResult() != target.hasMemoryOnlyResult())
    return false;
  if (actualSig.getFnEffects() != target.getFnEffects())
    return false;

  ArrayRef<Type> actualExplicitParams =
      actualSig.getInputParamTypes().drop_front(ctx.numStructAuxiliaryParams);
  ArrayRef<Type> targetExplicitParams = target.getInputParamTypes().drop_front(
      ctx.traitAuxiliaryParameters.size());
  SMLoc loc = shared.getTopLevelDecl().getLoc();
  SyntheticNode syntheticExpr(loc);

  SpecializeInf inference(shared.getTopLevelDecl(), &syntheticExpr,
                          target.getInputParamTypes(),
                          target.getParamListAttrs(), loc,
                          /*discardError=*/true);
  if (actualExplicitParams.size() != targetExplicitParams.size())
    return false;
  ParamRefRemapper remapper(actualFn.getInputParams());
  size_t actualAuxCount = ctx.numStructAuxiliaryParams;
  size_t targetAuxCount = ctx.traitAuxiliaryParameters.size();
  for (auto [index, actualParamType, targetParam] :
       llvm::enumerate(actualExplicitParams, targetExplicitParams)) {
    Type actualParam = remapper.replace(actualParamType);
    if (!isEqualCanon(actualParam, targetParam))
      return false;

    StringAttr actualParamName =
        actualFn.getInputParams()[index + actualAuxCount].getName();
    if (failed(inference.setInitialInferredValue(
            index + targetAuxCount,
            ParamDeclRefAttr::get(actualParamName, actualParam))))
      return false;
  }
  FailureOr<SmallVector<TypedAttr>> specialization =
      inference.inferSpecialization(target, actualFn);
  if (failed(specialization))
    return false;

  // Walk each target trait aux specialization. The inference produces these
  // bindings in the *wrapper's __call__* parameter space (e.g. _a + _b);
  // we need them in the struct space to call from the adaptee.
  ConformanceTableEntryMapper createConformanceTableEntry(actualFn, ctx);
  unsigned targetAuxStart = ctx.startingIndex;
  for (auto [offset, aliasAndParam] : llvm::enumerate(
           llvm::zip(ctx.traitAliases, ctx.traitAuxiliaryParameters))) {
    auto [aliasName, auxiliaryParameter] = aliasAndParam;
    TypedAttr rawBinding = (*specialization)[targetAuxStart + offset];
    if (!rawBinding || isa<UnboundAttr>(rawBinding))
      return false;

    rawBinding = makeExplicitUpcastBinding(rawBinding);
    auto mappedBinding = createConformanceTableEntry.map(rawBinding);
    if (mappedBinding.escapedParamName) {
      auto &error = inference.getMojoDiag(loc);
      error << "closure conformance alias '" << aliasName
            << "' cannot reference parameter "
            << mappedBinding.escapedParamName;

      inference.diag.release();
      return false;
    }

    if (!tryRecordSubstitution(adapteeParts.aliasSubstitutions, aliasName,
                               mappedBinding.binding))
      return false;

    // The adaptor's block-argument types reference target trait aux params.
    // Rewrite them into struct-level expressions so that, after symbol
    // binding (which substitutes the wrapper's __call__ aux with the same
    // struct-level expressions), the operand types match the callee's
    // expected types.
    adapteeParts.adapteeTypeMap[auxiliaryParameter.getName()] =
        mappedBinding.binding;
  }

  // The wrapper's __call__ has one type-level aux per struct alias (1:1 by
  // construction in createFnStructWrapper). Bind each one to its struct
  // alias value
  adapteeParts.fnLevelBindings.reserve(ctx.numStructAuxiliaryParams +
                                       targetExplicitParams.size());
  for (size_t offset = 0; offset < ctx.numStructAuxiliaryParams; ++offset) {
    TypedAttr aliasValue = ctx.getAliasRef(ctx.startingIndex + offset);
    if (!aliasValue)
      return false;
    adapteeParts.fnLevelBindings.push_back(aliasValue);
  }
  for (auto [index, explicitParamType] :
       llvm::enumerate(targetExplicitParams)) {
    StringAttr explicitParamName = target.getParamName(index + targetAuxCount);
    adapteeParts.fnLevelBindings.push_back(
        ParamDeclRefAttr::get(explicitParamName, explicitParamType));
  }

  return true;
}

} // namespace

void ClosureEmitter::buildCallAdaptorAndAddWitness(
    StructDeclOp structDeclOp, ASTDecl &structDecl, TraitDeclOp traitDeclOp,
    FnOp traitCallFn, FnOp structCallFn, const AdapteeParts &adapteeParts) {
  SharedState &shared = structDecl.getShared();
  MLIRContext *ctx = shared.getContext();
  ArrayRef<ParamDeclAttr> structParams = structDeclOp.getInputParams();
  assert(!structParams.empty() && "closure wrapper should have impl param");

  SymbolRefAttr traitSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation()));
  StringAttr adaptorNameAttr =
      StringAttr::get(ctx, "__call__$" + getFlattenedSymbolName(traitSymbol));
  auto [adaptorFnOp, adaptorParams, adaptorResult] =
      pushBackTraitFunctionImpl(traitCallFn, structDecl, true, adaptorNameAttr);
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr ref) -> TypedAttr {
    auto ptr = adapteeParts.adapteeTypeMap.find(ref.getName());
    if (ptr == adapteeParts.adapteeTypeMap.end())
      return ref;
    return ptr->second;
  });
  // Populate the adaptor body: rebind arguments and call original
  ImplicitLocOpBuilder b(adaptorFnOp.getLoc(), adaptorFnOp);
  b.setInsertionPointToEnd(&adaptorFnOp.getBodyRegion().front());
  SmallVector<Value> callOperands;
  SmallVector<TypedAttr> origins;
  Block &adaptorBlock = adaptorFnOp.getBodyRegion().front();
  SmallVector<Type> expectedTypes;
  expectedTypes.reserve(adaptorBlock.getNumArguments());
  for (BlockArgument arg : adaptorBlock.getArguments())
    expectedTypes.push_back(replacer.replace(arg.getType()));
  auto symbol = buildSymbolWithBindings(structCallFn, structParams,
                                        adapteeParts.fnLevelBindings);
  auto symbolSigGen = cast<FnTypeGeneratorType>(symbol.getType());
  auto calleeConventions = symbolSigGen.getArgConventions();

  for (auto [arg, targetType, conv] : llvm::zip(
           adaptorBlock.getArguments(), expectedTypes, calleeConventions)) {
    Value operand = arg;
    if (targetType != arg.getType())
      operand = RebindOp::create(b, targetType, operand);

    // Handle convention mismatches between the adaptor (trait signature) and
    // the callee (wrapper's __call__). Generic trait parameters use ReadMem
    // (ref), but concrete RegisterPassable types use ReadReg (value).
    if (!hasImplicitOrigin(conv) && isa<RefType>(operand.getType()))
      operand = RefLoadOp::create(b, operand);

    callOperands.push_back(operand);
    if (hasImplicitOrigin(conv))
      origins.push_back(cast<RefType>(operand.getType()).getOrigin());
  }
  auto callOp = LIT::CallOp::create(b, symbolSigGen.getResultType(), symbol,
                                    origins, callOperands);
  Value result = callOp.getResult(0);

  if (adapteeParts.needsResultConversion) {
    // The callee returns in-register but the adaptor expects a memory-only
    // result. Store the register value into the ByRefResult slot.
    Value resultSlot = adaptorBlock.getArguments().back();
    Type concreteSlotType = replacer.replace(resultSlot.getType());
    if (concreteSlotType != resultSlot.getType())
      resultSlot = RebindOp::create(b, concreteSlotType, resultSlot);
    RefStoreOp::create(b, result, resultSlot);
    IREmitter::emitNormalReturn(b);
  } else {
    Type resultType =
        cast<FnTypeGeneratorType>(symbol.getType()).getResultType();
    if (resultType != adaptorResult)
      result = RebindOp::create(b, adaptorResult, result);
    IREmitter::emitNormalReturn(b, result);
  }

  // Build the witness using the adaptor function
  SymbolConstantAttr adaptorSymbol = buildSymbol(adaptorFnOp, structParams);
  SmallVector<std::pair<StringRef, TypedAttr>> witnesses;
  witnesses.emplace_back(traitCallFn.getSymNameAttr(), adaptorSymbol);
  for (auto &[aliasName, aliasValue] : adapteeParts.aliasSubstitutions)
    witnesses.emplace_back(aliasName.getValue(), aliasValue);

  ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
  addConformanceTable(structDecl,
                      ClosureEmitter::ClosureParent(traitDeclOp, traitCallFn,
                                                    ClosureMethod::CALL),
                      witnesses, fileModule);
}

LogicalResult ClosureEmitter::checkStructCompatibility(ASTType structType,
                                                       ASTDecl *traitDecl,
                                                       bool rebind) {
  // Ensure that we have a valid closure trait and a struct metatype.
  TraitDeclOp traitDeclOp =
      llvm::dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
  if (!traitDeclOp)
    return failure();
  if (!traitDeclOp.getDefinesClosure())
    return failure();
  ASTDecl &structDecl = *structType.getDecl(shared);
  if (!structDecl.getIfOperation())
    return failure();

  StructDeclOp structDeclOp =
      dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  if (!structDeclOp)
    return failure();

  // does the struct already conform to the trait?
  SymbolRefAttr target = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation()));
  for (SymbolRefAttr currentTrait :
       structDeclOp.getCanonicalTrait().getSymbols()) {
    if (target == currentTrait) {
      return success();
    }
  }

  // This trait defines a closure which means it has a single call function.
  if (structDecl.resolvedness < DeclResolvedness::body) {
    if (failed(
            shared.declResolver->resolveBody(structDecl, structDecl.getLoc())))
      return failure();
  }
  StringRef name = "__call__";
  auto callDecls = structDecl.lookupInCurrentScope(name);
  // Resolve signatures for all __call__ declarations before creating the
  // OverloadSet, which requires DeclResolvedness::signature.
  for (ASTDecl *callDecl : callDecls) {
    if (failed(shared.declResolver->resolveSignature(*callDecl,
                                                     structDecl.getLoc())))
      return failure();
  }
  FnOp callFunction = getFnOpNamed(traitDeclOp, name);
  // get the call function in terms of the struct wrapper
  SyntheticNode syntheticNode(structDecl.getLoc());
  ASTType structSelfType = structDecl.getTypeDeclSelf();
  IREmitter emitter(structDecl, EC_Trait);
  FnTypeGeneratorType traitSignature = specializeSignature(
      callFunction, structSelfType.mlirType, *shared.declResolver);

  auto bindings = ParamBindings::getForDeclaredType(
      emitter.getDeclScope(), structSelfType, syntheticNode);
  // This could be a parametric function, we don't need to bind the parameter on
  // the function to test the compatibility.
  bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);
  OverloadSet ov(name, callDecls, std::move(bindings),
                 CallSyntax::kMethodCallSynthetic);
  /// Perform rebind on method that implements the trait function but with
  /// different argument names.
  auto [newWitness, _] = ov.filterOverloadSetForValueType(
      traitSignature, emitter.getDeclScope(), nullptr);
  if (newWitness) {
    if (rebind) {
      ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
      // Build witnesses including alias mappings.
      SmallVector<std::pair<StringRef, TypedAttr>> witnesses;
      witnesses.emplace_back(callFunction.getSymNameAttr(), newWitness.get());
      auto traitAliases = traitDeclOp.getFields().getOps<AliasDeclOp>();
      auto structAliases = structDeclOp.getFields().getOps<AliasDeclOp>();
      for (auto [traitAlias, structAlias] :
           llvm::zip(traitAliases, structAliases)) {
        StringRef aliasName = traitAlias.getParamDecl().getName().getValue();
        TypedAttr aliasValue = *structAlias.getValue();
        witnesses.emplace_back(aliasName, aliasValue);
      }
      addConformanceTable(structDecl,
                          ClosureEmitter::ClosureParent(
                              traitDeclOp, callFunction, ClosureMethod::CALL),
                          witnesses, fileModule);
    }

    return success();
  }

  // Exact Matching Failed. Check if we can conform to a trait by declaring
  // alias members. This requires conformance checked substitution.
  if (callDecls.empty())
    return failure();

  // Collect closure-specific alias names. Inherited AliasDeclOps (e.g.
  // `__del__is_trivial`) are cloned into the trait's fields by lazy body
  // resolution and are marked with `inheritedFrom`; skip them.
  SmallVector<StringAttr> traitAliasOps;
  for (AliasDeclOp aliasOp : traitDeclOp.getFields().getOps<AliasDeclOp>()) {
    if (aliasOp.getInheritedFrom())
      continue;
    traitAliasOps.push_back(aliasOp.getParamDecl().getName());
  }
  size_t traitAliasCount = traitAliasOps.size();
  SmallVector<ParamDeclAttr> auxiliaryParams;
  for (ParamDeclAttr auxiliaryParam :
       callFunction.getInputParams().take_front(traitAliasCount))
    auxiliaryParams.push_back(auxiliaryParam);
  unsigned startingIndex = 0;
  SmallVector<TypedAttr> structAliasOps;
  llvm::DenseSet<TypedAttr> uniqueNonInheritedAliasValues;
  for (AliasDeclOp aliasOp : structDeclOp.getFields().getOps<AliasDeclOp>()) {
    TypedAttr value = *aliasOp.getValue();
    structAliasOps.push_back(value);
    if (aliasOp.getInheritedFrom())
      continue;
    uniqueNonInheritedAliasValues.insert(value);
  }

  AuxiliaryParameters auxCtx{
      startingIndex, uniqueNonInheritedAliasValues.size(),
      std::move(auxiliaryParams), std::move(structAliasOps),
      std::move(traitAliasOps)};
  for (ASTDecl *callDecl : callDecls) {
    auto structCallFn = dyn_cast_or_null<FnOp>(callDecl->getIfOperation());
    if (!structCallFn)
      continue;
    if (failed(shared.declResolver->resolveSignature(*callDecl,
                                                     structDecl.getLoc())))
      continue;

    AdapteeParts adapteeParts;
    if (canFunctionSignatureMatchTraitParamInf(structCallFn, traitSignature,
                                               auxCtx, shared, adapteeParts)) {
      if (rebind)
        buildCallAdaptorAndAddWitness(structDeclOp, structDecl, traitDeclOp,
                                      callFunction, structCallFn, adapteeParts);

      return success();
    }
  }

  return failure();
}

LogicalResult
ClosureEmitter::augmentWitnessTablesToConformTo(ASTType structType,
                                                ASTDecl *traitDecl) {
  return checkStructCompatibility(structType, traitDecl, true);
}

LogicalResult ClosureEmitter::isCompatibleWith(ASTType structType,
                                               ASTDecl *traitDecl) {
  return checkStructCompatibility(structType, traitDecl, false);
}

void ClosureEmitter::addConformanceToDevicePassable(
    ASTDecl &structDecl, StructFieldOp devicePassedField, ParamDeclAttr impl,
    ParamDeclAttr originSet) {
  Type paramType = ParamType::get(ParamDeclRefAttr::get(impl));
  ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
  ASTDecl *devicePassableTrait =
      shared.getBuiltinDevicePassableTrait(structDecl.getLoc());
  if (!devicePassableTrait)
    return;
  // Ensure body is parsed and unresolved decls pulled in
  if (failed(shared.declResolver->resolveBody(*devicePassableTrait,
                                              devicePassableTrait->getLoc())))
    return;
  TraitDeclOp trait = cast<TraitDeclOp>(devicePassableTrait->getIfOperation());
  SymbolRefAttr devicePassableSymbol = devicePassableTrait->getSymbolRef();
  Type deviceTypeAliasType;

  // Resolve top-level members and collect the `device_type` alias type.
  for (auto &nameGroup : devicePassableTrait->getDeclsInScope()) {
    for (ASTDecl *funcFieldOrAlias : nameGroup.second) {
      if (failed(shared.declResolver->resolveBody(*funcFieldOrAlias,
                                                  funcFieldOrAlias->getLoc())))
        return;
      if (auto aliasOp = dyn_cast_if_present<AliasDeclOp>(
              funcFieldOrAlias->getIfOperation());
          aliasOp && aliasOp.getDeclName().getValue() == kDeviceType)
        deviceTypeAliasType = aliasOp.getType();
    }
  }

  assert(deviceTypeAliasType &&
         "DevicePassable trait should define device_type alias");
  SmallVector<std::pair<StringRef, TypedAttr>> devicePassableWitnesses;
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp);

  for (Operation &member : trait.getFields().getOps()) {
    if (auto function = dyn_cast<FnOp>(member)) {
      if (function.getSourceName() == kIsDeviceTypeConvertible) {
        auto [implementation, parameters, result] =
            pushBackTraitFunctionImpl(function, structDecl);
        b.setInsertionPointToStart(&implementation.getBodyRegion().front());
        assert(
            !parameters.empty() &&
            "expected _is_convertible_to_device_type to have type parameter");
        TypedAttr targetType = ParamDeclRefAttr::get(parameters.front());
        TypedAttr selfType =
            cast<TypedAttr>(PValue(structDecl.getTypeDeclSelf()).get());
        StringAttr traitName =
            b.getStringAttr(getFlattenedSymbolName(devicePassableSymbol));
        TypedAttr selfDeviceType = GetWitnessAttr::get(
            selfType, traitName, StringAttr::get(ctx, kDeviceType),
            deviceTypeAliasType);
        TypedAttr isConvertible =
            ParamOperatorAttr::get(POC::EQ, targetType, selfDeviceType);
        auto isConvertibleValue =
            KGEN::ParamConstantOp::create(b, isConvertible);
        IREmitter::emitNormalReturn(b, isConvertibleValue);
        devicePassableWitnesses.push_back({
            *function.getSymName(),
            buildSymbol(implementation, impl, originSet),
        });
        continue;
      }
      /// We already have AnyType members implemented, only implement those
      /// that are defined by DevicePassable.
      auto parent = function.getInheritedFrom();
      if (parent && parent != devicePassableSymbol)
        continue;
      if (function.getSourceName() == kToDeviceType) {
        auto [toDevice, params, result] =
            pushBackTraitFunctionImpl(function, structDecl);
        b.setInsertionPointToStart(&toDevice.getBodyRegion().front());
        assert(toDevice.getBodyRegion().getNumArguments() == 3);
        // get address
        Value targetArgument = toDevice.getBodyRegion().front().getArgument(2);
        StructType structType = cast<StructType>(targetArgument.getType());
        assert(structType.getParamValues().size() > 2 &&
               "expected pointer to be parameterized on element type");
        // UnsafePointer parameters:
        //   [mut: Bool, mlir_origin, type: AnyType, origin, ...]
        // The element type is at index 2.
        auto pointerElementType =
            dyn_cast<KGEN::TypeParamAttr>(structType.getParamValues()[2]);
        assert(pointerElementType &&
               "expected the pointer type's second parameter to "
               "indicate its element type");
        Value addressRef =
            StructExtractOp::create(
                b, KGEN::PointerType::get(pointerElementType.getTypeValue()),
                targetArgument, StringAttr::get(ctx, "address"))
                ->getResults()
                .front();
        Value address = POP::PointerBitcastOp::create(
            b, PointerType::get(paramType), addressRef);

        // Build a byref destination from the target address pointer
        auto immortal = b.getAttr<AnyOriginAttr>(/*isMut=*/true);
        Value targetRef = RefFromPointerOp::create(b, address, immortal,
                                                   /*startUninit=*/true,
                                                   /*endUninit=*/false);

        // get closure value
        Value selfArgument = toDevice.getBodyRegion().front().getArgument(0);
        Value closureMemberRef =
            RefStructGEROp::create(b, selfArgument, devicePassedField)
                ->getResults()
                .front();

        // Invoke T(copy=value)
        ASTDecl &moduleDecl = *structDecl.getNearestDeclOfType<FileModuleOp>();
        FnOp copyFn = copyParent.getDefiningOp(moduleDecl);
        FnTypeGeneratorType copySignature =
            specializeSignature(copyFn, paramType, *shared.declResolver);
        StringAttr parentName = copyParent.getFullSymbolName(moduleDecl);
        TypedAttr copySymbol =
            GetWitnessAttr::get(ctx, ParamDeclRefAttr::get(impl), parentName,
                                copyFn.getSymNameAttr(), copySignature);
        SmallVector<Value> operands{closureMemberRef, targetRef};
        SmallVector<TypedAttr> origins;
        origins.push_back(
            cast<RefType>(closureMemberRef.getType()).getOrigin());
        origins.push_back(cast<RefType>(targetRef.getType()).getOrigin());
        LIT::CallOp::create(b, copySignature.getResultType(), copySymbol,
                            origins, operands);
        auto noneAttr = KGEN::ParamConstantOp::create(
            b, KGEN::NoneAttr::get(b.getContext()));
        IREmitter::emitNormalReturn(b, noneAttr);

        devicePassableWitnesses.push_back(
            {*function.getSymName(), buildSymbol(toDevice, impl, originSet)});
        continue;
      }
      /// If this is a static method that returns a string, return the trait's
      /// source name.
      if (function.getIsStatic() &&
          function.getUserResultType() ==
              shared.lookupBuiltinType("String", structDecl,
                                       structDecl.getLoc())) {
        auto [implementation, parameters, result] =
            pushBackTraitFunctionImpl(function, structDecl);
        b.setInsertionPointToStart(&implementation.getBodyRegion().front());
        // Initialize the byref String result with the literal "closure".
        Block &block = implementation.getBodyRegion().front();
        OpBuilder ob(&block, block.begin());
        IREmitter emitter(structDecl, ob);
        SyntheticNode loc(structDecl.getLoc());

        // Build a StringLiteral["closure"] value.
        auto closureStr = StringAttr::get("closure", StringType::get(ctx));
        ASTType strLitType = shared.lookupBuiltinType(
            "StringLiteral", structDecl, structDecl.getLoc());
        auto strLitDecl =
            cast<StructDeclOp>(strLitType.getDecl(shared)->getIfOperation());
        Type boundStrLitType = strLitDecl.bindReference({closureStr});
        CValue literalValue = emitter.emitConstructorCall(
            ASTType(boundStrLitType),
            CallOperands(CallSyntax::kTypeCall, &loc, EC_CallArgValue));

        // Call String.__init__(literal) into the result slot.
        ASTType stringType =
            shared.lookupBuiltinType("String", structDecl, structDecl.getLoc());
        ExprDest resultDest(MLValue(block.getArguments().back()),
                            EC_ReturnValue);
        CallOperands ctorOperands(CallSyntax::kTypeCall, &loc,
                                  std::move(resultDest));
        ctorOperands.add(ASTExprAnd<CValue>{literalValue, &loc});
        emitter.emitConstructorCall(stringType, std::move(ctorOperands));
        auto noneAttr = KGEN::ParamConstantOp::create(
            b, KGEN::NoneAttr::get(b.getContext()));
        IREmitter::emitNormalReturn(b, noneAttr);
        devicePassableWitnesses.push_back(
            {*function.getSymName(),
             buildSymbol(implementation, impl, originSet)});

        continue;
      }
    }

    if (auto alias = dyn_cast<AliasDeclOp>(member)) {
      auto parent = alias.getInheritedFrom();
      if (parent && parent != devicePassableSymbol)
        continue;
      assert(alias.getDeclName().getValue().contains(kDeviceType) &&
             "we assume we are implementing device_type.");
      devicePassableWitnesses.push_back(
          {kDeviceType,
           TypeParamAttr::get(structDecl.getTypeDeclSelf().mlirType,
                              KGEN::TypeType::get(ctx))});
      continue;
    }
    llvm_unreachable(("unexpected member type '" +
                      member.getName().getStringRef().str() +
                      "' encountered in DevicePassable trait")
                         .c_str());
  }
  ClosureParent devicePassableParent(trait, {}, ClosureMethod::NONE);
  addConformanceTable(structDecl, devicePassableParent, devicePassableWitnesses,
                      fileModule);
}

TraitType ClosureEmitter::getWrapperTraitType(ASTDecl &traitDecl,
                                              ASTDecl &moduleDecl,
                                              bool isCopyable,
                                              TypeConvention typeConvention) {
  SmallVector<SymbolRefAttr> symbols;
  symbols.push_back(traitDecl.getSymbolRef());
  symbols.push_back(moveParent.getSymbolRef(moduleDecl));
  symbols.push_back(implicitlyDestructibleParent.getSymbolRef(moduleDecl));
  if (isCopyable) {
    symbols.push_back(copyParent.getSymbolRef(moduleDecl));
    symbols.push_back(implicitlyCopyableParent.getSymbolRef(moduleDecl));
  }

  if (typeConvention == TypeConvention::RegisterPassableTrivial) {
    symbols.push_back(trivialRegisterTypeParent.getSymbolRef(moduleDecl));
    symbols.push_back(registerPassableParent.getSymbolRef(moduleDecl));
    ASTDecl *devicePassableTrait =
        shared.getBuiltinDevicePassableTrait(traitDecl.getLoc());
    if (devicePassableTrait)
      symbols.push_back(devicePassableTrait->getSymbolRef());
  } else if (typeConvention == TypeConvention::RegisterPassable) {
    symbols.push_back(registerPassableParent.getSymbolRef(moduleDecl));
  }
  return TraitType::get(moduleDecl.getContext(), symbols);
}

void ClosureEmitter::enumerateWrapperTraits(SmallVectorImpl<char> &out,
                                            TraitType wrapperTraitType,
                                            ASTDecl &moduleDecl) {
  if (!parentOrdinals) {
    parentOrdinals.emplace();
    (*parentOrdinals)[moveParent.getSymbolRef(moduleDecl)] = 0;
    (*parentOrdinals)[implicitlyDestructibleParent.getSymbolRef(moduleDecl)] =
        1;
    (*parentOrdinals)[copyParent.getSymbolRef(moduleDecl)] = 2;
    (*parentOrdinals)[implicitlyCopyableParent.getSymbolRef(moduleDecl)] = 3;
    (*parentOrdinals)[trivialRegisterTypeParent.getSymbolRef(moduleDecl)] = 4;
    (*parentOrdinals)[registerPassableParent.getSymbolRef(moduleDecl)] = 6;
    ASTDecl *devicePassableTrait =
        shared.getBuiltinDevicePassableTrait(moduleDecl.getLoc());
    if (devicePassableTrait)
      (*parentOrdinals)[devicePassableTrait->getSymbolRef()] = 5;
  }
  llvm::raw_svector_ostream os(out);
  for (SymbolRefAttr symbol : wrapperTraitType.getSymbols().drop_front()) {
    auto it = parentOrdinals->find(symbol);
    assert(it != parentOrdinals->end() &&
           "wrapper trait symbol missing from parent ordinals");
    os << "_" << it->second;
  }
}

bool ClosureEmitter::isTypeRebindableTo(FuncTypeGeneratorType from,
                                        FuncTypeGeneratorType to) {
  if (from == to)
    return true;
  if (from.getInputParamTypes() != to.getInputParamTypes())
    return false;
  if (from.getBody() != to.getBody())
    return false;

  // Enforce parameter-name equality for every passing kind except
  // `Inferred`. Inferred parameters appear before the `+` separator in the
  // pog list and are not user-bindable, so their names are arbitrary
  // disambiguators that may legitimately differ between alpha-equivalent
  // generator types.
  PogListAttr fromPogs = from.getMetadata();
  PogListAttr toPogs = to.getMetadata();
  if (!fromPogs || !toPogs)
    return false;
  if (fromPogs == toPogs)
    return true;
  if (fromPogs.getOrigVariadicConvention() !=
      toPogs.getOrigVariadicConvention())
    return false;
  ArrayRef<PogMetadataAttr> a = fromPogs.getPogs();
  ArrayRef<PogMetadataAttr> b = toPogs.getPogs();
  assert(a.size() == b.size() &&
         "PogListAttr size invariant: tied to input-param-types count");
  for (auto [pa, pb] : llvm::zip(a, b)) {
    if (pa.getPassingKind() != pb.getPassingKind() ||
        pa.getVariadic() != pb.getVariadic() ||
        pa.getDefaultValue() != pb.getDefaultValue() ||
        pa.getConstraints() != pb.getConstraints() ||
        (pa.getPassingKind() != PassingKind::Inferred &&
         pa.getName() != pb.getName()))
      return false;
  }
  return true;
}
