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
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "IREmitter.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "SpecializeInf.h"
#include "Traits.h"
#include "mlir/IR/IRMapping.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/Compiler/OperationUtils.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using M::HLCF::ReturnOp;
using M::HLCF::UnreachableOp;
using namespace M::KGEN::LIT;

// File-local
namespace {
static constexpr char kToDeviceType[] = "_to_device_type";
static constexpr char kIsDeviceTypeConvertible[] =
    "_is_convertible_to_device_type";
static constexpr char kIsImplicitlyEncodableTo[] =
    "_is_implicitly_encodable_to";
static constexpr char kDeviceType[] = "device_type";

static bool usesClosurePipeline(FnOp fn) {
  return fn->getParentOfType<FnOp>() && !fn.isOptionalSymbol() &&
         !fn.getFuncTypeGenerator().isCapturing();
}
} // namespace

static FnOp getFnOpNamed(TraitDeclOp traitDecl, StringRef name) {
  for (FnOp candidate : traitDecl.getFields().getOps<FnOp>()) {
    StringRef sourceName = *candidate.getSourceName();
    if (sourceName.contains(name))
      return candidate;
  }
  return {};
}

// Instantiate the storage struct.
static VarDeclOp emitInitializerCall(ASTDecl &declScope,
                                     ImplicitLocOpBuilder &builder,
                                     Location location, StructDeclOp structDecl,
                                     ArrayRef<TypedAttr> paramArgs,
                                     ArrayRef<CValue> args, StringRef name) {
  LIT::StructType boundType = structDecl.bindReference(paramArgs);
  VarDeclOp var =
      VarDeclOp::create(builder, location, boundType, name,
                        declScope.mangleParamName(name), VarDeclKind::Var);

  IREmitter emitter(declScope, builder);
  SyntheticNode node(declScope.getLoc());
  ExprDest dest(MLValue(var), EC_ReturnValue);
  CallOperands operands(CallSyntax::kTypeCall, &node, std::move(dest));
  for (CValue arg : args)
    operands.add({arg, &node});
  emitter.emitConstructorCall(ASTType(boundType), std::move(operands));
  return var;
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
      if (convention == ArgConvention::ImmReg)
        return SRValue(bbArg);
      // Forward the moved argument.
      if (convention == ArgConvention::OwnedMem ||
          convention == ArgConvention::DeinitMem)
        return MRValue(bbArg);
      return CValue::getMValueForRef(bbArg);
    }();

    // Check the variadic kinds before the passing kind: a `**kwargs` argument
    // is keyword-only AND keyword-variadic, and must forward as a `**` splat.
    if (pog.isKwVarArg())
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kStarStar);
    else if (pog.isPosVarArg() || pog.isPack())
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kStar);
    else if (pog.getPassingKind() == PassingKind::KwOnly)
      callOperands.add(pog.getName(), {argValue, &syntheticExpr},
                       ArgUnpackStyle::kKeyword);
    else
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kPositional);
  }

  CValue callResult =
      emitter.emitCallUnchecked(callee, std::move(callOperands));
  // Forwarding reuses the normal call machinery; an unhandleable signature
  // surfaces as a failed call with the matcher's diagnostic -- propagate.
  if (!callResult)
    return failure();
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

static void
addConformanceTable(ASTDecl &structDecl,
                    const ClosureEmitter::ClosureParent &closureParent,
                    ArrayRef<std::pair<StringRef, TypedAttr>> witnesses) {
  // Insert the new witness into the conformance table.
  MLIRContext *ctx = structDecl.getContext();
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp.getContext());
  b.setInsertionPointToEnd(&structDeclOp.getBodyRegion().front());
  TraitDeclOp traitDeclOp = closureParent.getTrait(structDecl.getShared());
  TraitSymbolArrayAttr immediateParents = traitDeclOp.getImmediateParentsAttr();
  TraitSymbolAttr traitSymbol = closureParent.getSymbol();
  StringAttr parentName = closureParent.getFlattenedName();
  ConformanceOp witnessTable =
      ConformanceOp::create(b, traitSymbol, immediateParents);
  Block &block = witnessTable.getBody().emplaceBlock();
  b.setInsertionPointToStart(&block);
  for (auto [name, newWitness] : witnesses)
    WitnessOp::create(b, StringAttr::get(ctx, name), /*sym_visibility=*/nullptr,
                      newWitness);

  // Register the conformance with the ASTDecl so lookupInCurrentScope can find
  // it during constraint checking.
  ASTDecl &conformDecl = structDecl.getShared().getDeclResolver().addDecl(
      witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
  conformDecl.resolvedness = DeclResolvedness::signature;

  // Update the types of the struct wrapper.
  TraitType oldTraitType = structDeclOp.getCanonicalTrait();
  if (llvm::is_contained(oldTraitType.getSymbols(), traitSymbol))
    return;
  SmallVector<TraitSymbolAttr> symbols;
  llvm::append_range(symbols, oldTraitType.getSymbols());
  symbols.push_back(traitSymbol);
  canonicalizeTraitCompositionSymbols(structDecl.getShared(), symbols);

  TraitType traitType = TraitType::get(ctx, symbols);
  structDeclOp.setCanonicalTrait(traitType);
}

ClosureEmitter::ClosureParent
ClosureEmitter::getBuiltinParent(StringRef traitName, StringRef traitFnName,
                                 ClosureMethod closureMethod) {
  ASTDecl *traitDecl = shared.lookupBuiltinTrait(traitName, SMLoc());
  assert(traitDecl && "missing builtin closure parent trait");
  return ClosureParent(
      shared, cast<TraitDeclOp>(traitDecl->getIfOperation()).bindReference({}),
      traitFnName, closureMethod);
}

ClosureEmitter::ClosureEmitter(SharedState &shared)
    : FunctionEmitter(shared), ctx(shared.getContext()),
      selfName(StringAttr::get(ctx, "self")),
      copyName(StringAttr::get(ctx, "copy")) {}

TraitDeclOp ClosureEmitter::ClosureParent::getTrait(SharedState &shared) const {
  assert(symbol && "closure parent must name a trait");
  // This is a cached lookup.
  ASTDecl &traitDecl =
      shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
  return cast<TraitDeclOp>(traitDecl.getIfOperation());
}

ClosureEmitter::ClosureParent::ClosureParent(SharedState &shared,
                                             TraitSymbolAttr symbol,
                                             StringRef traitFnName,
                                             ClosureMethod closureMethod)
    : symbol(symbol), closureMethod(closureMethod) {
  if (traitFnName.empty())
    return;

  ASTDecl &traitDecl =
      shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
  if (traitDecl.resolvedness < DeclResolvedness::body) {
    [[maybe_unused]] bool outcome = succeeded(
        shared.declResolver->resolveBody(traitDecl, traitDecl.getLoc()));
    assert(outcome && "closure parent trait should not fail body resolution");
  }

  // A trait member carries no symbol name until its signature is resolved, and
  // that name is what identifies it here and keys its witness.
  for (auto [_, members] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *member : members) {
      [[maybe_unused]] bool outcome = succeeded(
          shared.declResolver->resolveSignature(*member, member->getLoc()));
      assert(outcome && "closure parent trait members should not fail "
                        "signature resolution");
    }
  }

  FnOp definingFn =
      getFnOpNamed(cast<TraitDeclOp>(traitDecl.getIfOperation()), traitFnName);
  assert(definingFn && "missing function in closure parent trait");
  witnessName = definingFn.getSymNameAttr();
  signature = definingFn.getFullSignature();
  inlineLevel = inlineLevelOrAutomatic(definingFn.getInlineLevel());
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

static std::pair<ASTDecl &, StructDeclOp>
createStruct(SharedState &shared, ASTDecl &moduleDecl, StringAttr name,
             ArrayRef<ParamDeclAttr> params, SMLoc loc,
             ArrayRef<PassingKind> passingKinds) {
  assert(passingKinds.size() == params.size() &&
         "passing kind per struct parameter");
  OpBuilder b(moduleDecl.getIfOperation()->getRegion(0));
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
  declOp->setAttrs(attrs.getDictionary(shared.getContext()));

  ASTDecl &structDecl = shared.declResolver->addFullyResolvedDecl(
      &*declOp, name, loc, &moduleDecl);

  structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(declOp));
  return {structDecl, declOp};
}

static bool isByReferenceCapture(CaptureConvention c) {
  switch (c) {
  case CaptureConvention::kConventionUnspecified:
  case CaptureConvention::kConventionMut:
  case CaptureConvention::kConventionRead:
  case CaptureConvention::kConventionRef:
    return true;
  case CaptureConvention::kConventionTrivialCopy:
  case CaptureConvention::kConventionCopy:
  case CaptureConvention::kConventionMove:
    return false;
  }
  return false;
}

static FailureOr<ASTType> getDeviceType(ASTType hostType, ASTDecl &scope,
                                        SharedState &shared);

static FailureOr<Type>
getReboundCaptureDeviceFieldType(ASTType captureStorageHostType,
                                 ASTDecl &scopeDecl, SharedState &shared) {
  FailureOr<ASTType> deviceCaptureType =
      getDeviceType(captureStorageHostType, scopeDecl, shared);
  if (failed(deviceCaptureType))
    return failure();

  ArrayRef<TypedAttr> captureBindings =
      captureStorageHostType.getParamBindings();
  if (captureBindings.empty())
    return getCanonicalType(*deviceCaptureType);
  ASTDecl *captureTypeDecl = captureStorageHostType.getDecl(shared);
  assert(captureTypeDecl && "expected declared type for parametric capture");
  auto structOp =
      dyn_cast_or_null<StructDeclOp>(captureTypeDecl->getIfOperation());
  assert(structOp && !structOp.getInputParams().empty() &&
         "expected parametric struct for rebound capture device field type");
  ParameterEvaluator evaluator =
      shared.getParameterEvaluator(structOp.getInputParams(), captureBindings);
  return getCanonicalType(evaluator.getReboundType(*deviceCaptureType));
}

static FailureOr<LIT::StructType>
createDeviceTypeStruct(SharedState &shared, ASTDecl &moduleDecl,
                       ASTDecl &storageStructDecl,
                       ArrayRef<Type> deviceFieldTypes) {
  MLIRContext *ctx = shared.getContext();
  auto storageStruct = cast<StructDeclOp>(storageStructDecl.getIfOperation());
  ArrayRef<ParamDeclAttr> structParams = storageStruct.getInputParams();
  StringAttr thunkKey = storageStruct.getSymNameAttr();
  auto creation = [&]() -> StructDeclOp {
    StringAttr deviceStructName = StringAttr::get(
        ctx, Twine(thunkKey.getValue()).concat(kClosureDeviceTypeSuffix));
    auto [deviceStructDecl, deviceStructOp] = createStruct(
        shared, moduleDecl, deviceStructName, structParams,
        storageStructDecl.getLoc(),
        SmallVector<PassingKind>(structParams.size(), PassingKind::Inferred));
    deviceStructOp.setClosureThunkKeyAttr(thunkKey);
    OpBuilder b(deviceStructOp.getRegion());
    b.setInsertionPointToStart(&deviceStructOp.getFields().front());
    for (auto [field, image] :
         llvm::zip(storageStruct.getFieldDecls(), deviceFieldTypes))
      addFieldOpAndDecl(field.getNameAttr(), image, deviceStructOp,
                        deviceStructDecl, b, *shared.declResolver);
    return deviceStructOp;
  };

  // TODO: this will always be top level decl after fully migrated. Should it be
  // keyed by captured value types (we can reuse the struct as long as closures
  // has the same captures types). It is probably not as important as we are
  // removing device passable conformance from closure anyway.
  StructDeclOp deviceStructOp =
      &moduleDecl == &shared.getTopLevelDecl()
          ? shared.getOrCreateClosureDeviceType(thunkKey, creation)
          : creation();
  SmallVector<TypedAttr> structBindings =
      llvm::map_to_vector(structParams, [](ParamDeclAttr param) -> TypedAttr {
        return ParamDeclRefAttr::get(param);
      });
  return deviceStructOp.bindReference(structBindings);
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
  FnMetaOriginDataAttr oldFnMetadata = sig.getFnMetaOriginData();
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
  auto metadata =
      FnMetaOriginDataAttr::get(ctx, oldFnMetadata.getNumImplicitOriginDecls(),
                                oldFnMetadata.getCaptureOrigins(),
                                oldFnMetadata.getIsNestedOriginsReadOnly(),
                                oldFnMetadata.getDefinesInteriorOrigins());
  return FuncTypeGeneratorType::get(
      sig.getInputParamTypes(),
      FunctionType::get(ctx, signatureInputs, sig.getResults()), argConventions,
      sig.getFnEffects(), metadata, sig.getParamListAttrs(), newArgListAttr);
}

static TraitType
getTraitType(SharedState &shared,
             SmallVector<ClosureEmitter::ClosureParent> &closureParents) {
  SmallVector<TraitSymbolAttr> symbols = llvm::map_to_vector(
      closureParents, [](const ClosureEmitter::ClosureParent &parent) {
        return parent.getSymbol();
      });
  canonicalizeTraitCompositionSymbols(shared, symbols);
  return TraitType::get(shared.getContext(), symbols);
}

/// If a parameter is captured in a signature it
/// becomes an inferred parameter on the struct. Collect such parameters.
static SmallVector<TypedAttr> getCaptureBindings(StructDeclOp structDeclOp) {
  ArrayRef<ParamDeclAttr> params = structDeclOp.getInputParams();
  ArrayRef<PogMetadataAttr> pogs =
      structDeclOp.getSignature().getParamListAttrs().getPogs();
  assert(params.size() == pogs.size() &&
         "struct params and POGs must agree in arity");
  SmallVector<TypedAttr> bindings;
  for (auto [param, pog] : llvm::zip_equal(params, pogs)) {
    if (pog.getPassingKind() == PassingKind::Inferred)
      bindings.push_back(ParamDeclRefAttr::get(param));
  }
  return bindings;
}

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

static size_t explicitParamCount(FnOp fn) {
  return fn.getInputParams().size() -
         fn.getFuncTypeGenerator().getNumImplicitOriginDecls();
}

/// Populate `wrapperFn` with a forwarding call to `callee`, rebinding each
/// block argument to the corresponding `expectedOperandTypes` entry when
/// needed.
static LogicalResult
emitCallForwarderBody(SharedState &shared, FnOp wrapperFn, ASTDecl &wrapperDecl,
                      TypedAttr callee, FnTypeGeneratorType calleeSig,
                      Type resultType, ArrayRef<Type> expectedOperandTypes,
                      bool skipFirst = false) {
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(wrapperFn.getLocScope());
  ImplicitLocOpBuilder bodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
      wrapperFn.getLoc(), wrapperFn.getBody());

  Block &block = wrapperFn.getBodyRegion().front();
  ArrayRef<BlockArgument> toForward = block.getArguments();
  if (skipFirst)
    toForward = toForward.drop_front();

  assert(toForward.size() == expectedOperandTypes.size() &&
         "forwarder arity must match expected operand types");
  SmallVector<Value> operands;
  operands.reserve(toForward.size());
  for (auto [arg, ty] : llvm::zip_equal(toForward, expectedOperandTypes))
    operands.push_back(arg.getType() != ty
                           ? Value(RebindOp::create(bodyBuilder, ty, arg))
                           : Value(arg));

  return emitForwardingCall(bodyBuilder, wrapperDecl, callee, calleeSig,
                            resultType, operands);
}

/// Synthesize the trait-shaped always-inline `__call__$trait` forwarder and
/// publish it as storage `__call__`.
static FnOp emitStorageCallWitness(
    ASTDecl &structDecl, StructDeclOp structOp, FnOp promotedCall, SMLoc smLoc,
    llvm::function_ref<std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>(
        ASTDecl &, bool, StringAttr)>
        pushBackTraitFn) {
  SharedState &shared = structDecl.getShared();
  MLIRContext *ctx = shared.getContext();

  ImplicitLocOpBuilder b(structOp.getLoc(), structOp);
  b.setInsertionPointToEnd(&structOp.getFields().front());
  StringAttr witnessName = StringAttr::get(ctx, "__call__$trait");
  auto [callWitness, callParameters, callResult] =
      pushBackTraitFn(structDecl, /*synthetic=*/true, witnessName);
  ASTDecl *callWitnessDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(callWitness));
  callWitnessDecl->resolvedness = DeclResolvedness::body;
  callWitness.setInlineLevelAttr(
      getInlineLevelAttr(callWitness.getContext(), InlineLevel::Always));

  const size_t promotedParams = explicitParamCount(promotedCall);
  assert(
      callParameters.size() >= promotedParams &&
      "trait-shaped witness cannot have fewer params than the promoted body");
  const size_t extraAux = callParameters.size() - promotedParams;

  // Map trait auxiliary parameters to the capture bindings of the storage
  // struct.
  SmallVector<TypedAttr> captureBindings = getCaptureBindings(structOp);
  assert(extraAux <= captureBindings.size() &&
         "trait aux must not exceed storage capture bindings");
  DenseMap<StringRef, TypedAttr> paramToAliasValue;
  for (auto [param, binding] :
       llvm::zip_equal(callParameters.take_front(extraAux),
                       ArrayRef(captureBindings).take_front(extraAux)))
    paramToAliasValue.insert({param.getName().getValue(), binding});

  mlir::AttrTypeReplacer aliasReplacer;
  aliasReplacer.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
    auto it = paramToAliasValue.find(paramRef.getName().getValue());
    if (it != paramToAliasValue.end())
      return it->second;
    return paramRef;
  });

  TypedAttr calleeSymbol = buildSymbol(promotedCall, structOp.getInputParams());
  SmallVector<TypedAttr> paramArgs;
  for (ParamDeclAttr param : callParameters.drop_front(extraAux)) {
    Type paramType = cast<Type>(aliasReplacer.replace(param.getType()));
    paramArgs.push_back(
        ParamOperatorAttr::getRebind(ParamDeclRefAttr::get(param), paramType));
  }
  if (!paramArgs.empty())
    calleeSymbol = BindParamsAttr::get(ctx, calleeSymbol, paramArgs,
                                       &shared.getEvaluationContext());

  // Linkage / compile-offload see the promoted body through this thunk.
  callWitness->setAttr(kTransparentThunkCalleeExprAttr, calleeSymbol);
  auto calleeSig = cast<FnTypeGeneratorType>(calleeSymbol.getType());
  if (failed(emitCallForwarderBody(shared, callWitness, *callWitnessDecl,
                                   calleeSymbol, calleeSig, callResult,
                                   calleeSig.getArguments()))) {
    shared.emitError(smLoc, "failed to emit trait-shaped __call__ forwarder");
    return {};
  }

  return callWitness;
}

/// Get a name for trait method parameter at idx, we just want a placeholder
/// here, the scheme that we use here does not matter (the decl/ref mapping is
/// what matters). Using a illegal user-space name to avoid collision.
inline static std::string getTraitMethodParamName(size_t idx) {
  return "Closure_Syn#" + llvm::utostr(idx);
}

std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
ClosureEmitter::pushBackTraitFunctionImpl(FnTypeGeneratorType traitFnSignature,
                                          ASTDecl &structDecl, bool synthetic,
                                          StringAttr fnName,
                                          SpecialFunctionKind specialFnID,
                                          InlineLevel inlineLevel) {
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp.getLoc(), structDeclOp);
  b.setInsertionPointToEnd(&structDeclOp.getFields().front());
  SharedState &shared = structDecl.getShared();
  // Wrapper signature is the signature of the method on the wrapper struct.
  // We create it by specializing the trait method by binding the struct type
  // to the self parameter.
  FnTypeGeneratorType wrapperSignature = specializeSignature(
      traitFnSignature, structDecl.getTypeDeclSelf(), *shared.declResolver);

  // Calculate the argument types and result types in terms of the named
  // parameters.
  ParamRefRemapper replacer;
  SmallVector<ParamDeclAttr> parameters;
  for (auto [idx, paramType] :
       llvm::enumerate(wrapperSignature.getInputParamTypes())) {
    parameters.push_back(ParamDeclAttr::get(getTraitMethodParamName(idx),
                                            replacer.replace(paramType)));
    replacer.appendParamDecl(parameters.back());
  }

  SmallVector<Type> argumentTypes;
  llvm::append_range(
      argumentTypes,
      llvm::map_range(wrapperSignature.getArguments(), [&](Type original) {
        return replacer.replace(original);
      }));
  Type result = replacer.replace(wrapperSignature.getResults().front());
  auto [op, decl] = synthesizeFunction(
      structDecl, fnName, parameters, wrapperSignature.getParamListAttrs(),
      argumentTypes, wrapperSignature.getArgConventions(),
      wrapperSignature.getArgListAttrs(), result, specialFnID,
      structDecl.getLoc(), b, wrapperSignature.getFnEffects(), "", synthetic,
      inlineLevel);
  size_t synthesizedOrigins =
      op.getFuncTypeGenerator().getNumImplicitOriginDecls();
  return {op, op.getInputParams().drop_back(synthesizedOrigins), result};
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
      baseSigGen.getFnMetaOriginData(), {});
  return SymbolConstantAttr::get(implSymbol, baseSigGen, {});
}

static ConformanceOp lookupConformanceTable(StructDeclOp op,
                                            SymbolRefAttr traitSymbol) {
  for (auto conformance : op.getFields().getOps<ConformanceOp>()) {
    if (conformance.getTraitSymbolAttr().getSymbol() == traitSymbol) {
      return conformance;
    }
  }

  assert(false && "conformance table should be present");
  return {};
}

static void
generateIsTrivialSpecialAlias(StringRef name, bool value, SharedState &shared,
                              ASTDecl &structDecl,
                              const ClosureEmitter::ClosureParent &parent) {
  auto ctx = shared.getContext();
  auto declOp = dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  auto conformanceOp = lookupConformanceTable(declOp, parent.getSymbolRef());

  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      declOp->getLoc(), &declOp.getBodyRegion().front());
  IREmitter emitter(structDecl, EC_AliasValue);
  SyntheticNode node(structDecl.getLoc());
  TypedAttr valueAttr =
      emitter
          .emitBool({BoolAttr::get(ctx, value), &node}, EC_OperatorOperandValue)
          .getIfPValue();

  ParamDeclAttr paramAttr =
      ParamDeclAttr::get(ctx, StringAttr::get(ctx, name), valueAttr.getType());
  AliasDeclOp aliasOp = LIT::AliasDeclOp::create(
      b, declOp.getBodyRegion().getLoc(), paramAttr, valueAttr);
  aliasOp.setInheritedFromAttr(parent.getSymbol());
  shared.declResolver->addFullyResolvedDecl(aliasOp, StringAttr::get(ctx, name),
                                            structDecl.getLoc(), &structDecl);

  b.setInsertionPointToEnd(&conformanceOp.getBody().front());
  WitnessOp::create(b, StringAttr::get(ctx, name), /*sym_visibility=*/nullptr,
                    valueAttr);
}

void ClosureEmitter::addTrivialClosureLifecycle(
    ASTDecl &structDecl, const ClosureParent &callParent) {
  auto declOp = cast<StructDeclOp>(structDecl.getIfOperation());
  declOp.setConvention(TypeConvention::RegisterPassableTrivial);

  ClosureParent movable = getMoveParent();
  ClosureParent copyable = getCopyParent();
  ClosureParent deinitable = getDeinitableParent();
  SmallVector<ClosureParent> parents{callParent,
                                     getAnyParent(),
                                     movable,
                                     copyable,
                                     getImplicitlyCopyableParent(),
                                     deinitable,
                                     getTrivialRegisterTypeParent(),
                                     getRegisterPassableParent()};
  TraitType traitType = getTraitType(shared, parents);
  declOp.setCanonicalTrait(traitType);

  ImplicitLocOpBuilder b(declOp->getLoc(), ctx);
  auto addLifecycleWitness = [&](const ClosureParent &parent, FnOp impl) {
    auto traitParent = parent.getTrait(shared);
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    TraitSymbolArrayAttr immediateParents =
        traitParent.getImmediateParentsAttr();
    TraitSymbolAttr parentTrait = parent.getSymbol();

    ConformanceOp witnessTable =
        ConformanceOp::create(b, parentTrait, immediateParents);
    ASTDecl &witnessDecl = shared.declResolver->addDecl(
        witnessTable, structDecl.getLoc(), parentTrait.getFlattenedName(),
        &structDecl, {}, {}, -1);
    witnessDecl.resolvedness = DeclResolvedness::body;
    Block &block = witnessTable.getBody().emplaceBlock();
    b.setInsertionPointToStart(&block);
    SymbolConstantAttr symbolConstant = buildSymbol(impl, declOp.getParams());
    WitnessOp::create(b, parent.getWitnessName(), /*sym_visibility=*/nullptr,
                      symbolConstant);
  };

  // The constructor is a no-op: the struct carries only compile-time
  // parameters, so there is no runtime state to initialize.
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
      NoneType::get(ctx), SpecialFunctionKind::kInit, structDecl.getLoc(), b,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true, InlineLevel::Always);
  b.setInsertionPointToStart(&initFnOp.getBodyRegion().front());
  IREmitter::emitNormalReturn(b);
  initDecl->resolvedness = DeclResolvedness::body;

  StructEmitter structEmitter(structDecl);

  // Empty __del__.
  auto delFnOp = structEmitter.synthesizeEmptyDtor();
  addLifecycleWitness(deinitable, delFnOp);

  // Empty move ctor.
  auto moveFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(true);
  declOp.setMoveInitAttr(getSymbolNoParamValues(declOp, moveFnOp));
  addLifecycleWitness(movable, moveFnOp);

  // Empty copy ctor.
  auto copyFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(false);
  declOp.setCopyInitAttr(getSymbolNoParamValues(declOp, copyFnOp));
  addLifecycleWitness(copyable, copyFnOp);

  // All of these operations are trivial in all cases; the struct has no
  // runtime fields.
  generateIsTrivialSpecialAlias("__del__is_trivial", true, shared, structDecl,
                                deinitable);
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", true, shared,
                                structDecl, movable);
  generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", true, shared,
                                structDecl, copyable);

  // Marker parents (AnyType, ImplicitlyCopyable, RegisterPassable,
  // TrivialRegisterPassable) declare no requirements of their own; their
  // inherited requirements (e.g. Copyable's copy init for ImplicitlyCopyable)
  // are witnessed in the declaring parent's ConformanceOp above. An empty
  // ConformanceOp per claimed trait is still required for
  // TypeConformsToTraitAttr::simplify() to verify conformance on concrete
  // closure types.
  for (const ClosureParent &parent : parents)
    if (parent.isEmpty())
      addConformanceTable(structDecl, parent, {});
}

LIT::StructType
ClosureEmitter::getInflatedClosureForFnSymbol(IREmitter &emitter, SMLoc loc,
                                              PValue fnPValue) {
  auto fnSig = cast<FnTypeGeneratorType>(getCanonicalType(fnPValue.getType()));
  auto toConform = emitter.bindParamsToClosureTraitFromSig(fnSig);

  llvm::SmallSetVector<ParamDeclRefAttr, 4> capturedParamRef;
  fnSig.walk([&](ParamDeclRefAttr ref) { capturedParamRef.insert(ref); });
  auto captures = capturedParamRef.takeVector();

  // Hoist the parameter ref to the extension struct, reuse the name so that we
  // don't need to remapped the name.
  SmallVector<ParamDeclAttr> structParams;
  SmallVector<TypedAttr> bindings;
  for (TypedAttr capture : captures) {
    auto ref = cast<ParamDeclRefAttr>(capture);
    structParams.push_back(ParamDeclAttr::get(ref.getName(), ref.getType()));
    bindings.push_back(ref);
  }
  structParams.push_back(ParamDeclAttr::get("#__CALL__#", fnSig));
  bindings.push_back(ParamOperatorAttr::getRebind(fnPValue, fnSig));

  // TODO: The cache key includes pog list, we can potentially strip in order to
  // get fewer inflated struct
  auto thunkKey = TypeAttr::get(fnSig);
  StructDeclOp inflatedDeclOp =
      shared.getOrCreateInflatedClosure(thunkKey, [&]() {
        std::string extName(kClosureInflatedPrefix);
        llvm::raw_string_ostream os(extName);
        generateConversionThunkName(os, {fnSig});

        auto [structDecl, declOp] =
            createStruct(shared, shared.getTopLevelDecl(),
                         StringAttr::get(ctx, extName), structParams, loc,
                         SmallVector<PassingKind>(structParams.size(),
                                                  PassingKind::PosOnly));
        declOp.setClosureThunkKeyAttr(thunkKey);
        auto fnName = StringAttr::get(shared.getContext(), "__call__");
        TypedAttr callee = ParamDeclRefAttr::get("#__CALL__#", fnSig);

        ClosureParent callParent(toConform, shared.getClosureFnSig(toConform),
                                 fnName, ClosureMethod::CALL);
        addTrivialClosureLifecycle(structDecl, callParent);

        auto [callMethod, parameters, result] = pushBackTraitFunctionImpl(
            callParent.getSignature(), structDecl, /*synthetic=*/false, fnName,
            SpecialFunctionKind::kNormal, callParent.getInlineLevel());
        callMethod.setInlineLevelAttr(
            getInlineLevelAttr(callMethod.getContext(), InlineLevel::Always));
        callMethod->setAttr(kTransparentThunkCalleeExprAttr, callee);

        assert(parameters.size() == fnSig.getInputParamTypes().size());
        if (!parameters.empty()) {
          SmallVector<TypedAttr> forwardParams;
          for (ParamDeclAttr param : parameters)
            forwardParams.push_back(ParamDeclRefAttr::get(param));
          callee = BindParamsAttr::get(ctx, callee, forwardParams,
                                       &shared.getEvaluationContext());
        }
        auto calleeSig = cast<FnTypeGeneratorType>(callee.getType());
        ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(fnName);
        assert(decls.size() == 1);
        if (failed(emitCallForwarderBody(
                shared, callMethod, *decls.front(), callee, calleeSig, result,
                calleeSig.getArguments(), /*skipFirst=*/true))) {
          llvm_unreachable("Internal Error: fail to forward closure call.");
        };
        addConformanceTable(
            structDecl, callParent,
            {{"__call__", buildSymbol(callMethod, structParams)}});

        addStorageConformanceToDevicePassable(structDecl, {}, extName);
        return declOp;
      });

  return inflatedDeclOp.bindReference(bindings);
}

bool ClosureEmitter::isInflatedClosureForFnSymbol(PValue fnSymbol,
                                                  LIT::StructType wrapper) {
  // The inflated struct's last parameter is the fnSymbol.
  return wrapper.getSymbolRef().getLeafReference().getValue().starts_with(
             kClosureInflatedPrefix) &&
         isEqualCanon(wrapper.getParamValues().back(), fnSymbol.get());
}

static bool hasCapturingParameterType(ArrayRef<ParamDeclAttr> params) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return llvm::any_of(params, [&](ParamDeclAttr param) {
    return walker.walk(param).wasInterrupted();
  });
}

/// Prepend a new implicit origin at index 0 in `sig`, shifting all existing
/// depth-local ImplicitOriginRefAttrs up by one and incrementing the
/// FnMetadata origin count. This is the implicit-origin analogue of
/// FnTypeGeneratorType::prependParams for explicit parameters.
static FnTypeGeneratorType prependImplicitOriginDecl(FnTypeGeneratorType sig) {
  struct OriginIndexShifter
      : public IndexParameterReplacer<OriginIndexShifter> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // Only shift refs that are scoped to this function (depth + 1 == depth
      // after the replacer increments depth for each nesting level).
      if (depth == 0)
        return {};
      if (auto ref = dyn_cast<ImplicitOriginRefAttr>(attr);
          ref && ref.getDepth() + 1 == depth)
        return ImplicitOriginRefAttr::get(ref.getDepth(), ref.getIndex() + 1,
                                          ref.getType());
      return {};
    }
  } shifter;
  sig = cast<FnTypeGeneratorType>(shifter.replace(sig));
  FnMetaOriginDataAttr oldMeta = sig.getFnMetaOriginData();
  FnMetaOriginDataAttr newMeta = FnMetaOriginDataAttr::get(
      sig.getContext(), oldMeta.getNumImplicitOriginDecls() + 1,
      oldMeta.getCaptureOrigins(), oldMeta.getIsNestedOriginsReadOnly(),
      oldMeta.getDefinesInteriorOrigins());
  return FnTypeGeneratorType::get(sig.getInputParamTypes(), sig.getValues(),
                                  sig.getArgConventions(), sig.getFnEffects(),
                                  newMeta, sig.getParamListAttrs(),
                                  sig.getArgListAttrs());
}

struct PromotedSignature {
  FnTypeGeneratorType signature;
  FunctionType functionType;
  Type selfRuntimeArgType;
  SmallVector<ParamDeclAttr> newParams;
};

/// Build the promoted signature for a closure being lifted to file scope.
static PromotedSignature buildPromotedSignature(
    SharedState &shared, FnTypeGeneratorType sig,
    ArrayRef<ParamDeclAttr> params, ArrayRef<ParamDeclAttr> prependedParams,
    std::optional<ClosureEmitter::PromotedClosureSelfArg> selfArg,
    TriBool capturingOverride = TriBool::unknown()) {
  MLIRContext *ctx = shared.getContext();
  size_t oldNumImplicitOrigins =
      sig.getFnMetaOriginData().getNumImplicitOriginDecls();
  assert(oldNumImplicitOrigins <= params.size());

  // Step 1: prepend a new origin slot for self and collect origin decls.
  SmallVector<ParamDeclAttr> implicitOriginDecls;
  ParamDeclAttr selfImplicitOriginDecl;
  if (selfArg) {
    selfImplicitOriginDecl = ParamDeclAttr::get(
        StringAttr::get(ctx, "__self_origin"), OriginType::get(ctx, true));
    implicitOriginDecls.push_back(selfImplicitOriginDecl);
    sig = prependImplicitOriginDecl(sig);
  }
  llvm::append_range(implicitOriginDecls,
                     params.take_back(oldNumImplicitOrigins));

  // Step 2: prepend explicit params for each captured parameter.
  SmallVector<ParamDeclAttr> explicitPrependedParams(prependedParams.begin(),
                                                     prependedParams.end());
  std::optional<IndexRefRemapper> prependParamRefRemapper;
  if (!explicitPrependedParams.empty()) {
    prependParamRefRemapper =
        std::make_optional<IndexRefRemapper>(explicitPrependedParams);
    sig = FnTypeGeneratorType::prependParams(sig, explicitPrependedParams);
  }

  // Step 3: prepend the self runtime argument so it becomes arg[0].
  if (selfArg) {
    Type selfArgType = selfArg->type;
    if (prependParamRefRemapper)
      selfArgType = prependParamRefRemapper->replace(selfArgType);

    Type selfRefType = RefType::get(
        selfArgType,
        ImplicitOriginRefAttr::get(0, 0, selfImplicitOriginDecl.getType()));
    sig = addClosureSelfArgToFunctionSignature(selfRefType, selfArg->convention,
                                               sig);
  }

  // Step 4: resolve depth-local index refs to named param/origin references.
  SmallVector<ParamDeclAttr> explicitParamDecls;
  llvm::append_range(explicitParamDecls, explicitPrependedParams);
  llvm::append_range(explicitParamDecls,
                     params.drop_back(oldNumImplicitOrigins));
  assert(explicitParamDecls.size() == sig.getInputParamTypes().size());
  FunctionType promotedFunctionType = replaceIndexRefsWithNamedRefs(
      sig.getValues(), explicitParamDecls, implicitOriginDecls);

  bool shouldBeCapturing;
  if (capturingOverride.isDefinite())
    shouldBeCapturing = capturingOverride.isTrue();
  else
    shouldBeCapturing =
        sig.isCapturing() || hasCapturingParameterType(explicitPrependedParams);
  FnTypeGeneratorType promotedSignature = FnTypeGeneratorType::get(
      sig.getInputParamTypes(), sig.getValues(), sig.getArgConventions(),
      sig.getFnEffects().setCapturing(shouldBeCapturing),
      sig.getFnMetaOriginData(), sig.getParamListAttrs(),
      sig.getArgListAttrs());

  Type selfRuntimeArgType;
  if (selfArg)
    selfRuntimeArgType = promotedFunctionType.getInputs().front();

  // Assemble the full param list when something changed: new capture params
  // were prepended or a self origin was inserted.
  SmallVector<ParamDeclAttr> newParams;
  if (!explicitPrependedParams.empty() || selfArg) {
    ArrayRef<ParamDeclAttr> oldExplicit =
        params.drop_back(oldNumImplicitOrigins);
    newParams.reserve(explicitPrependedParams.size() + oldExplicit.size() +
                      implicitOriginDecls.size());
    llvm::append_range(newParams, explicitPrependedParams);
    llvm::append_range(newParams, oldExplicit);
    llvm::append_range(newParams, implicitOriginDecls);
  }

  return {promotedSignature, promotedFunctionType, selfRuntimeArgType,
          std::move(newParams)};
}

ASTDecl *ClosureEmitter::promoteClosure(
    ASTDecl &nestedFnDecl, ArrayRef<ParamDeclAttr> prependedParams,
    std::optional<PromotedClosureSelfArg> selfArg, TriBool capturingOverride,
    ASTDecl *targetParent) {
  assert(nestedFnDecl.resolvedness == DeclResolvedness::body &&
         "nested decl must be fully resolved to promote");
  // Mark dead unparsed code as resolved to prevent resolution dependent on
  // the old parent scope, which is about to change below.
  for (auto &[_, children] : nestedFnDecl.getDeclsInScope()) {
    for (ASTDecl *child : children) {
      if (child->getParentDecl() == &nestedFnDecl &&
          child->resolvedness == DeclResolvedness::unparsed)
        child->resolvedness = DeclResolvedness::body;
    }
  }
  MLIRContext *ctx = shared.getContext();
  FnOp function = cast<FnOp>(nestedFnDecl.getIfOperation());
  SMLoc loc = nestedFnDecl.getLoc();
  if (!targetParent)
    targetParent = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();
  assert(targetParent && "expected a target parent for promotion");

  auto [promotedSignature, promotedFunctionType, selfRuntimeArgType,
        newParams] =
      buildPromotedSignature(shared, function.getFuncTypeGenerator(),
                             function.getParams(), prependedParams, selfArg,
                             capturingOverride);

  OpBuilder builder = targetParent->getDeclEndBuilder();
  function->moveBefore(builder.getInsertionBlock(),
                       builder.getInsertionPoint());
  function.setSymName(
      targetParent->mangleParamName(function.getSymName()->str()));

  // Update function attributes and body to reflect self argument addition.
  if (selfArg) {
    auto &entryBlock = function.getBodyRegion().front();
    Location selfArgLoc = function.getLoc();
    if (FileLineColLoc sourceLoc = DebugInfo::extractSourceLoc(selfArgLoc))
      selfArgLoc = sourceLoc;
    entryBlock.insertArgument(static_cast<unsigned>(0), selfRuntimeArgType,
                              selfArgLoc);

    if (ArrayAttr argAttrs = function.getFnArgAttrs();
        argAttrs && !argAttrs.empty()) {
      SmallVector<Attribute> newArgAttrs;
      newArgAttrs.push_back(ArrayAttr::get(ctx, {}));
      llvm::append_range(newArgAttrs, argAttrs);
      function.setFnArgAttrsAttr(ArrayAttr::get(ctx, newArgAttrs));
    }

    // Augment DISubroutineType with self argument.
    if (DebugInfo::DISubprogramAttr oldScope = function.getSubprogramScope()) {
      auto subroutineType =
          cast<DebugInfo::DISubroutineType>(oldScope.getType());
      SmallVector<DebugInfo::DIType> updatedArgTypes;
      updatedArgTypes.push_back(DebugInfo::DIUnspecifiedType::get(ctx, "self"));
      llvm::append_range(updatedArgTypes, subroutineType.getArgumentTypes());
      auto newSubroutineType = DebugInfo::DISubroutineType::get(
          ctx, subroutineType.getCallingConvention(), updatedArgTypes,
          subroutineType.getResultTypes());
      auto newScope = DebugInfo::DISubprogramAttr::get(
          oldScope.getCompileUnit(), oldScope.getScope(),
          oldScope.getSourceName(), oldScope.getLinkageName(),
          oldScope.getFile(), oldScope.getLine(), oldScope.getScopeLine(),
          oldScope.getSubprogramFlags(),
          cast<DebugInfo::DISubroutineType>(newSubroutineType));
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](DebugInfo::DISubprogramAttr sp) -> DebugInfo::DISubprogramAttr {
            if (sp == oldScope)
              return newScope;
            return sp;
          });
      replacer.recursivelyReplaceElementsIn(function, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true);
    }
  }
  function.setFuncTypeGenerator(promotedSignature);
  function.setFunctionType(promotedFunctionType);
  if (!newParams.empty())
    function.setParamsAttr(ParamDeclArrayAttr::get(ctx, newParams));
  // Transfer the linkage name to the promoted op: the mangled sym_name
  // above overwrites the original name, so preserve it so it survives
  // into elaboration.
  if (auto linkageName = function.getLinkageNameAttr())
    function.setLinkageNameAttr(linkageName);
  function.setNoDocRequired(true);
  function.setSynthetic(true);
  auto &decl = shared.declResolver->addFullyResolvedDecl(
      function, /*name=*/StringAttr(), loc, targetParent);
  // Transfer child decls from the original to the promoted decl. Since the op
  // was moved (not cloned), all mlir::Value pointers are still valid.
  decl.takeDecls(nestedFnDecl);
  // Register the lifted function to the symbol table.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(function, decl);
  assert(!existing && "unexpected redefinition of promoted closure");
  if (prependedParams.empty()) {
    nestedFnDecl.setIRValue(function);
    return &decl;
  }

  ArrayRef<ParamDeclAttr> promotedFnParams = function.getParams();
  ArrayRef<ParamDeclAttr> captureParams =
      promotedFnParams.take_front(prependedParams.size());
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
  nestedFnDecl.setIRValue(PValue(function.getFuncLiteralGenerator(
      shared.getEvaluationContext(),
      ParameterExprArrayAttr::get(ctx, bindings))));
  return &decl;
}

ASTDecl *ClosureEmitter::promoteClosure(
    ASTDecl &nestedFnDecl, ArrayRef<ParamDeclRefAttr> prependedParamRefs,
    std::optional<PromotedClosureSelfArg> selfArg, TriBool capturingOverride,
    ASTDecl *targetParent) {
  SmallVector<ParamDeclAttr> prependedParams =
      llvm::map_to_vector(prependedParamRefs, [](ParamDeclRefAttr paramRef) {
        return ParamDeclAttr::get(paramRef);
      });
  return promoteClosure(nestedFnDecl, prependedParams, selfArg,
                        capturingOverride, targetParent);
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

static bool isOutlinedInteriorOrigin(
    TypedAttr typed,
    const llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins) {
  if (!typed || !sugarIsa<OriginType>(typed.getType()))
    return false;
  TypedAttr stripped = OriginType::stripMutCastAndRebind(typed);
  if (!sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped))
    return false;
  return interiorOrigins.contains(cast<TypedAttr>(getCanonicalAttr(stripped)));
}

static bool hasOutlinedInteriorOrigin(
    Type type,
    const llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins) {
  WalkResult result = type.walk([&](Attribute attr) {
    auto typed = dyn_cast<TypedAttr>(attr);
    return isOutlinedInteriorOrigin(typed, interiorOrigins)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Given an attribute, update origin mutability information and register any
// interior origins for outlining. Returns false to stop recursion for origin
// typed nodes so cast operands and base origins of outlined interior origins
// aren't counted separately.
static bool checkOriginAndOutline(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    Attribute attr) {
  auto typed = dyn_cast<TypedAttr>(attr);
  if (!typed || !sugarIsa<OriginType>(typed.getType()))
    return true;

  // Strip mutcasts before classifying so an immutable use of a mutable
  // interior origin (mutcast(interior)) is not counted as a mutable use of
  // the outlined parameter. Mutability is still taken from `typed`.
  TypedAttr stripped = OriginType::stripMutCastAndRebind(typed);

  if (sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped)) {
    TypedAttr canon = cast<TypedAttr>(getCanonicalAttr(stripped));
    auto it = interiorOrigins.find(canon);
    if (it == interiorOrigins.end()) {
      StringAttr name = StringAttr::get(ctx, Twine("?__interior_origin_") +
                                                 Twine(interiorOrigins.size()));
      ParamDeclAttr param = ParamDeclAttr::get(name, canon.getType());
      it = interiorOrigins.insert({canon, ParamDeclRefAttr::get(param)}).first;
    }
    meetOriginMutability(originMutability, it->second.getName(),
                         OriginType::isMutableKnown(typed, false));
    return false;
  }

  if (auto ref = dyn_cast<ParamDeclRefAttr>(stripped)) {
    meetOriginMutability(originMutability, ref.getName(),
                         OriginType::isMutableKnown(typed, false));
    return false;
  }

  // Otherwise this is a derived origin such as a field projection,
  // which holds the origins it is derived from in its sub-elements. Keep
  // walking so those are counted at their own mutability; stopping here would
  // leave a mutably-used origin unrecorded and let it be wrongly promoted.
  return true;
}

template <typename AttrOrType>
static void checkOriginAndOutlineImpl(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    AttrOrType attrOrType) {
  if (!attrOrType)
    return;
  if constexpr (std::is_convertible_v<AttrOrType, Attribute>) {
    if (!checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                               attrOrType))
      return;
  }
  attrOrType.walkImmediateSubElements(
      [&](Attribute attribute) {
        checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins,
                                  attribute);
      },
      [&](Type type) {
        checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins, type);
      });
}

template <typename AttrOrType>
static void checkOriginAndOutline(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    AttrOrType attrOrType) {
  if constexpr (std::is_convertible_v<AttrOrType, Attribute>)
    attrOrType = cast<AttrOrType>(getCanonicalAttr(attrOrType));
  else
    attrOrType = cast<AttrOrType>(getCanonicalType(attrOrType));
  checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins, attrOrType);
}

static SmallPtrSet<StringAttr, 8>
collectPromotedOrigins(MLIRContext *ctx,
                       const DenseMap<StringAttr, bool> &originMutability,
                       SmallVectorImpl<ParamDeclAttr> &structParams,
                       SmallVectorImpl<TypedAttr> &structBindings) {
  SmallPtrSet<StringAttr, 8> promotedOriginNames;
  for (auto [index, param] : llvm::enumerate(structParams)) {
    StringAttr name = param.getName();
    auto originType = sugarDynCast<OriginType>(param.getType());
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

static TypedAttr
getOutlinedOriginRef(MLIRContext *ctx, ParamDeclRefAttr paramRef,
                     const SmallPtrSetImpl<StringAttr> &promotedOriginNames) {
  StringAttr name = paramRef.getName();
  Type originType = promotedOriginNames.contains(name)
                        ? OriginType::get(ctx, false)
                        : paramRef.getType();
  return ParamDeclRefAttr::get(name, originType);
}

static std::optional<TypedAttr>
getPromotedOriginRef(MLIRContext *ctx, TypedAttr origin,
                     const SmallPtrSetImpl<StringAttr> &promotedOriginNames) {
  auto originRef = dyn_cast<ParamDeclRefAttr>(origin);
  if (!originRef || !promotedOriginNames.contains(originRef.getName()))
    return std::nullopt;
  return ParamDeclRefAttr::get(originRef.getName(),
                               OriginType::get(ctx, false));
}

static LogicalResult outlineAndPromoteOrigins(
    SharedState &shared, SmallVectorImpl<StructDefFieldAttr> &fieldDecls,
    SmallVectorImpl<ParamDeclAttr> &allStructParams,
    SmallVectorImpl<TypedAttr> &structParamBindings, FnOp nestedFn,
    SmallVectorImpl<Type> &deviceCaptureFieldTypes, Location closureLoc) {
  MLIRContext *ctx = shared.getContext();
  assert(allStructParams.size() == structParamBindings.size() &&
         "expected parallel struct parameters and bindings");

  DenseMap<StringAttr, bool> originMutability;
  llvm::MapVector<TypedAttr, ParamDeclRefAttr> interiorOrigins;

  for (StructDefFieldAttr fieldDecl : fieldDecls)
    checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                          fieldDecl.getTypeValue());
  for (ParamDeclAttr param : allStructParams)
    checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                          param.getType());
  checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                        nestedFn.getFuncTypeGenerator());

  // Append fresh parameter declarations and their parent-scope bindings.
  for (auto &[interiorAttr, paramRef] : interiorOrigins) {
    allStructParams.push_back(
        ParamDeclAttr::get(paramRef.getName(), paramRef.getType()));
    structParamBindings.push_back(interiorAttr);
  }

  // Prune origin parameters in allStructParams that were not referenced outside
  // of interior origins.
  // If it were referenced outside the interior origin it would be registered in
  // the origin mutability table so if its not there its safe to assume its
  // unused and can be dropped.
  assert(allStructParams.size() == structParamBindings.size() &&
         "expected parallel struct parameters and bindings before pruning");
  SmallVector<ParamDeclAttr> prunedParams;
  SmallVector<TypedAttr> prunedBindings;
  for (auto [param, binding] :
       llvm::zip_equal(allStructParams, structParamBindings)) {
    if (sugarIsa<OriginType>(param.getType())) {
      if (!originMutability.contains(param.getName()))
        continue;
    }
    prunedParams.push_back(param);
    prunedBindings.push_back(binding);
  }
  allStructParams = std::move(prunedParams);
  structParamBindings = std::move(prunedBindings);

  // Promote origin parameters that are only read-only into immutable origins.
  SmallPtrSet<StringAttr, 8> promotedOriginNames = collectPromotedOrigins(
      ctx, originMutability, allStructParams, structParamBindings);

  if (interiorOrigins.empty() && promotedOriginNames.empty())
    return success();

  Location rewriteLoc = closureLoc;
  bool hadConflict = false;

  // Replace all interior origin occurrences and promoted origin references in a
  // single walk.
  mlir::AttrTypeReplacer originReplacer;
  originReplacer.addReplacement([&](SymbolConstantAttr sym)
                                    -> std::pair<Attribute, WalkResult> {
    bool bindsOutlinedInterior = false;
    bool changed = false;
    DenseMap<TypedAttr, TypedAttr> paramReplacements;
    SmallVector<TypedAttr> newParamValues;
    newParamValues.reserve(sym.getParamValues().size());
    for (TypedAttr pv : sym.getParamValues()) {
      bindsOutlinedInterior |= isOutlinedInteriorOrigin(pv, interiorOrigins);
      auto newPv = cast<TypedAttr>(originReplacer.replace(pv));
      if (newPv != pv) {
        changed = true;
        paramReplacements[cast<TypedAttr>(getCanonicalAttr(pv))] = newPv;
      }
      newParamValues.push_back(newPv);
    }

    // An interior origin in the signature that the call does not bind as a
    // parameter value is derived from the callee's own parameters, so rewriting
    // it would misstate the callee's signature. Keep it as is and reject if
    // that same interior origin was outlined (interior origins are spelled
    // inline, so preserved occurrences are indistinguishable from the ones that
    // need to be rewritten). Reject this case (for now).
    if (!bindsOutlinedInterior &&
        hasOutlinedInteriorOrigin(sym.getType(), interiorOrigins)) {
      hadConflict = true;
      shared.emitError(
          rewriteLoc,
          "cannot derive an interior origin from a captured container while "
          "an interior reference to the container is also captured");
      return {sym, WalkResult::skip()};
    }

    if (!changed)
      return {sym, WalkResult::skip()};

    mlir::AttrTypeReplacer symTypeReplacer;
    symTypeReplacer.addReplacement(
        [&](TypedAttr attr) -> std::optional<TypedAttr> {
          if (!sugarIsa<OriginType>(attr.getType()))
            return std::nullopt;
          TypedAttr canon = cast<TypedAttr>(getCanonicalAttr(attr));
          auto it = paramReplacements.find(canon);
          if (it != paramReplacements.end())
            return it->second;
          return getPromotedOriginRef(ctx,
                                      OriginType::stripMutCastAndRebind(attr),
                                      promotedOriginNames);
        });
    auto newType =
        cast<FuncTypeGeneratorType>(symTypeReplacer.replace(sym.getType()));
    return {SymbolConstantAttr::get(sym.getSymbol(), newType, newParamValues),
            WalkResult::skip()};
  });
  originReplacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<TypedAttr> {
        if (!sugarIsa<OriginType>(attr.getType()))
          return std::nullopt;

        TypedAttr stripped = OriginType::stripMutCastAndRebind(attr);
        if (sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped)) {
          auto it =
              interiorOrigins.find(cast<TypedAttr>(getCanonicalAttr(stripped)));
          if (it == interiorOrigins.end())
            return std::nullopt;
          if (attr != stripped)
            return std::nullopt;
          return getOutlinedOriginRef(ctx, it->second, promotedOriginNames);
        }

        return getPromotedOriginRef(ctx, stripped, promotedOriginNames);
      });

  // Rewrite the body before the storage struct so a conflict is diagnosed at
  // the operation that hit it, and so a rejected closure leaves the struct
  // alone.
  if (nestedFn) {
    nestedFn.walk([&](Operation *op) {
      rewriteLoc = op->getLoc();
      originReplacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                       /*replaceLocs=*/true,
                                       /*replaceTypes=*/true);
      return hadConflict ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (hadConflict)
      return failure();
    rewriteLoc = closureLoc;
  }

  for (StructDefFieldAttr &fieldDecl : fieldDecls) {
    auto newTypeValue =
        cast<TypedAttr>(originReplacer.replace(fieldDecl.getTypeValue()));
    fieldDecl = StructDefFieldAttr::get(fieldDecl.getName(), newTypeValue,
                                        fieldDecl.getAnnotations());
  }
  for (Type &fieldType : deviceCaptureFieldTypes)
    fieldType = cast<Type>(originReplacer.replace(fieldType));
  return failure(hadConflict);
}

static SmallVector<Type>
getConcreteStructFieldTypes(StructInstanceType structInstType,
                            ArrayRef<ParamDeclAttr> structParams,
                            ArrayRef<TypedAttr> structBindings) {
  ParameterEvaluator structEvaluator(structParams, structBindings);
  return llvm::map_to_vector(
      structInstType.getFields(), [&](StructDefFieldAttr field) -> Type {
        TypedAttr fieldType =
            structEvaluator.getReboundAttribute(field.getTypeValue());
        return ASTType(fieldType);
      });
}

static KGEN::StructType getMlirType(MLIRContext *ctx,
                                    StructInstanceType structInstType,
                                    ArrayRef<ParamDeclAttr> structParams,
                                    ArrayRef<TypedAttr> structBindings,
                                    TypeConvention convention) {
  SmallVector<Type> mlirFieldTypes =
      getConcreteStructFieldTypes(structInstType, structParams, structBindings);
  bool isMemOnly = convention == TypeConvention::MemoryOnly;
  return KGEN::StructType::get(ctx, mlirFieldTypes, isMemOnly);
}

bool ClosureEmitter::provenConformsToTrait(
    ASTType type, ASTDecl *traitDecl, SharedState &shared,
    ArrayRef<ConstraintAttr> callerAssumptions) {
  assert(traitDecl && "expected a trait declaration");
  auto trait = cast<TraitDeclOp>(traitDecl->getIfOperation());
  return type
      .doesConformTo(TraitType::get(getFullyResolvedSymbolRef(trait)), shared,
                     callerAssumptions)
      .isTrue();
}

static FailureOr<ASTType> getDeviceType(ASTType hostType, ASTDecl &scope,
                                        SharedState &shared) {
  ASTDecl *devicePassableDecl =
      shared.getBuiltinDevicePassableTrait(scope.getLoc());
  assert(devicePassableDecl && "could not find device passable dependency");

  SmallVector<ConstraintAttr> assumptions =
      ASTDecl::getAssumptionsFromScope(&scope);

  // A capture is device-encodable only if it conforms to DevicePassable;
  // resolve its device_type via GetWitness + fold attempt.
  if (!ClosureEmitter::provenConformsToTrait(hostType, devicePassableDecl,
                                             shared, assumptions))
    return failure();

  if (failed(shared.declResolver->resolveBody(*devicePassableDecl,
                                              scope.getLoc())))
    return failure();

  for (auto [_, decls] : devicePassableDecl->getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      if (failed(shared.declResolver->resolveSignature(*decl, scope.getLoc())))
        return failure();
    }
  }

  ArrayRef<ASTDecl *> aliasDecls = devicePassableDecl->lookupInCurrentScope(
      StringAttr::get(shared.getContext(), kDeviceType));
  if (aliasDecls.empty())
    return failure();

  auto aliasOp =
      dyn_cast_or_null<AliasDeclOp>(aliasDecls.front()->getIfOperation());
  if (!aliasOp || !aliasOp.getType())
    return failure();

  MLIRContext *ctx = shared.getContext();
  auto traitSymbol = TraitSymbolAttr::get(devicePassableDecl->getSymbolRef());
  TypedAttr deviceTypeWitness =
      shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
          PValue(hostType), traitSymbol, StringAttr::get(ctx, kDeviceType),
          aliasOp.getType());

  if (!deviceTypeWitness || !LIT::isTypeExpr(deviceTypeWitness))
    return failure();
  return ASTType(deviceTypeWitness);
}

static TypedAttr getRefLikeOrigin(Type type) {
  if (auto refType = dyn_cast<RefType>(type))
    return refType.getOrigin();
  if (auto refPackType = dyn_cast<RefPackType>(type))
    return refPackType.getOrigin();
  return nullptr;
}

static void
addOriginReplacements(mlir::AttrTypeReplacer &originReplacer,
                      const DenseMap<TypedAttr, TypedAttr> &originMap) {
  originReplacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<TypedAttr> {
        auto it = originMap.find(attr);
        if (it == originMap.end())
          return std::nullopt;
        return it->second;
      });
}

ASTDecl *ClosureEmitter::liftClosureIntoMethod(
    ASTDecl &nestedFnDecl, ASTDecl &storageStructDecl,
    PromotedClosureSelfArg selfArg,
    ArrayRef<StructDefFieldAttr> concreteFieldDecls,
    ArrayRef<Value> concreteFieldCaptures,
    ArrayRef<CaptureConvention> captureConventions,
    ArrayRef<Type> selfBoundFieldTypes, Location location) {
  MLIRContext *ctx = shared.getContext();
  // Nest under the storage struct as a method. Captured parameters already
  // live on the storage struct, so do not prepend them to the method.
  ASTDecl *promotedDecl = promoteClosure(
      nestedFnDecl, ArrayRef<ParamDeclAttr>{}, /*selfArg=*/selfArg,
      /*capturingOverride=*/TriBool::yes(),
      /*targetParent=*/&storageStructDecl);
  FnOp promotedCallFunction = cast<FnOp>(promotedDecl->getIfOperation());
  assert(concreteFieldDecls.size() == concreteFieldCaptures.size() &&
         "expected one capture value per closure field");
  assert(concreteFieldDecls.size() == captureConventions.size() &&
         "expected one capture convention per closure field");

  // Wire the captures in the promoted function body to the struct fields
  // accessed via the self argument.
  Block &callBody = promotedCallFunction.getBodyRegion().front();
  Value selfArgValue = callBody.getArgument(0);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&callBody);
  Location promotedBodyLoc = DebugInfo::extractSourceLoc(location);
  if (DebugInfo::DISubprogramAttr promotedScope =
          promotedCallFunction.getSubprogramScope())
    promotedBodyLoc = FusedLoc::get(ctx, promotedBodyLoc, promotedScope);

  DenseMap<Value, Value> captureReplacements;
  captureReplacements.reserve(concreteFieldCaptures.size());
  DenseMap<TypedAttr, TypedAttr> originMap;
  bool hadOriginConflict = false;
  for (auto [index, captureAndConvention] :
       llvm::enumerate(llvm::zip(concreteFieldCaptures, captureConventions))) {
    auto [capture, convention] = captureAndConvention;
    StringAttr fieldName = concreteFieldDecls[index].getName();
    auto selfRefType = cast<RefType>(selfArgValue.getType());
    Type fieldElementType = selfBoundFieldTypes[index];
    Type resultRefType = RefStructGEROp::getReboundFieldType(
        selfRefType, fieldName, fieldElementType);
    Value extractedRef =
        RefStructGEROp::create(bodyBuilder, promotedBodyLoc, resultRefType,
                               fieldName, selfArgValue)
            ->getResults()
            .front();

    Value replacement = extractedRef;
    if (!sugarIsa<RefType>(capture.getType()) ||
        isByReferenceCapture(convention))
      replacement =
          RefLoadOp::create(bodyBuilder, promotedBodyLoc, extractedRef);
    captureReplacements[capture] = replacement;

    TypedAttr oldOrigin = getRefLikeOrigin(capture.getType());
    TypedAttr newOrigin = getRefLikeOrigin(replacement.getType());
    if (oldOrigin && newOrigin && (newOrigin != oldOrigin)) {
      auto [it, inserted] = originMap.try_emplace(oldOrigin, newOrigin);
      if (!inserted && it->second != newOrigin)
        hadOriginConflict = true;
    }
  }
  if (hadOriginConflict)
    llvm::report_fatal_error(
        "conflicting capture origins while lifting closure into method");
  mlir::AttrTypeReplacer originReplacer;
  addOriginReplacements(originReplacer, originMap);
  promotedCallFunction.getBodyRegion().walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = captureReplacements.find(operand.get());
      if (it == captureReplacements.end())
        continue;
      Value replacement = it->second;
      Type expectedType = operand.get().getType();
      if (expectedType != replacement.getType() &&
          isEqualCanon(expectedType, replacement.getType())) {
        OpBuilder rebindBuilder(op);
        replacement = RebindOp::create(rebindBuilder, op->getLoc(),
                                       expectedType, replacement);
      }
      operand.set(replacement);
    }
    if (!originMap.empty())
      originReplacer.recursivelyReplaceElementsIn(op,
                                                  /*replaceAttrs=*/true,
                                                  /*replaceLocs=*/true,
                                                  /*replaceTypes=*/true);
  });

  return promotedDecl;
}

ClosureEmitter::Closure ClosureEmitter::liftClosure(
    ASTDecl &moduleDecl, SMLoc smLoc,
    SmallVector<ClosureParent> &closureParents, SymbolRefAttr parentSymbolRef,
    SmallVector<StructDefFieldAttr> &&concreteFieldDecls,
    SmallVector<Value> &&concreteFieldCaptures,
    SmallVector<CaptureConvention> &&concreteFieldCaptureConventions,
    SmallVector<ParamDeclAttr> &&concreteParams,
    SmallVector<TypedAttr> &&concreteStructBindings, StringAttr name,
    TypeConvention convention, SmallVector<Type> &&deviceCaptureFieldTypes,
    bool capturesEncodable, ASTDecl &nestedFnDecl) {
  Location location = shared.translateLocation(smLoc);
  MLIRContext *ctx = shared.getContext();

  SmallVector<Type> promotedDeviceCaptureFieldTypes =
      std::move(deviceCaptureFieldTypes);

  SmallVector<TypedAttr> selfRefParamValues = llvm::map_to_vector(
      concreteParams, [](ParamDeclAttr declAttr) -> TypedAttr {
        return ParamDeclRefAttr::get(declAttr);
      });
  SmallVector<StringAttr> paramNames = llvm::map_to_vector(
      concreteParams,
      [](ParamDeclAttr declAttr) -> StringAttr { return declAttr.getName(); });
  StructInstanceType structInstType = StructInstanceType::get(
      StringAttr::get(ctx, Twine(getFlattenedSymbolName(parentSymbolRef))
                               .concat("::")
                               .concat(name.getValue())),
      paramNames, selfRefParamValues, concreteFieldDecls,
      BoolAttr::get(ctx, convention == TypeConvention::MemoryOnly));
  KGEN::StructType kgenStructType = getMlirType(
      ctx, structInstType, concreteParams, concreteStructBindings, convention);
  SmallVector<Type> selfBoundFieldTypes = getConcreteStructFieldTypes(
      structInstType, concreteParams, selfRefParamValues);

  auto structName =
      StringAttr::get(ctx, Twine(kClosurePrefix)
                               .concat(getFlattenedSymbolName(parentSymbolRef))
                               .concat("::")
                               .concat(name.getValue())
                               .concat("::__storage"));

  // Create a StructType to serve as the self. The __call__ method will become
  // a method on the struct
  auto [structDecl, structOp] = createStruct(
      shared, moduleDecl, structName, concreteParams, smLoc,
      SmallVector<PassingKind>(concreteParams.size(), PassingKind::Inferred));
  structOp.setConvention(convention);
  TraitType traitType = getTraitType(shared, closureParents);
  structOp.setCanonicalTrait(traitType);
  OpBuilder structBuilder(structOp.getRegion());
  structBuilder.setInsertionPointToStart(&structOp.getFields().front());
  for (auto [index, fieldDecl] : llvm::enumerate(concreteFieldDecls))
    addFieldOpAndDecl(fieldDecl.getName(), selfBoundFieldTypes[index], structOp,
                      structDecl, structBuilder, *shared.declResolver);
  LIT::StructType closureStructType =
      structOp.bindReference(selfRefParamValues);
  assert(isa<FnOp>(nestedFnDecl.getIfOperation()) &&
         "expected nested closure declaration to be a function");
  PromotedClosureSelfArg selfArg{closureStructType, ArgConvention::ImmMem};
  ASTDecl *promotedCallDecl = liftClosureIntoMethod(
      nestedFnDecl, structDecl, selfArg, concreteFieldDecls,
      concreteFieldCaptures, concreteFieldCaptureConventions,
      selfBoundFieldTypes, location);
  FnOp promotedCallFunction = cast<FnOp>(promotedCallDecl->getIfOperation());
  StructEmitter structEmitter(structDecl);
  DenseMap<ClosureMethod, FnOp> methodImpls;
  methodImpls[ClosureMethod::CALL] = promotedCallFunction;
  auto synthesizeValueMethodBody = [&](bool isMove) -> FnOp {
    FnOp fn = structEmitter.synthesizeEmptyMoveOrCopyInit(/*isMove=*/isMove);
    ASTDecl *decl = shared.declResolver->getDeclForFuncSymbol(
        getFullyResolvedSymbolRef(fn));
    assert(decl && "synthesized value method must be registered");
    (void)structEmitter.populateMoveCopy(*decl, isMove);
    return fn;
  };
  for (const ClosureParent &closureParent : closureParents) {
    switch (closureParent.getClosureMethod()) {
    case ClosureMethod::DEL:
      methodImpls[ClosureMethod::DEL] = structEmitter.synthesizeEmptyDtor();
      break;
    case ClosureMethod::MOVE:
      methodImpls[ClosureMethod::MOVE] =
          synthesizeValueMethodBody(/*isMove=*/true);
      break;
    case ClosureMethod::COPY:
      methodImpls[ClosureMethod::COPY] =
          synthesizeValueMethodBody(/*isMove=*/false);
      break;
    default:
      break;
    }
  }

  ImplicitLocOpBuilder builder(location, ctx);

  // Synthesize the storage struct's initializer.
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ArgConvention> argConventions;

  // Each captured value becomes a positional-only constructor argument.
  assert(concreteFieldDecls.size() == selfBoundFieldTypes.size() &&
         "expected one bound field type per closure field");
  assert(concreteFieldDecls.size() == concreteFieldCaptureConventions.size() &&
         "expected one capture convention per closure field");
  size_t argCount = concreteFieldDecls.size() + 1;
  initArgumentTypes.reserve(argCount);
  argNames.reserve(argCount);
  argPassingKinds.reserve(argCount);
  argConventions.reserve(argCount);
  for (auto [index, fieldDecl] : llvm::enumerate(concreteFieldDecls)) {
    StringAttr fieldName = fieldDecl.getName();
    Type fieldType = selfBoundFieldTypes[index];
    CaptureConvention captureConvention =
        concreteFieldCaptureConventions[index];

    // `__init__`'s byref-result is always `self`; a capture of the same
    // spelling would alias both origins. Keep the field name; rename the arg.
    StringAttr initArgName = fieldName;
    if (fieldName.getValue() == "self")
      initArgName = StringAttr::get(ctx, "__capture_self");

    Type argType;
    ArgConvention argConvention;
    switch (captureConvention) {
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionRef:
      if (sugarIsa<RefType>(fieldType)) {
        argType = fieldType;
        argConvention = ArgConvention::Ref;
        break;
      }
      [[fallthrough]];
    case CaptureConvention::kConventionTrivialCopy:
    case CaptureConvention::kConventionCopy:
      argType = ASTType(fieldType).getRefForArgument(initArgName.getValue(),
                                                     /*isMut=*/false);
      argConvention = ArgConvention::ImmMem;
      break;
    case CaptureConvention::kConventionMove:
      argType = ASTType(fieldType).getRefForArgument(initArgName.getValue(),
                                                     /*isMut=*/true);
      argConvention = ArgConvention::OwnedMem;
      break;
    }

    initArgumentTypes.push_back(argType);
    argConventions.push_back(argConvention);
    argNames.push_back(initArgName);
    argPassingKinds.push_back(PassingKind::PosOnly);
  }

  // The trailing implicit `self` argument is the result slot the constructor
  // initializes.
  ASTType selfType = structDecl.getTypeDeclSelf();
  initArgumentTypes.push_back(
      selfType.getRefForArgument("self", /*isMut=*/true));
  argConventions.push_back(ArgConvention::ByRefResult);
  argNames.push_back(StringAttr::get(ctx, "self"));
  argPassingKinds.push_back(PassingKind::Implicit);

  // The initializer only stores the captures; it never accesses memory through
  // them. Mark it `@__unsafe_nested_origins_read_only` so that closure can
  // capture the same origin multiple times.
  builder.setInsertionPointToEnd(&structOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions, PogListAttr::get(ctx, argNames, argPassingKinds),
      NoneType::get(ctx), SpecialFunctionKind::kInit, smLoc, builder,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true,
      InlineLevel::Automatic, /*isNestedOriginsReadOnly=*/true);

  // Generate the constructor body.
  if (initFnOp) {
    Block *body = initFnOp.getBody();
    ImplicitLocOpBuilder bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(initFnOp.getLoc(), body);
    bodyBuilder.setInsertionPointToStart(body);
    IREmitter emitter(*initDecl, bodyBuilder);

    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(initFnOp.getLocScope());

    // The trailing implicit argument is the `self` result slot to initialize
    Value selfValue = body->getArgument(body->getNumArguments() - 1);
    SmallVector<StructFieldOp> fieldOps =
        llvm::to_vector(structOp.getFieldDecls());
    assert(fieldOps.size() == concreteFieldCaptureConventions.size() &&
           "expected one struct field per capture");

    for (auto [index, fieldOp] : llvm::enumerate(fieldOps)) {
      Value arg = body->getArgument(index);
      Value fieldRef = RefStructGEROp::create(bodyBuilder, selfValue, fieldOp)
                           ->getResults()
                           .front();
      Type fieldType = selfBoundFieldTypes[index];

      // Reference captures
      if (isByReferenceCapture(concreteFieldCaptureConventions[index]) &&
          sugarIsa<RefType>(fieldType)) {
        RefStoreOp::create(bodyBuilder, arg, fieldRef);
        continue;
      }

      // Value captures
      CValue argValue;
      switch (argConventions[index]) {
      case ArgConvention::ImmReg:
        argValue = SRValue(arg);
        break;
      case ArgConvention::ImmMem:
        argValue = MBValue(arg);
        break;
      case ArgConvention::OwnedMem:
        argValue = MRValue(arg);
        break;
      default:
        llvm_unreachable("unexpected argument convention for a value capture");
      }
      SyntheticNode node(structDecl.getLoc());
      emitter.emitStoreToLValue({argValue, &node}, MLValue(fieldRef),
                                EC_AttributeRefBase);
    }

    emitter.emitNormalReturn(initFnOp.getLoc(), /*returnVal=*/Value());
  }

  auto callParentIt = llvm::find_if(closureParents, [](const ClosureParent &p) {
    return p.getClosureMethod() == ClosureMethod::CALL;
  });
  assert(callParentIt != closureParents.end() &&
         "closure parents must include the call trait");
  const ClosureParent &callParent = *callParentIt;
  FnOp callWitness = emitStorageCallWitness(
      structDecl, structOp, promotedCallFunction, smLoc,
      [&](ASTDecl &decl, bool synthetic, StringAttr name) {
        // Storage has no `impl` param — do not redirect Self witnesses to it.
        return pushBackTraitFunctionImpl(
            callParent.getSignature(), decl, synthetic, name,
            SpecialFunctionKind::kNormal, callParent.getInlineLevel());
      });
  if (!callWitness)
    return {};
  methodImpls[ClosureMethod::CALL] = callWitness;

  StringAttr callName = StringAttr::get(ctx, "__call__");

  ASTDecl *callWitnessDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(callWitness));
  assert(callWitnessDecl && "call witness must be registered");
  shared.declResolver->attachDeclToParentNameTable(callWitnessDecl, callName);
  callWitness.setSourceNameAttr(callName);

  // Emit the conformance ops into the storage struct by finding the closure
  // method and FnOp associated with each parent trait.
  auto addWitnessTable = [&](const ClosureParent &closureParent) {
    TraitDeclOp traitParent = closureParent.getTrait(shared);
    builder.setInsertionPointToEnd(&structOp.getFields().front());
    TraitSymbolArrayAttr immediateParents =
        traitParent.getImmediateParentsAttr();
    StringAttr parentName = closureParent.getFlattenedName();
    ConformanceOp witnessTable = ConformanceOp::create(
        builder, closureParent.getSymbol(), immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();

    ASTDecl &conformDecl = shared.declResolver->addDecl(
        witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
    conformDecl.resolvedness = DeclResolvedness::signature;

    // Marker traits like AnyType have no methods -- empty ConformanceOp is
    // sufficient for TypeConformsToTraitAttr::simplify().
    if (closureParent.isEmpty())
      return;

    builder.setInsertionPointToStart(&block);
    ClosureMethod method = closureParent.getClosureMethod();
    auto it = methodImpls.find(method);
    assert(it != methodImpls.end() &&
           "non-marker closure method missing an implementation");

    TypedAttr symbol = buildSymbol(it->second, structOp.getInputParams());
    WitnessOp::create(builder, closureParent.getWitnessName(),
                      /*sym_visibility=*/nullptr, symbol);
  };

  for (const ClosureParent &closureParent : closureParents)
    addWitnessTable(closureParent);

  bool isTrivial = convention == TypeConvention::RegisterPassableTrivial;
  generateIsTrivialSpecialAlias("__del__is_trivial", isTrivial, shared,
                                structDecl, getDeinitableParent());
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", isTrivial, shared,
                                structDecl, getMoveParent());
  if (methodImpls.contains(ClosureMethod::COPY))
    generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", isTrivial, shared,
                                  structDecl, getCopyParent());
  LIT::StructType boundClosureStructType =
      structOp.bindReference(concreteStructBindings);
  TypedAttr typeParamAttr =
      TypeParamAttr::get(boundClosureStructType, kgenStructType, traitType);
  if (capturesEncodable) {
    unsigned numStorageFields = std::distance(structOp.getFieldDecls().begin(),
                                              structOp.getFieldDecls().end());
    assert(promotedDeviceCaptureFieldTypes.size() == numStorageFields &&
           "device field types must match storage struct fields");
    addStorageConformanceToDevicePassable(
        structDecl, promotedDeviceCaptureFieldTypes, name.getValue());
  }
  return Closure{&structDecl, promotedCallDecl, typeParamAttr};
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

static TypeConvention typeConventionOf(SharedState &shared,
                                       LIT::StructType structType) {
  ASTDecl &structDecl =
      shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  return structDeclOp.isRegisterPassableTrivial()
             ? TypeConvention::RegisterPassableTrivial
         : structDeclOp.isRegisterPassable() ? TypeConvention::RegisterPassable
                                             : TypeConvention::MemoryOnly;
}

static TypeConvention typeConventionOf(SharedState &shared, ParamType paramType,
                                       const Capture &capture,
                                       ASTDecl &nestedFnDecl) {
  // The captured value's type may have been refined in the capturing scope,
  // wrapping the parameter reference in a `DowncastAttr` that carries the
  // additional trait bounds (see the by-copy refinement in addCaptureValue).
  // Strip it to recover the underlying parameter reference.
  auto paramRef =
      dyn_cast<ParamDeclRefAttr>(DowncastAttr::strip(paramType.getParam()));
  if (!paramRef) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type is not a parameter reference.");
    return TypeConvention::Unspecified;
  }

  if (!sugarIsa<TraitType>(paramRef.getType())) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type constraint is not a trait.");
    return TypeConvention::Unspecified;
  }

  return ASTType(paramType).getRegisterPassability(nestedFnDecl.getLoc(),
                                                   shared);
}

Value ClosureEmitter::emitClosure(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                                  ArrayRef<Capture> captures, Location location,
                                  bool isCopyable,
                                  ArrayRef<ParamDeclRefAttr> paramCaptures) {
  // (1) Lift the nested function into a storage struct and instantiate it.
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  FnOp parent = nestedFn->getParentOfType<FnOp>();
  assert(parent && "expected the function to be a nested function");
  Block *closureInsertBlock = nestedFn->getBlock();
  Operation *closureInsertBefore = nestedFn->getNextNode();
  ImplicitLocOpBuilder builder(location, shared.getContext());
  builder.setInsertionPoint(nestedFn);
  MLIRContext *ctx = builder.getContext();
  StringAttr fnName = nestedFn.getSourceNameAttr();

  SmallVector<Value> captureValues;
  SmallVector<CValue> constructorArgs;
  SmallVector<CaptureConvention> captureConventions;

  TraitType anyType =
      shared.lookupBuiltinTraitType("AnyType", nestedFnDecl.getLoc());
  IREmitter emitter(*nestedFnDecl.getParentDecl(), builder);

  TypeConvention highestCaptureConvention =
      TypeConvention::RegisterPassableTrivial;
  SmallVector<StructDefFieldAttr> fieldDecls;
  SmallVector<ParamDeclAttr> allStructParams;
  SmallVector<TypedAttr> structParamBindings;
  SmallVector<Type> deviceCaptureFieldTypes;

  SmallPtrSet<StringAttr, 8> byValueCapturedOriginParamNames;
  auto updateCaptureConvention = [&](TypeConvention captureConventionMet) {
    highestCaptureConvention =
        meetCaptureConvention(highestCaptureConvention, captureConventionMet);
  };
  bool allCapturesEncodable = true;
  for (const Capture &capture : captures) {
    Value value = capture.getValue().getMlirValue();
    captureValues.push_back(value);
    if (capture.getCaptureConvention() == CaptureConvention::kConventionMove &&
        sugarIsa<RefType>(value.getType()) &&
        sugarCast<RefType>(value.getType()).isMutableKnown(true))
      constructorArgs.push_back(MRValue(value));
    else
      constructorArgs.push_back(capture.getValue());
    captureConventions.push_back(capture.getCaptureConvention());

    SyntheticNode synthNode(nestedFnDecl.getLoc());
    ExprDest dest(anyType, EC_Type);
    PValue captureTypeValue =
        emitter
            .emitImplicitConversionToType({value.getType(), &synthNode},
                                          anyType, dest)
            .getIfPValue();
    auto captureTypeAttr = cast<TypedAttr>(captureTypeValue.get());
    auto captureName = StringAttr::get(ctx, capture.getSpelling());
    auto captureConvention = capture.getCaptureConvention();
    Type mlirType = value.getType();
    if (auto refType = sugarDynCast<LIT::RefType>(mlirType))
      mlirType = refType.getElementType();
    switch (captureConvention) {
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionRef: {
      // Mutability casts should have been emitted during parse time.
      // TODO: Pointers are register passable, so this demotion
      // should become unnecessary once downstream passes are fixed.
      TypeConvention captureConventionMet =
          (sugarIsa<LIT::RefType>(value.getType())
               ? ASTType(
                     sugarCast<LIT::RefType>(value.getType()).getElementType())
               : ASTType(value.getType()))
              .getRegisterPassability(nestedFnDecl.getLoc(), shared);
      updateCaptureConvention(captureConventionMet);
      break;
    }
    case CaptureConvention::kConventionTrivialCopy:
      break;
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionMove: {
      if (auto refType = sugarDynCast<LIT::RefType>(value.getType())) {
        if (auto captureOriginParam = dyn_cast<ParamDeclRefAttr>(
                OriginType::stripMutCastAndRebind(refType.getOrigin())))
          byValueCapturedOriginParamNames.insert(captureOriginParam.getName());
      }
      // Copy/move captures materialize storage for the captured value itself,
      // not for a reference wrapper. Use the pointee as the field type.
      if (sugarIsa<LIT::RefType>(value.getType()))
        captureTypeAttr = TypeParamAttr::get(mlirType, anyType);

      if (auto structType = sugarDynCast<StructType>(mlirType)) {
        updateCaptureConvention(typeConventionOf(shared, structType));
      } else if (sugarIsa<TraitType>(mlirType)) {
        shared.emitError(nestedFnDecl.getLoc(),
                         "cannot capture a value of trait type yet because "
                         "existentials are not implemented.");
        return {};
      } else if (auto paramType = sugarDynCast<ParamType>(mlirType)) {
        updateCaptureConvention(
            typeConventionOf(shared, paramType, capture, nestedFnDecl));
      }
      break;
    }
    }
    fieldDecls.push_back(StructDefFieldAttr::get(captureName, captureTypeAttr));
    if (allCapturesEncodable) {
      // A by-reference capture stores a host pointer (LIT::RefType) as its
      // storage field, while its device field type is computed from the
      // pointee. The two disagree in `encode_fields` (ref != pointee device
      // type), so a reference is not device-encodable.
      if (isByReferenceCapture(captureConvention)) {
        allCapturesEncodable = false;
      } else {
        FailureOr<Type> deviceFieldType = getReboundCaptureDeviceFieldType(
            ASTType(mlirType), nestedFnDecl, shared);
        if (failed(deviceFieldType))
          allCapturesEncodable = false;
        else
          deviceCaptureFieldTypes.push_back(*deviceFieldType);
      }
    }
  }
  // TODO(MOCO-4045): DevicePassable conformance currently requires a
  // register-passable storage struct.
  if (allCapturesEncodable &&
      highestCaptureConvention == TypeConvention::MemoryOnly)
    allCapturesEncodable = false;

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

  // (1) Outline interior origins and promote origin parameters in a unified
  // pass.
  if (!allCapturesEncodable)
    deviceCaptureFieldTypes.clear();
  if (failed(outlineAndPromoteOrigins(shared, fieldDecls, allStructParams,
                                      structParamBindings, nestedFn,
                                      deviceCaptureFieldTypes, location)))
    return {};

  // Bind the closure signature after origin is promoted.
  SmallVector<ClosureParent> closureParents;
  TraitSymbolAttr boundSymbol =
      emitter.bindParamsToClosureTraitFromSig(nestedFn.getFuncTypeGenerator());
  closureParents.emplace_back(boundSymbol, shared.getClosureFnSig(boundSymbol),
                              StringAttr::get(shared.getContext(), "__call__"),
                              ClosureMethod::CALL);
  closureParents.append(
      {getMoveParent(), getDeinitableParent(), getAnyParent()});

  if (isCopyable) {
    closureParents.push_back(getCopyParent());
    closureParents.push_back(getImplicitlyCopyableParent());
  }
  if (highestCaptureConvention == TypeConvention::RegisterPassableTrivial) {
    closureParents.push_back(getTrivialRegisterTypeParent());
    closureParents.push_back(getRegisterPassableParent());
  } else if (highestCaptureConvention == TypeConvention::RegisterPassable)
    closureParents.push_back(getRegisterPassableParent());

  // Storage bindings passed to initializer call.
  SmallVector<TypedAttr> storageParamBindings = structParamBindings;

  Closure liftedClosure = liftClosure(
      moduleDecl, nestedFnDecl.getLoc(), closureParents,
      SymbolRefAttr::get(
          ctx,
          getFlattenedSymbolName(getFullyResolvedSymbolRefUpTo<FileModuleOp>(
              cast<mlir::SymbolOpInterface>(parent.getOperation())))),
      std::move(fieldDecls), std::move(captureValues),
      std::move(captureConventions), std::move(allStructParams),
      std::move(structParamBindings), fnName, highestCaptureConvention,
      std::move(deviceCaptureFieldTypes), allCapturesEncodable, nestedFnDecl);
  if (!liftedClosure.structDecl)
    return {};

  // The nested closure function is moved into the storage struct as a method.
  // Emit closure materialization ops back in the original parent function body.
  if (closureInsertBefore &&
      closureInsertBefore->getBlock() == closureInsertBlock)
    builder.setInsertionPoint(closureInsertBefore);
  else
    builder.setInsertionPointToEnd(closureInsertBlock);

  // Instantiate the storage struct directly through its synthesized `__init__`,
  // passing the captured values as the positional arguments.
  StructDeclOp storageStructOp =
      cast<StructDeclOp>(liftedClosure.structDecl->getIfOperation());
  VarDeclOp storageVar = emitInitializerCall(
      *nestedFnDecl.getParentDecl(), builder, location, storageStructOp,
      storageParamBindings, /*args=*/constructorArgs, fnName.getValue());

  return MLValue(storageVar);
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
// anything. If `upperBound` is set, the walk includes that decl and then stops.
static FailureOr<ASTDecl *> findCapture(SharedState &shared, StringRef name,
                                        llvm::SMLoc loc, ASTDecl &scope,
                                        ASTDecl *upperBound = nullptr) {
  auto nameAttr = StringAttr::get(shared.getContext(), name);
  ASTDecl *current = &scope;
  do {
    FailureOr<ASTDecl *> result = partialLookup(nameAttr, *current, loc);
    if (failed(result))
      return failure();
    if (result.value()) {
      // Error already diagnosed.
      if (result.value()->isErroneous())
        return failure();
      return result.value();
    }
    if (current == upperBound)
      break;
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
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  ASTDecl *fnParentDecl = closure.getParentDecl()->getNearestDeclOfType<FnOp>();
  auto parentFn = cast<FnOp>(fnParentDecl->getIfOperation());
  ASTDecl *result = nullptr;
  if (usesClosurePipeline(parentFn)) {
    auto localMaybe = findCapture(shared, name, location,
                                  *closure.getParentDecl(), fnParentDecl);
    if (failed(localMaybe))
      return nullptr;
    result = localMaybe.value();
    if (!result) {
      result = addCaptureValue(shared, *fnParentDecl, name, location);
      if (!result)
        return nullptr;
    }
  } else {
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

  // Capture materialization is inserted into the enclosing function. Stamp
  // those ops with that function's debug scope, using the inner closure's
  // file/line only. Keeping the inner subprogram on the loc would lower as
  // an inlined location and fail LLVM's dbg-value verifier.
  emitter.builder->setInsertionPoint(closure.getIfOperation());
  DebugInfo::DIBuilder::ScopeGuard diGuard;
  Location captureLoc = funcOp.getLoc();
  if (auto fileLoc = captureLoc->findInstanceOf<FileLineColLoc>())
    captureLoc = fileLoc;
  if (shared.diBuilder) {
    diGuard = shared.diBuilder->pushScopeGuard(parentFn.getLocScope());
    captureLoc = shared.diBuilder->createScopedLoc(captureLoc);
  }

  CValue valueInParent = ASTDeclToCValue(result, *emitter.builder, captureLoc);
  if (!valueInParent) {
    shared.emitError(location, "'")
        << name << "' does not name a capturable value";
    return nullptr;
  }

  CaptureConvention convention;
  /// The captureValue is a map of the valueInParent. For example, the
  /// valueInParent may be an immutable borrowed value. If this value is
  /// captured by copy the capturedValue in the body of the closure is a
  /// mutable owned value. Since the captured value does not exist until
  /// later, we have to create a temporary value to represent the change in
  /// the properties of the value in the body of the closure.
  CValue captureValue;

  auto captureByRef = [&](CValue value, TriBool mutability) -> CValue {
    // Ensure we are not capturing an immutable reference by mutable
    // reference.
    if (auto refType = sugarDynCast<RefType>(value.getType().mlirType)) {
      // If the mutability is not specified or the reference type match the
      // specified mutability, return the original value.
      OriginType originType = refType.getOriginType();
      if (mutability.isUnknown() ||
          originType.isMutableKnown(mutability.isTrue()))
        return value;

      if (originType.isMutableKnown(false)) {
        // mutable capture of an immutable reference, error.
        shared.emitError(location, "Cannot capture ")
            << name << " by mut because it could be immutable";
        return {};
      }

      if (originType.isMutableKnown(true)) {
        // convert a mut ref to immut ref
        auto refImmutOp = LIT::RefImmutOp::create(*emitter.builder, captureLoc,
                                                  valueInParent.getMlirValue());
        return MBValue(refImmutOp->getResult(0));
      }
    }

    // Not a reference capture, then it must be a read effect.
    if (mutability.isFalse())
      return value;

    shared.emitError(location, "register passible value '")
        << name << "' can not be captured by "
        << (mutability.isDefinite() ? "'mut'" : "'ref'")
        << ". Do you mean 'imm'?";
    return {};
  };

  // Apply scope-based type refinement before by-value capture checks. A
  // parameter may gain extra `conforms_to` constraints in the capturing scope;
  // without rebinding to that refined type, move/copy validation would still
  // see the original generic bound.
  if (parsedConvention == CaptureConvention::kConventionMove ||
      parsedConvention == CaptureConvention::kConventionCopy) {
    SyntheticNode refineNode(location);
    valueInParent =
        maybeEmitRefinementRebind({valueInParent, &refineNode}, emitter);
  }

  switch (parsedConvention) {
  case CaptureConvention::kConventionMove: {
    Type type = valueInParent.getType().mlirType;
    if (auto ref = sugarDynCast<RefType>(valueInParent.getType().mlirType))
      type = ref.getElementType();
    if (!ASTType(type).isMovable(closure.getLoc(), shared, *fnParentDecl)) {
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
    if (sugarIsa<RefType>(valueInParent.getType().mlirType))
      captureValue = CValue::getMValueForRef(valueInParent.getMlirValue());
    else
      captureValue = MRValue(valueInParent.getMlirValue());
    break;
  }
  case CaptureConvention::kConventionCopy: {
    ASTType originalType = valueInParent.getRValueType();
    if (originalType.isTrivial(closure.getLoc(), shared)) {
      // Remap to trivial copy convention to avoid storing symbols.
      convention = CaptureConvention::kConventionTrivialCopy;
      // if we are capturing by mutable copy and its trivial do not capture
      // the reference.
      if (sugarIsa<RefType>(valueInParent.getType())) {
        SyntheticNode node(result->getLoc());
        ExprDest dest(EC_Capture);
        captureValue = emitter.emitRValue(
            {CValue::getMValueForRef(valueInParent.getMlirValue()), &node},
            dest);
      } else {
        captureValue = valueInParent;
      }
    } else {
      convention = parsedConvention;
      if (auto refType =
              sugarDynCast<RefType>(valueInParent.getType().mlirType)) {
        OriginType originType = refType.getOriginType();
        if (originType.isMutableKnown(false)) {
          auto refImmutOp = LIT::RefImmutOp::create(
              *emitter.builder, captureLoc, valueInParent.getMlirValue());
          captureValue = MBValue(refImmutOp->getResult(0));
        }
      }
      ExprDest dest(EC_Capture);
      SyntheticNode node(result->getLoc());
      ASTExprAnd<CValue> valueInParentExpr{valueInParent, &node};
      LValue copiedOrMovedValue =
          dest.getLValueForResult(valueInParentExpr.expr->getLoc(),
                                  valueInParentExpr.ir.getRValueType(),
                                  /*allowIncompatibleTypes=*/false,
                                  /*requireMLValue=*/false, emitter);
      emitter.emitStoreToLValue(valueInParentExpr, copiedOrMovedValue,
                                dest.getContext());
      // Diagnose an uncopyable capture.
      if (!originalType.isCopyable(closure.getLoc(), shared,
                                   /*isImplicit=*/false, *fnParentDecl)) {
        shared.emitError(location, "cannot capture ")
            << name << " by copy because it is not copyable.";
        return nullptr;
      }
      captureValue = copiedOrMovedValue;
    }
    break;
  }
  case CaptureConvention::kConventionMut:
  case CaptureConvention::kConventionRead:
  case CaptureConvention::kConventionRef: {
    convention = parsedConvention;
    auto mutability = [convention]() -> TriBool {
      if (convention == CaptureConvention::kConventionRef)
        return TriBool::unknown();
      return TriBool::fromBool(convention == CaptureConvention::kConventionMut);
    }();
    captureValue = captureByRef(valueInParent, mutability);
    if (!captureValue)
      return nullptr;
    break;
  }
  case CaptureConvention::kConventionTrivialCopy:
    llvm_unreachable("trivial copy is derived from by-copy, not parsed");
  case CaptureConvention::kConventionUnspecified:
    shared.emitError(
        location, "Could not infer capture convention of the captured value ")
        << name;
    return nullptr;
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

PValue ClosureEmitter::createParamClosureExtensionType(
    IREmitter &emitter, ASTExprAnd<CValue> srcTypeVal,
    TraitSymbolAttr srcClosureInst, TraitSymbolAttr tgtClosureInst) {
  auto newSelfVal = ParamDeclRefAttr::get(
      "#Closure_Ext#", shared.declResolver->getCanonicalTrait(srcClosureInst));
  auto srcSig =
      LIT::specializeSignature(shared.getClosureFnSig(srcClosureInst),
                               ASTType(newSelfVal), shared.getDeclResolver());
  auto tgtSig =
      LIT::specializeSignature(shared.getClosureFnSig(tgtClosureInst),
                               ASTType(newSelfVal), shared.getDeclResolver());
  assert(srcSig && tgtSig);

  auto callName = StringAttr::get(shared.getContext(), "__call__");

  // This is the function that we try to convert, the selfVal is synthetic,
  // guaranteed to be un-foldable.
  auto toConvert =
      GetWitnessAttr::get(newSelfVal, srcClosureInst, callName, srcSig);

  ExprDest dest(EC_TypeParamValue);
  TypedAttr converted =
      emitter
          .emitImplicitConversionToType({PValue(toConvert), srcTypeVal.expr},
                                        tgtSig, dest)
          .getIfPValue();
  assert(converted && "trait convertibility must have been tested");

  // This is the bridging thunk generated.
  auto thunkSymbol =
      cast<SymbolConstantAttr>(ParamOperatorAttr::stripRebind(converted));

  // Collecting all the parameter
  SmallVector<TypedAttr> thunkParams;
  for (auto param : thunkSymbol.getParamValues()) {
    if (sugarIsa<UnboundAttr>(param))
      continue;
    thunkParams.push_back(param);
  }

  assert(thunkSymbol.getParamValues().size() - thunkParams.size() ==
         tgtSig.getInputParamTypes().size());

  // The generated thunk always append an extra parameter for the function.
  assert(isEqualCanon(toConvert,
                      ParamOperatorAttr::stripRebind(thunkParams.back())));
  thunkParams.pop_back();

#ifndef MODULAR_PRODUCTION
  llvm::SmallSetVector<ParamDeclRefAttr, 4> capturedParamRef;
  getCanonicalType(toConvert.getType()).walk([&](ParamDeclRefAttr ref) {
    capturedParamRef.insert(ref);
  });

  // The captures parameter must lined up with the bound parameter in the
  // thunk.
  auto captures = capturedParamRef.takeVector();
  for (auto [t, c] : llvm::zip_equal(thunkParams, captures))
    assert(isEqualCanon(t, c) && "mismatched captures?");
#endif

  // Hoist the parameter ref to the extension struct, reuse the name so that we
  // don't need to remapped the name.
  SmallVector<ParamDeclAttr> structParams;
  SmallVector<TypedAttr> bindings;
  for (TypedAttr capture : thunkParams) {
    auto ref = cast<ParamDeclRefAttr>(capture);
    structParams.push_back(ParamDeclAttr::get(ref.getName(), ref.getType()));
    // Every capture binds back to the value it stood for at the conversion
    // site; the synthetic `Self` binds to the closure being extended.
    if (isEqualCanon(ref, newSelfVal))
      bindings.push_back(
          UpcastAttr::get(ref.getType(), srcTypeVal.ir.getIfPValue()));
    else
      bindings.push_back(ref);
  }

  // TODO: The cache key includes pog list, we can potentially strip in order to
  // get fewer extension struct
  auto thunkKey =
      ArrayAttr::get(ctx, {TypeAttr::get(tgtSig), TypeAttr::get(srcSig)});
  StructDeclOp extDeclOp =
      shared.getOrCreateParamClosureExtension(thunkKey, [&] {
        std::string extName(kClosureExtensionPrefix);
        llvm::raw_string_ostream os(extName);
        generateConversionThunkName(os, {tgtSig, srcSig});

        auto [structDecl, declOp] = createStruct(
            shared, shared.getTopLevelDecl(), StringAttr::get(ctx, extName),
            structParams, srcTypeVal.expr->getLoc(),
            SmallVector<PassingKind>(structParams.size(),
                                     PassingKind::PosOnly));
        declOp.setClosureThunkKeyAttr(thunkKey);

        // The bridging thunk is the whole conformance; there is nothing to
        // store.
        declOp.setConvention(TypeConvention::RegisterPassable);
        addConformanceTable(
            structDecl,
            ClosureParent(tgtClosureInst,
                          shared.getClosureFnSig(tgtClosureInst), callName,
                          ClosureMethod::CALL),
            {{callName.getValue(), converted}});
        return declOp;
      });

  StructType extType = extDeclOp.bindReference(bindings);
  return PValue(TypeParamAttr::get(extType, StructMetaType::get(extType)));
}

/// Push `fnOp`'s debug scope for the duration of emitting its body.
static DebugInfo::DIBuilder::ScopeGuard pushFnDebugScope(SharedState &shared,
                                                         FnOp fnOp) {
  if (!shared.diBuilder)
    return {};
  return shared.diBuilder->pushScopeGuard(fnOp.getLocScope());
}

static void populateDevicePassableTypeName(FnOp implementation,
                                           ASTDecl &structDecl,
                                           TypedAttr closureName) {
  MLIRContext *ctx = structDecl.getContext();
  Block &block = implementation.getBodyRegion().front();
  ImplicitLocOpBuilder b(implementation.getLoc(), implementation);
  b.setInsertionPointToStart(&block);
  IREmitter emitter(structDecl, b);
  SyntheticNode loc(structDecl.getLoc());

  ASTType strLitType = structDecl.getShared().lookupBuiltinType(
      "StringLiteral", structDecl, structDecl.getLoc());
  auto strLitDecl = cast<StructDeclOp>(
      strLitType.getDecl(structDecl.getShared())->getIfOperation());
  Type boundStrLitType = strLitDecl.bindReference({closureName});
  CValue literalValue = emitter.emitConstructorCall(
      ASTType(boundStrLitType),
      CallOperands(CallSyntax::kTypeCall, &loc, EC_CallArgValue));

  ASTType stringType = structDecl.getShared().lookupBuiltinType(
      "String", structDecl, structDecl.getLoc());
  ExprDest resultDest(MLValue(block.getArguments().back()), EC_ReturnValue);
  CallOperands ctorOperands(CallSyntax::kTypeCall, &loc, std::move(resultDest));
  ctorOperands.add(ASTExprAnd<CValue>{literalValue, &loc});
  emitter.emitConstructorCall(stringType, std::move(ctorOperands));
  auto noneAttr = KGEN::ParamConstantOp::create(b, KGEN::NoneAttr::get(ctx));
  IREmitter::emitNormalReturn(b, noneAttr);
}

static void emitIsConvertibleToDeviceTypeBody(
    FnOp implementation, ArrayRef<ParamDeclAttr> parameters,
    ImplicitLocOpBuilder &b, TypedAttr deviceTypeAttr) {
  b.setInsertionPointToStart(&implementation.getBodyRegion().front());
  assert(!parameters.empty() &&
         "expected _is_convertible_to_device_type to have type parameter");
  TypedAttr targetType = ParamDeclRefAttr::get(parameters.front());
  TypedAttr isConvertible = ParamIdenticalAttr::get(targetType, deviceTypeAttr);
  auto isConvertibleValue = KGEN::ParamConstantOp::create(b, isConvertible);
  IREmitter::emitNormalReturn(b, isConvertibleValue);
}

// TODO: replace clone with witness entry that binds self parameter once self
// parameter of traits becomes function level.
static void cloneTraitDefaultBody(FnOp implementation, FnOp traitFn,
                                  ASTDecl &structDecl) {
  implementation.getBodyRegion().getBlocks().clear();
  IRMapping mapping;
  traitFn.getBodyRegion().cloneInto(&implementation.getBodyRegion(), mapping);

  // Map `traitFn`'s parameters to `implementation` parameter.
  DenseMap<StringAttr, StringAttr> paramNames;
  for (auto [traitParam, implParam] :
       llvm::zip(traitFn.getInputParams(), implementation.getInputParams()))
    paramNames.insert({traitParam.getName(), implParam.getName()});

  ASTType selfType = structDecl.getTypeDeclSelf();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
    if (auto it = paramNames.find(paramRef.getName()); it != paramNames.end())
      return ParamDeclRefAttr::get(it->second, paramRef.getType());
    return paramRef;
  });
  replacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
    SmallString<64> buf;
    llvm::raw_svector_ostream os(buf);
    getWitness.getTypeValue().print(os);
    if (!StringRef(buf).contains("_Self"))
      return getWitness;
    return GetWitnessAttr::get(PValue(selfType), getWitness.getTraitSymbol(),
                               getWitness.getWitnessName(),
                               getWitness.getType());
  });
  implementation.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/false, /*replaceUses=*/false);
  });
}

static AliasDeclOp getDeviceTypeAlias(SharedState &shared, llvm::SMLoc loc) {
  ASTDecl *devicePassableTrait = shared.getBuiltinDevicePassableTrait(loc);
  assert(devicePassableTrait && "DevicePassable trait should be present");
  ArrayRef<ASTDecl *> aliasDecls = devicePassableTrait->lookupInCurrentScope(
      StringAttr::get(shared.getContext(), kDeviceType));
  assert(aliasDecls.size() == 1 &&
         "DevicePassable trait should define one device_type alias");
  return cast<AliasDeclOp>(aliasDecls.front()->getIfOperation());
}

void ClosureEmitter::addConformanceToDevicePassable(
    ASTDecl &structDecl, const DevicePassablePopulators &populators) {
  ASTDecl *devicePassableTrait =
      shared.getBuiltinDevicePassableTrait(structDecl.getLoc());
  if (!devicePassableTrait)
    return;
  if (failed(shared.declResolver->resolveBody(*devicePassableTrait,
                                              devicePassableTrait->getLoc())))
    return;
  TraitDeclOp trait = cast<TraitDeclOp>(devicePassableTrait->getIfOperation());
  for (auto &nameGroup : devicePassableTrait->getDeclsInScope()) {
    for (ASTDecl *funcFieldOrAlias : nameGroup.second) {
      if (failed(shared.declResolver->resolveBody(*funcFieldOrAlias,
                                                  funcFieldOrAlias->getLoc())))
        return;
    }
  }

  SmallVector<std::pair<StringRef, TypedAttr>> devicePassableWitnesses;
  TypedAttr deviceTypeWitness = populators.deviceType();

  for (Operation &member : trait.getFields().getOps()) {
    if (auto function = dyn_cast<FnOp>(member)) {
      FailureOr<SymbolConstantAttr> witness =
          [&]() -> FailureOr<SymbolConstantAttr> {
        if (function.getSourceName() == kIsDeviceTypeConvertible)
          return populators.isConvertible(function);
        if (function.getSourceName() == kIsImplicitlyEncodableTo)
          return populators.isEncodable(function);
        if (function.getSourceName() == kToDeviceType)
          return populators.toDeviceType(function);
        if (function.getIsStatic() &&
            function.getUserResultType() ==
                shared.lookupBuiltinType("String", structDecl,
                                         structDecl.getLoc()))
          return populators.typeName(function);
        llvm_unreachable("unexpected function in DevicePassable trait");
      }();
      if (failed(witness))
        return;
      devicePassableWitnesses.push_back(
          {*function.getSymName(), std::move(*witness)});
      continue;
    }

    if (auto alias = dyn_cast<AliasDeclOp>(member)) {
      assert(alias.getDeclName().getValue() == kDeviceType &&
             "unexpected alias in DevicePassable trait");
      devicePassableWitnesses.push_back({kDeviceType, deviceTypeWitness});
      continue;
    }
    llvm_unreachable(("unexpected member type '" +
                      member.getName().getStringRef().str() +
                      "' encountered in DevicePassable trait")
                         .c_str());
  }
  ClosureParent devicePassableParent(shared, trait.bindReference({}), "",
                                     ClosureMethod::NONE);
  addConformanceTable(structDecl, devicePassableParent,
                      devicePassableWitnesses);
}

void ClosureEmitter::addStorageConformanceToDevicePassable(
    ASTDecl &structDecl, ArrayRef<Type> deviceCaptureFieldTypes,
    StringRef name) {
  ASTDecl *fileModule = structDecl.getNearestDeclOfType<FileModuleOp>();
  if (!fileModule) {
    // for parametric trait based closure, the struct decl is put within the top
    // level decl.
    fileModule = &structDecl.getShared().getTopLevelDecl();
  }
  MLIRContext *ctx = structDecl.getContext();
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp);
  FailureOr<LIT::StructType> deviceType = createDeviceTypeStruct(
      shared, *fileModule, structDecl, deviceCaptureFieldTypes);
  if (failed(deviceType))
    return;

  TypedAttr deviceTypeValue;
  auto populateIsConvertible =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, parameters, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        inlineLevelOrAutomatic(function.getInlineLevel()));
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard =
        pushFnDebugScope(shared, implementation);
    emitIsConvertibleToDeviceTypeBody(implementation, parameters, b,
                                      deviceTypeValue);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateIsEncodable =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, parameters, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl,
        /*synthetic=*/true, function.getSymNameAttr(),
        function.getSpecialFunctionKind(),
        inlineLevelOrAutomatic(function.getInlineLevel()));
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard =
        pushFnDebugScope(shared, implementation);
    cloneTraitDefaultBody(implementation, function, structDecl);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateToDeviceType =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [toDevice, params, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        inlineLevelOrAutomatic(function.getInlineLevel()));
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard =
        pushFnDebugScope(shared, toDevice);
    b.setInsertionPointToStart(&toDevice.getBodyRegion().front());
    assert(toDevice.getBodyRegion().getNumArguments() == 3);

    Value selfArgument = toDevice.getBodyRegion().front().getArgument(0);
    Value encoderRef = toDevice.getBodyRegion().front().getArgument(1);
    Value targetArgument = toDevice.getBodyRegion().front().getArgument(2);

    IREmitter emitter(structDecl, b);
    SyntheticNode syntheticNode(structDecl.getLoc());
    ExprDest dest(EC_ReturnValue);
    CallOperands callOperands(CallSyntax::kMethodCall, &syntheticNode,
                              std::move(dest));
    CValue encoderValue = CValue::getMValueForRef(encoderRef);
    callOperands.add({encoderValue, &syntheticNode});
    callOperands.add({CValue::getMValueForRef(selfArgument), &syntheticNode});
    callOperands.add({SRValue(targetArgument), &syntheticNode});
    OverloadSet overloads = OverloadSet::lookup(
        structDecl, encoderValue.getRValueType(), "encode_closure_state",
        &syntheticNode, CallSyntax::kMethodCall);
    overloads.paramBindings.add(&syntheticNode, PValue(deviceTypeValue),
                                StringAttr::get(ctx, "DeviceStructType"));
    auto calleeResult = overloads.filterOverloadSet(
        callOperands, /*emitDiagnosticOnFailure=*/true, emitter);
    if (!calleeResult.isYes())
      return failure();
    CValue callResult = emitter.emitIndirectCall(calleeResult.getYes(),
                                                 std::move(callOperands));
    if (!callResult)
      return failure();
    auto noneAttr =
        KGEN::ParamConstantOp::create(b, KGEN::NoneAttr::get(b.getContext()));
    IREmitter::emitNormalReturn(b, noneAttr);

    return buildSymbol(toDevice, structDeclOp.getInputParams());
  };
  auto populateTypeName = [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, _, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        inlineLevelOrAutomatic(function.getInlineLevel()));
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard =
        pushFnDebugScope(shared, implementation);
    auto closureName = StringAttr::get(name, StringType::get(ctx));
    populateDevicePassableTypeName(implementation, structDecl, closureName);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateDeviceType = [&]() {
    deviceTypeValue = TypeParamAttr::get(
        *deviceType, getDeviceTypeAlias(shared, structDecl.getLoc()).getType());
    return deviceTypeValue;
  };
  DevicePassablePopulators populators{populateIsConvertible,
                                      populateIsEncodable, populateToDeviceType,
                                      populateTypeName, populateDeviceType};
  addConformanceToDevicePassable(structDecl, populators);
}
