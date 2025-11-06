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
#include "CallEmission.h"
#include "IREmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "Traits.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/NameMangling.h"
#include "Support/Compiler/OperationUtils.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// File-local
namespace {
static constexpr char kToDeviceType[] = "_to_device_type";
}

static FnOp getFnOpNamed(TraitDeclOp traitDecl, StringRef name) {
  for (FnOp candidate : traitDecl.getFields().getOps<FnOp>()) {
    StringRef sourceName = *candidate.getSourceName();
    if (sourceName.contains(name))
      return candidate;
  }
  return {};
}

static void updateLocationScope(Location nestedLocation,
                                Location callFuncLocation,
                                Operation *replacementTarget) {
  DebugInfo::DISubprogramAttr subprogramAttrOfCallFunc;
  if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
          callFuncLocation)) {
    subprogramAttrOfCallFunc = fusedLoc.getMetadata();
    DebugInfo::DISubprogramAttr subprogramAttrOfOriginalFunc;
    if (auto fusedLocOriginal =
            dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
                nestedLocation))
      subprogramAttrOfOriginalFunc = fusedLocOriginal.getMetadata();

    // After cloning the DI attributes will be referencing the original
    // function. We need it to reference the new function. Traverse each
    // operation and attributes recursively to update all the DI attributes.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](DebugInfo::DISubprogramAttr sp) {
      if (subprogramAttrOfOriginalFunc == sp)
        return subprogramAttrOfCallFunc;
      return sp;
    });
    replacer.recursivelyReplaceElementsIn(replacementTarget, true, true);
  }
}

ClosureEmitter::ClosureEmitter(SharedState &shared)
    : FunctionEmitter(shared), ctx(shared.getContext()),
      selfName(StringAttr::get(ctx, "self")),
      otherName(StringAttr::get(ctx, "other")),
      dtorFieldAttr(StringAttr::get(ctx, "dtor")),
      copyFieldAttr(StringAttr::get(ctx, "_copy")),
      callFieldAttr(StringAttr::get(ctx, "call")),
      callMethodAttr(StringAttr::get(ctx, "closureCallMethod")),
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))),
      moveParent("Movable", "__moveinit__", ClosureMethod::MOVE),
      anyParent("AnyType", "__del__", ClosureMethod::DEL),
      copyParent("Copyable", "__copyinit__", ClosureMethod::COPY),
      implicitlyCopyableParent("ImplicitlyCopyable", "", ClosureMethod::NONE) {}

TraitDeclOp ClosureEmitter::ClosureParent::getTrait(ASTDecl &moduleDecl) {
  if (trait)
    return trait;
  SharedState &shared = moduleDecl.getShared();
  auto traitDeclParent =
      shared.lookupBuiltinTrait(traitName, &moduleDecl, moduleDecl.getLoc());
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

static Value loadField(ImplicitLocOpBuilder &b, Value self,
                       StructFieldOp field) {
  return RefLoadOp::create(b, RefStructGEROp::create(b, self, field));
}
static void storeField(ImplicitLocOpBuilder &b, Value self, Value value,
                       StructFieldOp field) {
  RefStoreOp::create(b, value, RefStructGEROp::create(b, self, field));
}
static void storeField(ImplicitLocOpBuilder &b, Value self, Value value,
                       StringAttr name) {
  auto resultTy = RefStructGEROp::getReboundFieldType(
      cast<RefType>(self.getType()), name, value.getType());
  auto fieldRef = RefStructGEROp::create(b, resultTy, name, self);
  RefStoreOp::create(b, value, fieldRef);
}

static std::pair<ASTDecl &, StructDeclOp>
createStruct(SharedState &shared, ASTDecl &moduleDecl, StringAttr name,
             ArrayRef<ParamDeclAttr> params, SMLoc loc) {
  auto module = cast_or_null<FileModuleOp>(moduleDecl.getIfOperation());
  OpBuilder b(module.getRegion());
  SmallVector<StringAttr> paramNames;
  for (ParamDeclAttr param : params) {
    paramNames.push_back(StringAttr::get(
        b.getContext(), demangleParameterName(param.getName())));
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
  PogListAttr argListAttr = oldFnMetadata.getArgListAttrs();
  llvm::append_range(signatureInputs, sig.getArguments());
  llvm::append_range(argConventions, sig.getArgConventions());
  llvm::append_range(argPogs, argListAttr.getPogs());
  assert(argPogs.size() == argConventions.size());

  // A closure signature is not escaping because its 'escaping' state is
  // captured in the self argument we are inserting in this function.
  auto metadata = FnMetadataAttr::get(
      argListAttr.cloneWith(argPogs), oldFnMetadata.getNumImplicitOriginDecls(),
      oldFnMetadata.getCaptureOrigins(),
      oldFnMetadata.getIsNestedOriginExclusivityCheckingDisabled(),
      oldFnMetadata.getConstraints());
  return FuncTypeGeneratorType::get(
      sig.getInputParamTypes(),
      FunctionType::get(ctx, signatureInputs, sig.getResults()), argConventions,
      sig.getFnEffects().setEscaping(false), metadata, sig.getMetadata());
}

/// ```mojo
/// fn __init__(f: fn_ptr_type, out self):
///     self.field0 = f
///     self.dtor = __closure_wrapper_noop_dtor
///     self.copy = __closure_wrapper_noop_copy
///     fn call_impl(field0: !kgen.pointer<none>, *args):
///         return (fn_ptr_type)(field0)(*args)
///     self.call = call_impl
/// ```
void ClosureEmitter::synthesizeWrapperFnPtrCtor(ASTDecl &decl, ASTType selfType,
                                                FnTypeGeneratorType sig) {
  // Skip this if builtins are not found.
  if (!shared.hasBuiltinModule())
    return;

  // Declare the function.
  FnTypeGeneratorType fnPtrType =
      sig.getWithBody(sig.getBody().getWithFnEffects(
          sig.getBody().getFnEffects().setEscaping(false)));
  auto b = ImplicitLocOpBuilder::atBlockEnd(
      translateLocation(decl.getLoc()),
      &cast<StructDeclOp>(decl.getIfOperation()).getFields().front());
  auto argListAttrs =
      PogListAttr::get(ctx, {otherName, selfName},
                       {PassingKind::PosOrKw, PassingKind::Implicit});
  auto [func, _] = synthesizeFunction(
      decl, "__init__", /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(ctx),
      {fnPtrType, selfType.getRefForArgument("self", /*isMut=*/true)},
      {ArgConvention::ReadReg, ArgConvention::ByRefResult}, argListAttrs,
      shared.getNoneType(), SpecialFunctionKind::kInit, decl.getLoc(), b);
  func.setInlineLevel(InlineLevel::Always);

  Value self = func.getArgument(1);
  b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());

  // Store the function pointer into the pointer field.
  Value opaqueFnPtr =
      POP::PointerBitcastOp::create(b, opaquePtrType, func.getArgument(0));
  storeField(b, self, opaqueFnPtr, b.getStringAttr("field0"));

  // Use the no-op destructor and copy constructor.
  ArrayRef<ASTDecl *> dtor = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_dtor", decl.getLoc());
  ArrayRef<ASTDecl *> copy = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_copy", decl.getLoc());
  if (dtor.empty() || copy.empty())
    return;

  Value dtorRef = CreateClosureOp::create(
      b, cast<FnOp>(dtor.front()->getIfOperation())
             .getBoundReference(shared.getEvaluationContext()));
  Value copyRef = CreateClosureOp::create(
      b, cast<FnOp>(copy.front()->getIfOperation())
             .getBoundReference(shared.getEvaluationContext()));
  storeField(b, self, dtorRef, b.getStringAttr("dtor"));
  storeField(b, self, copyRef, b.getStringAttr("_copy"));

  // Generate the 'call_impl' function that performs the indirect call.
  FnTypeGeneratorType callImplType = addClosureSelfArgToFunctionSignature(
      opaquePtrType, ArgConvention::ReadReg, fnPtrType);
  StringAttr lambdaName = b.getStringAttr("call_impl");
  auto [callImpl, callDecl] = synthesizeFunction(
      decl, lambdaName, /*params=*/{}, callImplType.getMetadata(),
      callImplType.getArguments(), callImplType.getArgConventions(),
      callImplType.getArgListAttrs(), fnPtrType.getResultType(),
      SpecialFunctionKind::kNormal, decl.getLoc(), b, fnPtrType.getFnEffects());
  auto paramDecl =
      ParamDeclAttr::get(lambdaName, callImpl.getFuncTypeGenerator());
  callImpl.setParamDeclAttr(paramDecl);

  // Store it into the call field.
  storeField(b, self,
             CreateClosureOp::create(b, ParamDeclRefAttr::get(paramDecl)),
             b.getStringAttr("call"));
  IREmitter::emitNormalReturn(b);

  // Populate the lambda.
  b = ImplicitLocOpBuilder::atBlockBegin(callImpl.getLoc(), callImpl.getBody());
  Value fnPtr =
      POP::PointerBitcastOp::create(b, fnPtrType, callImpl.getArgument(0));
  SmallVector<TypedAttr> origins;
  for (ParamDeclAttr originDecl : callImpl.getParams())
    origins.push_back(ParamDeclRefAttr::get(originDecl));
  SmallVector<Value> callArgs;
  llvm::append_range(callArgs, callImpl.getArguments());
  auto callIndirect =
      CallIndirectOp::create(b, fnPtrType.getResultType(), fnPtr, origins,
                             ArrayRef(callArgs).drop_front());
  IREmitter::emitNormalReturn(b, callIndirect.getResult(0));
}

namespace {
// ParamIndexRefReplacer converts parameter references by index (e.g., *(0,1))
// to references by name (e.g., "b").
//
// Problem:
// In Mojo parametric functions, parameter types can refer to earlier parameters
// using indices. For example, in "fn f[T: Baz, b: T]", the type of 'b' refers
// to 'T' via an index. When creating function types outside this context, we
// need named references instead.
//
// Example:
// Input
// Parameter types: [!lit.trait<@Baz>, !kgen.param<*(0,0)>]
// Parameter names: ["T", "b"]
//
// Output (canonical types):
// {"T": !lit.trait<@Baz>, "b": !kgen.param<"T">}
//
// Solution:
// We recursively traverse attributes, replacing ParamIndexRefAttr with
// ParamDeclRefAttr using a map from indices to parameter declarations.
// The recursion terminates because MLIR attributes are directed acyclic graphs.
struct ParamIndexRefReplacer
    : public IndexParameterReplacer<ParamIndexRefReplacer> {
  using Base = IndexParameterReplacer<ParamIndexRefReplacer>;
  ParamIndexRefReplacer(ArrayRef<ParamDeclAttr> declarations) {
    for (auto [i, p] : llvm::enumerate(declarations))
      parameters.insert({i, p.getName()});
  }
  ParamIndexRefReplacer(ArrayRef<PogMetadataAttr> declarations) {
    for (auto [i, p] : llvm::enumerate(declarations))
      parameters.insert({i, p.getName()});
  }
  Attribute tryReplace(Attribute attr, size_t depth) {
    auto indexRef = dyn_cast<ParamIndexRefAttr>(attr);
    if (!indexRef || indexRef.getDepth() != depth)
      return nullptr;
    auto it = parameters.find(indexRef.getIndex());
    if (it == parameters.end())
      return nullptr;

    StringRef paramName = it->second;
    Type mappedType = Base::replace(indexRef.getType());
    return ParamDeclRefAttr::get(paramName, mappedType);
  }
  Type tryReplace(Type t, size_t) { return {}; }
  DenseMap<unsigned, StringRef> parameters;
};
} // namespace

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
  ParamIndexRefReplacer replacer(pogAttrs);
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
static void
getUnwrappedOperands(ImplicitLocOpBuilder &b, FnOp op, Type wrapperType,
                     StructFieldOp wrappedField,
                     llvm::SmallDenseSet<StringRef> const &explicitParameters,
                     SmallVector<Value> &operands,
                     SmallVector<TypedAttr> &origins) {
  MLIRContext *ctx = b.getContext();
  // If we map a value to its field, save the lifetime name so we can map the
  // origins as well.
  DenseMap<StringRef, StringAttr> originToField;
  for (Value arg : op.getBodyRegion().front().getArguments()) {
    // replace wrapper type with impl type
    RefType refType = dyn_cast<RefType>(arg.getType());
    if (!refType) {
      operands.push_back(arg);
      continue;
    }

    if (refType.getElementType() == wrapperType) {
      ParamDeclRefAttr originReference =
          dyn_cast<ParamDeclRefAttr>(refType.getOrigin());
      assert(originReference && "There should not be parameter expressions "
                                "in the signature of wrapper functions");
      operands.push_back(
          RefStructGEROp::create(b, arg, wrappedField)->getResults().front());
      originToField[originReference.getName().getValue()] =
          wrappedField.getNameAttr();
    } else {
      operands.push_back(arg);
    }
  }

  // Since this is a wrapper we know all the origins of the function must be
  // bound to the single call op in the body.
  SmallVector<ParamDeclAttr> allParams = op.collectAllParams(true);
  for (ParamDeclAttr param : allParams) {
    if (explicitParameters.contains(param.getName().getValue()))
      continue;
    auto originType = dyn_cast<OriginType>(param.getType());
    if (!originType)
      continue;
    ParamDeclRefAttr originRef =
        ParamDeclRefAttr::get(param.getName(), param.getType());
    TypedAttr originArg;
    auto ptr = originToField.find(originRef.getName().getValue());
    if (ptr != originToField.end())
      originArg = OriginFieldAttr::get(ctx, originRef, ptr->second, originType);
    else
      originArg = originRef;
    origins.push_back(originArg);
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

static SymbolConstantAttr buildSymbol(FnOp impl, ParamDeclAttr implType,
                                      ParamDeclAttr originSetParam) {
  MLIRContext *ctx = impl.getContext();
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));
  // Build symbol by binding struct level parameters and explicit parameters.
  FuncTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  SmallVector<TypedAttr> params;
  params.push_back(ParamDeclRefAttr::get(implType));
  params.push_back(ParamDeclRefAttr::get(originSetParam));
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

std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
ClosureEmitter::pushBackTraitFunctionImpl(FnOp traitFnOp, ASTDecl &structDecl) {
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp.getLoc(), structDeclOp);
  b.setInsertionPointToEnd(&structDeclOp.getFields().front());
  SharedState &shared = structDecl.getShared();
  // Wrapper signature is the signature of the method on the wrapper struct.
  // We create it by specializing the trait method by binding the struct type
  // to the self parameter.
  FnTypeGeneratorType wrapperSignature = specializeSignature(
      traitFnOp, structDecl.getTypeDeclSelf(), *shared.declResolver);

  // Calculate the argument types and result types in terms of the named
  // parameters. Since the name of the parameters have not changed from the
  // trait definition, we can avoid another remap of the indexed types in
  // parameters and instead reuse the trait function's input parameters.
  ArrayRef<ParamDeclAttr> parameters =
      ArrayRef<ParamDeclAttr>(traitFnOp.getInputParams())
          .take_front(traitFnOp.getInputParams().size() -
                      wrapperSignature.getNumImplicitOriginDecls());
  ParamIndexRefReplacer replacer(parameters);
  SmallVector<Type> argumentTypes;
  llvm::append_range(
      argumentTypes,
      llvm::map_range(wrapperSignature.getArguments(), [&](Type original) {
        return replacer.replace(original);
      }));
  Type result = replacer.replace(wrapperSignature.getResults().front());
  auto [op, decl] = synthesizeFunction(
      structDecl, traitFnOp.getSourceNameAttr(), parameters,
      wrapperSignature.getParamListAttrs(), argumentTypes,
      wrapperSignature.getArgConventions(), wrapperSignature.getArgListAttrs(),
      result, traitFnOp.getSpecialFunctionKind(), structDecl.getLoc(), b,
      wrapperSignature.getFnEffects().setUnified(false).setRegisterPassable(
          false),
      "", true, traitFnOp.getInlineLevel());
  if (traitFnOp.getSelfDeinit())
    op.setSelfDeinit(true);
  return {op, parameters, result};
}

ASTDecl *ClosureEmitter::createStructWrapper(
    ASTDecl &moduleDecl, StringRef baseName, ASTDecl &traitDecl,
    SMLoc smLocation, TypeConvention typeConvention, bool isCopyable) {
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
      moveParent, anyParent};
  if (isCopyable) {
    closureParents.push_back(copyParent);
    closureParents.push_back(implicitlyCopyableParent);
  }

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

  // Create a struct with a single field of type "impl".
  std::pair<ASTDecl &, StructDeclOp> pair = createStruct(
      shared, moduleDecl,
      StringAttr::get(
          b.getContext(),
          baseName + "_wrapper" + (isCopyable ? "_copyable" : "") +
              (typeConvention == TypeConvention::RegisterPassableTrivial
                   ? "_devicePassable"
                   : "")),
      implParameters, smLocation);
  ASTDecl &structDecl = pair.first;
  StructDeclOp declOp = pair.second;
  declOp.setConvention(typeConvention);
  addFieldsToStruct(declOp, structDecl,
                    KGEN::ParamType::get(ParamDeclRefAttr::get(implType)),
                    *shared.declResolver);
  StructFieldOp wrappedField = *declOp.getFieldDecls().begin();

  // Populate the wrapper methods with a call to the result of a witness lookup.
  auto populateTraitFn = [&](ClosureParent &closureParent) -> FnOp {
    FnOp traitFnOp = closureParent.getDefiningOp(moduleDecl);
    b.setInsertionPointToEnd(&declOp.getFields().front());
    FnTypeGeneratorType wrappedSignature =
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
    getUnwrappedOperands(b, op, wrapperType, wrappedField, explicitParameters,
                         operands, origins);
    StringAttr parentName = closureParent.getFullSymbolName(moduleDecl);
    TypedAttr symbol = GetWitnessAttr::get(
        ctx, ParamDeclRefAttr::get(implType.getName(), implType.getType()),
        parentName, traitFnOp.getSymNameAttr(), wrappedSignature);
    SmallVector<TypedAttr> paramArgs;
    llvm::append_range(
        paramArgs,
        llvm::map_range(parameters, [](ParamDeclAttr p) -> TypedAttr {
          return ParamDeclRefAttr::get(p);
        }));
    auto callOp = LIT::CallOp::create(
        b, result,
        BindParamsAttr::get(symbol, paramArgs, &shared.getEvaluationContext()),
        origins, operands);
    IREmitter::emitNormalReturn(b, callOp.getResult(0));
    return op;
  };
  auto getSymbolNoParamValues = [&](FnOp impl) {
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
  };
  DenseMap<StringRef, FnOp> nameToImpl;
  for (ClosureParent &closureParent : closureParents) {
    if (!closureParent.isEmpty()) {
      FnOp impl = populateTraitFn(closureParent);
      switch (closureParent.getClosureMethod()) {
      case ClosureMethod::COPY:
        declOp.setCopyInitAttr(getSymbolNoParamValues(impl));
        break;
      case ClosureMethod::MOVE:
        declOp.setMoveInitAttr(getSymbolNoParamValues(impl));
        break;
      case ClosureMethod::DEL:
        declOp.setDestructorAttr(getSymbolNoParamValues(impl));
        break;
      default:
        break;
      }
      nameToImpl.insert({closureParent.getDefiningOpName(), impl});
    }
  }

  // Emit conformance tables
  StringAttr moveParentStrAttr;
  auto addWitnessEntry = [&](TraitDeclOp traitParent, FnOp fnOp) {
    StringRef name = *fnOp.getSourceName();
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    SymbolRefArrayAttr immediateParents = traitParent.getImmediateParentsAttr();
    SymbolRefAttr parentSymbol = getFullyResolvedSymbolRef(
        cast<mlir::SymbolOpInterface>(traitParent.getOperation()));
    StringAttr parentName =
        b.getStringAttr(getFlattenedSymbolName(parentSymbol));
    if (name == "__moveinit__")
      moveParentStrAttr = parentName;

    ConformanceOp witnessTable =
        ConformanceOp::create(b, parentName, parentSymbol, immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();
    b.setInsertionPointToStart(&block);
    assert(nameToImpl.contains(name) &&
           "expected all trait ops to be implemented");
    FnOp impl = nameToImpl[name];
    SymbolConstantAttr symbolConstant =
        buildSymbol(impl, implType, originSetParam);
    WitnessOp::create(b, fnOp.getSymNameAttr(), symbolConstant);

    return witnessTable;
  };

  llvm::StringMap<ConformanceOp> parentWitnesses;
  for (ClosureParent &closureParent : closureParents) {
    if (!closureParent.isEmpty()) {
      auto tbl = addWitnessEntry(closureParent.getTrait(moduleDecl),
                                 closureParent.getDefiningOp(moduleDecl));
      parentWitnesses[closureParent.getDefiningOpName()] = tbl;
    }
  }

  assert(moveParentStrAttr && "closures are expected to conform to move");
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<PogMetadataAttr> argPogs;
  SmallVector<ArgConvention> argConventions;

  initArgumentTypes.reserve(2);
  argPogs.reserve(2);
  argConventions.reserve(2);

  // the constructor takes an instance of type "impl" and an instance of type
  // "self"
  Type refInitImplType = ASTType((paramType)).getRefForArgument(implName, true);
  argConventions.push_back(ArgConvention::OwnedMem);
  initArgumentTypes.push_back(refInitImplType);
  argPogs.push_back(PogMetadataAttr::get(StringAttr::get(ctx, implName),
                                         PassingKind::PosOnly));

  RefType refSelfType = ASTType(structDecl.getTypeDeclSelf())
                            .getRefForArgument(selfName.getValue(), true);
  argConventions.push_back(ArgConvention::ByRefResult);
  initArgumentTypes.push_back(refSelfType);
  argPogs.push_back(PogMetadataAttr::get(selfName, PassingKind::Implicit));
  b.setInsertionPointToEnd(&declOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions, PogListAttr::get(ctx, argPogs), NoneType::get(ctx),
      SpecialFunctionKind::kInit, smLocation, b, {}, "", true,
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
  SmallVector<TypedAttr> origins;
  llvm::SmallDenseSet<StringRef> explicitParameters;
  getUnwrappedOperands(b, initFnOp, refSelfType.getElementType(), wrappedField,
                       explicitParameters, operands, origins);
  LIT::CallOp::create(b, moveSignature.getResultType(), moveSymbol, origins,
                      operands);
  IREmitter::emitNormalReturn(b);
  declOp.setCanonicalTrait(traitType);

  if (typeConvention == TypeConvention::RegisterPassableTrivial)
    addConformanceToDevicePassable(structDecl, wrappedField, implType,
                                   originSetParam);

  // Generate is-trivial special aliases
  auto generateIsTrivialSpecialAlias = [&](StringRef name,
                                           ClosureParent parent) {
    bool value = typeConvention == TypeConvention::RegisterPassableTrivial;
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    TypedAttr valueAttr = BoolAttr::get(ctx, value);
    ParamDeclAttr paramAttr = ParamDeclAttr::get(
        ctx, StringAttr::get(ctx, name), valueAttr.getType());
    AliasDeclOp aliasOp = LIT::AliasDeclOp::create(
        b, declOp.getBodyRegion().getLoc(), paramAttr, valueAttr);
    aliasOp.setInheritedFromAttr(parent.getSymbolRef(moduleDecl));
    shared.declResolver->addFullyResolvedDecl(
        aliasOp, StringAttr::get(ctx, name), structDecl.getLoc(), &structDecl);

    // Look up the existing conformance table for this trait. We already created
    // these earlier.
    assert(parentWitnesses.contains(parent.getDefiningOpName()) &&
           "parent witness table should already exist");
    auto conformanceOp = parentWitnesses[parent.getDefiningOpName()];
    b.setInsertionPointToEnd(&conformanceOp.getBody().front());
    WitnessOp::create(b, StringAttr::get(ctx, name), valueAttr);
  };

  generateIsTrivialSpecialAlias("__del__is_trivial", anyParent);
  generateIsTrivialSpecialAlias("__moveinit__is_trivial", moveParent);
  if (isCopyable)
    generateIsTrivialSpecialAlias("__copyinit__is_trivial", copyParent);

  return &structDecl;
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

ASTDecl *
ClosureEmitter::createClosureTrait(ASTDecl &moduleDecl, StringAttr name,
                                   FnTypeGeneratorType dependentSignatureType,
                                   SMLoc nestedFunctionOrTypeLocation,
                                   InlineLevel inlineLevel) {
  // Generate the movable, destructable closure trait, populating the trait
  // definition with the single characteristic "__call__" method.
  SmallVector<ClosureParent> parents{moveParent, anyParent};
  auto populate = [&](ASTDecl &decl,
                      DenseSet<std::pair<StringAttr, StringAttr>> &functions) {
    TraitDeclOp closureTrait = cast<TraitDeclOp>(decl.getIfOperation());
    RefType refType = decl.getTypeDeclSelf().getRefForArgument("self", true);
    FnTypeGeneratorType sig = addClosureSelfArgToFunctionSignature(
        refType, ArgConvention::ReadMem, dependentSignatureType);
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
        closureTrait.getLoc(), &closureTrait.getFields().front());
    SmallVector<ParamDeclAttr> parameters(
        populateParametersFromFnGeneratorType(sig));
    auto callName = StringAttr::get(ctx, "__call__");
    // Calculate the argument types and result types in terms of the named
    // parameters.
    ParamIndexRefReplacer replacer(parameters);
    SmallVector<Type> argumentTypes;
    llvm::append_range(argumentTypes,
                       llvm::map_range(sig.getArguments(), [&](Type original) {
                         return replacer.replace(original);
                       }));
    Type result = replacer.replace(sig.getResults().front());
    // TODO: remove capturing when legacy closures are removed.
    auto [fnOp, fnDecl] = synthesizeFunction(
        decl, callName, parameters, sig.getParamListAttrs(), argumentTypes,
        sig.getArgConventions(), sig.getArgListAttrs(), result,
        SpecialFunctionKind::kNormal, nestedFunctionOrTypeLocation, builder,
        sig.getFnEffects()
            .setUnified(false)
            .setRegisterPassable(false)
            .setCapturing(true),
        "", true, inlineLevel);
    builder.setInsertionPointToEnd(&fnOp.getBodyRegion().front());
    UnreachableOp::create(builder);
    functions.insert({callName, fnOp.getSymNameAttr()});
  };
  auto createTraitFn = [&]() -> ASTDecl * {
    auto [closureTrait, traitDecl] = createTraitOp(
        moduleDecl, name, parents, nestedFunctionOrTypeLocation, populate);
    return traitDecl;
  };
  return getOrCreateClosureTrait(dependentSignatureType, createTraitFn);
}

StructDeclOp ClosureEmitter::createClosureWrapperStructDecl(
    ASTDecl &moduleDecl, StringAttr name,
    FnTypeGeneratorType dependentSignatureType,
    SMLoc nestedFunctionOrTypeLocation) {
  SmallVector<Type> fieldTypes{opaquePtrType};

  SmallVector<ParamDeclAttr> wrapperDecls;
  ParserParameterEvaluator evaluator(shared);
  SmallVector<TypedAttr> paramValues;
  for (auto [i, type] :
       llvm::enumerate(dependentSignatureType.getInputParamTypes())) {
    wrapperDecls.push_back(
        ParamDeclAttr::get(StringAttr::get(getContext(), "p" + Twine(i)),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(wrapperDecls.back()));
    evaluator.appendIndexBinding(paramValues.back());
  }

  auto [structDecl, declOp] =
      createStruct(shared, moduleDecl, name, wrapperDecls, moduleDecl.getLoc());
  addFieldsToStruct(declOp, structDecl, opaquePtrType, *shared.declResolver);
  declOp.setClosureSignature(dependentSignatureType);

  StructEmitter structEmitter(structDecl);
  Type noneType = shared.getNoneType();

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  auto dtorMetadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, {selfName}, {PassingKind::PosOnly}));
  auto dtorSig = FuncTypeGeneratorType::get(
      /*paramTypes=*/{}, b.getFunctionType(opaquePtrType, noneType),
      ArgConvention::ReadReg,
      /*effects=*/{}, dtorMetadata, PogListAttr::get(ctx));

  auto dtor = addFieldOpAndDecl(dtorFieldAttr, dtorSig, declOp, structDecl, b,
                                getDeclResolver());

  // Create Copy Member.
  auto fnType =
      b.getType<FunctionType>(ArrayRef<Type>{opaquePtrType}, opaquePtrType);
  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, {otherName}, {PassingKind::PosOnly}));
  auto cpySignatureType = FuncTypeGeneratorType::get(
      /*paramTypes=*/{}, fnType, {ArgConvention::ReadReg},
      /*effects=*/{}, metadata, PogListAttr::get(ctx));
  auto copy = addFieldOpAndDecl(copyFieldAttr, cpySignatureType, declOp,
                                structDecl, b, getDeclResolver());

  dependentSignatureType = dependentSignatureType.getSpecializedGenerator(
      paramValues, &shared.getEvaluationContext(),
      translateLocation(nestedFunctionOrTypeLocation));
  auto sigMetadata =
      FnMetadataAttr::get(dependentSignatureType.getArgListAttrs(),
                          dependentSignatureType.getNumImplicitOriginDecls());
  Type resultType = dependentSignatureType.getResults().front();
  FunctionType functionType =
      b.getFunctionType(dependentSignatureType.getArguments(), resultType);
  FnTypeGeneratorType signatureType = FuncTypeGeneratorType::get(
      /*paramTypes=*/{}, functionType,
      dependentSignatureType.getArgConventions(),
      dependentSignatureType.getFnEffects(), sigMetadata,
      PogListAttr::get(ctx));

  // Add the call member
  FnTypeGeneratorType callMemberSignatureType =
      addClosureSelfArgToFunctionSignature(
          opaquePtrType, ArgConvention::ReadReg, signatureType);
  auto callMember = addFieldOpAndDecl(callFieldAttr, callMemberSignatureType,
                                      declOp, structDecl, b, getDeclResolver());

  std::optional<ValueInfo> stubs =
      structEmitter.addMissingValueMemberStubsToStruct(
          /*forceGenerateDestructor=*/true);
  assert(stubs && "expected the stubs on a purely synthetic class to succeed.");

  FnOp copyCtr = stubs->copyinit;
  SymbolConstantAttr copyCtrRef =
      copyCtr.getBoundSymbolRef(shared.getEvaluationContext());
  ASTDecl *copyCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(copyCtrRef.getSymbol());

  FnOp moveCtr = stubs->moveinit;
  SymbolConstantAttr moveCtrRef =
      moveCtr.getBoundSymbolRef(shared.getEvaluationContext());
  ASTDecl *moveCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(moveCtrRef.getSymbol());

  // Populate destructor.
  {
    FnOp destructor = stubs->del;
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
        destructor.getLoc(), destructor.getBody());
    Value dtorSelf = destructor.getBody()->getArgument(0);
    Value dtorImpl = loadField(b, dtorSelf, impl);
    Value callee = loadField(b, dtorSelf, dtor);
    CallIndirectOp::create(b, noneType, callee,
                           /*implicitOrigins=*/ArrayRef<TypedAttr>(), dtorImpl);
  }

  // Populate the copy constructor.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(copyCtr.getLocScope());
    ImplicitLocOpBuilder b =
        ImplicitLocOpBuilder::atBlockBegin(copyCtr.getLoc(), copyCtr.getBody());
    Value copySelf = copyCtr.getBody()->getArgument(1);
    Value copyExisting = copyCtr.getBody()->getArgument(0);
    Value existingImpl = loadField(b, copyExisting, impl);
    Value funcPtr = loadField(b, copySelf, copy);
    auto call = CallIndirectOp::create(
        b, opaquePtrType, funcPtr, /*implicitOrigins=*/ArrayRef<TypedAttr>(),
        existingImpl);
    storeField(b, copySelf, call.getResult(0), impl);
  }
  // Copy all the fields over as well.
  if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    return {};

  // Populate move constructor
  if (failed(structEmitter.populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
    return {};

  // Add the __call__ Method.
  ASTType selfType = structDecl.getTypeDeclSelf();
  auto refToSelfType = selfType.getRefForArgument("self", /*isMut=*/false);
  FnTypeGeneratorType closureMethodSignatureType =
      addClosureSelfArgToFunctionSignature(
          refToSelfType, ArgConvention::ReadMem, signatureType);
  // The __call__ method is effectively the in-source body of the function. Mark
  // it as *not* synthetic so that debugging will step into the body.
  auto [callMethod, _] = structEmitter.synthesizeMethodInStruct(
      "__call__", closureMethodSignatureType.getArguments(),
      closureMethodSignatureType.getArgConventions(),
      closureMethodSignatureType.getArgListAttrs(), resultType,
      SpecialFunctionKind::kNormal, closureMethodSignatureType.getFnEffects(),
      /*suffix=*/"", /*synthetic=*/false);

  // Populate the body of ClosureWrapper::__call__.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(callMethod.getLocScope());
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        callMethod.getLoc(), callMethod.getBody());
    Value callSelf = callMethod.getBody()->getArgument(0);

    // Load self, but pass the rest unmodified.
    SmallVector<Value> arguments;
    arguments.push_back(loadField(builder, callSelf, impl));
    llvm::append_range(arguments,
                       callMethod.getBody()->getArguments().drop_front());
    Value callMemberPtr = loadField(builder, callSelf, callMember);

    SmallVector<TypedAttr> implicitOrigins;
    auto calleeSig = cast<FnTypeGeneratorType>(callMemberPtr.getType());
    for (auto [arg, conv] : llvm::zip(arguments, calleeSig.getArgConventions()))
      if (hasImplicitOrigin(conv))
        implicitOrigins.push_back(cast<RefType>(arg.getType()).getOrigin());

    auto callResult = CallIndirectOp::create(builder, resultType, callMemberPtr,
                                             implicitOrigins, arguments);
    IREmitter::emitNormalReturn(builder, callResult.getResult(0));
  }

  synthesizeWrapperFnPtrCtor(structDecl, selfType, dependentSignatureType);
  return declOp;
}

StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    ASTDecl &moduleDecl, ArrayRef<Capture> captures,
    ArrayRef<ParamDeclRefAttr> paramCaptures, ASTDecl &nestedFnDecl,
    FnTypeGeneratorType wrapperSig) {
  FileModuleOp fileModuleOp = cast<FileModuleOp>(moduleDecl.getIfOperation());
  auto implName =
      StringAttr::get(ctx, "`_CI_" + fileModuleOp.getSymName() + "_escaping" +
                               Twine(moduleDecl.getNextUniqueID()));

  // Create map from the parent name to the index of the parameter in the
  // closure struct.
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  wrapperSig = nestedFn.getFuncTypeGenerator();
  if (!wrapperSig.getInputParamTypes().empty()) {
    shared.emitError(
        nestedFnDecl.getLoc(),
        "add parameters of nested function to parent function and capture "
        "them: parameters declared in nested functions are not supported yet");
    return {};
  }

  // Collect the types of the capture values.
  SmallVector<Type> fieldTypes =
      llvm::map_to_vector(captures, [](const Capture &capture) {
        return capture.getValue().getRValueType().mlirType;
      });

  // Check for parameter closure captures.
  bool hasParamClosureCaptures = false;
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  for (ParamDeclRefAttr pc : paramCaptures)
    hasParamClosureCaptures |= walker.walk(pc.getType()).wasInterrupted();

  // Create the closure impl struct with the field types. Add the capture
  // parameters as parameter decls to the generated struct. This way, parameter
  // references within the body do not have to be renamed.
  auto paramDecls =
      llvm::map_to_vector(paramCaptures, [](ParamDeclRefAttr ref) {
        return ParamDeclAttr::get(ref);
      });
  auto [structDecl, declOp] = createStruct(shared, moduleDecl, implName,
                                           paramDecls, nestedFnDecl.getLoc());

  StructEmitter structEmitter(structDecl);

  // Generate the __call__ method.

  // Build the call signature from the closure signature. This means inserting
  // the self argument in the correct location.
  unsigned callArgCount = wrapperSig.getNumArguments() + 1;
  SmallVector<Type> callInputTypes;
  callInputTypes.reserve(callArgCount);
  SmallVector<ArgConvention> callConventions;
  callConventions.reserve(callArgCount);
  SmallVector<PogMetadataAttr> callPogs;
  callPogs.reserve(callArgCount);

  // Currently Closure Impls are not register passable, so use ReadMem
  // convention.
  ASTType structSelfType = structDecl.getTypeDeclSelf();
  callInputTypes.push_back(
      ASTType(structSelfType).getRefForArgument("self", /*isMut=*/false));
  callConventions.push_back(ArgConvention::ReadMem);
  callPogs.push_back(
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));

  llvm::append_range(callInputTypes, nestedFn.getFunctionType().getInputs());
  llvm::append_range(callConventions, wrapperSig.getArgConventions());
  llvm::append_range(callPogs, wrapperSig.getArgListAttrs().getPogs());

  Type closureResultType = wrapperSig.getResults().front();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(declOp.getLoc(),
                                                  &declOp.getFields().front());
  auto [callFunc, _] = structEmitter.synthesizeMethodInStruct(
      "__call__", callInputTypes, callConventions,
      PogListAttr::get(ctx, callPogs), closureResultType,
      SpecialFunctionKind::kNormal,
      wrapperSig.getFnEffects().setEscaping(false));
  callFunc.setInlineLevel(InlineLevel::Always);

  // Add and register its fields as fully resolved decls.
  addFieldsToStruct(declOp, structDecl, fieldTypes, *shared.declResolver);

  // Build the init method. This only needs the captured arguments. Populate the
  // function argument information.

  // All arguments as positional-only.
  SmallVector<PassingKind> initSigPassingKinds(captures.size(),
                                               PassingKind::PosOnly);
  // Fill the types and conventions based on the register-passabilities.
  SmallVector<StringAttr> initSigNames;
  SmallVector<Type> initSigTypes;
  SmallVector<ArgConvention> initSigConventions;
  unsigned fieldNameIdx = 0;
  for (const Capture &capture : captures) {
    // If this is a reference capture, then we are capturing the address of the
    // value in the closure, otherwise we are taking an RValue that is either
    // copied or moved.
    bool isRef = capture.isRef();
    ASTType rvalueType = capture.getValue().getRValueType();
    initSigNames.push_back(StringAttr::get(ctx, "fld" + Twine(fieldNameIdx++)));
    // FIXME: By-reference captures should be capturable either as by-imm-ref or
    // by-mut-ref.  Right now we type check var captures as mutable but codegen
    // them as immutable references!
    if (isRef && rvalueType.isTrivial(nestedFnDecl.getLoc(), shared)) {
      initSigConventions.push_back(ArgConvention::ReadReg);
      initSigTypes.push_back(rvalueType);
    } else {
      initSigConventions.push_back(isRef ? ArgConvention::ReadMem
                                         : ArgConvention::OwnedMem);
      initSigTypes.push_back(rvalueType.getRefForArgument(
          initSigNames.back().str(), /*isMut=*/!isRef));
    }
  }

  // Add "out self" at the end.
  initSigNames.push_back(selfName);
  initSigTypes.push_back(
      ASTType(structSelfType).getRefForArgument("self", /*isMut=*/true));
  initSigConventions.push_back(ArgConvention::ByRefResult);
  initSigPassingKinds.push_back(PassingKind::Implicit);

  // FIXME: This can't use the simple form of 'synthesizeMemberwiseInit' because
  // 'ref' captures are modeled wrong: we're storing the /values/ in the closure
  // instead of the /addresses/. fieldTypes above should be adding a layer of
  // lit.ref, which would allow us to use the simple form.
  FnOp initFunc = structEmitter.synthesizeFieldwiseInit(
      initSigTypes, initSigConventions,
      PogListAttr::get(ctx, initSigNames, initSigPassingKinds),
      shared.getNoneType());
  if (!initFunc) // This can fail when the members aren't copy/moveable.
    return {};

  // Add the copy and move constructors and dtor.
  (void)structEmitter.addMissingValueMemberStubsToStruct(
      /*forceGenerateDestructor=*/true);

  builder =
      ImplicitLocOpBuilder::atBlockBegin(initFunc.getLoc(), initFunc.getBody());

  StructFieldOp paramField;
  IREmitter emitter(nestedFnDecl, builder);
  SyntheticNode loc(nestedFnDecl.getLoc());
  if (hasParamClosureCaptures) {
    // Propagate the 'capturing' bit to the init function.
    FnTypeGeneratorType oldSig = initFunc.getFuncTypeGenerator();
    initFunc.setFuncTypeGenerator(
        oldSig.getWithBody(oldSig.getBody().getWithFnEffects(
            oldSig.getFnEffects().setCapturing(true))));

    // Declare an extra field to carry the parametric closure captures.
    ASTType clType = shared.getBuiltinCaptureListType(nestedFnDecl.getLoc());
    TypedAttr bound = callFunc.getBoundReference(
        shared.getEvaluationContext(),
        ParameterExprArrayAttr::get(
            getContext(), cast<StructType>(structSelfType).getParamValues()));
    clType = cast<LIT::StructType>(clType).bindAll(
        {TypeParamAttr::get(bound.getType(), TypeType::get(getContext())),
         bound});
    auto b = OpBuilder::atBlockBegin(declOp.getBody());
    paramField =
        addFieldOpAndDecl(StringAttr::get(ctx, "param_capture"), clType, declOp,
                          structDecl, b, getDeclResolver());

    // Emit IR to generate the capture list and store it into self. Bind the
    // call function reference to itself.
    auto selfArg = initFunc.getArgument(initFunc.getNumArguments() - 1);
    Value target = RefStructGEROp::create(builder, selfArg, paramField);
    ValueDest dest(MLValue(target), EC_Assignment);
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(initFunc.getLocScope());
    emitter.emitConstructorCall(clType, {}, loc, CallSyntax::kDirectCall, dest);
  }

  // Populate the body of the call op.
  declOp->setAttr(callMethodAttr,
                  callFunc.getBoundReference(shared.getEvaluationContext()));
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(callFunc.getLocScope());

  // Take the body of the nested function.
  callFunc.getBody()->erase();
  callFunc.getBodyRegion().takeBody(nestedFn.getBodyRegion());
  Location callFuncLocation = callFunc.getLoc();
  updateLocationScope(nestedFn->getLoc(), callFuncLocation, callFunc);

  builder =
      ImplicitLocOpBuilder::atBlockBegin(callFunc.getLoc(), callFunc.getBody());
  Value selfArg = callFunc.getBodyRegion().insertArgument(
      0U, callFunc.getFunctionType().getInput(0), callFuncLocation);

  if (paramField) {
    // Emit the `kgen.capture_list.expand` into the call if required.
    Value target = RefStructGEROp::create(builder, selfArg, paramField);
    emitter.builder = builder;
    ValueDest dest(EC_Assignment);
    emitter.emitNamedMethodCall("expand", {{{MBValue(target), loc}}}, dest,
                                CallSyntax::kMethodCall, loc);
  }
  for (auto [capture, fieldOp] :
       llvm::zip(captures, llvm::drop_begin(declOp.getFieldDecls(),
                                            hasParamClosureCaptures))) {
    Value target = RefStructGEROp::create(builder, selfArg, fieldOp);
    // If the capture is an SValue then it lives in register.
    if (capture.getValue().isSValue())
      target = RefLoadOp::create(builder, target);

    // If the reference types disagree, the cast to fix the origin.
    // FIXME: This isn't great.  We should really /replace/ the original
    // origins with the self origin.  For example, when rewriting something
    // like:
    //      fn outer(a: MemType):
    //         fn inner():
    //           use(a)
    // the capture will use 'a' with its own `a origin implicitly generated on
    // the outer type.  However, after rewriting it to a struct, we get
    // something like this:
    //      fn closure(self: CaptureStruct):
    //        use(self.a)
    // which now has the origin (and mutability) of 'self'.
    Value captureValue = capture.getValue().getMlirValue();
    // FIXME: This should use emitRebindOpIfNeeded, but it is introducing
    // mutability with the rebind!
    if (captureValue.getType() != target.getType())
      target = RebindOp::create(builder, captureValue.getType(), target);

    assert(captureValue.getType() == target.getType() &&
           "Capture body rewrite problem");
    replaceAllUsesInRegionWith(captureValue, target, callFunc.getBodyRegion());
  }
  shared.deleteDecl(nestedFnDecl);
  return declOp;
}

/// Given a Closure struct and parameter values, create the specialized self
/// type.
static Type makeClosureImplSelfType(StructDeclOp closureImpl,
                                    ArrayRef<TypedAttr> paramRefs) {
  return closureImpl.bindReference(paramRefs);
}

static SymbolConstantAttr createTypedSymbol(SymbolConstantAttr symbol,
                                            ArrayRef<ParamDeclAttr> parameters,
                                            SharedState &shared) {
  SmallVector<TypedAttr> paramReferences =
      llvm::map_to_vector(parameters, [](ParamDeclAttr attr) -> TypedAttr {
        return ParamDeclRefAttr::get(attr);
      });
  auto paramRefs =
      ParameterExprArrayAttr::get(symbol.getContext(), paramReferences);
  auto [specializedSignature, _] = getUnboundSpecializedSignature(
      symbol.getType(), paramRefs, &shared.getEvaluationContext());
  return SymbolConstantAttr::get(symbol.getSymbol(), specializedSignature,
                                 paramReferences);
}

/// Generate the code to allocate heap memory for the given pointer type.
static Value allocateHeapMemory(PointerType ptrType, ImplicitLocOpBuilder &b) {
  TypedAttr elementType = TypeParamAttr::get(
      ptrType.getElementType(), TypeType::get(ptrType.getContext()));
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value sizeOf = ParamConstantOp::create(
      b, ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = ParamConstantOp::create(
      b, ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return POP::AlignedAllocOp::create(b, ptrType, ValueRange{alignOf, sizeOf});
}

TopLevelTypes
ClosureEmitter::collectTopLevelFunctionTypes(StructDeclOp closureWrapper) {
  TopLevelTypes topLevelTypes;
  for (StructFieldOp fieldOp : closureWrapper.getFieldDecls()) {
    StringAttr name = fieldOp.getNameAttr();
    if (name == callFieldAttr)
      topLevelTypes.callFuncFieldType = fieldOp.getType();
    else if (name == copyFieldAttr)
      topLevelTypes.copyFuncFieldType = fieldOp.getType();
    else if (name == dtorFieldAttr)
      topLevelTypes.delFuncFieldType = fieldOp.getType();
  }
  assert(topLevelTypes.callFuncFieldType &&
         "All closure wrapper initializers must have a top "
         "level call function associated with them");
  assert(topLevelTypes.copyFuncFieldType &&
         "All closure wrapper initializers must have a top "
         "level delete function associated with them");
  assert(topLevelTypes.delFuncFieldType &&
         "All closure wrapper initializers must have a top "
         "level copy function associated with them");
  return topLevelTypes;
}

/// Helper to return an array of demangled names for the given declarations.
static SmallVector<StringAttr>
getDemangledNames(ArrayRef<ParamDeclAttr> decls) {
  return llvm::map_to_vector(decls, [](ParamDeclAttr p) {
    return StringAttr::get(p.getContext(), demangleParameterName(p.getName()));
  });
}

static FnOp findInitInStruct(StructDeclOp structOp, ArrayRef<Type> operands) {
  size_t expectedNumInputs = operands.size() + 1;

  for (auto candidate : structOp.getOps<FnOp>()) {
    SpecialFunctionKind kind = candidate.getSpecialFunctionKind();
    if (kind != SpecialFunctionKind::kInit ||
        candidate.getBody()->getArguments().size() != expectedNumInputs)
      continue;

    bool isMatch = true;
    for (auto [existing, proposed] :
         llvm::zip(candidate.getFuncTypeGenerator().getArguments().slice(1),
                   operands)) {
      if (existing != proposed) {
        isMatch = false;
        break;
      }
    }
    if (isMatch)
      return candidate;
  }
  return {};
}

FnOp ClosureEmitter::createWrapperInitWithImpl(ASTDecl &moduleDecl,
                                               StructDeclOp closureWrapper,
                                               StructDeclOp closureImpl,
                                               SMLoc loc) {
  // The __init__ will take self and the impl. We first build the types. Add the
  // parameter references captured only in the body to the signature of the
  // constructor. Pass the ones captured in the signature from the wrapper to
  // the impl type.
  SmallVector<TypedAttr> totalParams;
  SmallVector<TypedAttr> wrapperParams;
  SmallVector<ParamDeclAttr> initParams;
  // We know from the walk order that the first N impl parameters are the
  // wrapper parameters.
  ArrayRef<ParamDeclAttr> wrapperParamDecls = closureWrapper.getParams();
  for (ParamDeclAttr param : wrapperParamDecls) {
    auto ref = ParamDeclRefAttr::get(param);
    totalParams.push_back(ref);
    wrapperParams.push_back(ref);
  }
  for (ParamDeclAttr param :
       closureImpl.getParams().drop_front(wrapperParamDecls.size())) {
    totalParams.push_back(ParamDeclRefAttr::get(param));
    initParams.push_back(param);
  }

  // Bind the impl struct to the declared parameters.
  ASTType closureImplType = makeClosureImplSelfType(closureImpl, totalParams);
  auto closureImplRefType =
      closureImplType.getRefForArgument("existing", /*isMut=*/true);

  // Create unique names for parameters.
  if (auto init = findInitInStruct(closureWrapper, closureImplRefType))
    return init;
  ASTType wrapperType = makeClosureImplSelfType(closureWrapper, wrapperParams);
  SmallVector<Type> argTypes{closureImplRefType, wrapperType.getRefForArgument(
                                                     "self", /*isMut=*/true)};

  // Then build all other information needed for the __init__ signature.
  ArgConvention argConventions[] = {ArgConvention::OwnedMem,
                                    ArgConvention::ByRefResult};
  StringAttr argNames[] = {StringAttr::get(ctx, "impl"), selfName};
  PassingKind argPassingKinds[] = {PassingKind::PosOnly, PassingKind::Implicit};
  SmallVector<PassingKind> paramPassingKindsOfInit(initParams.size(),
                                                   PassingKind::Implicit);
  auto paramListAttrsOfInit = PogListAttr::get(
      ctx, getDemangledNames(initParams), paramPassingKindsOfInit);
  auto argListAttrsOfInit = PogListAttr::get(ctx, argNames, argPassingKinds);
  ASTDecl &closureDecl =
      *ASTType(ASTDecl::computeSelfTypeForStruct(closureWrapper))
           .getDecl(shared);
  auto [init, _] = StructEmitter(closureDecl)
                       .synthesizeMethodInStruct(
                           "__init__", initParams, paramListAttrsOfInit,
                           argTypes, argConventions, argListAttrsOfInit,
                           shared.getNoneType(), SpecialFunctionKind::kInit);
  init.setInlineLevel(InlineLevel::Always);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(init.getLocScope());

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());

  // Allocate memory on heap and copy argument into allocated memory.
  Value target = allocateHeapMemory(PointerType::get(closureImplType), builder);
  Value source = init.getBody()->getArgument(0);

  // TODO(references): Move closures off pointers to correct origins.
  auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/true);
  Value targetRef = RefFromPointerOp::create(builder, target, immortal,
                                             /*startUninit=*/true,
                                             /*endUninit=*/false);

  // Move the contents of the injected impl into the heap memory.
  IREmitter emitter(moduleDecl, builder);
  SyntheticNode node(moduleDecl.getLoc());
  emitter.emitStoreToLValue({MRValue(source), &node}, MLValue(targetRef),
                            EC_Assignment);

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(1);
  Value erasedType =
      POP::PointerBitcastOp::create(builder, opaquePtrType, target);
  storeField(builder, self, erasedType, implField);
  auto generateName = [&](StringRef prefix) {
    return (closureWrapper.getSymName() + prefix + closureImpl.getSymName())
        .str();
  };
  TopLevelTypes topLevelTypes = collectTopLevelFunctionTypes(closureWrapper);
  auto setMember = [&](FnOp topLevelFunc, StringAttr fieldName,
                       Type fieldType) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    TypedAttr funcSymbol = topLevelFunc.getBoundReference(
        shared.getEvaluationContext(),
        ParameterExprArrayAttr::get(ctx, totalParams));
    if (funcSymbol.getType() != fieldType)
      funcSymbol = ParamOperatorAttr::getRebind(funcSymbol, fieldType);
    auto createClosure =
        CreateClosureOp::create(builder, funcSymbol, ValueRange());
    storeField(builder, init.getArgument(1), createClosure, fieldName);
  };

  // Create the top level copy constructor.
  // The copy constructor takes the Wrapper instance and the impl of the other.
  SmallVector<ParamDeclAttr> topLevelParams;
  for (TypedAttr param : totalParams) {
    auto declRef = cast<ParamDeclRefAttr>(param);
    topLevelParams.push_back(ParamDeclAttr::get(declRef));
  }

  SmallVector<PassingKind> paramPassingKinds(closureImpl.getParams().size(),
                                             PassingKind::Implicit);
  auto paramListAttrs = PogListAttr::get(ctx, getDemangledNames(topLevelParams),
                                         paramPassingKinds);
  auto argListAttrs =
      PogListAttr::get(ctx, {otherName}, {PassingKind::PosOnly});
  auto fileModuleOp = cast<FileModuleOp>(moduleDecl.getIfOperation());
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  auto [topLevelCopyInit, copyInitDecl] = synthesizeFunction(
      moduleDecl, generateName("_copyinit_"), topLevelParams, paramListAttrs,
      {opaquePtrType}, {ArgConvention::ReadReg}, argListAttrs, opaquePtrType,
      SpecialFunctionKind::kNormal, loc, builder);

  SmallVector<TypedAttr> topLevelParamRefs;
  for (auto [i, p] : llvm::enumerate(totalParams))
    topLevelParamRefs.push_back(
        ParamDeclRefAttr::get(topLevelCopyInit.getParams()[i]));
  auto closureImplTopLevelType =
      makeClosureImplSelfType(closureImpl, topLevelParamRefs);
  auto closureImplTopLevelPtrType = PointerType::get(closureImplTopLevelType);

  // Populate copy init body.
  {
    builder = ImplicitLocOpBuilder::atBlockBegin(topLevelCopyInit.getLoc(),
                                                 topLevelCopyInit.getBody());
    SmallVector<PassingKind> paramPassingKinds(topLevelParams.size(),
                                               PassingKind::PosOnly);
    Block *body = topLevelCopyInit.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard =
          shared.diBuilder->pushScopeGuard(topLevelCopyInit.getLocScope());

    // Allocate memory on heap and call copy constructor
    Value target = allocateHeapMemory(closureImplTopLevelPtrType, builder);
    Value existingPtr = POP::PointerBitcastOp::create(
        builder, closureImplTopLevelPtrType, body->getArgument(0));

    // TODO(references): move closures to references and correct origins.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/true);
    Value targetRef = RefFromPointerOp::create(builder, target, immortal,
                                               /*startUninit=*/true,
                                               /*endUninit=*/false);
    Value existingRef = RefFromPointerOp::create(builder, existingPtr, immortal,
                                                 /*startUninit=*/false,
                                                 /*endUninit=*/false);

    // Copy the existing value into the target.
    // TODO: Use nicer expr emitter for the result expr.
    IREmitter emitter(*copyInitDecl, builder);
    emitter.emitStoreToLValue({MLValue(existingRef), &node}, MLValue(targetRef),
                              EC_Assignment);

    // Return the allocated and populated impl.
    auto loc = topLevelCopyInit.getLoc();
    Value erasedType = POP::PointerBitcastOp::create(*emitter.builder, loc,
                                                     opaquePtrType, target);
    emitter.emitNormalReturn(loc, erasedType);
    setMember(topLevelCopyInit, copyFieldAttr, topLevelTypes.copyFuncFieldType);
  }

  // Create top level destructor.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  auto [topLevelDtor, dtorDecl] = synthesizeFunction(
      moduleDecl, generateName("_dtor_"), topLevelParams, paramListAttrs,
      opaquePtrType, ArgConvention::ReadReg,
      PogListAttr::get(ctx, {selfName}, {PassingKind::PosOnly}),
      shared.getNoneType(), SpecialFunctionKind::kNormal, loc, builder);

  // Populate destructor body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(),
                                               topLevelDtor.getBody());
    Block *body = topLevelDtor.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard =
          shared.diBuilder->pushScopeGuard(topLevelDtor.getLocScope());

    // Cast the opaque pointer back to the closure impl type.
    Value implPtr = POP::PointerBitcastOp::create(
        builder, closureImplTopLevelPtrType, body->getArgument(0));

    // TODO(references): Move closures off pointers.
    // This takes ownership of the pointer, telling checklifetimes that the
    // value should be destroyed by the exit of the function.  ASAP destruction
    // will make sure it is immediately destroyed because there are no uses.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/true);
    (void)RefFromPointerOp::create(builder, implPtr, immortal,
                                   /*startUninit=*/false,
                                   /*endUninit=*/true);

    // Free the memory we allocated on the heap to store the closure.
    POP::AlignedFreeOp::create(builder, implPtr);
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(), body);
    IREmitter::emitNormalReturn(builder);
  }

  // Set the member.
  setMember(topLevelDtor, dtorFieldAttr, topLevelTypes.delFuncFieldType);

  // Create the __call__ function.
  assert(closureWrapper.getClosureSignature().has_value() &&
         "The closure signature should have been set at creation time");
  FuncTypeGeneratorType functionSignature =
      *closureWrapper.getClosureSignature();
  FnTypeGeneratorType closureSignature = addClosureSelfArgToFunctionSignature(
      opaquePtrType, ArgConvention::ReadReg, functionSignature);
  assert(closureSignature.getResults().size() == 1);
  closureSignature = closureSignature.getSpecializedGenerator(
      ArrayRef(topLevelParamRefs).take_front(wrapperParamDecls.size()),
      &shared.getEvaluationContext(), translateLocation(loc));

  Type resultType = closureSignature.getResults().front();

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  auto [topLevelCall, callDecl] = synthesizeFunction(
      moduleDecl, generateName("_call_"), topLevelParams, paramListAttrs,
      closureSignature.getArguments(), closureSignature.getArgConventions(),
      closureSignature.getArgListAttrs(), resultType,
      SpecialFunctionKind::kNormal, loc, builder,
      closureSignature.getFnEffects());

  // Populate the __call__ body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCall.getLoc(),
                                               topLevelCall.getBody());
    Block *body = topLevelCall.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard =
          shared.diBuilder->pushScopeGuard(topLevelCall.getLocScope());

    // Cast the opaque pointer back to the closure impl type.
    Value closureArg = body->getArgument(0);
    Value implPtr = POP::PointerBitcastOp::create(
        builder, closureImplTopLevelPtrType, closureArg);

    // FIXME: Thread a origin through correctly.

    // TODO(references): Move closures off pointers.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/false);
    Value implRef = RefFromPointerOp::create(builder, implPtr, immortal,
                                             /*startUninit=*/false,
                                             /*endUninit=*/false);

    // Call the __call__ on the closure impl.
    assert(closureImpl->hasAttr(callMethodAttr) &&
           "Closure Impls are generated with a __call__ method.");
    SymbolConstantAttr symbol =
        closureImpl->getAttrOfType<SymbolConstantAttr>(callMethodAttr);
    SmallVector<Value> args;
    args.push_back(implRef);
    for (unsigned i = 1 /*implPtr*/, e = closureSignature.getNumArguments();
         i != e; ++i)
      args.push_back(topLevelCall.getArgument(i));

    SymbolConstantAttr typedSymbol =
        createTypedSymbol(symbol, topLevelParams, shared);

    SmallVector<TypedAttr> implicitOrigins;
    auto finalSig = cast<FnTypeGeneratorType>(typedSymbol.getType());
    for (auto [arg, conv] : llvm::zip(args, finalSig.getArgConventions()))
      if (hasImplicitOrigin(conv))
        implicitOrigins.push_back(cast<RefType>(arg.getType()).getOrigin());

    Value result =
        CallOp::create(builder, resultType, typedSymbol, implicitOrigins, args)
            .getResult(0);
    IREmitter::emitNormalReturn(builder, result);
  }
  setMember(topLevelCall, callFieldAttr, topLevelTypes.callFuncFieldType);

  builder = ImplicitLocOpBuilder::atBlockEnd(init.getLoc(), init.getBody());
  IREmitter::emitNormalReturn(builder);
  return init;
}

static SymbolRefAttr
getFullyResolvedSymbolRefUpToFileModule(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  do {
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
  } while ((op = dyn_cast<mlir::SymbolOpInterface>(op->getParentOp())) &&
           !isa<FileModuleOp>(op));
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

TypedAttr ClosureEmitter::addWitnessTablesToClosure(
    ASTDecl &moduleDecl, SMLoc smLoc, FnOp parent, ClosureType closureType,
    SmallVector<ClosureParent> &closureParents) {
  // create kgen.struct.generator
  Location location = shared.translateLocation(smLoc);
  MLIRContext *ctx = shared.getContext();
  SymbolRefAttr parentSymbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(parent.getOperation()));
  ParamClosureType paramClosureType =
      KGEN::ParamClosureType::get(ctx, parentSymbolRef, closureType.getName());

  SmallVector<ParamDeclAttr> closureParams;
  auto capturesParam = ParamDeclAttr::get("CAPTURES", paramClosureType);
  closureParams.push_back(capturesParam);
  ParamDeclArrayAttr parameters = ParamDeclArrayAttr::get(ctx, closureParams);
  ImplicitLocOpBuilder builder(location, ctx);
  builder.setInsertionPointToStart(
      &cast<FileModuleOp>(moduleDecl.getIfOperation()).getBodyRegion().front());
  TraitType traitType = getTraitType(closureParents, moduleDecl);
  auto structGen = StructGeneratorOp::create(
      builder,
      StringAttr::get(
          ctx,
          Twine(getFlattenedSymbolName(getFullyResolvedSymbolRefUpToFileModule(
                    cast<mlir::SymbolOpInterface>(parent.getOperation()))))
              .concat("::")
              .concat(closureType.getName().getValue())),
      parameters, closureType, traitType);
  Block *structGenBody = builder.createBlock(&structGen.getRegion());

  // Emit the conformance ops into the struct gen body by finding the closure
  // method and FnOp associated with the parent trait.
  auto addWitnessTable = [&](ClosureParent &closureParent) {
    ClosureMethod method = closureParent.getClosureMethod();
    TraitDeclOp traitParent = closureParent.getTrait(moduleDecl);
    builder.setInsertionPointToStart(structGenBody);
    SymbolRefArrayAttr immediateParents = traitParent.getImmediateParentsAttr();
    SymbolRefAttr parentSymbol = closureParent.getSymbolRef(moduleDecl);
    StringAttr parentName = closureParent.getFullSymbolName(moduleDecl);
    ConformanceOp witnessTable = ConformanceOp::create(
        builder, parentName, parentSymbol, immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&block);

    FnOp fnOp = closureParent.getDefiningOp(moduleDecl);
    FnTypeGeneratorType sig =
        specializeSignature(fnOp, closureType, *shared.declResolver);
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
    paramValues.push_back(ParamDeclRefAttr::get(capturesParam));

    TypedAttr symbol = ClosureSymbolAttr::get(
        ctx, parentSymbolRef, closureType.getName(),
        ClosureMethodAttr::get(ctx, method), paramValues, sig);
    WitnessOp::create(builder, fnOp.getSymNameAttr(), symbol);
  };

  for (ClosureParent &closureParent : closureParents) {
    if (!closureParent.isEmpty())
      addWitnessTable(closureParent);
  }

  // create a SymbolRefAttr from the StructGeneratorOp
  SymbolRefAttr structGenSymbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(structGen.getOperation()));
  // Type value contains the reference to the struct gen op with the witness
  // table.
  ClosureAttr packedParamCaptures =
      KGEN::ClosureAttr::get(ctx, paramClosureType);
  auto typeValue = KGEN::TypeValueType::get(
      ctx, TypeGeneratorRefAttr::get(ctx, structGenSymbolRef,
                                     {packedParamCaptures}, traitType));
  auto typeParamAttr = TypeParamAttr::get(typeValue, closureType, traitType);
  return typeParamAttr;
}

Value ClosureEmitter::emitClosureOp(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                                    ArrayRef<Capture> captures,
                                    StructDeclOp wrapper, TraitDeclOp trait,
                                    Location location, bool isCopyable) {
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
  Location opLoc = FusedLoc::get(
      ctx, fileOnlyLoc,
      cast<FnOp>(symbolParent->getIfOperation()).getSubprogramScope());

  // TODO: use effect to determine the memory kind for the closure
  bool isRegPassable = nestedFn.getFuncTypeGenerator().isRegisterPassable();
  KGEN::ClosureType closureType =
      ClosureType::get(ctx, symbolParent->getSymbolRef(), fnName,
                       isRegPassable ? ClosureMemoryKind::REGISTER_PASSABLE
                                     : ClosureMemoryKind::NONESCAPING);
  SmallVector<Attribute> captureInfo;
  SmallVector<Value> captureValues;
  SmallVector<TypedAttr> origins;
  for (const Capture &capture : captures) {
    Value value = capture.getValue().getMlirValue();
    captureValues.push_back(value);
    UnitAttr isMove;
    auto captureConvention = capture.getCaptureConvention();
    switch (captureConvention) {
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionRead: {
      // Mutability casts should have been emitted during parse time.
      if (auto refType = dyn_cast<LIT::RefType>(value.getType())) {
        origins.push_back(refType.getOrigin());
        captureInfo.push_back(refType.getOrigin());
      } else {
        captureInfo.push_back(UnitAttr::get(ctx));
      }
      break;
    }
    case CaptureConvention::kConventionTrivialCopy:
      captureInfo.push_back(UnitAttr::get(ctx));
      break;
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionMove: {
      Type mlirType;
      if (auto refType = dyn_cast<LIT::RefType>(value.getType()))
        mlirType = refType.getElementType();
      else
        mlirType = value.getType();

      if (auto structType = dyn_cast<StructType>(mlirType)) {
        SymbolConstantAttr del, move, copy;
        ASTDecl &structDecl =
            shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
        StructDeclOp structDeclOp =
            cast<StructDeclOp>(structDecl.getIfOperation());
        if (isRegPassable && !structDeclOp.isRegisterPassable() &&
            !structDeclOp.isRegisterPassableTrivial()) {
          shared.emitError(
              nestedFnDecl.getLoc(),
              "cannot capture " + capture.getSpelling() +
                  " by copy or move because it is not register passable and "
                  "your closure is marked as register passable.");
        }
        if (structDeclOp.getDestructor().has_value())
          del = *structDeclOp.getDestructor();
        if (!del) {
          ArrayRef<ASTDecl *> results =
              structDecl.lookupInCurrentScope("__del__");
          if (results.size() == 1) {
            FnOp destructor = dyn_cast<FnOp>(results.front()->getIfOperation());
            if (destructor)
              del = destructor.getBoundSymbolRef(shared.getEvaluationContext());
          }
        }
        if (structDeclOp.getMoveInit().has_value())
          move = *structDeclOp.getMoveInit();
        if (structDeclOp.getCopyInit().has_value())
          copy = *structDeclOp.getCopyInit();

        if (captureConvention == CaptureConvention::kConventionCopy && !copy) {
          shared.emitError(nestedFnDecl.getLoc(),
                           "cannot capture " + capture.getSpelling() +
                               " by copy because it is not copyable.");
          return {};
        }
        if (captureConvention == CaptureConvention::kConventionMove) {
          if (!move) {
            shared.emitError(nestedFnDecl.getLoc(),
                             "cannot capture " + capture.getSpelling() +
                                 " by move because it is not movable.");

            return {};
          }
          isMove = UnitAttr::get(ctx);
        }
        if (!del)
          shared.emitError(nestedFnDecl.getLoc(),
                           "cannot capture " + capture.getSpelling() +
                               " because it is not destructable.");
        // rebind the parameterized symbol
        auto paramValues = structType.getParamValues();
        auto paramArray = ParameterExprArrayAttr::get(ctx, paramValues);
        auto bind = [&](SymbolConstantAttr sym) -> SymbolConstantAttr {
          if (!sym)
            return sym;
          // Resolve the FnOp, then bind concrete params on the symbol.
          ASTDecl *fnDecl =
              shared.declResolver->getDeclForFuncSymbol(sym.getSymbol());
          auto fnOp = cast<FnOp>(fnDecl->getIfOperation());
          SymbolConstantAttr bound =
              fnOp.getBoundSymbolRef(shared.getEvaluationContext(), paramArray);
          return bound;
        };

        copy = bind(copy);
        move = bind(move);
        del = bind(del);
        MemSymbolTripleAttr memTriple =
            MemSymbolTripleAttr::get(ctx, copy, move, del, isMove);
        captureInfo.push_back(memTriple);
        break;
      } else if (auto traitType = dyn_cast<TraitType>(mlirType)) {
        shared.emitError(nestedFnDecl.getLoc(),
                         "cannot capture a value of trait type yet because "
                         "existentials are not implemented.");
        return {};
      } else {
        // this is a trivially copyable/movable type
        captureInfo.push_back(UnitAttr::get(ctx));
      }
      break;
    }
    }
  }
  StringAttr originAttr =
      nestedFnDecl.getParentDecl()->mangleParamName(fnName.getValue());
  SmallVector<ClosureParent> closureParents{
      ClosureParent(trait, getFnOpNamed(trait, "__call__"),
                    ClosureMethod::CALL),
      moveParent, anyParent};
  if (isCopyable) {
    closureParents.push_back(copyParent);
    closureParents.push_back(implicitlyCopyableParent);
  }

  TypedAttr witnessTable = addWitnessTablesToClosure(
      moduleDecl, nestedFnDecl.getLoc(), parent, closureType, closureParents);
  ParamDeclAttr origin =
      ParamDeclAttr::get(originAttr, OriginType::get(ctx, true));
  auto refType = RefType::get(closureType, ParamDeclRefAttr::get(origin));
  FnTypeGeneratorType original = nestedFn.getFuncTypeGenerator();
  // TODO: Remove capturing when legacy closures are removed
  FnTypeGeneratorType withoutUnified = FnTypeGeneratorType::get(
      original.getInputParamTypes(), original.getValues(),
      original.getArgConventions(),
      original.getFnEffects()
          .setUnified(false)
          .setRegisterPassable(false)
          .setCapturing(true),
      original.getFnMetadata(), original.getMetadata());
  auto closure = LIT::ClosureInitOp::create(
      builder, opLoc, refType, withoutUnified, nestedFn.getFunctionType(),
      ValueRange(captureValues), ArrayAttr::get(ctx, captureInfo),
      nestedFn.getInputParams(), nestedFn.getInlineLevel(), origin,
      witnessTable, nestedFn.getSubprogramScope());

  closure.getBodyRegion().takeBody(nestedFn.getBodyRegion());

  // (2) Create the wrapper instance and populate it with the closure init op
  // value.

  // The wrapper takes ownership of the closure.
  OwnershipUseOp::create(builder, location, closure);

  // Create the wrapper instance by emitting a call to the Wrapper
  // constructor.
  auto originSet = OriginSetAttr::get(ctx, origins);
  LIT::StructType closureWrapperType =
      wrapper.bindReference({witnessTable, originSet});
  VarDeclOp var = VarDeclOp::create(
      builder, location, closureWrapperType, fnName.getValue(),
      nestedFnDecl.getParentDecl()->mangleParamName(fnName.getValue()),
      VarDeclKind::Var);
  SmallVector<Value> operands({closure->getResult(0), var});
  SmallVector<TypedAttr> implicitOrigins(
      {ParamDeclRefAttr::get(origin), var.getType().getOrigin()});
  FnOp init;
  for (auto fn : wrapper.getFields().getOps<FnOp>()) {
    if (fn.getSourceName() == "__init__") {
      assert(!init && "Wrapper has exactly one constructor.");
      init = fn;
    }
  }
  assert(init && "Wrapper has exactly one constructor but could not find it.");
  SymbolRefAttr symbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(init.getOperation()));
  FnTypeGeneratorType fullSig =
      LIT::getFullSignature(wrapper, init.getFuncTypeGenerator());
  SmallVector<TypedAttr> paramArgs;
  paramArgs.reserve(2);
  paramArgs.push_back(closure.getTypeValue());
  paramArgs.push_back(originSet);
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
    if (var.getKind() != VarDeclKind::Ref)
      return MLValue(var);
    auto ref = RefLoadOp::create(builder, loc, var);
    return CValue::getMValueForRef(ref);
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

ASTDecl *ClosureEmitter::addCaptureValue(ASTDecl &closure, SMLoc location,
                                         StringRef name,
                                         CaptureConvention parsedConvention,
                                         IREmitter &emitter,
                                         ASTDecl *signatureDecl) {
  SharedState &shared = emitter.shared;
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  LookupResult lookup = shared.lookupAndResolveDecl(
      name, location, *closure.getParentDecl(), true);
  if (!lookup.isSuccess()) {
    shared.emitError(location, "reference to an unknown value: ") << name;
    return nullptr;
  }
  ArrayRef<ASTDecl *> results = lookup.getIfSuccess();
  if (results.size() > 1) {
    shared.emitError(location, "ambiguous captured value: ") << name;
    return nullptr;
  }
  ASTDecl *result = results.front();
  if (auto pval = result->getIfIRValue().getIfPValue()) {
    shared.emitError(location, "value ")
        << name << " is a parameter and does not need a capture convention";
    return nullptr;
  }
  CValue valueInParent =
      ASTDeclToCValue(result, *emitter.builder, funcOp->getLoc());
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
  auto parentFn = funcOp->getParentOfType<FnOp>();
  DebugInfo::DIBuilder::ScopeGuard diGuard;
  if (shared.diBuilder)
    diGuard = shared.diBuilder->pushScopeGuard(parentFn.getLocScope());
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
        ValueDest dest(EC_Capture);
        captureValue = emitter.emitRValue(
            {MLValue(valueInParent.getMlirValue()), node}, dest);
      } else {
        captureValue = valueInParent;
      }
    } else {
      convention = parsedConvention;
      if (auto refType = dyn_cast<RefType>(valueInParent.getType().mlirType)) {
        OriginType originType = refType.getOriginType();
        if (originType.isMutableKnown(false)) {
          auto refImmutOp = LIT::RefImmutOp::create(
              *emitter.builder, funcOp.getLoc(), valueInParent.getMlirValue());
          captureValue = MBValue(refImmutOp->getResult(0));
        }
      }
      ValueDest dest(EC_Capture);
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
  case CaptureConvention::kConventionMut: {
    convention = parsedConvention;
    captureValue = valueInParent;
    // Ensure we are not capturing an immutable reference by mutable
    // reference.
    if (auto refType = dyn_cast<RefType>(valueInParent.getType().mlirType)) {
      OriginType originType = refType.getOriginType();
      if (originType.isMutableKnown(false)) {
        shared.emitError(location, "Cannot capture ")
            << name << " by mut because the value is immutable";
        return nullptr;
      }
    }
    break;
  }
  case CaptureConvention::kConventionRead: {
    convention = parsedConvention;
    captureValue = valueInParent;
    if (auto refType = dyn_cast<RefType>(valueInParent.getType().mlirType)) {
      OriginType originType = refType.getOriginType();
      if (originType.isMutableKnown(true)) {
        auto refImmutOp = LIT::RefImmutOp::create(
            *emitter.builder, parentFn.getLoc(), valueInParent.getMlirValue());
        captureValue = MBValue(refImmutOp->getResult(0));
      } else {
        captureValue = valueInParent;
      }
    }
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
  // Update the types of the struct wrapper.
  SymbolRefAttr symbol = closureParent.getSymbolRef(fileModule);
  TraitType oldTraitType = structDeclOp.getCanonicalTrait();
  SmallVector<SymbolRefAttr> symbols;
  llvm::append_range(symbols, oldTraitType.getSymbols());
  symbols.push_back(symbol);
  TraitType traitType = TraitType::get(ctx, symbols);
  structDeclOp.setCanonicalTrait(traitType);
}

static void addConformanceTable(ASTDecl &structDecl,
                                ClosureEmitter::ClosureParent closureParent,
                                TypedAttr newWitness, StringRef name,
                                ASTDecl &fileModule) {
  SmallVector<std::pair<StringRef, TypedAttr>> witnesses({{name, newWitness}});
  addConformanceTable(structDecl, closureParent, witnesses, fileModule);
}

LogicalResult
ClosureEmitter::augmentWitnessTablesToConformTo(ASTType structType,
                                                ASTDecl *traitDecl) {
  // Ensure that we have a valid closure trait and a struct metatype.
  TraitDeclOp traitDeclOp =
      llvm::dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
  if (!traitDeclOp)
    return failure();
  if (!traitDeclOp.getDefinesClosure())
    return failure();
  StructMetaType anyStruct = dyn_cast<StructMetaType>(structType);
  if (!anyStruct)
    return failure();
  ASTDecl &structDecl =
      shared.declResolver->getDeclForTypeSymbol(anyStruct.getSymbol());
  StructDeclOp structDeclOp =
      dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  if (!structDeclOp)
    return failure();

  // does the struct already conform to the trait?
  SymbolRefAttr target = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation()));
  for (SymbolRefAttr currentTrait :
       structDeclOp.getCanonicalTrait().getSymbols()) {
    if (target == currentTrait)
      return success();
  }

  // This trait defines a closure which means it has a single call function.
  StringRef name = "__call__";
  auto callDecls = structDecl.lookupInCurrentScope(name);
  FnOp callFunction = getFnOpNamed(traitDeclOp, name);
  // get the call function in terms of the struct wrapper
  SyntheticNode syntheticNode(structDecl.getLoc());
  ASTType structSelfType = structDecl.getTypeDeclSelf();
  IREmitter emitter(structDecl, EC_Trait);
  FnTypeGeneratorType traitSignature = specializeSignature(
      callFunction, structSelfType.mlirType, *shared.declResolver);
  auto bindings = ParamBindings::getForDeclaredType(
      emitter.getDeclScope(), structSelfType, syntheticNode);
  OverloadSet ov(name, callDecls, std::move(bindings), syntheticNode,
                 CallSyntax::kMethodCallSynthetic);
  /// Perform rebind on method that implements the trait function but with
  /// different argument names.
  PValue newWitness = ov.filterOverloadSetForValueType(
      traitSignature, emitter.getDeclScope(), nullptr);
  if (newWitness) {
    ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
    addConformanceTable(structDecl,
                        ClosureEmitter::ClosureParent(traitDeclOp, callFunction,
                                                      ClosureMethod::CALL),
                        newWitness.get(), callFunction.getSymNameAttr(),
                        fileModule);
    return success();
  }
  return failure();
}

void ClosureEmitter::addConformanceToDevicePassable(
    ASTDecl &structDecl, StructFieldOp devicePassedField, ParamDeclAttr impl,
    ParamDeclAttr originSet) {
  Type paramType = ParamType::get(ParamDeclRefAttr::get(impl));
  ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
  ASTDecl *devicePassableTrait =
      shared.getBuiltinDevicePassableTrait(structDecl.getLoc());
  // Ensure body is parsed and unresolved decls pulled in
  if (failed(shared.declResolver->resolveBody(*devicePassableTrait,
                                              devicePassableTrait->getLoc())))
    return;
  // Ensure the top level members are at least signature resolved.
  for (auto &nameGroup : devicePassableTrait->getDeclsInScope()) {
    for (ASTDecl *funcFieldOrAlias : nameGroup.second)
      if (failed(shared.declResolver->resolveSignature(
              *funcFieldOrAlias, funcFieldOrAlias->getLoc())))
        return;
  }
  if (!devicePassableTrait)
    return;
  TraitDeclOp trait = cast<TraitDeclOp>(devicePassableTrait->getIfOperation());
  SymbolRefAttr devicePassableSymbol = devicePassableTrait->getSymbolRef();
  SmallVector<std::pair<StringRef, TypedAttr>> devicePassableWitnesses;
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp);

  for (Operation &member : trait.getFields().getOps()) {
    if (auto function = dyn_cast<FnOp>(member)) {
      /// We already have AnyType members implemented, only implement those that
      /// are defined by DevicePassable.
      auto parent = function.getInheritedFrom();
      if (parent && parent != devicePassableSymbol)
        continue;
      if (function.getSourceName() == kToDeviceType) {
        auto [toDevice, params, result] =
            pushBackTraitFunctionImpl(function, structDecl);
        b.setInsertionPointToStart(&toDevice.getBodyRegion().front());
        assert(toDevice.getBodyRegion().getNumArguments() == 2);
        // get address
        Value targetArgument = toDevice.getBodyRegion().front().getArgument(1);
        StructType structType = cast<StructType>(targetArgument.getType());
        assert(structType.getParamValues().size() > 1 &&
               "expected pointer to be parameterized on element type");
        auto pointerElementType =
            dyn_cast<KGEN::TypeParamAttr>(structType.getParamValues().front());
        assert(pointerElementType &&
               "expected a the pointer type's first parameter to "
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

        // Invoke Copyable.__copyinit__(existing: read_mem, self:
        // byref_result)
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
              shared.getBuiltinStringType(structDecl, structDecl.getLoc())) {
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
        ASTType strLitType =
            shared.getBuiltinStringLiteralType(structDecl, structDecl.getLoc());
        auto strLitDecl =
            cast<StructDeclOp>(strLitType.getDecl(shared)->getIfOperation());
        Type boundStrLitType = strLitDecl.bindReference({closureStr});
        ValueDest litDest(EC_CallArgValue);
        CValue literalValue = emitter.emitConstructorCall(
            ASTType(boundStrLitType), CallOperands(), &loc,
            CallSyntax::kTypeCall, litDest);

        // Call String.__init__(literal) into the byref result slot.
        ASTType stringType =
            shared.getBuiltinStringType(structDecl, structDecl.getLoc());
        ValueDest resultDest(MLValue(block.getArguments().back()),
                             EC_ReturnValue);
        CallOperands ctorOperands;
        ctorOperands.add(ASTExprAnd<CValue>{literalValue, &loc});
        emitter.emitConstructorCall(stringType, std::move(ctorOperands), &loc,
                                    CallSyntax::kTypeCall, resultDest);
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
      assert(alias.getDeclName().getValue().contains("device_type") &&
             "we assume we are implementing device_type.");
      devicePassableWitnesses.push_back(
          {"device_type",
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
