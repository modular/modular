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

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

ClosureEmitter::ClosureEmitter(ASTDecl &moduleDecl, SharedState &shared)
    : FunctionEmitter(shared), ctx(shared.getContext()), moduleDecl(moduleDecl),
      node(moduleDecl.getLoc()),
      fileModuleOp(cast<FileModuleOp>(moduleDecl.getIfOperation())),
      selfName(StringAttr::get(ctx, "self")),
      otherName(StringAttr::get(ctx, "other")),
      dtorFieldAttr(StringAttr::get(ctx, "dtor")),
      copyFieldAttr(StringAttr::get(ctx, "_copy")),
      callFieldAttr(StringAttr::get(ctx, "call")),
      callMethodAttr(StringAttr::get(ctx, "closureCallMethod")),
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))) {}

static StructFieldOp addFieldOpAndDecl(StringAttr name, Type type,
                                       StructDeclOp structOp,
                                       ASTDecl &structDecl, OpBuilder &b,
                                       DeclResolver &declResolver) {
  auto field = b.create<StructFieldOp>(structOp.getLoc(), name, type);
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
  return b.create<RefLoadOp>(b.create<RefStructGEROp>(self, field));
}
static void storeField(ImplicitLocOpBuilder &b, Value self, Value value,
                       StructFieldOp field) {
  b.create<RefStoreOp>(value, b.create<RefStructGEROp>(self, field));
}
static void storeField(ImplicitLocOpBuilder &b, Value self, Value value,
                       StringAttr name) {
  auto resultTy = RefStructGEROp::getReboundFieldType(
      cast<RefType>(self.getType()), name, value.getType());
  auto fieldRef = b.create<RefStructGEROp>(resultTy, name, self);
  b.create<RefStoreOp>(value, fieldRef);
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
      b.create<StructDeclOp>(shared.diags.translateLocation(loc), name);
  declOp.setIsSynthetic(true);

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
      oldFnMetadata.getIsNestedOriginExclusivityCheckingDisabled());
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
      b.create<POP::PointerBitcastOp>(opaquePtrType, func.getArgument(0));
  storeField(b, self, opaqueFnPtr, b.getStringAttr("field0"));

  // Use the no-op destructor and copy constructor.
  ArrayRef<ASTDecl *> dtor = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_dtor", decl.getLoc());
  ArrayRef<ASTDecl *> copy = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_copy", decl.getLoc());
  if (dtor.empty() || copy.empty())
    return;

  Value dtorRef = b.create<CreateClosureOp>(
      cast<FnOp>(dtor.front()->getIfOperation())
          .getBoundReference(shared.getEvaluationContext()));
  Value copyRef = b.create<CreateClosureOp>(
      cast<FnOp>(copy.front()->getIfOperation())
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
             b.create<CreateClosureOp>(ParamDeclRefAttr::get(paramDecl)),
             b.getStringAttr("call"));
  IREmitter::emitNormalReturn(b);

  // Populate the lambda.
  b = ImplicitLocOpBuilder::atBlockBegin(callImpl.getLoc(), callImpl.getBody());
  Value fnPtr =
      b.create<POP::PointerBitcastOp>(fnPtrType, callImpl.getArgument(0));
  SmallVector<TypedAttr> origins;
  for (ParamDeclAttr originDecl : callImpl.getParams())
    origins.push_back(ParamDeclRefAttr::get(originDecl));
  SmallVector<Value> callArgs;
  llvm::append_range(callArgs, callImpl.getArguments());
  auto callIndirect =
      b.create<CallIndirectOp>(fnPtrType.getResultType(), fnPtr, origins,
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
    StringAttr name, ArrayRef<StringRef> parentNames,
    SMLoc nestedFunctionOrTypeLocation,
    llvm::function_ref<
        void(ASTDecl &traitDecl,
             DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
        populateTrait) {
  auto module = cast<FileModuleOp>(moduleDecl.getIfOperation());
  OpBuilder b(module.getRegion());
  MLIRContext *ctx = b.getContext();
  Location location =
      shared.diags.translateLocation(nestedFunctionOrTypeLocation);
  StringRef originalName = name.getValue();
  auto closureTrait =
      b.create<TraitDeclOp>(location, StringAttr::get(ctx, originalName));
  ASTDecl &traitDecl = shared.declResolver->addFullyResolvedDecl(
      &*closureTrait, name, nestedFunctionOrTypeLocation, &moduleDecl);
  closureTrait.setDefinesClosure(true);
  // Populate the trait with parent and self methods.
  SmallVector<SymbolRefAttr> parents;
  for (StringRef parent : parentNames) {
    if (auto anyType = shared.lookupBuiltinTrait(parent, &moduleDecl,
                                                 nestedFunctionOrTypeLocation))
      parents.push_back(anyType->getSymbolRef());
  }

  DenseSet<SymbolRefAttr> immediateParents;
  for (auto p : parents)
    immediateParents.insert(p);
  (void)shared.declResolver->addSelfTypeToTrait(closureTrait, traitDecl,
                                                parents, immediateParents);
  DenseSet<std::pair<StringAttr, StringAttr>> existingFns;
  populateTrait(traitDecl, existingFns);
  shared.declResolver->addParentDeclsToTrait(closureTrait, traitDecl,
                                             existingFns);
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
          b.create<RefStructGEROp>(arg, wrappedField)->getResults().front());
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

StructDeclOp ClosureEmitter::createStructWrapper(StringRef baseName,
                                                 ASTDecl &traitDecl,
                                                 SMLoc smLocation) {
  StringRef implName = "impl";
  TraitDeclOp trait = cast<TraitDeclOp>(traitDecl.getIfOperation());
  auto module = cast<FileModuleOp>(moduleDecl.getIfOperation());
  Location location = shared.diags.translateLocation(smLocation);
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(location, module->getBlock());
  b.setInsertionPointAfter(trait);
  MLIRContext *ctx = b.getContext();

  // Create the trait type.
  StringRef root = moduleDecl.getSymbolRef().getRootReference();
  SmallVector<FlatSymbolRefAttr> path(
      moduleDecl.getSymbolRef().getNestedReferences());
  path.push_back(FlatSymbolRefAttr::get(trait.getSymNameAttr()));
  SymbolRefAttr symbol = SymbolRefAttr::get(ctx, root, path);
  TraitType traitType = TraitType::get(ctx, symbol);

  // Give the struct a parameter "impl" of metatype trait.
  SmallVector<ParamDeclAttr> implParameters;
  ParamDeclAttr implType = ParamDeclAttr::get(implName, traitType);
  Type paramType = ParamType::get(ParamDeclRefAttr::get(implType));
  implParameters.push_back(implType);
  ASTType selfType(paramType);

  // Create a struct with a single field of type "impl".
  std::pair<ASTDecl &, StructDeclOp> pair =
      createStruct(shared, moduleDecl,
                   StringAttr::get(b.getContext(), baseName + "_wrapper"),
                   implParameters, smLocation);
  ASTDecl &structDecl = pair.first;
  StructDeclOp declOp = pair.second;
  addFieldsToStruct(declOp, structDecl,
                    KGEN::ParamType::get(ParamDeclRefAttr::get(implType)),
                    *shared.declResolver);
  StructFieldOp wrappedField = *declOp.getFieldDecls().begin();

  // Populate the wrapper methods with a call to the result of a vtable lookup.
  auto populateTraitFn = [&](FnOp traitFnOp) {
    b.setInsertionPointToEnd(&declOp.getFields().front());
    FnTypeGeneratorType wrappedSignature =
        specializeSignature(traitFnOp, selfType, *shared.declResolver);

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
                        wrappedSignature.getNumImplicitOriginDecls());
    ParamIndexRefReplacer replacer(parameters);
    SmallVector<Type> argumentTypes;
    llvm::append_range(
        argumentTypes,
        llvm::map_range(wrapperSignature.getArguments(), [&](Type original) {
          return replacer.replace(original);
        }));
    Type result = replacer.replace(wrapperSignature.getResults().front());
    auto [op, decl] =
        synthesizeFunction(structDecl, traitFnOp.getSourceNameAttr(),
                           parameters, wrapperSignature.getParamListAttrs(),
                           argumentTypes, wrapperSignature.getArgConventions(),
                           wrapperSignature.getArgListAttrs(), result,
                           traitFnOp.getSpecialFunctionKind(), smLocation, b,
                           wrapperSignature.getFnEffects().setUnified(false),
                           "", true, traitFnOp.getInlineLevel());

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

    TypedAttr symbol = ParamOperatorAttr::get(
        POC::GetVTableEntry,
        {ParamDeclRefAttr::get(implType.getName(), implType.getType()),
         StringAttr::get(op.getSourceNameAttr().getValue(),
                         StringType::get(ctx))},
        wrappedSignature);
    SmallVector<TypedAttr> paramArgs;
    llvm::append_range(
        paramArgs,
        llvm::map_range(parameters, [](ParamDeclAttr p) -> TypedAttr {
          return ParamDeclRefAttr::get(p);
        }));
    auto callOp = b.create<LIT::CallOp>(
        result,
        shared.getEvaluationContext().getBindParamsAttr(symbol, paramArgs),
        origins, operands);
    ValueRange results = callOp.getResults();
    // if this is a del, mark the self as destroyed. If it is a move, mark the
    // existing as destroyed.
    SpecialFunctionKind kind =
        (SpecialFunctionKind)traitFnOp.getSpecialFnKind();
    if (kind == SpecialFunctionKind::kMoveInit ||
        kind == SpecialFunctionKind::kDel) {
      auto arg0 = op.getBodyRegion().front().getArgument(0);
      b.create<LIT::OwnershipMarkDestroyedOp>(arg0);
    }
    b.create<LIT::ReturnOp>(results);
    b.create<LIT::EndFnOp>();
  };
  for (auto decls : traitDecl.getDeclsInScope()) {
    for (auto method : decls.second) {
      auto fnOp = dyn_cast_or_null<FnOp>(method->getIfOperation());
      if (!fnOp)
        continue;
      populateTraitFn(fnOp);
    }
  }

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
  StringAttr moveName = StringAttr::get("__moveinit__", StringType::get(ctx));
  FnOp moveFn;
  for (auto fnOp : trait.getFields().getOps<FnOp>()) {
    if (fnOp.getSourceName() == "__moveinit__") {
      moveFn = fnOp;
      break;
    }
  }
  assert(moveFn && "closures are movable but cannot find the move function.");
  FnTypeGeneratorType moveSignature =
      specializeSignature(moveFn, paramType, *shared.declResolver);
  b.setInsertionPointToStart(&initFnOp.getBodyRegion().front());
  TypedAttr moveSymbol = ParamOperatorAttr::get(
      POC::GetVTableEntry,
      {ParamDeclRefAttr::get(implType.getName(), implType.getType()), moveName},
      moveSignature);
  SmallVector<Value> operands;
  SmallVector<TypedAttr> origins;
  llvm::SmallDenseSet<StringRef> explicitParameters;
  getUnwrappedOperands(b, initFnOp, refSelfType.getElementType(), wrappedField,
                       explicitParameters, operands, origins);
  b.create<LIT::CallOp>(moveSignature.getResultType(), moveSymbol, origins,
                        operands);
  ValueRange results =
      b.create<ParamConstantOp>(NoneAttr::get(ctx))->getResults();
  b.create<LIT::ReturnOp>(results);
  b.create<LIT::EndFnOp>();
  declOp.setCanonicalTrait(traitType);
  return declOp;
}

std::pair<StructDeclOp, TraitDeclOp>
ClosureEmitter::createParametricClosureWrapperStructDecl(
    StringAttr name, FnTypeGeneratorType dependentSignatureType,
    SMLoc nestedFunctionOrTypeLocation, InlineLevel inlineLevel) {
  // Generate the movable, destructable closure trait, populating the trait
  // definition with the single characteristic "__call__" method.
  SmallVector<StringRef> parents{"Movable", "AnyType"};
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
    auto [fnOp, fnDecl] = synthesizeFunction(
        decl, callName, parameters, sig.getParamListAttrs(), argumentTypes,
        sig.getArgConventions(), sig.getArgListAttrs(), result,
        SpecialFunctionKind::kNormal, nestedFunctionOrTypeLocation, builder,
        sig.getFnEffects().setUnified(false), "", true, inlineLevel);
    builder.setInsertionPointToEnd(&fnOp.getBodyRegion().front());
    builder.create<UnreachableOp>();
    functions.insert({callName, fnOp.getSymNameAttr()});
  };
  auto [closureTrait, traitDecl] =
      createTraitOp(name, parents, nestedFunctionOrTypeLocation, populate);

  // Now create a wrapper struct that conforms to the trait we created.
  return {createStructWrapper(name.getValue(), *traitDecl,
                              nestedFunctionOrTypeLocation),
          closureTrait};
}

StructDeclOp ClosureEmitter::createClosureWrapperStructDecl(
    StringAttr name, FnTypeGeneratorType dependentSignatureType,
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
      paramValues, translateLocation(nestedFunctionOrTypeLocation),
      &shared.getEvaluationContext());
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
    b.create<CallIndirectOp>(noneType, callee,
                             /*implicitOrigins=*/ArrayRef<TypedAttr>(),
                             dtorImpl);
  }

  // Populate the copy constructor.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(copyCtr.getLocScope());
    // we want to insert before return at end of function. LIT::ReturnOp is not
    // a terminator though, so let's find it and set it.
    ImplicitLocOpBuilder b =
        ImplicitLocOpBuilder::atBlockBegin(copyCtr.getLoc(), copyCtr.getBody());
    auto returnOps = copyCtr.getBody()->getOps<LIT::ReturnOp>();
    assert(std::distance(returnOps.begin(), returnOps.end()) == 1 &&
           "copy should have exactly one return op.");
    b.setInsertionPoint(*returnOps.begin());
    Value copySelf = copyCtr.getBody()->getArgument(1);
    Value copyExisting = copyCtr.getBody()->getArgument(0);
    Value existingImpl = loadField(b, copyExisting, impl);
    Value funcPtr = loadField(b, copySelf, copy);
    auto call = b.create<CallIndirectOp>(
        opaquePtrType, funcPtr, /*implicitOrigins=*/ArrayRef<TypedAttr>(),
        existingImpl);
    storeField(b, copySelf, call.getResult(0), impl);
  }
  // Copy all the fields over as well.
  if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    return {};

  // Populate move constructor.
  {
    // Take the impl from the existing.
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(moveCtr.getLocScope());
    ImplicitLocOpBuilder b =
        ImplicitLocOpBuilder::atBlockBegin(moveCtr.getLoc(), moveCtr.getBody());
    Value moveExisting = moveCtr.getBody()->getArgument(0);
    auto opaquePointerTypeAttr = M::PointerAttr::get(ctx, 0, opaquePtrType);
    Value nullPtr =
        b.create<ParamConstantOp>(opaquePtrType, opaquePointerTypeAttr);
    storeField(b, moveExisting, nullPtr, impl);
  }
  // Move all the fields over as well.
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

    auto callResult = builder.create<CallIndirectOp>(
        resultType, callMemberPtr, implicitOrigins, arguments);
    IREmitter::emitNormalReturn(builder, callResult.getResult(0));
  }

  synthesizeWrapperFnPtrCtor(structDecl, selfType, dependentSignatureType);
  return declOp;
}

StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    ArrayRef<Capture> captures, ArrayRef<ParamDeclRefAttr> paramCaptures,
    ASTDecl &nestedFnDecl, FnTypeGeneratorType wrapperSig) {
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
    Value target = builder.create<RefStructGEROp>(selfArg, paramField);
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
  DebugInfo::DISubprogramAttr subprogramAttrOfCallFunc;

  if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
          callFuncLocation)) {
    subprogramAttrOfCallFunc = fusedLoc.getMetadata();
    DebugInfo::DISubprogramAttr subprogramAttrOfOriginalFunc;
    if (auto fusedLocOriginal =
            dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
                nestedFn.getLoc()))
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
    replacer.recursivelyReplaceElementsIn(callFunc, true, true);
  }

  builder =
      ImplicitLocOpBuilder::atBlockBegin(callFunc.getLoc(), callFunc.getBody());
  Value selfArg = callFunc.getBodyRegion().insertArgument(
      0U, callFunc.getFunctionType().getInput(0), callFuncLocation);

  if (paramField) {
    // Emit the `kgen.capture_list.expand` into the call if required.
    Value target = builder.create<RefStructGEROp>(selfArg, paramField);
    emitter.builder = builder;
    ValueDest dest(EC_Assignment);
    emitter.emitNamedMethodCall("expand", {{{MBValue(target), loc}}}, dest,
                                CallSyntax::kMethodCall, loc);
  }
  for (auto [capture, fieldOp] :
       llvm::zip(captures, llvm::drop_begin(declOp.getFieldDecls(),
                                            hasParamClosureCaptures))) {
    Value target = builder.create<RefStructGEROp>(selfArg, fieldOp);
    // If the capture is an SValue then it lives in register.
    if (capture.getValue().isSValue())
      target = builder.create<RefLoadOp>(target);

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
    if (captureValue.getType() != target.getType())
      target = builder.create<RebindOp>(captureValue.getType(), target);

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
  Value sizeOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return b.create<POP::AlignedAllocOp>(ptrType, ValueRange{alignOf, sizeOf});
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

FnOp ClosureEmitter::createWrapperInitWithImpl(StructDeclOp closureWrapper,
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
  auto [initTmp, _] =
      StructEmitter(closureDecl)
          .addVoidMethod("__init__", argTypes, argConventions,
                         argListAttrsOfInit, SpecialFunctionKind::kInit,
                         initParams, paramListAttrsOfInit);
  auto init = initTmp;
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
  Value targetRef = builder.create<RefFromPointerOp>(target, immortal,
                                                     /*startUninit=*/true,
                                                     /*endUninit=*/false);

  // Move the contents of the injected impl into the heap memory.
  IREmitter emitter(moduleDecl, builder);
  emitter.emitStoreToLValue({MRValue(source), &node}, MLValue(targetRef),
                            EC_Assignment);

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(1);
  Value erasedType =
      builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
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
      funcSymbol = ParamOperatorAttr::get(POC::Rebind, funcSymbol, fieldType);
    auto createClosure =
        builder.create<CreateClosureOp>(funcSymbol, ValueRange());
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
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(),
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
    Value existingPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, body->getArgument(0));

    // TODO(references): move closures to references and correct origins.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/true);
    Value targetRef = builder.create<RefFromPointerOp>(target, immortal,
                                                       /*startUninit=*/true,
                                                       /*endUninit=*/false);
    Value existingRef = builder.create<RefFromPointerOp>(existingPtr, immortal,
                                                         /*startUninit=*/false,
                                                         /*endUninit=*/false);

    // Copy the existing value into the target.
    // TODO: Use nicer expr emitter for the result expr.
    IREmitter emitter(*copyInitDecl, builder);
    emitter.emitStoreToLValue({MLValue(existingRef), &node}, MLValue(targetRef),
                              EC_Assignment);

    // Return the allocated and populated impl.
    auto loc = topLevelCopyInit.getLoc();
    Value erasedType = emitter.builder->create<POP::PointerBitcastOp>(
        loc, opaquePtrType, target);
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
    Value implPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, body->getArgument(0));

    // TODO(references): Move closures off pointers.
    // This takes ownership of the pointer, telling checklifetimes that the
    // value should be destroyed by the exit of the function.  ASAP destruction
    // will make sure it is immediately destroyed because there are no uses.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/true);
    (void)builder.create<RefFromPointerOp>(implPtr, immortal,
                                           /*startUninit=*/false,
                                           /*endUninit=*/true);

    // Free the memory we allocated on the heap to store the closure.
    builder.create<POP::AlignedFreeOp>(implPtr);
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
      translateLocation(loc), &shared.getEvaluationContext());

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
    Value implPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, closureArg);

    // FIXME: Thread a origin through correctly.

    // TODO(references): Move closures off pointers.
    auto immortal = builder.getAttr<AnyOriginAttr>(/*isMut=*/false);
    Value implRef = builder.create<RefFromPointerOp>(implPtr, immortal,
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
        builder.create<CallOp>(resultType, typedSymbol, implicitOrigins, args)
            .getResult(0);
    IREmitter::emitNormalReturn(builder, result);
  }
  setMember(topLevelCall, callFieldAttr, topLevelTypes.callFuncFieldType);
  return init;
}

static TypedAttr generateVTableAttr(SharedState &shared, Location location,
                                    FnOp parent, ClosureType closureType,
                                    TraitDeclOp trait) {
  MLIRContext *ctx = parent->getContext();
  SmallVector<VTableEntryAttr> entries;
  SymbolRefAttr parentSymbolRef = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(parent.getOperation()));
  for (auto fnOp : trait.getFields().getOps<FnOp>()) {
    std::optional<StringRef> sourceNameMaybe = fnOp.getSourceName();
    assert(sourceNameMaybe.has_value() &&
           "Expected the function to have a source name");
    StringRef sourceName = *sourceNameMaybe;
    ClosureMethod method;
    if (sourceName == "__moveinit__")
      method = ClosureMethod::MOVE;
    else if (sourceName == "__call__")
      method = ClosureMethod::CALL;
    else if (sourceName == "__del__")
      method = ClosureMethod::DEL;
    else
      assert(false && "closures should only define del, call, and move but "
                      "trait specifies unrecognized method");
    FnTypeGeneratorType sig =
        specializeSignature(fnOp, closureType, *shared.declResolver);
    SmallVector<TypedAttr> paramValues;
    TypedAttr symbol = ClosureSymbolAttr::get(
        ctx, parentSymbolRef, closureType.getName(),
        ClosureMethodAttr::get(ctx, method), paramValues, sig);
    entries.push_back(
        VTableEntryAttr::get(StringAttr::get(ctx, sourceName), symbol));
  }
  auto typeParamAttr = TypeParamAttr::get(
      closureType,
      TraitType::get(getFullyResolvedSymbolRef(
          cast<mlir::SymbolOpInterface>(trait.getOperation()))),
      VTableAttr::get(ctx, entries));
  return typeParamAttr;
}

Value ClosureEmitter::emitClosureOp(ASTDecl &nestedFnDecl,
                                    ArrayRef<Capture> captures,
                                    StructDeclOp wrapper, TraitDeclOp trait,
                                    Location location) {
  // (1) Create the closure instance.
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  FnOp parent = nestedFn->getParentOfType<FnOp>();
  assert(parent && "expected the function to be a nested function");
  ImplicitLocOpBuilder builder(location, shared.getContext());
  builder.setInsertionPoint(nestedFn);
  MLIRContext *ctx = builder.getContext();
  StringAttr fnName = nestedFn.getSourceNameAttr();
  // TODO: use effect to determine the memory kind for the closure
  KGEN::ClosureType closureType =
      ClosureType::get(ctx, nestedFnDecl.getParentDecl()->getSymbolRef(),
                       fnName, ClosureMemoryKind::NONESCAPING);
  // TODO: support explicit capture semantics. For now just default to by value.
  SmallVector<Attribute> captureInfo;
  SmallVector<Value> captureValues;
  SmallVector<TypedAttr> origins;
  for (const Capture &capture : captures) {
    captureValues.push_back(capture.getValue().getMlirValue());
    captureInfo.push_back(UnitAttr::get(ctx));
  }
  StringAttr originAttr =
      nestedFnDecl.getParentDecl()->mangleParamName(fnName.getValue());
  TypedAttr vTable =
      generateVTableAttr(shared, location, parent, closureType, trait);
  ParamDeclAttr origin =
      ParamDeclAttr::get(originAttr, OriginType::get(ctx, true));
  auto refType = RefType::get(closureType, ParamDeclRefAttr::get(origin));
  FnTypeGeneratorType original = nestedFn.getFuncTypeGenerator();
  FnTypeGeneratorType withoutUnified = FnTypeGeneratorType::get(
      original.getInputParamTypes(), original.getValues(),
      original.getArgConventions(), original.getFnEffects().setUnified(false),
      original.getFnMetadata(), original.getMetadata());
  auto closure = builder.create<LIT::ClosureInitOp>(
      location, refType, withoutUnified, nestedFn.getFunctionType(),
      ValueRange(captureValues), ArrayAttr::get(ctx, captureInfo),
      OriginSetAttr::get(ctx, origins), nestedFn.getInputParams(),
      nestedFn.getInlineLevel(), origin, vTable);
  closure.getBodyRegion().takeBody(nestedFn.getBodyRegion());

  // (2) Create the wrapper instance and populate it with the closure init op
  // value.

  // The wrapper takes ownership of the closure.
  builder.create<OwnershipUseOp>(location, closure);

  // Create the wrapper instance by emitting a call to the Wrapper constructor.
  LIT::StructType closureWrapperType =
      wrapper.bindReference(closure.getVtableAttr());
  VarDeclOp var = builder.create<VarDeclOp>(
      location, closureWrapperType, fnName.getValue(),
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
  paramArgs.push_back(closure.getVtableAttr());
  auto boundSig = fullSig.getSpecializedGenerator(paramArgs, location);
  TypedAttr symbol = SymbolConstantAttr::get(symbolRef, boundSig, paramArgs);
  builder.create<LIT::CallOp>(location, boundSig.getBody().getResults(), symbol,
                              implicitOrigins, operands);
  return MLValue(var);
}

ClosureEmitter::ClosureSignatureInfo
ClosureEmitter::remapClosureParameters(ASTDecl &decl,
                                       TypeCheckedParamList &paramList) {
  ClosureSignatureInfo signatureInfo;
  MLIRContext *ctx = paramList.shared.getContext();

  // Check the parsed parameters for function typed parameters. Such parameters
  // will trigger an argument/field augmentation.
  for (auto [smLoc, paramDecl, pog] :
       llvm::zip(paramList.locations, paramList.paramDeclAttrs,
                 paramList.getParamListAttr().getPogs())) {
    auto traitType = dyn_cast<TraitType>(paramDecl.getType());
    if (!traitType)
      continue;

    // Not all traits are registered with the shared state but since closures
    // have a single symbol they should be registered. This must not be a
    // closure.
    ASTDecl *traitDecl = shared.declResolver->getTraitDecl(traitType);
    if (!traitDecl)
      continue;

    // The decl associated with the trait type is a parameter, probably the
    // type. This is not the case with closures, so this is not a closure.
    if (!traitDecl->getIfOperation())
      continue;
    auto traitDeclOp = llvm::dyn_cast<TraitDeclOp>(traitDecl->getIfOperation());
    assert(traitDeclOp &&
           "expected a trait decl op associated with the trait type");
    if (!traitDeclOp.getDefinesClosure())
      continue;

    // We have a parameter that is a closure. That means we need an implicit
    // argument.

    // Note that because we can get a type from a value but we cannot get a
    // value from a type, we rebind the parameter name to the runtime value.
    // During emission, we will use the expression context to determine what
    // type of value to emit.
    assert(succeeded(shared.declResolver->replaceNameAssociatedWithParameter(
               pog.getName(),
               StringAttr::get(ctx, pog.getName().getValue() + "```"), decl)) &&
           "expected the parameters to be parsed and registered "
           "before performing closure analysis.");
    Type elementType =
        ParamType::get(ParamDeclRefAttr::get(pog.getName(), traitType));

    RefType refType =
        ASTType(elementType).getRefForArgument(pog.getName().getValue(), false);
    signatureInfo.argTypes.push_back(refType);
    ParamDeclRefAttr originRef =
        dyn_cast<ParamDeclRefAttr>(refType.getOrigin());
    assert(originRef &&
           "expected a param ref for the origin entry of the ref type");
    signatureInfo.implicitOriginDecls.push_back(
        ParamDeclAttr::get(originRef.getName(), originRef.getType()));

    signatureInfo.argPogs.push_back(
        PogMetadataAttr::get(pog.getName(), PassingKind::PosOnly));
    signatureInfo.argConventions.push_back(ArgConvention::ReadMem);
    signatureInfo.locations.push_back(smLoc);
  }
  return signatureInfo;
}

FnOpAttributes
ClosureEmitter::computeFunctionSignature(ASTDecl &decl, StringAttr baseName,
                                         TypeCheckedFnSignature &tcSignature) {
  FnTypeGeneratorType signature;
  FunctionType functionType;
  MLIRContext *ctx = shared.getContext();

  ClosureSignatureInfo signatureInfo =
      remapClosureParameters(decl, tcSignature.paramList);

  FnOp funcOp = cast<FnOp>(decl.getIfOperation());

  // If there are closures, add block arguments.
  if (signatureInfo.argTypes.size()) {
    for (auto [pog, argType, location] :
         llvm::reverse(llvm::zip(signatureInfo.argPogs, signatureInfo.argTypes,
                                 signatureInfo.locations))) {
      BlockArgument bbArg = funcOp.getBodyRegion().front().insertArgument(
          (unsigned)0, argType, shared.diags.translateLocation(location));
      shared.getDeclResolver().addFullyResolvedDecl(
          CValue::getMValueForRef(bbArg), pog.getName(), location, &decl);
    }

    for (auto [parsedArg, argType] :
         llvm::zip(tcSignature.argList.parsedArgs, tcSignature.fullArgTypes)) {
      signatureInfo.argConventions.push_back(parsedArg.kgenConvention);
      signatureInfo.argPogs.push_back(PogMetadataAttr::get(
          parsedArg.name, parsedArg.getKWArgHandlingAsPassingKind()));
      signatureInfo.argTypes.push_back(argType);
    }
    unsigned numImplicitOrigins = signatureInfo.implicitOriginDecls.size() +
                                  tcSignature.implicitOriginDecls.size();
    auto paramPogList = tcSignature.paramList.getParamListAttr();
    functionType = FunctionType::get(ctx, signatureInfo.argTypes,
                                     tcSignature.fullResultType.mlirType);
    signature = FnTypeGeneratorType::remapToFuncTypeGenerator(
        tcSignature.paramList.paramDeclAttrs, functionType,
        signatureInfo.argConventions, tcSignature.argList.effects,
        FnMetadataAttr::get(
            ctx, PogListAttr::get(ctx, signatureInfo.argPogs),
            numImplicitOrigins,
            getOriginsAccessibleByParams(paramPogList,
                                         tcSignature.paramList.paramDeclAttrs,
                                         shared, tcSignature.captureOrigins),
            tcSignature.isNestedOriginExclusivityCheckingDisabled),
        paramPogList);
  } else {
    // Otherwise, there are no closures so no transformation is needed.
    signature = tcSignature.getFnTypeGeneratorType();
    functionType = tcSignature.getFunctionType();
  }

  if (!signature)
    return {{}, 0, {}};
  unsigned numberOfImplicitClosureArgs =
      signatureInfo.implicitOriginDecls.size();
  llvm::append_range(signatureInfo.implicitOriginDecls,
                     tcSignature.implicitOriginDecls);
  tcSignature.implicitOriginDecls =
      std::move(signatureInfo.implicitOriginDecls);

  return {signature, numberOfImplicitClosureArgs, functionType};
}
