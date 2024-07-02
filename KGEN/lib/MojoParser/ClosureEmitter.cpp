//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ClosureEmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
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
    : StructEmitter(shared), ctx(shared.getContext()), moduleDecl(moduleDecl),
      node(moduleDecl.getLoc()), fileModuleOp(cast<FileModuleOp>(moduleDecl)),
      selfName(StringAttr::get(ctx, "self")),
      otherName(StringAttr::get(ctx, "other")),
      dtorFieldAttr(StringAttr::get(ctx, "dtor")),
      copyFieldAttr(StringAttr::get(ctx, "copy")),
      callFieldAttr(StringAttr::get(ctx, "call")),
      callMethodAttr(StringAttr::get(ctx, "closureCallMethod")),
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))),
      opaqueRefType(
          RefType::getImmortal(KGEN::NoneType::get(ctx), /*isMut=*/true)) {}

static void addFieldsToStruct(StructDeclOp structOp, ArrayRef<Type> fields) {
  OpBuilder b(structOp.getRegion());
  b.setInsertionPointToStart(&structOp.getFields().front());
  for (auto [i, type] : llvm::enumerate(fields))
    b.create<StructFieldOp>(structOp.getLoc(), "field" + Twine(i), type);
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
  b.create<RefStoreOp>(
      value, b.create<RefStructGEROp>(
                 cast<RefType>(self.getType()).getWithElement(value.getType()),
                 name, self));
}

static std::pair<ASTDecl &, StructDeclOp>
createStruct(SharedState &shared, ASTDecl &moduleDecl, StringAttr name,
             ArrayRef<ParamDeclAttr> params) {
  auto module = cast<FileModuleOp>(moduleDecl);
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

  StructDeclOp declOp = b.create<StructDeclOp>(module.getLoc(), name);
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
      &*declOp, name, moduleDecl.getLoc(), &moduleDecl);

  structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(declOp));
  return {structDecl, declOp};
}

/// Given a signature of a function, create a new signature by inserting a
/// closure argument at index 0 with the given convention.
static LITSignatureType
addClosureSelfArgToFunctionSignature(Type closureType, ArgConvention convention,
                                     LITSignatureType sig) {
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
  FnMetadataAttr oldMetadata = sig.getMetadata();
  PogListAttr argListAttr = oldMetadata.getArgListAttrs();
  llvm::append_range(signatureInputs, sig.getArguments());
  llvm::append_range(argConventions, sig.getArgConventions());
  llvm::append_range(argPogs, argListAttr.getPogs());
  assert(argPogs.size() == argConventions.size());

  // A closure signature is not escaping because its 'escaping' state is
  // captured in the self argument we are inserting in this function.
  auto metadata = FnMetadataAttr::get(
      argListAttr.cloneWith(argPogs), oldMetadata.getParamListAttrs(),
      oldMetadata.getNumImplicitLifetimeDecls());
  return SignatureType::get(
      FunctionType::get(ctx, signatureInputs, sig.getResults()),
      sig.getParamTypes(), /*resultParamTypes=*/{}, argConventions,
      sig.getFnEffects().setEscaping(false), metadata);
}

/// ```mojo
/// fn __init__(inout self, f: fn_ptr_type):
///     self.field0 = f
///     self.dtor = __closure_wrapper_noop_dtor
///     self.copy = __closure_wrapper_noop_copy
///     fn call_impl(field0: !kgen.pointer<none>, *args):
///         return (fn_ptr_type)(field0)(*args)
///     self.call = call_impl
/// ```
void ClosureEmitter::synthesizeWrapperFnPtrCtor(ASTDecl &decl, ASTType selfType,
                                                LITSignatureType sig) {
  // Skip this if builtins are not found.
  if (!shared.hasBuiltinModule())
    return;

  // Declare the function.
  LITSignatureType fnPtrType =
      sig.getWithFnEffects(sig.getFnEffects().setEscaping(false));
  auto b = ImplicitLocOpBuilder::atBlockEnd(
      translateLocation(decl.getLoc()),
      &cast<StructDeclOp>(decl).getFields().front());
  auto argListAttrs = PogListAttr::get(
      ctx, {selfName, otherName}, {PassingKind::PosOrKw, PassingKind::PosOrKw});
  LIT::FuncOp func = createFunction(
      decl, "__init__", /*params=*/{},
      /*paramListAttrs=*/PogListAttr::get(ctx),
      {selfType.getRefForArgument("self", /*isMut=*/true), fnPtrType},
      {ArgConvention::InitSelf, ArgConvention::BorrowedInReg}, argListAttrs,
      noneType, SpecialFunctionKind::kInit, decl.getLoc(), b);
  func.setInlineLevel(InlineLevel::Always);
  shared.declResolver->addFullyResolvedDecl(&*func, "__init__", decl.getLoc(),
                                            &decl);
  Value self = func.getArgument(0);
  b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());

  // Store the function pointer into the pointer field.
  Value opaqueFnPtr =
      b.create<POP::PointerBitcastOp>(opaquePtrType, func.getArgument(1));
  storeField(b, self, opaqueFnPtr, b.getStringAttr("field0"));

  // Use the no-op destructor and copy constructor.
  ArrayRef<ASTDecl *> dtor = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_dtor", decl.getLoc());
  ArrayRef<ASTDecl *> copy = shared.getBuiltinFunction(
      decl, "builtin._closure", "__closure_wrapper_noop_copy", decl.getLoc());
  if (dtor.empty() || copy.empty())
    return;

  Value dtorRef = b.create<CreateClosureOp>(
      cast<LIT::FuncOp>(dtor.front()).getBoundReference());
  Value copyRef = b.create<CreateClosureOp>(
      cast<LIT::FuncOp>(copy.front()).getBoundReference());
  storeField(b, self, dtorRef, b.getStringAttr("dtor"));
  storeField(b, self, copyRef, b.getStringAttr("copy"));

  // Generate the 'call_impl' function that performs the indirect call.
  LITSignatureType callImplType = addClosureSelfArgToFunctionSignature(
      opaquePtrType, ArgConvention::BorrowedInReg, fnPtrType);
  StringAttr lambdaName = b.getStringAttr("call_impl");
  LIT::FuncOp callImpl = createFunction(
      decl, lambdaName, /*params=*/{}, callImplType.getParamListAttrs(),
      callImplType.getArguments(), callImplType.getArgConventions(),
      callImplType.getArgListAttrs(), fnPtrType.getResultType(),
      SpecialFunctionKind::kNormal, decl.getLoc(), b, fnPtrType.getFnEffects());
  auto paramDecl = ParamDeclAttr::get(lambdaName, callImpl.getSignature());
  callImpl.setParamDeclAttr(paramDecl);

  // Store it into the call field.
  storeField(b, self,
             b.create<CreateClosureOp>(ParamDeclRefAttr::get(paramDecl)),
             b.getStringAttr("call"));
  b.create<LIT::ReturnOp>(Value(b.create<ParamConstantOp>(NoneAttr::get(ctx))));
  b.create<EndFuncOp>();

  // Populate the lambda.
  b = ImplicitLocOpBuilder::atBlockBegin(callImpl.getLoc(), callImpl.getBody());
  Value fnPtr =
      b.create<POP::PointerBitcastOp>(fnPtrType, callImpl.getArgument(0));
  SmallVector<TypedAttr> lifetimes;
  for (ParamDeclAttr lifetimeDecl : callImpl.getParams())
    lifetimes.push_back(ParamDeclRefAttr::get(lifetimeDecl));
  SmallVector<Value> callArgs;
  llvm::append_range(callArgs, callImpl.getArguments());
  auto callIndirect =
      b.create<CallIndirectOp>(fnPtrType.getResultType(), fnPtr, lifetimes,
                               ArrayRef(callArgs).drop_front());
  b.create<LIT::ReturnOp>(callIndirect.getResult(0));
  b.create<EndFuncOp>();
}

StructDeclOp ClosureEmitter::createClosureWrapperStructDecl(
    StringAttr name, LITSignatureType dependentSignatureType,
    SMLoc nestedFunctionOrTypeLocation) {
  SmallVector<Type> fieldTypes{opaquePtrType};

  if (!dependentSignatureType.getResultParamTypes().empty()) {
    shared.emitError(
        nestedFunctionOrTypeLocation,
        "result parameters in escaping closures are not supported yet");
    return {};
  }
  SmallVector<ParamDeclAttr> wrapperDecls;
  ParserParamEvaluator evaluator(getDeclResolver());
  SmallVector<TypedAttr> paramValues;
  for (auto [i, type] :
       llvm::enumerate(dependentSignatureType.getParamTypes())) {
    wrapperDecls.push_back(
        ParamDeclAttr::get(StringAttr::get(getContext(), "p" + Twine(i)),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(wrapperDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }

  auto [structDecl, declOp] =
      createStruct(shared, moduleDecl, name, wrapperDecls);
  addFieldsToStruct(declOp, opaquePtrType);
  declOp.setClosureSignature(dependentSignatureType);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  auto dtorMetadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, {selfName}, {PassingKind::PosOnly}));
  auto dtorSig = SignatureType::get(b.getFunctionType(opaquePtrType, noneType),
                                    ArgConvention::BorrowedInReg,
                                    /*effects=*/{}, dtorMetadata);
  auto dtor = b.create<StructFieldOp>(declOp.getLoc(), dtorFieldAttr, dtorSig);

  // Create Copy Member.
  auto fnType =
      b.getType<FunctionType>(ArrayRef<Type>{opaquePtrType}, opaquePtrType);
  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, {otherName}, {PassingKind::PosOnly}));
  auto cpySignatureType = SignatureType::get(
      fnType, {ArgConvention::BorrowedInReg}, /*effects=*/{}, metadata);
  auto copy =
      b.create<StructFieldOp>(declOp.getLoc(), copyFieldAttr, cpySignatureType);

  dependentSignatureType = dependentSignatureType.getSpecializedSignature(
      paramValues, translateLocation(nestedFunctionOrTypeLocation));
  auto sigMetadata =
      FnMetadataAttr::get(dependentSignatureType.getArgListAttrs(),
                          dependentSignatureType.getNumImplicitLifetimeDecls());
  Type resultType = dependentSignatureType.getResults().front();
  FunctionType functionType =
      b.getFunctionType(dependentSignatureType.getArguments(), resultType);
  LITSignatureType signatureType = SignatureType::get(
      functionType, {}, {}, dependentSignatureType.getArgConventions(),
      dependentSignatureType.getFnEffects(), sigMetadata);

  // Add the call member
  LITSignatureType callMemberSignatureType =
      addClosureSelfArgToFunctionSignature(
          opaquePtrType, ArgConvention::BorrowedInReg, signatureType);
  auto callMember = b.create<StructFieldOp>(declOp.getLoc(), callFieldAttr,
                                            callMemberSignatureType);

  for (StructFieldOp field : declOp.getFieldDecls()) {
    shared.declResolver->addFullyResolvedDecl(field.getOperation(),
                                              field.getNameAttr(),
                                              structDecl.getLoc(), &structDecl);
  }

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      structDecl, /*generateFieldwiseInit=*/false,
      /*forceGenerateDestructor=*/true);
  assert(stubs && "expected the stubs on a purely synthetic class to succeed.");
  LIT::FuncOp destructor = stubs->dtor;
  declOp.setDestructorAttr(destructor.getBoundSymbolRef());

  LIT::FuncOp copyCtr = stubs->copyCtr;
  SymbolConstantAttr copyCtrRef = copyCtr.getBoundSymbolRef();
  declOp.setCopyInitAttr(copyCtrRef);
  ASTDecl *copyCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(copyCtrRef.getSymbol());

  LIT::FuncOp moveCtr = stubs->moveCtr;
  SymbolConstantAttr moveCtrRef = moveCtr.getBoundSymbolRef();
  declOp.setMoveInitAttr(moveCtrRef);
  ASTDecl *moveCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(moveCtrRef.getSymbol());

  // Populate destructor.
  {
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
        destructor.getLoc(), destructor.getBody());
    Value dtorSelf = destructor.getBody()->getArgument(0);
    Value dtorImpl = loadField(b, dtorSelf, impl);
    Value callee = loadField(b, dtorSelf, dtor);
    b.create<CallIndirectOp>(noneType, callee,
                             /*implicitLifetimes=*/ArrayRef<TypedAttr>(),
                             dtorImpl);
  }

  // Populate the copy constructor.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(copyCtr.getLocScope());
    Location translatedLocation =
        shared.translateLocation(copyCtrDecl->getLoc());
    // we want to insert before return at end of function. LIT::ReturnOp is not
    // a terminator though, so let's find it and set it.
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
        translatedLocation, copyCtr.getBody());
    auto returnOps = copyCtr.getBody()->getOps<LIT::ReturnOp>();
    assert(std::distance(returnOps.begin(), returnOps.end()) == 1 &&
           "copy should have exactly one return op.");
    b.setInsertionPoint(*returnOps.begin());
    Value copySelf = copyCtr.getBody()->getArgument(0);
    Value copyExisting = copyCtr.getBody()->getArgument(1);
    Value existingImpl = loadField(b, copyExisting, impl);
    Value funcPtr = loadField(b, copySelf, copy);
    auto call = b.create<CallIndirectOp>(
        opaquePtrType, funcPtr, /*implicitLifetimes=*/ArrayRef<TypedAttr>(),
        existingImpl);
    storeField(b, copySelf, call.getResult(0), impl);
  }
  if (failed(populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    return {};

  // Populate move constructor.
  {
    // Take the impl from the existing.
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(moveCtr.getLocScope());
    Location translatedLocation =
        shared.translateLocation(moveCtrDecl->getLoc());
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
        translatedLocation, moveCtr.getBody());
    Value moveExisting = moveCtr.getBody()->getArgument(1);
    auto opaquePointerTypeAttr = M::PointerAttr::get(ctx, 0, opaquePtrType);
    Value nullPtr =
        b.create<ParamConstantOp>(opaquePtrType, opaquePointerTypeAttr);
    storeField(b, moveExisting, nullPtr, impl);
  }
  if (failed(populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
    return {};

  // Add the __call__ Method.
  ASTType selfType = structDecl.getTypeDeclSelf();
  auto refToSelfType = selfType.getRefForArgument("self", /*isMut=*/false);
  LITSignatureType closureMethodSignatureType =
      addClosureSelfArgToFunctionSignature(
          refToSelfType, ArgConvention::BorrowedInMem, signatureType);
  // The __call__ method is effectively the in-source body of the function. Mark
  // it as *not* synthetic so that debugging will step into the body.
  auto [callMethod, _] = synthesizeMethodInStruct(
      "__call__", closureMethodSignatureType.getArguments(),
      closureMethodSignatureType.getArgConventions(),
      closureMethodSignatureType.getArgListAttrs(), resultType, structDecl,
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

    SmallVector<TypedAttr> implicitLifetimes;
    auto calleeSig = cast<SignatureType>(callMemberPtr.getType());
    for (auto [arg, conv] : llvm::zip(arguments, calleeSig.getArgConventions()))
      if (SignatureType::hasImplicitLifetime(conv))
        implicitLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());

    auto callResult = builder.create<CallIndirectOp>(
        resultType, callMemberPtr, implicitLifetimes, arguments);
    ExprEmitter::emitNormalReturn(builder, callResult.getResult(0), callMethod);
    builder.create<LIT::EndFuncOp>();
  }

  synthesizeWrapperFnPtrCtor(structDecl, selfType, dependentSignatureType);
  return declOp;
}

/// Generate a Closure Implementation Struct, a struct that contains the
/// capture list.
StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    SMLoc location, ASTDecl &nestedFnDecl,
    ArrayRef<ParamDeclRefAttr> paramCaptures, LITSignatureType wrapperSig) {
  auto implName =
      StringAttr::get(ctx, "`_CI_" + fileModuleOp.getSymName() + "_escaping" +
                               Twine(moduleDecl.getNextUniqueID()));

  // Create map from the parent name to the index of the parameter in the
  // closure struct.
  FuncOp nestedFn = cast<LIT::FuncOp>(nestedFnDecl);
  wrapperSig = nestedFn.getSignature();
  if (wrapperSig.getNumParams()) {
    shared.emitError(
        location,
        "add parameters of nested function to parent function and capture "
        "them: parameters declared in nested functions are not supported yet");
    return {};
  }

  // Collect the types of the capture values.
  auto captures = shared.getCaptureRangeInScope(nestedFnDecl);
  SmallVector<Type> fieldTypes = llvm::map_to_vector(
      llvm::make_second_range(captures), [](const Capture &capture) {
        return capture.getValue().getRValueType().mlirType;
      });

  // Check for parameter closure captures.
  bool hasParamClosureCaptures = false;
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SignatureType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  for (ParamDeclRefAttr pc : paramCaptures)
    hasParamClosureCaptures |= walker.walk(pc.getType()).wasInterrupted();

  // Create the closure impl struct with the field types. Add the capture
  // parameters as parameter decls to the generated struct. This way, parameter
  // references within the body do not have to be renamed.
  auto [structDecl, declOp] =
      createStruct(shared, moduleDecl, implName,
                   llvm::map_to_vector(paramCaptures, [](ParamDeclRefAttr ref) {
                     return ParamDeclAttr::get(ref);
                   }));

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

  // Currently Closure Impls are not register passable, so use BorrowedInMem
  // convention.
  ASTType structSelfType = structDecl.getTypeDeclSelf();
  callInputTypes.push_back(
      ASTType(structSelfType).getRefForArgument("self", /*isMut=*/false));
  callConventions.push_back(ArgConvention::BorrowedInMem);
  callPogs.emplace_back(
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));

  llvm::append_range(callInputTypes, nestedFn.getFunctionType().getInputs());
  llvm::append_range(callConventions, wrapperSig.getArgConventions());
  llvm::append_range(callPogs, wrapperSig.getArgListAttrs().getPogs());

  Type closureResultType = wrapperSig.getResults().front();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(declOp.getLoc(),
                                                  &declOp.getFields().front());
  auto [callFunc, _] = synthesizeMethodInStruct(
      "__call__", callInputTypes, callConventions,
      PogListAttr::get(ctx, callPogs), closureResultType, structDecl,
      SpecialFunctionKind::kNormal,
      wrapperSig.getFnEffects().setEscaping(false));
  callFunc.setInlineLevel(InlineLevel::Always);

  // Add and register its fields as fully resolved decls.
  addFieldsToStruct(declOp, fieldTypes);
  for (StructFieldOp field : declOp.getFieldDecls()) {
    shared.declResolver->addFullyResolvedDecl(&*field, field.getNameAttr(),
                                              structDecl.getLoc(), &structDecl);
  }

  // Build the init method. This only needs the captured arguments. Populate the
  // function argument information.

  // All arguments as positional-only.
  SmallVector<PassingKind> initSigPassingKinds(1 + captures.size(),
                                               PassingKind::PosOnly);
  // Fill the types and conventions based on the register-passabilities.
  SmallVector<StringAttr> initSigNames{selfName};
  SmallVector<Type> initSigTypes{
      ASTType(structSelfType).getRefForArgument("self", /*isMut=*/true)};
  SmallVector<ArgConvention> initSigConventions{ArgConvention::InitSelf};
  unsigned fieldNameIdx = 0;
  for (auto &[decl, capture] : captures) {
    // If this is a reference capture, then we are capturing the address of the
    // value in the closure, otherwise we are taking an RValue that is either
    // copied or moved.
    bool isRef = capture.isRef();
    ASTType rvalueType = capture.getValue().getRValueType();
    initSigNames.push_back(StringAttr::get(ctx, "fld" + Twine(fieldNameIdx++)));
    // FIXME: By-reference captures should be capturable either as by-imm-ref or
    // by-mut-ref.  Right now we type check var captures as mutable but codegen
    // them as immutable references!
    if (rvalueType.isRegisterPassable(decl->getLoc(), shared)) {
      initSigConventions.push_back(isRef ? ArgConvention::BorrowedInReg
                                         : ArgConvention::OwnedInReg);
      initSigTypes.push_back(rvalueType);
    } else {
      initSigConventions.push_back(isRef ? ArgConvention::BorrowedInMem
                                         : ArgConvention::OwnedInMem);
      initSigTypes.push_back(rvalueType.getRefForArgument(
          initSigNames.back().str(), /*isMut=*/!isRef));
    }
  }

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      structDecl, /*generateFieldwiseInit=*/false,
      /*forceGenerateDestructor=*/true);
  LIT::FuncOp initFunc = synthesizeMemberwiseInit(
      structDecl, initSigTypes, initSigConventions,
      PogListAttr::get(ctx, initSigNames, initSigPassingKinds));
  builder =
      ImplicitLocOpBuilder::atBlockBegin(initFunc.getLoc(), initFunc.getBody());

  StructFieldOp paramField;
  ExprEmitter emitter(shared, nestedFnDecl, builder);
  SyntheticNode loc(nestedFnDecl.getLoc());
  if (hasParamClosureCaptures) {
    // Propagate the 'capturing' bit to the init function.
    LITSignatureType oldSig = initFunc.getSignature();
    initFunc.setSignature(
        oldSig.getWithFnEffects(oldSig.getFnEffects().setCapturing(true)));

    // Declare an extra field to carry the parametric closure captures.
    ASTType clType = shared.getBuiltinCaptureListType(nestedFnDecl.getLoc());
    TypedAttr bound = callFunc.getBoundReference(ParameterExprArrayAttr::get(
        getContext(), cast<DeclRefType>(structSelfType).getParamValues()));
    clType = BindTypeAttr::get(
        PValue(clType),
        {TypeConstantAttr::get(bound.getType(), TypeType::get(getContext())),
         bound});
    auto b = OpBuilder::atBlockBegin(declOp.getBody());
    paramField =
        b.create<StructFieldOp>(initFunc.getLoc(), "param_capture", clType);

    // Emit IR to generate the capture list and store it into self. Bind the
    // call function reference to itself.
    auto selfArg = initFunc.getArgument(0);
    Value target = builder.create<RefStructGEROp>(selfArg, paramField);
    ValueDest dest(MLValue(target), EC_Assignment);
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(initFunc.getLocScope());
    emitter.emitConstructorCall(clType, {}, loc, CallSyntax::kDirectCall, dest);
  }

  LIT::FuncOp copyCtr = stubs->copyCtr;
  SymbolConstantAttr copyCtrRef = copyCtr.getBoundSymbolRef();
  ASTDecl *copyCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(copyCtrRef.getSymbol());
  LIT::FuncOp moveCtr = stubs->moveCtr;
  SymbolConstantAttr moveCtrRef = moveCtr.getBoundSymbolRef();
  ASTDecl *moveCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(moveCtrRef.getSymbol());

  // Try to create a closure copy constructor if possible.
  if (failed(populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    shared.deleteDecl(*copyCtrDecl);
  else
    declOp.setCopyInitAttr(copyCtrRef);

  // Try to create a closure move constructor if possible.
  if (failed(populateMoveCopy(*moveCtrDecl, true)))
    shared.deleteDecl(*moveCtrDecl);
  else
    declOp.setMoveInitAttr(moveCtrRef);

  if (LIT::FuncOp dtor = stubs->dtor)
    declOp.setDestructorAttr(dtor.getBoundSymbolRef());

  // Populate the body of the call op.
  declOp->setAttr(callMethodAttr, callFunc.getBoundReference());
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
  for (auto [declAndCapture, fieldOp] :
       llvm::zip(captures, llvm::drop_begin(declOp.getFieldDecls(),
                                            hasParamClosureCaptures))) {
    auto [decl, capture] = declAndCapture;
    Value target = builder.create<RefStructGEROp>(selfArg, fieldOp);
    // If the capture is an SValue then it lives in register.
    if (capture.getValue().isSValue())
      target = builder.create<RefLoadOp>(target);

    // If the reference types disagree, the cast to fix the lifetime.
    // FIXME: This isn't great.  We should really /replace/ the original
    // lifetimes with the self lifetime.  For example, when rewriting something
    // like:
    //      fn outer(a: MemType):
    //         fn inner():
    //           use(a)
    // the capture will use 'a' with its own `a lifetime implicitly generated on
    // the outer type.  However, after rewriting it to a struct, we get
    // something like this:
    //      fn closure(self: CaptureStruct):
    //        use(self.a)
    // which now has the lifetime (and mutability) of 'self'.
    Value captureValue = capture.getMlirValue();
    if (captureValue.getType() != target.getType()) {
      auto captureType = cast<RefType>(captureValue.getType());
      auto targetRef = cast<RefType>(target.getType());

      // The lifetime won't be defined in the extracted function, so stub it
      // out.  The mutability may also differ, so we just hack this.
      assert(isa<ParamDeclRefAttr>(captureType.getLifetime()) &&
             "FIXME: Doesn't support complex lifetime captures yet");
      auto expectedLifetime = cast<ParamDeclRefAttr>(captureType.getLifetime());

      builder.create<ParamDeclareOp>(
          ParamDeclAttr::get(expectedLifetime),
          LifetimeMutCastAttr::get(targetRef.getLifetime(),
                                   expectedLifetime.getType()));
      target = builder.create<RebindOp>(captureValue.getType(), target);
    }

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

static SymbolConstantAttr
createTypedSymbol(SymbolConstantAttr symbol,
                  ArrayRef<ParamDeclAttr> parameters) {
  SmallVector<TypedAttr> paramReferences =
      llvm::map_to_vector(parameters, [](ParamDeclAttr attr) -> TypedAttr {
        return ParamDeclRefAttr::get(attr);
      });
  auto paramRefs =
      ParameterExprArrayAttr::get(symbol.getContext(), paramReferences);
  auto [specializedSignature, _] =
      getUnboundSpecializedSignature(symbol.getType(), paramRefs);
  return SymbolConstantAttr::get(symbol.getSymbol(), paramReferences,
                                 specializedSignature);
}

/// Generate the code to allocate heap memory for the given pointer type.
static Value allocateHeapMemory(PointerType ptrType, ImplicitLocOpBuilder &b) {
  TypedAttr elementType = TypeConstantAttr::get(
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

LIT::FuncOp
ClosureEmitter::createWrapperInitWithImpl(StructDeclOp closureWrapper,
                                          StructDeclOp closureImpl, SMLoc loc) {
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
  SmallVector<Type> argTypes{
      wrapperType.getRefForArgument("self", /*isMut=*/true),
      closureImplRefType};

  // Then build all other information needed for the __init__ signature.
  SmallVector<ArgConvention> argConventions{ArgConvention::InitSelf,
                                            ArgConvention::OwnedInMem};
  SmallVector<StringAttr> argNames{selfName, StringAttr::get(ctx, "impl")};
  SmallVector<PassingKind> argPassingKinds(2, PassingKind::PosOnly);
  SmallVector<PassingKind> paramPassingKindsOfInit(initParams.size(),
                                                   PassingKind::PosOnly);
  auto paramListAttrsOfInit = PogListAttr::get(
      ctx, getDemangledNames(initParams), paramPassingKindsOfInit);
  auto argListAttrsOfInit = PogListAttr::get(ctx, argNames, argPassingKinds);
  FuncOp init = addVoidMethod(
      *ASTType(ASTDecl::computeSelfTypeForStruct(closureWrapper))
           .getDecl(shared),
      "__init__", argTypes, argConventions, argListAttrsOfInit,
      SpecialFunctionKind::kInit, initParams, paramListAttrsOfInit);
  init.setInlineLevel(InlineLevel::Always);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(init.getLocScope());

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());

  // Allocate memory on heap and copy argument into allocated memory.
  Value target = allocateHeapMemory(PointerType::get(closureImplType), builder);
  Value source = init.getBody()->getArgument(1);

  // TODO(references): Move closures off pointers to correct lifetimes.
  auto immortal = builder.getAttr<LifetimeAttr>(/*isMut=*/true);
  Value targetRef = builder.create<RefFromPointerOp>(target, immortal,
                                                     /*startUninit=*/true,
                                                     /*endUninit=*/false);

  // Copy the contents of the injected impl into the heap memory.
  ExprEmitter emitter(shared, moduleDecl, builder);
  ValueDest implDest(MLValue(targetRef), EC_Assignment);
  emitter.emitResult(MRValue(source), &node, implDest);

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(0);
  Value erasedType =
      builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
  storeField(builder, self, erasedType, implField);
  auto generateName = [&](StringRef prefix) {
    return (closureWrapper.getSymName() + prefix + closureImpl.getSymName())
        .str();
  };
  TopLevelTypes topLevelTypes = collectTopLevelFunctionTypes(closureWrapper);
  auto setMember = [&](LIT::FuncOp topLevelFunc, StringAttr fieldName,
                       Type fieldType) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    TypedAttr funcSymbol = topLevelFunc.getBoundReference(
        ParameterExprArrayAttr::get(ctx, totalParams));
    if (funcSymbol.getType() != fieldType)
      funcSymbol = ParamOperatorAttr::get(POC::Rebind, funcSymbol, fieldType);
    auto createClosure =
        builder.create<CreateClosureOp>(funcSymbol, ValueRange());
    storeField(builder, init.getArgument(0), createClosure, fieldName);
  };

  // Create the top level copy constructor.
  // The copy constructor takes the Wrapper instance and the impl of the other.
  SmallVector<ParamDeclAttr> topLevelParams;
  for (TypedAttr param : totalParams) {
    auto declRef = cast<ParamDeclRefAttr>(param);
    topLevelParams.push_back(ParamDeclAttr::get(declRef));
  }

  SmallVector<PassingKind> paramPassingKinds(closureImpl.getParams().size(),
                                             PassingKind::PosOnly);
  auto paramListAttrs = PogListAttr::get(ctx, getDemangledNames(topLevelParams),
                                         paramPassingKinds);
  auto argListAttrs =
      PogListAttr::get(ctx, {otherName}, {PassingKind::PosOnly});
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCopyInit = createFunction(
      moduleDecl, generateName("_copyinit_"), topLevelParams, paramListAttrs,
      {opaquePtrType}, {ArgConvention::BorrowedInReg}, argListAttrs,
      opaquePtrType, SpecialFunctionKind::kNormal, loc, builder);

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

    // TODO(references): move closures to references and correct lifetimes.
    auto immortal = builder.getAttr<LifetimeAttr>(/*isMut=*/true);
    Value targetRef = builder.create<RefFromPointerOp>(target, immortal,
                                                       /*startUninit=*/true,
                                                       /*endUninit=*/false);
    Value existingRef = builder.create<RefFromPointerOp>(existingPtr, immortal,
                                                         /*startUninit=*/false,
                                                         /*endUninit=*/false);

    // Copy the existing value into the target.
    ValueDest copyDest(MLValue(targetRef), EC_Assignment);
    ExprEmitter emitter(shared, moduleDecl, builder);
    emitter.emitResult(MBValue(existingRef), &node, copyDest);

    // Return the allocated and populated impl.
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(), body);
    Value erasedType =
        builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
    ExprEmitter::emitNormalReturn(builder, erasedType, topLevelCopyInit);
    builder.create<LIT::EndFuncOp>();
    setMember(topLevelCopyInit, copyFieldAttr, topLevelTypes.copyFuncFieldType);
  }

  // Create top level destructor.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelDtor = createFunction(
      moduleDecl, generateName("_dtor_"), topLevelParams, paramListAttrs,
      opaquePtrType, ArgConvention::BorrowedInReg,
      PogListAttr::get(ctx, {selfName}, {PassingKind::PosOnly}), noneType,
      SpecialFunctionKind::kNormal, loc, builder);

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
    auto immortal = builder.getAttr<LifetimeAttr>(/*isMut=*/true);
    (void)builder.create<RefFromPointerOp>(implPtr, immortal,
                                           /*startUninit=*/false,
                                           /*endUninit=*/true);

    // Free the memory we allocated on the heap to store the closure.
    builder.create<POP::AlignedFreeOp>(implPtr);
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), topLevelDtor);
    builder.create<LIT::EndFuncOp>();
  }

  // Set the member.
  setMember(topLevelDtor, dtorFieldAttr, topLevelTypes.delFuncFieldType);

  // Create the __call__ function.
  assert(closureWrapper.getClosureSignature().has_value() &&
         "The closure signature should have been set at creation time");
  SignatureType functionSignature = *closureWrapper.getClosureSignature();
  LITSignatureType closureSignature = addClosureSelfArgToFunctionSignature(
      opaquePtrType, ArgConvention::BorrowedInReg, functionSignature);
  assert(closureSignature.getResults().size() == 1);
  closureSignature = closureSignature.getSpecializedSignature(
      ArrayRef(topLevelParamRefs).take_front(wrapperParamDecls.size()),
      translateLocation(loc));

  Type resultType = closureSignature.getResults().front();

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCall = createFunction(
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

    // FIXME: Thread a lifetime through correctly.

    // TODO(references): Move closures off pointers.
    auto immortal = builder.getAttr<LifetimeAttr>(/*isMut=*/false);
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

    SymbolConstantAttr typedSymbol = createTypedSymbol(symbol, topLevelParams);

    SmallVector<TypedAttr> implicitLifetimes;
    auto finalSig = cast<SignatureType>(typedSymbol.getType());
    for (auto [arg, conv] : llvm::zip(args, finalSig.getArgConventions()))
      if (SignatureType::hasImplicitLifetime(conv))
        implicitLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());

    Value result =
        builder.create<CallOp>(resultType, typedSymbol, implicitLifetimes, args)
            .getResult(0);
    ExprEmitter::emitNormalReturn(builder, result, topLevelCall);
    builder.create<LIT::EndFuncOp>();
  }
  setMember(topLevelCall, callFieldAttr, topLevelTypes.callFuncFieldType);
  return init;
}

Value Capture::getMlirValue() const {
  if (value.isSValue())
    return value.getSValueRegister();
  if (value.isMValue())
    return value.getMValueReference();
  return {};
}
