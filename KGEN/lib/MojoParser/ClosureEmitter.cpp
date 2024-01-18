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

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
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
      ptrToImplName(StringAttr::get(ctx, "ptrToImpl")),
      dtorFieldAttr(StringAttr::get(ctx, "dtor")),
      copyFieldAttr(StringAttr::get(ctx, "copy")),
      callFieldAttr(StringAttr::get(ctx, "call")),
      callMethodAttr(StringAttr::get(ctx, "closureCallMethod")),
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))),
      opaqueRefType(RefType::getImmortal(true, KGEN::NoneType::get(ctx))) {}

StringAttr ClosureEmitter::getClosureNameFromType(StringRef prefix,
                                                  FileModuleOp fileModuleOp,
                                                  SignatureType signatureType) {
  // Note: Add the trailing "escaping" so that the type alias gets picked up.
  return StringAttr::get(fileModuleOp.getContext(),
                         prefix + fileModuleOp.getSymName() + "_" +
                             ASTType(signatureType).getAsString() + "_wrapper");
}

static void addFieldsToStruct(StructDeclOp structOp, ArrayRef<Type> fields) {
  OpBuilder b(structOp.getRegion());
  b.setInsertionPointToStart(&structOp.getFields().front());
  for (auto [i, type] : llvm::enumerate(fields))
    b.create<StructFieldOp>(structOp.getLoc(), "field" + Twine(i), type);
}

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 ArrayRef<ParamDeclAttr> inputParams) {
  OpBuilder b(module.getRegion());
  SmallVector<StringAttr> inputParamNames(inputParams.size(),
                                          StringAttr::get(b.getContext()));
  // TODO: The type may contain decl references that need to be remapped.
  SmallVector<PassingKind> passingKinds(inputParams.size(),
                                        PassingKind::PosOnly);

  StructDeclOp declOp = b.create<StructDeclOp>(module.getLoc(), nameAttr);
  declOp.setIsSynthetic(true);

  // Set attributes in bulk.
  NamedAttrList attrs = declOp->getAttrDictionary();
  attrs.set(declOp.getInputParamsAttrName(),
            b.getAttr<ParamDeclArrayAttr>(inputParams));
  SmallVector<Type> inputParamTypes = llvm::map_to_vector(
      inputParams, [](ParamDeclAttr decl) { return decl.getType(); });
  auto sig = TypeSignatureType::remapToSignature(
      [&]() -> InFlightDiagnostic {
        llvm_unreachable("unexpected invalid signature");
      },
      ParamDeclArrayAttr::get(b.getContext(), inputParams), inputParamNames,
      passingKinds, /*defaultPosParams=*/{}, /*defaultKwOnlyParams=*/{},
      /*paramVarArg=*/false);
  attrs.set(declOp.getSignatureAttrName(), TypeAttr::get(sig));
  declOp->setAttrs(attrs.getDictionary(module.getContext()));
  return declOp;
}

/// Given a signature of a function, create a new signature by inserting a
/// closure argument at index 0 or 1 (depending on the result type) with the
/// given convention.
static LITSignatureType addClosureSelfArgToFunctionSignature(
    Type closureType, ValueInputConvention convention, LITSignatureType sig) {
  MLIRContext *ctx = sig.getContext();

  unsigned callArgCount = sig.getNumInputs() + 1;
  SmallVector<Type> callMemberSignatureInputs;
  callMemberSignatureInputs.reserve(callArgCount);
  SmallVector<ValueInputConvention> callMemberInputConventions;
  callMemberInputConventions.reserve(callArgCount);
  SmallVector<StringAttr> callMemberArgNames;
  callMemberArgNames.reserve(callArgCount);
  SmallVector<PassingKind> callMemberArgPassingKinds;
  callMemberArgPassingKinds.reserve(callArgCount);

  // Add result slot if necessary.
  bool hasResultSlot = sig.hasMemoryOnlyResult();
  if (hasResultSlot) {
    callMemberSignatureInputs.push_back(sig.getValueInputs()[0]);
    callMemberInputConventions.push_back(ValueInputConvention::ByRefResult);
    callMemberArgNames.push_back(StringAttr::get(ctx));
    callMemberArgPassingKinds.push_back(PassingKind::PosOnly);
  }
  // Add self.
  callMemberSignatureInputs.push_back(closureType);
  callMemberInputConventions.push_back(convention);
  callMemberArgNames.push_back(StringAttr::get(ctx));
  callMemberArgPassingKinds.push_back(PassingKind::PosOnly);
  // Add the rest of the arguments.
  llvm::append_range(callMemberSignatureInputs,
                     sig.getValueInputs().drop_front(hasResultSlot));
  llvm::append_range(callMemberInputConventions,
                     sig.getInputConventions().drop_front(hasResultSlot));
  llvm::append_range(callMemberArgNames,
                     sig.getArgNames().drop_front(hasResultSlot));
  llvm::append_range(callMemberArgPassingKinds,
                     sig.getArgPassingKinds().drop_front(hasResultSlot));

  // A closure signature is not escaping because its 'escaping' state is
  // captured in the self argument we are inserting in this function.

  assert(callMemberArgNames.size() == callMemberInputConventions.size());
  FnMetadataAttr metadata = sig.getMetadata().cloneWith(
      callMemberArgNames, callMemberArgPassingKinds);
  return SignatureType::get(
      FunctionType::get(ctx, callMemberSignatureInputs, sig.getValueResults()),
      sig.getInputParamTypes(), sig.getResultParamTypes(),
      callMemberInputConventions, sig.getFnEffects().setEscaping(false),
      metadata);
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
  ParameterEvaluator evaluator;
  SmallVector<TypedAttr> paramValues;
  for (auto [i, type] :
       llvm::enumerate(dependentSignatureType.getInputParamTypes())) {
    wrapperDecls.push_back(
        ParamDeclAttr::get(StringAttr::get(getContext(), "p" + Twine(i)),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(wrapperDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }
  StructDeclOp declOp = createStruct(fileModuleOp, name, wrapperDecls);
  addFieldsToStruct(declOp, opaquePtrType);
  declOp.setClosureSignature(dependentSignatureType);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  auto dtorMetadata = FnMetadataAttr::get(
      ctx, {selfName}, {PassingKind::PosOnly}, /*numImplicitLifetimeDecls=*/0);
  auto dtorSig = SignatureType::get(b.getFunctionType(opaquePtrType, noneType),
                                    ValueInputConvention::OwnedInReg,
                                    /*effects=*/{}, dtorMetadata);
  auto dtor = b.create<StructFieldOp>(declOp.getLoc(), dtorFieldAttr, dtorSig);

  // Create Copy Member.
  auto fnType = b.getType<FunctionType>(
      ArrayRef<Type>({PointerType::get(opaquePtrType), opaquePtrType}),
      noneType);
  auto metadata =
      FnMetadataAttr::get(ctx, {ptrToImplName, otherName},
                          {PassingKind::PosOnly, PassingKind::PosOnly},
                          /*numImplicitLifetimeDecls=*/0);
  auto cpySignatureType =
      SignatureType::get(fnType,
                         {ValueInputConvention::BorrowedInReg,
                          ValueInputConvention::BorrowedInReg},
                         /*effects=*/{}, metadata);
  auto copy =
      b.create<StructFieldOp>(declOp.getLoc(), copyFieldAttr, cpySignatureType);

  dependentSignatureType = dependentSignatureType.getSpecializedSignature(
      paramValues, translateLocation(nestedFunctionOrTypeLocation));
  auto sigMetadata =
      FnMetadataAttr::get(ctx, dependentSignatureType.getArgNames(),
                          dependentSignatureType.getArgPassingKinds(),
                          dependentSignatureType.getNumImplicitLifetimeDecls());
  Type resultType = dependentSignatureType.getValueResults().front();
  FunctionType functionType =
      b.getFunctionType(dependentSignatureType.getValueInputs(), resultType);
  LITSignatureType signatureType = SignatureType::get(
      functionType, {}, {}, dependentSignatureType.getInputConventions(),
      dependentSignatureType.getFnEffects(), sigMetadata);

  // Add the call member
  bool hasResultSlot = dependentSignatureType.hasMemoryOnlyResult();
  LITSignatureType callMemberSignatureType =
      addClosureSelfArgToFunctionSignature(
          opaquePtrType, ValueInputConvention::BorrowedInReg, signatureType);
  auto callMember = b.create<StructFieldOp>(declOp.getLoc(), callFieldAttr,
                                            callMemberSignatureType);

  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      &*declOp, name, moduleDecl.getLoc(), &moduleDecl);
  for (StructFieldOp field : declOp.getFieldDecls())
    shared.declResolver->addFullyResolvedDecl(
        field.getOperation(), field.getNameAttr(), astDecl.getLoc(), &astDecl);

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      astDecl, /*generateFieldwiseInit=*/false,
      /*forceGenerateDestructor=*/true);
  assert(stubs && "expected the stubs on a purely synthetic class to succeed.");
  LIT::FuncOp destructor = stubs->getDestructor();
  declOp.setDestructorAttr(destructor.getBoundSymbolRef());

  LIT::FuncOp copyCtr = stubs->getCopyConstructor();
  SymbolConstantAttr copyCtrRef = copyCtr.getBoundSymbolRef();
  declOp.setCopyInitAttr(copyCtrRef);
  ASTDecl *copyCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(copyCtrRef.getSymbol());

  LIT::FuncOp moveCtr = stubs->getMoveConstructor();
  SymbolConstantAttr moveCtrRef = moveCtr.getBoundSymbolRef();
  declOp.setMoveInitAttr(moveCtrRef);
  ASTDecl *moveCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(moveCtrRef.getSymbol());

  // Populate destructor.
  {
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        destructor.getLoc(), destructor.getBody());
    Value dtorSelf = destructor.getBody()->getArgument(0);
    // Return early if the impl is null.
    Value dtorImpl = builder.create<RefLoadOp>(
        builder.create<RefStructGEROp>(dtorSelf, impl));
    Type scalarIndex = POP::SIMDType::get(
        1,
        DTypeConstantAttr::get(ctx, KGENDType(KGENDType::ExtraCases::index)));
    Value zero = builder.create<ParamConstantOp>(builder.getIndexAttr(0));
    Value dtorImplAsIndex = builder.create<POP::CastToBuiltinOp>(
        builder.getIndexType(),
        builder.create<POP::PointerToIndexOp>(scalarIndex, dtorImpl));
    Value isEqualToZero = builder.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::EQ, dtorImplAsIndex, zero);
    auto ifOp = builder.create<HLCF::IfOp>(isEqualToZero);
    auto insertionPoint = builder.saveInsertionPoint();

    // If false, the impl is not null. Continue to destruction.
    if (ifOp.getElseRegion().empty())
      ifOp.getElseRegion().push_back(new Block);
    builder =
        ImplicitLocOpBuilder::atBlockEnd(ifOp.getLoc(), &ifOp.getElseBlock());
    builder.create<HLCF::YieldOp>();

    // If true, the impl is null and no destruction is needed.
    if (ifOp.getThenRegion().empty())
      ifOp.getThenRegion().push_back(new Block);
    builder =
        ImplicitLocOpBuilder::atBlockEnd(ifOp.getLoc(), &ifOp.getThenBlock());
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), destructor);
    builder.create<HLCF::YieldOp>();
    builder.restoreInsertionPoint(insertionPoint);
    Value callee = builder.create<RefLoadOp>(
        builder.create<RefStructGEROp>(dtorSelf, dtor));
    builder.create<CallSignatureOp>(noneType, callee,
                                    /*implicitLifetimes=*/ArrayRef<TypedAttr>(),
                                    dtorImpl);
  }

  // Populate the copy constructor.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = copyCtr.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
    Location translatedLocation =
        shared.translateLocation(copyCtrDecl->getLoc());
    // we want to insert before return at end of function. LIT::ReturnOp is not
    // a terminator though, so let's find it and set it.
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        translatedLocation, copyCtr.getBody());
    auto returnOps = copyCtr.getBody()->getOps<LIT::ReturnOp>();
    assert(std::distance(returnOps.begin(), returnOps.end()) == 1 &&
           "copy should have exactly one return op.");
    builder.setInsertionPoint(*returnOps.begin());
    Value copySelf = copyCtr.getBody()->getArgument(0);
    Value copyExisting = copyCtr.getBody()->getArgument(1);
    Value existingImpl = builder.create<RefStructGEROp>(copyExisting, impl);
    auto loadedExistingImpl = builder.create<RefLoadOp>(existingImpl);
    auto funcPtrRef = builder.create<RefStructGEROp>(copySelf, copy);
    Value refToImpl = builder.create<RefStructGEROp>(copySelf, impl);
    auto loadedFuncPtr = builder.create<RefLoadOp>(funcPtrRef);
    refToImpl = builder.create<RefToPointerOp>(refToImpl);
    builder.create<CallSignatureOp>(noneType, loadedFuncPtr,
                                    /*implicitLifetimes=*/ArrayRef<TypedAttr>(),
                                    ValueRange{refToImpl, loadedExistingImpl});
  }
  if (failed(populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    return {};

  // Populate move constructor.
  {
    // Take the impl from the existing.
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = moveCtr.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
    Location translatedLocation =
        shared.translateLocation(moveCtrDecl->getLoc());
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        translatedLocation, moveCtr.getBody());
    Value moveExisting = moveCtr.getBody()->getArgument(1);
    auto opaquePointerTypeAttr = M::PointerAttr::get(ctx, 0, opaquePtrType);
    Value nullPtr =
        builder.create<ParamConstantOp>(opaquePtrType, opaquePointerTypeAttr);
    builder.create<RefStoreOp>(
        nullPtr, builder.create<RefStructGEROp>(moveExisting, impl));
  }
  if (failed(populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
    return {};

  // Add the __call__ Method.
  ASTType selfType = ASTDecl::computeSelfTypeForStruct(declOp);
  auto refToSelfType = selfType.getRefForArgument("self", /*isMut=*/false);
  LITSignatureType closureMethodSignatureType =
      addClosureSelfArgToFunctionSignature(
          refToSelfType, ValueInputConvention::BorrowedInMem, signatureType);
  auto [callMethod, callDecl] = synthesizeMethodInStruct(
      "__call__", closureMethodSignatureType.getValueInputs(),
      closureMethodSignatureType.getInputConventions(),
      closureMethodSignatureType.getArgNames(),
      closureMethodSignatureType.getArgPassingKinds(), resultType, astDecl,
      SpecialFunctionKind::kNormal, closureMethodSignatureType.getFnEffects());

  // Populate the body of ClosureWrapper::__call__.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = callMethod.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
    Location translatedLocation = shared.translateLocation(callDecl.getLoc());
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        translatedLocation, callMethod.getBody());
    Value callSelf = hasResultSlot ? callMethod.getBody()->getArgument(1)
                                   : callMethod.getBody()->getArgument(0);
    SmallVector<Value> arguments;
    if (hasResultSlot) {
      Value destArg = callMethod.getBody()->getArgument(0);
      arguments.push_back(destArg);
    }

    arguments.push_back(builder.create<RefLoadOp>(
        builder.create<RefStructGEROp>(callSelf, impl)));

    for (unsigned i = 1 + hasResultSlot, e = callMethod.getNumArguments();
         i < e; i++)
      arguments.push_back(callMethod.getBody()->getArgument(i));

    auto getCallMember = builder.create<RefStructGEROp>(callSelf, callMember);
    auto callMemberPtr = builder.create<RefLoadOp>(getCallMember);

    SmallVector<TypedAttr> implicitLifetimes;
    auto calleeSig = cast<SignatureType>(callMemberPtr.getType());
    for (auto [arg, conv] :
         llvm::zip(arguments, calleeSig.getInputConventions()))
      if (SignatureType::hasAddress(conv))
        implicitLifetimes.push_back(cast<RefType>(arg.getType()).getLifetime());

    auto callResult = builder.create<CallSignatureOp>(
        resultType, callMemberPtr, implicitLifetimes, arguments);
    ExprEmitter::emitNormalReturn(builder, callResult.getResult(0), callMethod);
    builder.create<LIT::EndFuncOp>();
  }
  return declOp;
}

/// Generate a Closure Implementation Struct, a struct that contains the
/// capture list.
StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    SMLoc location, ASTDecl &nestedFnDecl,
    ArrayRef<ParamDeclRefAttr> paramCaptures, LITSignatureType wrapperSig) {
  // FIXME: Add another counter for closures?
  auto implName = moduleDecl.getAnonymousLifetimeFor(
      "_CI_" + fileModuleOp.getSymName() + "_escaping");

  // Create map from the parent name to the index of the parameter in the
  // closure struct.
  FuncOp nestedFn = cast<LIT::FuncOp>(nestedFnDecl);
  wrapperSig = nestedFn.getSignature();
  if (wrapperSig.getNumInputParams()) {
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
  StructDeclOp declOp =
      createStruct(fileModuleOp, implName,
                   llvm::map_to_vector(paramCaptures, [](ParamDeclRefAttr ref) {
                     return ParamDeclAttr::get(ref);
                   }));
  // Register the struct as a fully resolved decl.
  ASTDecl &structDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), implName, moduleDecl.getLoc(), &moduleDecl);

  // Generate the __call__ method.

  // Build the call signature from the closure signature. This means inserting
  // the self argument in the correct location.
  unsigned callArgCount = wrapperSig.getNumInputs() + 1;
  SmallVector<Type> callInputTypes;
  callInputTypes.reserve(callArgCount);
  SmallVector<ValueInputConvention> callConventions;
  callConventions.reserve(callArgCount);
  SmallVector<StringAttr> callNames;
  callNames.reserve(callArgCount);
  SmallVector<PassingKind> callPassingKinds;
  callPassingKinds.reserve(callArgCount);

  // Move by ref result argument to front before self argument.
  bool hasByRefReturn = wrapperSig.hasMemoryOnlyResult();
  if (hasByRefReturn) {
    assert(wrapperSig.getInputConvention(0) ==
           ValueInputConvention::ByRefResult);
    callInputTypes.push_back(nestedFn.getFunctionType().getInput(0));
    callConventions.push_back(ValueInputConvention::ByRefResult);
    callNames.push_back(StringAttr::get(ctx));
    callPassingKinds.push_back(PassingKind::PosOnly);
  }

  // Currently Closure Impls are not register passable, so use BorrowedInMem
  // convention.
  auto structSelfType = ASTDecl::computeSelfTypeForStruct(declOp);
  RefType implRefType =
      ASTType(structSelfType).getRefForArgument("self", /*isMut=*/false);
  callInputTypes.push_back(implRefType);
  callConventions.push_back(ValueInputConvention::BorrowedInMem);
  callNames.push_back(StringAttr::get(ctx));
  callPassingKinds.push_back(PassingKind::PosOnly);

  llvm::append_range(
      callInputTypes,
      nestedFn.getFunctionType().getInputs().drop_front(hasByRefReturn));
  llvm::append_range(
      callConventions,
      wrapperSig.getInputConventions().drop_front(hasByRefReturn));
  llvm::append_range(callNames,
                     wrapperSig.getArgNames().drop_front(hasByRefReturn));
  llvm::append_range(
      callPassingKinds,
      wrapperSig.getArgPassingKinds().drop_front(hasByRefReturn));

  Type closureResultType = wrapperSig.getValueResults().front();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(declOp.getLoc(),
                                                  &declOp.getFields().front());
  auto [callFunc, _] = synthesizeMethodInStruct(
      "__call__", callInputTypes, callConventions, callNames, callPassingKinds,
      closureResultType, structDecl, SpecialFunctionKind::kNormal,
      wrapperSig.getFnEffects().setEscaping(false));

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
  SmallVector<Type> initSigTypes{implRefType.getWithMutability(true)};
  SmallVector<ValueInputConvention> initSigConventions{
      ValueInputConvention::InitSelf};
  unsigned fieldNameIdx = 0;
  for (auto &[decl, capture] : captures) {
    bool move = capture.isMoveCapture();
    ASTType rvalueType = capture.getValue().getRValueType();
    initSigNames.push_back(StringAttr::get(ctx, "fld" + Twine(fieldNameIdx++)));
    if (rvalueType.isRegisterPassable(decl->getLoc(), shared)) {
      initSigConventions.push_back(move ? ValueInputConvention::OwnedInReg
                                        : ValueInputConvention::BorrowedInReg);
      initSigTypes.push_back(rvalueType);
    } else {
      initSigConventions.push_back(move ? ValueInputConvention::OwnedInMem
                                        : ValueInputConvention::BorrowedInMem);
      initSigTypes.push_back(rvalueType.getRefForArgument(
          initSigNames.back().str(), /*isMut=*/move));
    }
  }

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      structDecl, /*generateFieldwiseInit=*/false,
      /*forceGenerateDestructor=*/hasParamClosureCaptures);
  LIT::FuncOp initFunc =
      synthesizeMemberwiseInit(structDecl, initSigTypes, initSigConventions,
                               initSigNames, initSigPassingKinds);
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
    if (DebugInfo::DIScopeAttr spAttr = initFunc.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
    emitter.emitConstructorCall(clType, {}, &loc, CallSyntax::kDirectCall,
                                dest);
  }

  LIT::FuncOp copyCtr = stubs->getCopyConstructor();
  SymbolConstantAttr copyCtrRef = copyCtr.getBoundSymbolRef();
  ASTDecl *copyCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(copyCtrRef.getSymbol());
  LIT::FuncOp moveCtr = stubs->getMoveConstructor();
  SymbolConstantAttr moveCtrRef = moveCtr.getBoundSymbolRef();
  ASTDecl *moveCtrDecl =
      shared.declResolver->getDeclForFuncSymbol(moveCtrRef.getSymbol());

  // Try to create a closure copy constructor if possible.
  if (failed(populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
    copyCtr.erase();
  else
    declOp.setCopyInitAttr(copyCtrRef);

  // Try to create a closure move constructor if possible.
  if (failed(populateMoveCopy(*moveCtrDecl, true)))
    moveCtr.erase();
  else
    declOp.setMoveInitAttr(moveCtrRef);

  if (LIT::FuncOp dtor = stubs->getDestructor())
    declOp.setDestructorAttr(dtor.getBoundSymbolRef());

  // Populate the body of the call op.
  declOp->setAttr(callMethodAttr, callFunc.getBoundReference());
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = callFunc.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

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
      hasByRefReturn, callFunc.getFunctionType().getInput(hasByRefReturn),
      callFuncLocation);

  if (paramField) {
    // Emit the `kgen.capture_list.expand` into the call if required.
    Value target = builder.create<RefStructGEROp>(selfArg, paramField);
    emitter.builder = builder;
    ValueDest dest(EC_Assignment);
    emitter.emitNamedMethodCall("expand", {{{MBValue(target), &loc}}}, dest,
                                CallSyntax::kMethodCall, &loc);
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
    // which now has the lifetime of 'self'.
    Value captureValue = capture.getMlirValue();
    if (captureValue.getType() != target.getType()) {
      auto captureType = cast<RefType>(captureValue.getType());
      auto targetRef = cast<RefType>(target.getType());

      // The lifetime won't be defined in the extracted function, so stub it
      // out.  The mutability may also differ.
      assert(isa<ParamDeclRefAttr>(captureType.getLifetime()) &&
             "FIXME: Doesn't support complex lifetime captures yet");
      auto expectedLifetime = cast<ParamDeclRefAttr>(captureType.getLifetime());

      builder.create<ParamDeclareOp>(ParamDeclAttr::get(expectedLifetime),
                                     targetRef.getLifetime());
      target = builder.create<RebindOp>(captureValue.getType(), target);
    }

    assert(captureValue.getType() == target.getType() &&
           "Capture body rewrite problem");
    replaceAllUsesInRegionWith(captureValue, target, callFunc.getBodyRegion());
  }
  nestedFn->erase();
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

/// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
/// instance. The 'fromImplToWrapperParameterIndexMap' allows the caller to
/// specify which parameters of the ClosureWrapper should be bound to the
/// ClosureImpl.
LIT::FuncOp ClosureEmitter::createWrapperInitWithImpl(
    StructDeclOp closureWrapper, StructDeclOp closureImpl,
    SmallDenseMap<unsigned, unsigned> fromImplToWrapperParameterIndexMap,
    SMLoc loc) {
  // The __init__ will take self and the impl. We first build the types. Add the
  // parameter references captured only in the body to the signature of the
  // constructor. Pass the ones captured in the signature from the wrapper to
  // the impl type.
  SmallVector<TypedAttr> totalInputParams;
  SmallVector<TypedAttr> wrapperParams;
  SmallVector<ParamDeclAttr> initParams;
  // We know from the walk order that the first N impl parameters are the
  // wrapper parameters.
  ArrayRef<ParamDeclAttr> wrapperParamDecls = closureWrapper.getInputParams();
  for (ParamDeclAttr param : wrapperParamDecls) {
    auto ref = ParamDeclRefAttr::get(param);
    totalInputParams.push_back(ref);
    wrapperParams.push_back(ref);
  }
  for (ParamDeclAttr param :
       closureImpl.getInputParams().drop_front(wrapperParamDecls.size())) {
    totalInputParams.push_back(ParamDeclRefAttr::get(param));
    initParams.push_back(param);
  }

  // Bind the impl struct to the declared parameters.
  ASTType closureImplType =
      makeClosureImplSelfType(closureImpl, totalInputParams);
  auto closureImplRefType =
      closureImplType.getRefForArgument("existing", /*mut=*/true);

  SmallVector<PassingKind> paramPassingKinds(
      closureImpl.getInputParams().size(), PassingKind::PosOnly);

  // Create unique names for parameters.
  if (auto init = findInitInStruct(closureWrapper, closureImplRefType))
    return init;
  ASTType wrapperType = makeClosureImplSelfType(closureWrapper, wrapperParams);
  SmallVector<Type> argTypes{
      wrapperType.getRefForArgument("self", /*mut=*/true), closureImplRefType};

  // Then build all other information needed for the __init__ signature.
  SmallVector<ValueInputConvention> argConventions{
      ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem};
  SmallVector<StringAttr> argNames{selfName, StringAttr::get(ctx, "impl")};
  SmallVector<PassingKind> argPassingKinds(2, PassingKind::PosOnly);
  SmallVector<PassingKind> paramPassingKindsOfInit(initParams.size(),
                                                   PassingKind::PosOnly);
  FuncOp init = addVoidMethod(
      *ASTType(ASTDecl::computeSelfTypeForStruct(closureWrapper))
           .getDecl(shared),
      "__init__", argTypes, argConventions, argNames, argPassingKinds,
      SpecialFunctionKind::kInit, initParams, paramPassingKindsOfInit);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = init.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());

  // Allocate memory on heap and copy argument into allocated memory.
  Value target = allocateHeapMemory(PointerType::get(closureImplType), builder);
  Value source = init.getBody()->getArgument(1);

  // TODO(references): Move closures off pointers to correct lifetimes.
  auto immortal = builder.getAttr<LifetimeAttr>();
  Value targetRef =
      builder.create<RefFromPointerOp>(/*isMut=*/true, target, immortal,
                                       /*startUninit=*/true,
                                       /*endUninit=*/false);

  // Copy the contents of the injected impl into the heap memory.
  ExprEmitter emitter(shared, moduleDecl, builder);
  ValueDest implDest(MLValue(targetRef), EC_Assignment);
  emitter.emitResult(MRValue(source), &node, implDest);

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(0);
  Value refToImpl = builder.create<RefStructGEROp>(self, implField);
  Value erasedType =
      builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
  builder.create<RefStoreOp>(erasedType, refToImpl);
  auto generateName = [&](StringRef prefix) {
    return (closureWrapper.getSymName() + prefix + closureImpl.getSymName())
        .str();
  };
  TopLevelTypes topLevelTypes = collectTopLevelFunctionTypes(closureWrapper);
  auto setMember = [&](LIT::FuncOp topLevelFunc, StringAttr fieldName,
                       Type fieldType) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    auto selfVal = init.getBody()->getArgument(0);
    auto funcMember = builder.create<RefStructGEROp>(
        cast<RefType>(selfVal.getType()).getWithElement(fieldType), fieldName,
        selfVal);
    TypedAttr funcSymbol = topLevelFunc.getBoundReference(
        ParameterExprArrayAttr::get(ctx, totalInputParams));
    if (funcSymbol.getType() != fieldType)
      funcSymbol = ParamOperatorAttr::get(POC::Rebind, funcSymbol, fieldType);
    auto createClosure =
        builder.create<CreateClosureOp>(funcSymbol, ValueRange());
    builder.create<RefStoreOp>(createClosure, funcMember);
  };

  // Create the top level copy constructor.
  // The copy constructor takes the Wrapper instance and the impl of the other.
  SmallVector<ParamDeclAttr> topLevelInputParams;
  for (TypedAttr param : totalInputParams) {
    auto declRef = cast<ParamDeclRefAttr>(param);
    topLevelInputParams.push_back(ParamDeclAttr::get(declRef));
  }

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCopyInit = createFunction(
      generateName("_copyinit_"), topLevelInputParams, paramPassingKinds,
      {PointerType::get(opaquePtrType), opaquePtrType},
      {ValueInputConvention::BorrowedInReg,
       ValueInputConvention::BorrowedInReg},
      {ptrToImplName, otherName}, {PassingKind::PosOnly, PassingKind::PosOnly},
      noneType, SpecialFunctionKind::kNormal, loc, builder);

  SmallVector<TypedAttr> topLevelInputParamRefs;
  for (auto [i, p] : llvm::enumerate(totalInputParams))
    topLevelInputParamRefs.push_back(
        ParamDeclRefAttr::get(topLevelCopyInit.getInputParams()[i]));
  auto closureImplTopLevelType =
      makeClosureImplSelfType(closureImpl, topLevelInputParamRefs);
  auto closureImplTopLevelPtrType = PointerType::get(closureImplTopLevelType);

  // Populate copy init body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(),
                                               topLevelCopyInit.getBody());
    SmallVector<PassingKind> paramPassingKinds(topLevelInputParams.size(),
                                               PassingKind::PosOnly);
    Block *body = topLevelCopyInit.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelCopyInit.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Allocate memory on heap and call copy constructor
    Value target = allocateHeapMemory(closureImplTopLevelPtrType, builder);
    Value existingPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, body->getArgument(1));

    // TODO(references): move closures to references and correct lifetimes.
    auto immortal = builder.getAttr<LifetimeAttr>();
    Value targetRef =
        builder.create<RefFromPointerOp>(/*isMut=*/true, target, immortal,
                                         /*startUninit=*/true,
                                         /*endUninit=*/false);
    Value existingRef =
        builder.create<RefFromPointerOp>(/*isMut=*/true, existingPtr, immortal,
                                         /*startUninit=*/false,
                                         /*endUninit=*/false);

    // Copy the existing value into the target.
    ValueDest copyDest(MLValue(targetRef), EC_Assignment);
    ExprEmitter emitter(shared, moduleDecl, builder);
    emitter.emitResult(MBValue(existingRef), &node, copyDest);

    // Store the allocated and populated impl into the closure wrapper.
    Value ptrToImpl = topLevelCopyInit.getBody()->getArgument(0);
    Value erasedType =
        builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
    builder.create<POP::StoreOp>(erasedType, ptrToImpl);

    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), topLevelCopyInit);
    builder.create<LIT::EndFuncOp>();
    setMember(topLevelCopyInit, copyFieldAttr, topLevelTypes.copyFuncFieldType);
  }

  // Create top level destructor.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelDtor = createFunction(
      generateName("_dtor_"), topLevelInputParams, paramPassingKinds,
      opaquePtrType, ValueInputConvention::OwnedInReg, selfName,
      PassingKind::PosOnly, noneType, SpecialFunctionKind::kNormal, loc,
      builder);

  // Populate destructor body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(),
                                               topLevelDtor.getBody());
    Block *body = topLevelDtor.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelDtor.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Cast the opaque pointer back to the closure impl type.
    Value implPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, body->getArgument(0));

    // TODO(references): Move closures off pointers.
    // This takes ownership of the pointer, telling checklifetimes that the
    // value should be destroyed by the exit of the function.  ASAP destruction
    // will make sure it is immediately destroyed because there are no uses.
    auto immortal = builder.getAttr<LifetimeAttr>();
    (void)builder.create<RefFromPointerOp>(/*isMut=*/true, implPtr, immortal,
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
      opaquePtrType, ValueInputConvention::BorrowedInReg, functionSignature);
  assert(closureSignature.getValueResults().size() == 1);
  closureSignature = closureSignature.getSpecializedSignature(
      ArrayRef(topLevelInputParamRefs).take_front(wrapperParamDecls.size()),
      translateLocation(loc));

  Type resultType = closureSignature.getValueResults().front();

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCall = createFunction(
      generateName("_call_"), topLevelInputParams, paramPassingKinds,
      closureSignature.getValueInputs(), closureSignature.getInputConventions(),
      closureSignature.getArgNames(), closureSignature.getArgPassingKinds(),
      resultType, SpecialFunctionKind::kNormal, loc, builder,
      closureSignature.getFnEffects());

  // Populate the __call__ body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCall.getLoc(),
                                               topLevelCall.getBody());
    Block *body = topLevelCall.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelCall.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Cast the opaque pointer back to the closure impl type.
    bool hasMemoryOnlyResult = closureSignature.hasMemoryOnlyResult();
    Value closureArg = body->getArgument(hasMemoryOnlyResult);
    Value implPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelPtrType, closureArg);

    // FIXME: Thread a lifetime through correctly.

    // TODO(references): Move closures off pointers.
    auto immortal = builder.getAttr<LifetimeAttr>();
    Value implRef =
        builder.create<RefFromPointerOp>(/*isMut=*/false, implPtr, immortal,
                                         /*startUninit=*/false,
                                         /*endUninit=*/false);

    // Call the __call__ on the closure impl.
    assert(closureImpl->hasAttr(callMethodAttr) &&
           "Closure Impls are generated with a __call__ method.");
    SymbolConstantAttr symbol =
        closureImpl->getAttrOfType<SymbolConstantAttr>(callMethodAttr);
    SmallVector<Value> args;
    if (hasMemoryOnlyResult)
      args.push_back(topLevelCall.getArgument(0));
    args.push_back(implRef);
    for (unsigned i = hasMemoryOnlyResult + 1 /*implPtr*/,
                  e = closureSignature.getNumInputs();
         i < e; ++i)
      args.push_back(topLevelCall.getArgument(i));

    SymbolConstantAttr typedSymbol =
        createTypedSymbol(symbol, topLevelInputParams);

    SmallVector<TypedAttr> implicitLifetimes;
    auto finalSig = cast<SignatureType>(typedSymbol.getType());
    for (auto [arg, conv] : llvm::zip(args, finalSig.getInputConventions()))
      if (SignatureType::hasAddress(conv))
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
