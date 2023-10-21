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
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))) {}

StringAttr ClosureEmitter::getClosureNameFromType(StringRef prefix,
                                                  FileModuleOp fileModuleOp,
                                                  SignatureType signatureType) {
  // Note: Add the trailing "escaping" so that the type alias gets picked up.
  return StringAttr::get(fileModuleOp.getContext(),
                         prefix + fileModuleOp.getSymName() + "_" +
                             ASTType(signatureType).getAsString() +
                             "_escaping");
}

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 ArrayRef<Type> fields, Location location,
                                 ArrayRef<Type> inputParameters) {
  OpBuilder b(module.getRegion());
  StructDeclOp declOp = b.create<StructDeclOp>(location, nameAttr);
  if (declOp.getFields().empty())
    declOp.getFields().push_back(new Block());
  b.setInsertionPointToStart(&declOp.getFields().front());
  for (auto [i, type] : llvm::enumerate(fields))
    b.create<StructFieldOp>(location, "field" + Twine(i++), type);
  SmallVector<ParamDeclAttr> inputParams;
  SmallVector<StringAttr> inputParamNames;

  // TODO: The type may contain decl references that need to be remapped.
  SmallVector<PassingKind> parameterPassingKinds(inputParameters.size(),
                                                 PassingKind::PosOnly);
  for (auto [index, paramType] : llvm::enumerate(inputParameters)) {
    ParamDeclAttr paramDecl =
        ParamDeclAttr::get("p" + std::to_string(index), paramType);
    inputParams.push_back(paramDecl);
    inputParamNames.push_back(paramDecl.getName());
  }
  declOp.setInputParams(inputParams);
  declOp.setParamNames(inputParamNames);
  declOp.setParamPassingKindsAttr(
      PassingKindArrayAttr::get(module.getContext(), parameterPassingKinds));
  return declOp;
}

LITSignatureType ClosureEmitter::addClosureSelfArgToFunctionSignature(
    Type closureType, LITSignatureType sig) const {
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
  callMemberInputConventions.push_back(ValueInputConvention::BorrowedInMem);
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
    StringAttr name, LITSignatureType signatureType,
    SMLoc nestedFunctionOrTypeLocation) {
  MLIRContext *ctx = fileModuleOp.getContext();
  SmallVector<Type> fieldTypes{opaquePtrType};

  if (!signatureType.getResultParamTypes().empty() ||
      !signatureType.getInputParamTypes().empty()) {
    shared.emitError(
        nestedFunctionOrTypeLocation,
        "declared parameters in escaping closures are not supported yet");
    return {};
  }
  StructDeclOp declOp =
      createStruct(fileModuleOp, name, fieldTypes, fileModuleOp.getLoc(),
                   signatureType.getInputParamTypes());
  declOp.setClosureSignature(signatureType);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  auto dtorMetadata =
      FnMetadataAttr::get(ctx, {selfName}, {PassingKind::PosOnly});
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
                          {PassingKind::PosOnly, PassingKind::PosOnly});
  auto cpySignatureType =
      SignatureType::get(fnType,
                         {ValueInputConvention::BorrowedInReg,
                          ValueInputConvention::BorrowedInMem},
                         /*effects=*/{}, metadata);
  auto copy =
      b.create<StructFieldOp>(declOp.getLoc(), copyFieldAttr, cpySignatureType);

  // Add the call member
  bool hasResultSlot = signatureType.hasMemoryOnlyResult();
  LITSignatureType callMemberSignatureType =
      addClosureSelfArgToFunctionSignature(opaquePtrType, signatureType);
  auto callMember = b.create<StructFieldOp>(declOp.getLoc(), callFieldAttr,
                                            callMemberSignatureType);

  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), declOp.getDeclName(), moduleDecl.getLoc(),
      &moduleDecl);
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
    Value dtorImpl = builder.create<POP::LoadOp>(
        builder.create<StructGEPOp>(dtorSelf, impl));
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
    builder.create<CallSignatureOp>(
        noneType,
        builder.create<POP::LoadOp>(
            builder.create<StructGEPOp>(dtorSelf, dtor)),
        ValueRange({dtorImpl}));
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
    Value existingImpl = builder.create<StructGEPOp>(copyExisting, impl);
    auto loadedExistingImpl = builder.create<POP::LoadOp>(existingImpl);
    auto funcPtrPtr = builder.create<StructGEPOp>(copySelf, copy);
    auto ptrToImpl = builder.create<StructGEPOp>(copySelf, impl);
    auto loadedFuncPtr = builder.create<POP::LoadOp>(funcPtrPtr);
    builder.create<CallSignatureOp>(
        noneType, loadedFuncPtr,
        ArrayRef<Value>{ptrToImpl, loadedExistingImpl});
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
    Value copyExisting = moveCtr.getBody()->getArgument(1);
    auto opaquePointerTypeAttr = M::PointerAttr::get(ctx, 0, opaquePtrType);
    Value nullPtr =
        builder.create<ParamConstantOp>(opaquePtrType, opaquePointerTypeAttr);
    builder.create<POP::StoreOp>(
        nullPtr, builder.create<StructGEPOp>(copyExisting, impl));
  }
  if (failed(populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
    return {};

  // Add the __call__ Method.
  Type selfType = ASTDecl::computeSelfTypeForStruct(declOp);
  auto ptrToSelfType = PointerType::get(selfType);
  LITSignatureType closureMethodSignatureType =
      addClosureSelfArgToFunctionSignature(ptrToSelfType, signatureType);
  auto [callMethod, callDecl] = synthesizeMethodInStruct(
      "__call__", /*inputParameters=*/{}, /*paramPassingKinds=*/{},
      closureMethodSignatureType.getValueInputs(),
      closureMethodSignatureType.getInputConventions(),
      callMemberSignatureType.getArgNames(),
      callMemberSignatureType.getArgPassingKinds(),
      signatureType.getValueResults().front(), astDecl,
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
    if (hasResultSlot)
      arguments.push_back(callMethod.getBody()->getArgument(0));

    arguments.push_back(builder.create<POP::LoadOp>(
        builder.create<StructGEPOp>(callSelf, impl)));

    for (unsigned i = 1 + hasResultSlot, e = callMethod.getNumArguments();
         i < e; i++)
      arguments.push_back(callMethod.getBody()->getArgument(i));

    assert(callMemberSignatureType.getValueResults().size() == 1);
    auto getCallMember =
        builder.create<StructGEPOp>(PointerType::get(callMemberSignatureType),
                                    callMember.getNameAttr(), callSelf);
    auto callResult = builder.create<CallSignatureOp>(
        callMemberSignatureType.getValueResults().front(),
        builder.create<POP::LoadOp>(getCallMember), arguments);
    ExprEmitter::emitNormalReturn(builder, callResult.getResult(0), callMethod);
    builder.create<LIT::EndFuncOp>();
  }
  return declOp;
}

StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    SMLoc location, ASTDecl &nestedFunctionDecl,
    CaptureTraversableMap capturedParams, ClosureCache &cache) {
  MLIRContext *ctx = shared.getContext();
  FuncOp nestedFunction = dyn_cast<LIT::FuncOp>(nestedFunctionDecl);
  assert(nestedFunction && "a function must back the nestedFunctionDecl");
  FuncOp parentFunction = nestedFunction->getParentOfType<FuncOp>();
  assert(parentFunction && "a nested function must have a function parent");

  LITSignatureType closureWrapperSignature = nestedFunction.getSignature();
  if (!closureWrapperSignature.getResultParamTypes().empty() ||
      !closureWrapperSignature.getInputParamTypes().empty()) {
    shared.emitError(
        fileModuleOp.getLoc(),
        "declared parameters in escaping closures are not supported yet");
    return {};
  }
  size_t wrapperNumArgs = closureWrapperSignature.getNumInputs();

  auto captureRange = shared.getCaptureRangeInScope(nestedFunctionDecl);
  unsigned captureCount = llvm::size(captureRange);

  SmallVector<Type> closureImplSigTypes;
  closureImplSigTypes.reserve(captureCount + wrapperNumArgs);
  SmallVector<ValueInputConvention> closureImplSigConventions;
  closureImplSigConventions.reserve(captureCount + wrapperNumArgs);
  SmallVector<StringAttr> closureImplSigArgNames;
  closureImplSigArgNames.reserve(captureCount + wrapperNumArgs);
  SmallVector<PassingKind> closureImplSigArgPassingKinds;
  closureImplSigArgPassingKinds.reserve(captureCount + wrapperNumArgs);

  SmallVector<Type> fieldTypes;
  fieldTypes.reserve(captureCount);
  // TODO: Enable expression of how to capture.
  for (const auto &[i, declCaptureIter] : llvm::enumerate(captureRange)) {
    Capture capture = declCaptureIter.second;
    bool move = capture.isMoveCapture();
    ASTType rvalueType = capture.getValue().getRValueType();

    if (ASTType(rvalueType).isRegisterPassable(location, shared)) {
      closureImplSigConventions.push_back(
          move ? ValueInputConvention::OwnedInReg
               : ValueInputConvention::BorrowedInReg);
      closureImplSigTypes.push_back(rvalueType);
    } else {
      closureImplSigConventions.push_back(
          move ? ValueInputConvention::OwnedInMem
               : ValueInputConvention::BorrowedInMem);
      closureImplSigTypes.push_back(PointerType::get(rvalueType));
    }
    fieldTypes.push_back(rvalueType);

    closureImplSigArgNames.push_back(StringAttr::get(ctx, "fld" + Twine(i)));
  }
  closureImplSigArgPassingKinds.append(captureCount, PassingKind::PosOnly);

  // Create the closure impl signature from the captures and the wrapper
  // signature.
  llvm::append_range(closureImplSigTypes,
                     closureWrapperSignature.getValueInputs());
  llvm::append_range(closureImplSigConventions,
                     closureWrapperSignature.getInputConventions());
  llvm::append_range(closureImplSigArgNames,
                     closureWrapperSignature.getArgNames());
  llvm::append_range(closureImplSigArgPassingKinds,
                     closureWrapperSignature.getArgPassingKinds());

  // Captured parameters should be a part of the closure's key. Incorporate it
  // into the signature.
  SmallVector<Type> closureImplInputParams(
      closureWrapperSignature.getInputParamTypes().getValue());
  SmallVector<StringAttr> closureImplInputParamNames(
      closureWrapperSignature.getParamNames());
  SmallVector<PassingKind> closureImplInputParamPassingKinds(
      closureWrapperSignature.getParamPassingKinds());
  if (!closureWrapperSignature.getResultParamTypes().empty()) {
    shared.emitError(location,
                     "Result parameters not supported in closures yet.");
    return {};
  }

  for (auto [capturedParamName, capturedParamType] : capturedParams) {
    closureImplInputParams.push_back(capturedParamType);
    closureImplInputParamNames.push_back(capturedParamName);
    closureImplInputParamPassingKinds.push_back(PassingKind::PosOnly);
  }

  auto metadata = FnMetadataAttr::get(
      ctx, closureImplSigArgNames, closureImplSigArgPassingKinds,
      closureImplInputParamNames, closureImplInputParamPassingKinds,
      /*defaultArguments=*/{},
      /*defaultParameters=*/{});
  auto fnType = FunctionType::get(ctx, closureImplSigTypes,
                                  closureWrapperSignature.getValueResults());
  auto closureImplSignature = LITSignatureType::get(
      fnType, TypeArrayAttr::get(ctx, closureImplInputParams),
      closureWrapperSignature.getResultParamTypes(), closureImplSigConventions,
      closureWrapperSignature.getFnEffects(), metadata);

  std::pair<LITSignatureType, StringAttr> key(
      closureImplSignature, fileModuleOp.getSymNameAttrName());
  if (auto existing = cache.getExisting(key)) {
    nestedFunction->erase();
    return existing;
  }

  StringAttr name =
      getClosureNameFromType("_CI_", fileModuleOp, closureImplSignature);

  // Create map from the parent name to the index of the parameter in the
  // closure struct.
  DenseMap<StringAttr, unsigned> parentRefToClosureImplParamIndex;
  if (closureWrapperSignature.getNumInputParams() != 0)
    shared.emitError(
        location,
        "Add parameters of nested function to parent function and capture "
        "them. Parameters declared in nested functions are not supported yet");

  for (auto [index, capturedParam] : llvm::enumerate(capturedParams))
    parentRefToClosureImplParamIndex[capturedParam.first] = index;

  StructDeclOp declOp =
      createStruct(fileModuleOp, name, fieldTypes, fileModuleOp.getLoc(),
                   closureImplSignature.getInputParamTypes());
  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), declOp.getDeclName(), moduleDecl.getLoc(),
      &moduleDecl);

  for (StructFieldOp field : declOp.getFieldDecls())
    shared.declResolver->addFullyResolvedDecl(
        field.getOperation(), field.getNameAttr(), astDecl.getLoc(), &astDecl);

  // Build the init method. This only needs the captured arguments, so we drop
  // the args from the wrapper.
  auto ptrToClosureImplType =
      PointerType::get(ASTDecl::computeSelfTypeForStruct(declOp));
  SmallVector<Type> initSigTypes{ptrToClosureImplType};
  llvm::append_range(initSigTypes,
                     llvm::drop_end(closureImplSigTypes, wrapperNumArgs));

  SmallVector<ValueInputConvention> initSigConventions{
      ValueInputConvention::InitSelf};
  llvm::append_range(initSigConventions,
                     llvm::drop_end(closureImplSigConventions, wrapperNumArgs));

  SmallVector<StringAttr> initSigNames{selfName};
  llvm::append_range(initSigNames,
                     llvm::drop_end(closureImplSigArgNames, wrapperNumArgs));

  SmallVector<PassingKind> initSigPassingKinds{PassingKind::PosOnly};
  llvm::append_range(
      initSigPassingKinds,
      llvm::drop_end(closureImplSigArgPassingKinds, wrapperNumArgs));

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      astDecl, /*generateFieldwiseInit=*/false);
  synthesizeMemberwiseInit(astDecl, initSigTypes, initSigConventions,
                           initSigNames, initSigPassingKinds);

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

  // Generate the __call__ method.

  // Build the call signature from the closure signature. This means inserting
  // the self argument in the correct location.
  unsigned callArgCount = closureWrapperSignature.getNumInputs() + 1;
  SmallVector<Type> callInputTypes;
  callInputTypes.reserve(callArgCount);
  SmallVector<ValueInputConvention> callConventions;
  callConventions.reserve(callArgCount);
  SmallVector<StringAttr> callNames;
  callNames.reserve(callArgCount);
  SmallVector<PassingKind> callPassingKinds;
  callPassingKinds.reserve(callArgCount);

  // Move by ref result argument to front before self argument.
  bool hasByRefReturn = closureWrapperSignature.hasMemoryOnlyResult();
  if (hasByRefReturn) {
    callInputTypes.push_back(closureWrapperSignature.getValueInputs()[0]);
    callConventions.push_back(closureWrapperSignature.getInputConvention(0));
    callNames.push_back(StringAttr::get(ctx));
    callPassingKinds.push_back(PassingKind::PosOnly);
  }

  // Currently Closure Impls are not register passable, so use BorrowInMem
  // convention.
  callInputTypes.push_back(ptrToClosureImplType);
  callConventions.push_back(ValueInputConvention::BorrowedInMem);
  callNames.push_back(StringAttr::get(ctx));
  callPassingKinds.push_back(PassingKind::PosOnly);

  llvm::append_range(
      callInputTypes,
      closureWrapperSignature.getValueInputs().drop_front(hasByRefReturn));
  llvm::append_range(
      callConventions,
      closureWrapperSignature.getInputConventions().drop_front(hasByRefReturn));
  llvm::append_range(
      callNames,
      closureWrapperSignature.getArgNames().drop_front(hasByRefReturn));
  llvm::append_range(
      callPassingKinds,
      closureWrapperSignature.getArgPassingKinds().drop_front(hasByRefReturn));

  assert(closureImplSignature.getValueResults().size() == 1 &&
         "Multiple outputs are not supported.");
  Type closureResultType = closureImplSignature.getValueResults().front();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(declOp.getLoc(),
                                                  &declOp.getFields().front());
  LIT::FuncOp callFunc = createFunction(
      "__call__", /*inputParameters=*/{}, /*paramPassingKinds=*/{},
      callInputTypes, callConventions, callNames, callPassingKinds,
      closureResultType, SpecialFunctionKind::kNormal, location, builder,
      closureImplSignature.getFnEffects().setEscaping(false));
  declOp->setAttr(callMethodAttr, callFunc.getBoundReference());
  // Populate the body of the call op.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = callFunc.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Take the body of the nested function.
  IRMapping mapping;
  callFunc.getBody()->erase();
  callFunc.getBodyRegion().takeBody(nestedFunction.getBodyRegion());
  Location callFuncLocation = callFunc.getLoc();
  DebugInfo::DISubprogramAttr subprogramAttrOfCallFunc;

  if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
          callFuncLocation)) {
    subprogramAttrOfCallFunc = fusedLoc.getMetadata();
    DebugInfo::DISubprogramAttr subprogramAttrOfOriginalFunc;
    if (auto fusedLocOriginal =
            dyn_cast<mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(
                nestedFunction.getLoc()))
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
  // Replace parent parameter references with references to the closure impl's
  // parameters.
  mlir::AttrTypeReplacer capturedParamReplacer;
  auto replaceParamDeclRef = [&parentRefToClosureImplParamIndex,
                              &declOp](ParamDeclRefAttr paramRef) {
    if (parentRefToClosureImplParamIndex.contains(paramRef.getName())) {
      return ParamDeclRefAttr::get(
          declOp.getInputParams()
              [parentRefToClosureImplParamIndex[paramRef.getName()]]);
    }
    return paramRef;
  };
  capturedParamReplacer.addReplacement(replaceParamDeclRef);
  void recursivelyReplaceElementsIn(Operation * op, bool replaceAttrs = true,
                                    bool replaceLocs = false,
                                    bool replaceTypes = false);
  capturedParamReplacer.recursivelyReplaceElementsIn(
      callFunc, /*replaceAttrs=*/true, /*replaceLocs=*/false,
      /*replaceTypes=*/true);
  builder =
      ImplicitLocOpBuilder::atBlockBegin(callFunc.getLoc(), callFunc.getBody());
  Value selfArg = callFunc.getBodyRegion().insertArgument(
      hasByRefReturn, ptrToClosureImplType, callFuncLocation);
  for (auto [declAndCapture, fieldOp] :
       llvm::zip(captureRange, declOp.getFieldDecls())) {
    auto [decl, capture] = declAndCapture;
    Value target = builder.create<StructGEPOp>(selfArg, fieldOp);
    // If the rvalue type matches the real type, then it lives in register.
    if (capture.getValue().getRValueType().isEqualCanon(
            capture.getValue().getType()))
      target = builder.create<POP::LoadOp>(target);

    Value captureValue = capture.getMlirValue();
    // FIXME(references): properly capture things as references instead of
    // capturing by raw pointer.
    if (isa<RefType>(captureValue.getType())) {
      target = builder
                   .create<mlir::UnrealizedConversionCastOp>(
                       captureValue.getType(), target)
                   .getResult(0);
    }

    replaceAllUsesInRegionWith(captureValue, target, callFunc.getBodyRegion());
  }
  cache.storeClosure(key, declOp);
  nestedFunction->erase();
  return declOp;
}

Type ClosureEmitter::makeClosureImplSelfType(
    StructDeclOp closureImpl, SmallVector<ParamDeclAttr> paramValues) {
  Type closureImplType = ASTDecl::computeSelfTypeForStruct(closureImpl);
  if (DeclRefType declRef = dyn_cast<DeclRefType>(closureImplType)) {
    ASTDecl &typeDecl =
        shared.declResolver->getDeclForTypeSymbol(declRef.getSymbol());
    SmallVector<ParamBindAttr> bindingValues;
    for (auto [i, param] : llvm::enumerate(paramValues)) {
      TypedAttr typedAttr =
          ParamDeclRefAttr::get(param.getName(), param.getType());
      bindingValues.push_back(ParamBindAttr::get(
          closureImpl.getInputParams()[i].getName(), typedAttr));
    }
    closureImplType = DeclRefType::get(
        typeDecl.getSymbolRef(),
        ParamBindArrayAttr::get(shared.getContext(), bindingValues));
  }
  return closureImplType;
}

static SymbolConstantAttr
createTypedSymbol(SymbolConstantAttr symbol,
                  ArrayRef<ParamDeclAttr> parameters) {
  SmallVector<TypedAttr> paramReferences(parameters.size());
  for (auto [i, paramDecl] : llvm::enumerate(parameters))
    paramReferences[i] = ParamDeclRefAttr::get(paramDecl);
  auto paramRefs =
      ParameterExprArrayAttr::get(symbol.getContext(), paramReferences);
  auto [specializedSignature, paramExpArray] =
      getUnboundSpecializedSignature(symbol.getType(), paramRefs);
  return SymbolConstantAttr::get(
      symbol.getSymbol(),
      ParameterExprArrayAttr::get(symbol.getContext(), paramReferences),
      specializedSignature);
}

/// Generate the code to allocate heap memory for the given pointer type.
static Value allocateHeapMemory(PointerType ptrType, ImplicitLocOpBuilder &b) {
  TypedAttr elementType = TypeConstantAttr::get(ptrType.getElementAsType());
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value sizeOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetSizeOf, {elementType, target}));
  Value alignOf = b.create<ParamConstantOp>(
      ParamOperatorAttr::get(POC::GetAlignOf, {elementType, target}));
  return b.create<POP::AlignedAllocOp>(ptrType, ValueRange{alignOf, sizeOf});
}

LIT::FuncOp ClosureEmitter::createWrapperInitWithImpl(
    StructDeclOp closureWrapper, StructDeclOp closureImpl, SMLoc location) {
  // The __init__ will take self and the impl. We first build the types.
  SmallVector<ParamDeclAttr> inputParams;
  auto [line, col] = shared.getSourceMgr().getLineAndColumn(location);
  std::string prefix = mangleParameter("", line, col);
  for (ParamDeclAttr p : closureImpl.getInputParams()) {
    inputParams.push_back(ParamDeclAttr::get(
        StringAttr::get(shared.getContext(), prefix + p.getName().str()),
        p.getType()));
  }
  auto closureImplType =
      PointerType::get(makeClosureImplSelfType(closureImpl, inputParams));

  SmallVector<PassingKind> paramPassingKinds(
      closureImpl.getInputParams().size(), PassingKind::PosOnly);

  // Create unique names for parameters.
  if (auto init = findInitInStruct(closureWrapper, closureImplType))
    return init;
  Type wrapperType = ASTDecl::computeSelfTypeForStruct(closureWrapper);
  SmallVector<Type> argTypes{PointerType::get(wrapperType), closureImplType};

  // Then build all other information needed for the __init__ signature.
  SmallVector<ValueInputConvention> argConventions{
      ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem};
  SmallVector<StringAttr> argNames{selfName, StringAttr::get(ctx, "impl")};
  SmallVector<PassingKind> argPassingKinds(2, PassingKind::PosOnly);
  FuncOp init = addVoidMethod(
      *ASTType(ASTDecl::computeSelfTypeForStruct(closureWrapper))
           .getDecl(shared),
      "__init__", inputParams, paramPassingKinds, argTypes, argConventions,
      argNames, argPassingKinds, SpecialFunctionKind::kInit);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = init.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());

  // Allocate memory on heap and copy argument into allocated memory.
  Value target = allocateHeapMemory(closureImplType, builder);
  Value source = init.getBody()->getArgument(1);

  // Copy the contents of the injected impl into the heap memory.
  ExprEmitter emitter(shared, moduleDecl, builder);
  ValueDest implDest(MLValue(target), EC_Assignment);
  emitter.emitResult(MRValue(source), &node, implDest);

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(0);
  Value ptrToImpl = builder.create<LIT::StructGEPOp>(
      PointerType::get(opaquePtrType), implField.getNameAttr(), self);
  Value erasedType =
      builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
  builder.create<POP::StoreOp>(erasedType, ptrToImpl);
  auto generateName = [&](StringRef prefix) {
    return (closureWrapper.getDeclName().str() + prefix +
            closureImpl.getDeclName().str())
        .str();
  };
  SmallVector<TypedAttr> bindings;
  for (ParamDeclAttr initParam : inputParams)
    bindings.push_back(
        ParamDeclRefAttr::get(initParam.getName(), initParam.getType()));
  auto parameterExprArrayAttr =
      ParameterExprArrayAttr::get(shared.getContext(), bindings);
  Type topLevelCallType;
  Type topLevelCopyType;
  Type topLevelDelType;
  for (StructFieldOp fieldOp : closureWrapper.getFieldDecls()) {
    StringAttr name = fieldOp.getNameAttr();
    if (name == callFieldAttr)
      topLevelCallType = fieldOp.getType();
    else if (name == copyFieldAttr)
      topLevelCopyType = fieldOp.getType();
    else if (name == dtorFieldAttr)
      topLevelDelType = fieldOp.getType();
  }
  assert(topLevelCallType && "All closure wrapper initializers must have a top "
                             "level call function associated with them");
  assert(topLevelDelType && "All closure wrapper initializers must have a top "
                            "level delete function associated with them");
  assert(topLevelCopyType && "All closure wrapper initializers must have a top "
                             "level copy function associated with them");
  auto setMember = [&](LIT::FuncOp topLevelFunc, StringAttr fieldName,
                       Type fieldType) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    auto funcMember = builder.create<StructGEPOp>(
        PointerType::get(fieldType), fieldName, init.getBody()->getArgument(0));
    TypedAttr funcSymbol =
        topLevelFunc.getBoundReference(ParameterExprArrayAttr::get(
            shared.getContext(), parameterExprArrayAttr));
    if (funcSymbol.getType() != fieldType)
      funcSymbol = ParamOperatorAttr::get(POC::Rebind, funcSymbol, fieldType);
    auto createClosure =
        builder.create<CreateClosureOp>(funcSymbol, ValueRange());
    builder.create<POP::StoreOp>(createClosure, funcMember);
  };

  // Create the top level copy constructor.
  // The copy constructor takes the Wrapper instance and the impl of the other.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCopyInit = createFunction(
      generateName("_copyinit_"), inputParams, paramPassingKinds,
      {PointerType::get(opaquePtrType), opaquePtrType},
      {ValueInputConvention::BorrowedInReg,
       ValueInputConvention::BorrowedInMem},
      {ptrToImplName, otherName}, {PassingKind::PosOnly, PassingKind::PosOnly},
      noneType, SpecialFunctionKind::kNormal, location, builder);
  // Populate copy init body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(),
                                               topLevelCopyInit.getBody());
    Block *body = topLevelCopyInit.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelCopyInit.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Allocate memory on heap and call copy constructor
    Value target = allocateHeapMemory(closureImplType, builder);
    Value existingPtr = builder.create<POP::PointerBitcastOp>(
        closureImplType, body->getArgument(1));

    ValueDest copyDest(MLValue(target), EC_Assignment);
    ExprEmitter emitter(shared, moduleDecl, builder);
    emitter.emitResult(MBValue(existingPtr), &node, copyDest);

    // Store the allocated and populated impl into the closure wrapper.
    Value ptrToImpl = topLevelCopyInit.getBody()->getArgument(0);
    Value erasedType =
        builder.create<POP::PointerBitcastOp>(opaquePtrType, target);
    builder.create<POP::StoreOp>(erasedType, ptrToImpl);

    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), topLevelCopyInit);
    builder.create<LIT::EndFuncOp>();
    setMember(topLevelCopyInit, copyFieldAttr, topLevelCopyType);
  }

  // Create top level destructor.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelDtor = createFunction(
      generateName("_dtor_"), inputParams, paramPassingKinds, opaquePtrType,
      ValueInputConvention::OwnedInReg, selfName, PassingKind::PosOnly,
      noneType, SpecialFunctionKind::kNormal, location, builder);

  // Populate destructor body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(),
                                               topLevelDtor.getBody());
    Block *body = topLevelDtor.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelDtor.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Cast the opaque pointer back to the closure impl type.
    Value implPtr = builder.create<POP::PointerBitcastOp>(closureImplType,
                                                          body->getArgument(0));
    builder.create<OwnershipEndLifetimeOp>(builder.getLoc(), implPtr,
                                           /*isRegister=*/false);

    // Free the memory we allocated on the heap to store the closure.
    builder.create<POP::AlignedFreeOp>(implPtr);
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), topLevelDtor);
    builder.create<LIT::EndFuncOp>();
  }

  // Set the member.
  setMember(topLevelDtor, dtorFieldAttr, topLevelDelType);

  // Create the __call__ function.
  assert(closureWrapper.getClosureSignature().has_value() &&
         "The closure signature should have been set at creation time");
  SignatureType functionSignature = *closureWrapper.getClosureSignature();
  LITSignatureType closureSignature =
      addClosureSelfArgToFunctionSignature(opaquePtrType, functionSignature);
  assert(closureSignature.getValueResults().size() == 1);

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCall = createFunction(
      generateName("_call_"), inputParams, paramPassingKinds,
      closureSignature.getValueInputs(), closureSignature.getInputConventions(),
      closureSignature.getArgNames(), closureSignature.getArgPassingKinds(),
      closureSignature.getValueResults().front(), SpecialFunctionKind::kNormal,
      location, builder, closureSignature.getFnEffects());

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
    Value implPtr =
        builder.create<POP::PointerBitcastOp>(closureImplType, closureArg);
    // Call the __call__ on the closure impl.
    assert(closureImpl->hasAttr(callMethodAttr) &&
           "Closure Impls are generated with a __call__ method.");
    SymbolConstantAttr symbol =
        closureImpl->getAttrOfType<SymbolConstantAttr>(callMethodAttr);
    SmallVector<Value> args;
    if (hasMemoryOnlyResult)
      args.push_back(topLevelCall.getArgument(0));
    args.push_back(implPtr);
    for (unsigned i = hasMemoryOnlyResult + 1,
                  e = closureSignature.getNumInputs();
         i < e; ++i)
      args.push_back(topLevelCall.getArgument(i));
    SymbolConstantAttr typedSymbol = createTypedSymbol(symbol, inputParams);
    Value result =
        builder
            .create<CallOp>(
                typedSymbol.getType().getValueResults(), typedSymbol,
                /*resultParamTypes=*/ParamDeclArrayAttr::get(ctx, {}), args)
            .getResult(0);
    ExprEmitter::emitNormalReturn(builder, result, topLevelDtor);
    builder.create<LIT::EndFuncOp>();
  }
  setMember(topLevelCall, callFieldAttr, topLevelCallType);
  return init;
}

Value Capture::getMlirValue() const {
  if (auto v = value.getIfMLValue())
    return v;
  if (auto v = value.getIfXLValue())
    return v;
  if (auto v = value.getIfXBValue())
    return v;
  if (auto v = value.getIfMBValue())
    return v;
  if (auto v = value.getIfSBValue())
    return v;
  if (auto v = value.getIfXRValue())
    return v;
  if (auto v = value.getIfMRValue())
    return v;
  if (auto v = value.getIfSRValue())
    return v;

  return {};
}
