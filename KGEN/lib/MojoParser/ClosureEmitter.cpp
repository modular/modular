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
      opaquePtrType(PointerType::get(KGEN::NoneType::get(ctx))) {}

StringAttr ClosureEmitter::getClosureNameFromType(StringRef prefix,
                                                  FileModuleOp fileModuleOp,
                                                  SignatureType signatureType) {
  // Note: Add the trailing "escaping" so that the type alias gets picked up.
  return StringAttr::get(fileModuleOp.getContext(),
                         prefix + fileModuleOp.getSymName() + "_" +
                             ASTType(signatureType).getAsString() + "_wrapper");
}

static void addFieldsToStruct(StructDeclOp structDeclOp, ArrayRef<Type> fields,
                              Location location) {
  OpBuilder b(structDeclOp.getRegion());
  b.setInsertionPointToStart(&structDeclOp.getFields().front());
  for (auto [i, type] : llvm::enumerate(fields))
    b.create<StructFieldOp>(location, "field" + Twine(i), type);
}

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 ArrayRef<Type> fields, Location location,
                                 ArrayRef<ParamDeclAttr> inputParams) {
  OpBuilder b(module.getRegion());
  SmallVector<StringAttr> inputParamNames(inputParams.size(),
                                          StringAttr::get(b.getContext()));
  // TODO: The type may contain decl references that need to be remapped.
  SmallVector<PassingKind> passingKinds(inputParams.size(),
                                        PassingKind::PosOnly);

  StructDeclOp declOp = b.create<StructDeclOp>(location, nameAttr);
  declOp.setIsSynthetic(true);
  addFieldsToStruct(declOp, fields, location);

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
      passingKinds, /*defaults=*/{}, /*paramVarArg=*/false);
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
  StructDeclOp declOp = createStruct(fileModuleOp, name, fieldTypes,
                                     fileModuleOp.getLoc(), wrapperDecls);
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
        /*implicitLifetimes=*/ArrayRef<TypedAttr>(), dtorImpl);
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
    builder.create<CallSignatureOp>(noneType, loadedFuncPtr,
                                    /*implicitLifetimes=*/ArrayRef<TypedAttr>(),
                                    ValueRange{ptrToImpl, loadedExistingImpl});
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
      addClosureSelfArgToFunctionSignature(
          ptrToSelfType, ValueInputConvention::BorrowedInMem, signatureType);
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
    SmallVector<TypedAttr> implicitLifetimes;
    if (hasResultSlot) {
      Value destArg = callMethod.getBody()->getArgument(0);
      arguments.push_back(destArg);
      implicitLifetimes.push_back(
          cast<RefType>(destArg.getType()).getLifetime());
    }

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
        resultType, builder.create<POP::LoadOp>(getCallMember),
        implicitLifetimes, arguments);
    ExprEmitter::emitNormalReturn(builder, callResult.getResult(0), callMethod);
    builder.create<LIT::EndFuncOp>();
  }
  return declOp;
}

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
  SmallVector<Type> paramCaptureListTypes;
  for (ParamDeclRefAttr pc : paramCaptures) {
    // Check for parameter closure captures.
    // TODO - To do this properly, we need to be able to recursively check types
    // for SignatureTypes, eg. they could be inside an option or a list or
    // struct or whatnot at arbitrary depth.  For now, we just do the case of an
    // unwrapped SignatureType.
    if (auto sig = dyn_cast<SignatureType>(pc.getType());
        sig && sig.isCapturing()) {
      auto type = CaptureListType::get(pc.getContext(), pc);
      paramCaptureListTypes.push_back(type);
      fieldTypes.push_back(type);
    }
  }

  // Create the closure impl struct with the field types. Add the capture
  // parameters as parameter decls to the generated struct. This way, parameter
  // references within the body do not have to be renamed.
  StructDeclOp declOp =
      createStruct(fileModuleOp, implName, fieldTypes, fileModuleOp.getLoc(),
                   llvm::map_to_vector(paramCaptures, [](ParamDeclRefAttr ref) {
                     return ParamDeclAttr::get(ref);
                   }));

  // Register the struct and its fields as fully resolved decls.
  ASTDecl &structDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), implName, moduleDecl.getLoc(), &moduleDecl);
  for (StructFieldOp field : declOp.getFieldDecls()) {
    shared.declResolver->addFullyResolvedDecl(&*field, field.getNameAttr(),
                                              structDecl.getLoc(), &structDecl);
  }

  SmallVector<StructFieldOp> paramClosureCaptureFieldDecls;
  SmallVector<StructFieldOp> normalCaptureFieldDecls;
  for (auto [i, fieldDecl] : llvm::enumerate(declOp.getFieldDecls())) {
    if (i < captures.size())
      normalCaptureFieldDecls.push_back(fieldDecl);
    else
      paramClosureCaptureFieldDecls.push_back(fieldDecl);
  }

  // Build the init method. This only needs the captured arguments. Populate the
  // function argument information.
  auto implPtrType =
      PointerType::get(ASTDecl::computeSelfTypeForStruct(declOp));
  // All arguments as positional-only.
  SmallVector<PassingKind> initSigPassingKinds(1 + captures.size(),
                                               PassingKind::PosOnly);
  // Fill the types and conventions based on the register-passabilities.
  SmallVector<StringAttr> initSigNames{selfName};
  SmallVector<Type> initSigTypes{implPtrType};
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
      initSigTypes.push_back(PointerType::get(rvalueType));
    }
  }

  std::optional<GeneratedStubs> stubs = addMissingValueMemberStubsToStruct(
      structDecl, /*generateFieldwiseInit=*/false,
      /*forceGenerateDestructor=*/!paramCaptureListTypes.empty());
  LIT::FuncOp initFunc = synthesizeMemberwiseInit(
      structDecl, initSigTypes, initSigConventions, initSigNames,
      initSigPassingKinds, normalCaptureFieldDecls);
  if (!paramCaptureListTypes.empty()) {
    LITSignatureType oldSig = initFunc.getSignature();
    LITSignatureType capturingInitSymbol = LITSignatureType::get(
        oldSig.getValues(), oldSig.getInputParamTypes(),
        oldSig.getResultParamTypes(), oldSig.getInputConventions(),
        oldSig.getFnEffects().setCapturing(true), oldSig.getMetadata());
    initFunc.setSignature(capturingInitSymbol);
  }
  auto builder =
      ImplicitLocOpBuilder::atBlockBegin(initFunc.getLoc(), initFunc.getBody());
  for (auto [clType, fieldDecl] :
       llvm::zip(paramCaptureListTypes, paramClosureCaptureFieldDecls)) {
    auto selfArg = initFunc.getArgument(0);
    auto captureList = builder.create<CaptureListCreate>(clType);
    Value target = builder.create<StructGEPOp>(selfArg, fieldDecl);
    builder.create<POP::StoreOp>(captureList, target);
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

  if (LIT::FuncOp dtor = stubs->getDestructor()) {
    auto builder =
        ImplicitLocOpBuilder::atBlockBegin(dtor.getLoc(), dtor.getBody());
    auto selfArg = dtor.getArgument(0);
    for (auto fieldDecl : paramClosureCaptureFieldDecls) {
      Value target = builder.create<StructGEPOp>(selfArg, fieldDecl);
      target = builder.create<POP::LoadOp>(target);
      builder.create<CaptureListDestroy>(target);
    }
    declOp.setDestructorAttr(dtor.getBoundSymbolRef());
  }

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
    // wrapperSig.getValueInputs()[0]);
    callConventions.push_back(ValueInputConvention::ByRefResult);
    callNames.push_back(StringAttr::get(ctx));
    callPassingKinds.push_back(PassingKind::PosOnly);
  }

  // Currently Closure Impls are not register passable, so use BorrowedInMem
  // convention.
  callInputTypes.push_back(implPtrType);
  callConventions.push_back(ValueInputConvention::BorrowedInMem);
  callNames.push_back(StringAttr::get(ctx));
  callPassingKinds.push_back(PassingKind::PosOnly);

  llvm::append_range(callInputTypes,
                     wrapperSig.getValueInputs().drop_front(hasByRefReturn));
  llvm::append_range(
      callConventions,
      wrapperSig.getInputConventions().drop_front(hasByRefReturn));
  llvm::append_range(callNames,
                     wrapperSig.getArgNames().drop_front(hasByRefReturn));
  llvm::append_range(
      callPassingKinds,
      wrapperSig.getArgPassingKinds().drop_front(hasByRefReturn));

  Type closureResultType = wrapperSig.getValueResults().front();
  builder = ImplicitLocOpBuilder::atBlockEnd(declOp.getLoc(),
                                             &declOp.getFields().front());
  LIT::FuncOp callFunc = createFunction(
      "__call__", /*inputParameters=*/{}, /*paramPassingKinds=*/{},
      callInputTypes, callConventions, callNames, callPassingKinds,
      closureResultType, SpecialFunctionKind::kNormal, location, builder,
      wrapperSig.getFnEffects().setEscaping(false));
  declOp->setAttr(callMethodAttr, callFunc.getBoundReference());
  // Populate the body of the call op.
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
      hasByRefReturn, implPtrType, callFuncLocation);
  for (auto fieldDecl : paramClosureCaptureFieldDecls) {
    Value target = builder.create<StructGEPOp>(selfArg, fieldDecl);
    target = builder.create<POP::LoadOp>(target);
    builder.create<CaptureListExpand>(target);
  }
  for (auto [declAndCapture, fieldOp] :
       llvm::zip(captures, normalCaptureFieldDecls)) {
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
      ptrType.getElementType(), AnyRegTypeType::get(ptrType.getContext()));
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
  auto closureImplType =
      PointerType::get(makeClosureImplSelfType(closureImpl, totalInputParams));

  SmallVector<PassingKind> paramPassingKinds(
      closureImpl.getInputParams().size(), PassingKind::PosOnly);

  // Create unique names for parameters.
  if (auto init = findInitInStruct(closureWrapper, closureImplType))
    return init;
  Type wrapperType = makeClosureImplSelfType(closureWrapper, wrapperParams);
  SmallVector<Type> argTypes{PointerType::get(wrapperType), closureImplType};

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
    return (closureWrapper.getSymName() + prefix + closureImpl.getSymName())
        .str();
  };
  TopLevelTypes topLevelTypes = collectTopLevelFunctionTypes(closureWrapper);
  auto setMember = [&](LIT::FuncOp topLevelFunc, StringAttr fieldName,
                       Type fieldType) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    auto funcMember = builder.create<StructGEPOp>(
        PointerType::get(fieldType), fieldName, init.getBody()->getArgument(0));
    TypedAttr funcSymbol = topLevelFunc.getBoundReference(
        ParameterExprArrayAttr::get(ctx, totalInputParams));
    if (funcSymbol.getType() != fieldType)
      funcSymbol = ParamOperatorAttr::get(POC::Rebind, funcSymbol, fieldType);
    auto createClosure =
        builder.create<CreateClosureOp>(funcSymbol, ValueRange());
    builder.create<POP::StoreOp>(createClosure, funcMember);
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
  auto closureImplTopLevelType = PointerType::get(
      makeClosureImplSelfType(closureImpl, topLevelInputParamRefs));

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
    Value target = allocateHeapMemory(closureImplTopLevelType, builder);
    Value existingPtr = builder.create<POP::PointerBitcastOp>(
        closureImplTopLevelType, body->getArgument(1));

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
        closureImplTopLevelType, body->getArgument(0));
    //  TODO(references)
    auto destTy =
        RefType::getRefForPointerHACK(closureImplTopLevelType, /*isMut=*/false);
    auto castOp =
        builder.create<mlir::UnrealizedConversionCastOp>(destTy, implPtr)
            .getResult(0);

    builder.create<OwnershipEndLifetimeOp>(castOp, /*isRegister=*/false);

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
        closureImplTopLevelType, closureArg);
    // Call the __call__ on the closure impl.
    assert(closureImpl->hasAttr(callMethodAttr) &&
           "Closure Impls are generated with a __call__ method.");
    SymbolConstantAttr symbol =
        closureImpl->getAttrOfType<SymbolConstantAttr>(callMethodAttr);
    SmallVector<Value> args;
    SmallVector<TypedAttr> implicitLifetimes;
    if (hasMemoryOnlyResult) {
      Value destArg = topLevelCall.getArgument(0);
      args.push_back(destArg);
      implicitLifetimes.push_back(
          cast<RefType>(destArg.getType()).getLifetime());
    }
    args.push_back(implPtr);
    for (unsigned i = hasMemoryOnlyResult + 1,
                  e = closureSignature.getNumInputs();
         i < e; ++i)
      args.push_back(topLevelCall.getArgument(i));
    SymbolConstantAttr typedSymbol =
        createTypedSymbol(symbol, topLevelInputParams);

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
