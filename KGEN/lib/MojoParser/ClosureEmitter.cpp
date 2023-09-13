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
#include "ASTDecl.h"
#include "ExprEmitter.h"
#include "IRValues.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

StringAttr ClosureEmitter::getClosureNameFromType(StringRef prefix,
                                                  FileModuleOp fileModuleOp,
                                                  SignatureType signatureType) {
  std::string base(prefix);
  llvm::raw_string_ostream stream(base);
  stream << fileModuleOp.getSymName() << "_";
  stream << DeclResolver::getMangledName(
      StringAttr::get(fileModuleOp.getContext(), ""), signatureType);
  for (FnEffects effect : {FnEffects::Throws, FnEffects::Async})
    if (bitEnumContainsAny(signatureType.getFnEffects(), effect))
      stream << effect;
  return StringAttr::get(signatureType.getContext(), stream.str());
}

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 ArrayRef<Type> fields, Location location) {
  OpBuilder b(module.getRegion());
  StructDeclOp declOp = b.create<StructDeclOp>(location, nameAttr);
  if (declOp.getFields().empty())
    declOp.getFields().push_back(new Block());
  b.setInsertionPointToStart(&declOp.getFields().front());
  unsigned i = 0;
  for (Type type : fields)
    b.create<StructFieldOp>(
        location,
        StringAttr::get(b.getContext(), "field" + std::to_string(i++)), type,
        nullptr);
  return declOp;
}

/// Given a signature of a function, create a new signature by inserting a
/// closure argument at index 0 or 1 depending on the result type.
static SignatureType
addClosureSelfArgToFunctionSignature(Type closureType,
                                     SignatureType functionType) {
  unsigned callArgCount = functionType.getNumInputs() + 1;
  SmallVector<Type> callMemberSignatureInputs;
  callMemberSignatureInputs.reserve(callArgCount);
  SmallVector<ValueInputConvention> callMemberInputConventions;
  callMemberInputConventions.reserve(callArgCount);
  SmallVector<StringAttr> callMemberArgNames;
  callMemberArgNames.reserve(callArgCount);
  // Add result slot if necessary.
  bool hasResultSlot = functionType.hasMemoryOnlyResult();
  MLIRContext *ctx = functionType.getContext();
  if (hasResultSlot) {
    callMemberSignatureInputs.push_back(functionType.getValueInputs()[0]);
    callMemberInputConventions.push_back(ValueInputConvention::ByRefResult);
    callMemberArgNames.push_back(StringAttr::get(ctx, "__result__"));
  }
  // Add self.
  callMemberSignatureInputs.push_back(closureType);
  callMemberInputConventions.push_back(ValueInputConvention::BorrowedInMem);
  callMemberArgNames.push_back(StringAttr::get(ctx, "self"));
  // Add the rest of the arguments.
  for (unsigned j = hasResultSlot, e = functionType.getNumInputs(); j < e;
       j++) {
    callMemberSignatureInputs.push_back(functionType.getValueInputs()[j]);
    callMemberInputConventions.push_back(functionType.getInputConvention(j));
    callMemberArgNames.push_back(functionType.getArgName(j));
  }
  // A closure signature is not escaping because its 'escaping' state is
  // captured in the self argument we are inserting in this function.

  assert(callMemberArgNames.size() == callMemberInputConventions.size());
  auto metadata = FnMetadataAttr::get(
      functionType.getContext(), StringArrayAttr::get(ctx, callMemberArgNames),
      callMemberInputConventions, {},
      bitEnumClear(functionType.getFnEffects(), FnEffects::Escaping));
  return SignatureType::get(
      functionType.getInputParamTypes(), functionType.getResultParamTypes(),
      FunctionType::get(functionType.getContext(), callMemberSignatureInputs,
                        functionType.getValueResults()),
      metadata);
}

StructDeclOp
ClosureEmitter::createClosureWrapperStructDecl(StringAttr name,
                                               SignatureType signatureType) {
  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = PointerType::get(emptyList);
  SmallVector<Type> fieldTypes;
  fieldTypes.push_back(opaquePointer);
  StructDeclOp declOp =
      createStruct(fileModuleOp, name, fieldTypes, fileModuleOp.getLoc());
  TypeAttr signatureAttr = TypeAttr::get(signatureType);
  declOp.setClosureSignatureAttr(signatureAttr);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  auto dtorMetadata = b.getAttr<FnMetadataAttr>(
      b.getAttr<StringArrayAttr>(b.getStringAttr("self")),
      ArrayRef<ValueInputConvention>{ValueInputConvention::OwnedInReg},
      ArrayRef<TypedAttr>{}, FnEffects());
  auto dtorSig = SignatureType::get(TypeArrayAttr(), {},
                                    b.getFunctionType(opaquePointer, noneType),
                                    dtorMetadata);
  auto dtor =
      b.create<StructFieldOp>(declOp.getLoc(), dtorFieldAttr, dtorSig, nullptr);
  SmallVector<Type> callInputTypes;
  callInputTypes.push_back(opaquePointer);
  llvm::append_range(callInputTypes, signatureType.getValueInputs());
  Type selfType = ASTDecl::computeSelfTypeForStruct(declOp);
  auto ptrToSelfType = PointerType::get(selfType);

  // Create Copy Member.
  Type opaquePtrType = PointerType::get(
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1)));
  SignatureType cpySignatureType = SignatureType::get(
      {}, {},
      b.getType<FunctionType>(
          ArrayRef<Type>({PointerType::get(opaquePtrType), opaquePtrType}),
          noneType),
      FnMetadataAttr::get(
          b.getContext(),
          b.getAttr<StringArrayAttr>(SmallVector<StringAttr>{
              b.getStringAttr("ptrToImpl"), b.getStringAttr("other")}),
          {ValueInputConvention::BorrowedInReg,
           ValueInputConvention::BorrowedInMem},
          ArrayRef<TypedAttr>(), FnEffects()));
  auto copy = b.create<StructFieldOp>(declOp.getLoc(), copyFieldAttr,
                                      cpySignatureType, nullptr);

  // Add the call member
  bool hasResultSlot = signatureType.hasMemoryOnlyResult();
  SignatureType callMemberSignatureType =
      addClosureSelfArgToFunctionSignature(opaquePointer, signatureType);
  auto callMember = b.create<StructFieldOp>(declOp.getLoc(), callFieldAttr,
                                            callMemberSignatureType, nullptr);

  ASTDecl &parent = shared.declResolver->getDeclForTypeSymbol(
      SymbolRefAttr::get(fileModuleOp.getDeclName()));
  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), declOp.getDeclName(), parent.getLoc(), &parent);
  for (StructFieldOp field : declOp.getFieldDecls())
    shared.declResolver->addFullyResolvedDecl(
        field.getOperation(), field.getNameAttr(), astDecl.getLoc(), &astDecl);

  GeneratedStubs stubs = structEmitter.addMissingValueMemberStubsToStruct(
      declOp, parent.getLoc(), astDecl, /*generateFieldwiseInit*/ false,
      /*forceGenerateDestructor*/ true);
  assert(stubs && "expected the stubs on a purely synthetic class to succeed.");
  LIT::FuncOp destructor = stubs.getDestructor();
  LIT::FuncOp copyCtr = stubs.getCopyConstrucotr();
  LIT::FuncOp moveCtr = stubs.getMoveConstructor();

  // Populate methods.
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
      destructor.getLoc(), destructor.getBody());
  Value dtorSelf = destructor.getBody()->getArgument(0);
  // Return early if the impl is null.
  Value dtorImpl =
      builder.create<POP::LoadOp>(builder.create<StructGEPOp>(dtorSelf, impl));
  Type scalarIndex = POP::SIMDType::get(
      1, DTypeConstantAttr::get(builder.getContext(),
                                KGENDType(KGENDType::ExtraCases::index)));
  Value zero = builder.create<ParamConstantOp>(
      IntegerAttr::get(IndexType::get(builder.getContext()), 0));
  Value dtorImplAsIndex = builder.create<POP::CastToBuiltinOp>(
      IndexType::get(builder.getContext()),
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
      builder,
      builder.create<ParamConstantOp>(builder.getAttr<LIT::NoneAttr>()),
      destructor);
  builder.create<HLCF::YieldOp>();

  builder.restoreInsertionPoint(insertionPoint);
  builder.create<CallSignatureOp>(
      noneType,
      builder.create<POP::LoadOp>(builder.create<StructGEPOp>(dtorSelf, dtor)),
      ValueRange({dtorImpl}));

  // Populate the copy constructor.
  {
    llvm::SaveAndRestore saveAndRestore(builder);
    builder.setLoc(copyCtr.getLoc());
    builder.setInsertionPoint(copyCtr.getBody(), copyCtr.getBody()->begin());
    Value copySelf = copyCtr.getBody()->getArgument(0);
    Value copyExisting = copyCtr.getBody()->getArgument(1);
    Value existingImpl = builder.create<StructGEPOp>(copyExisting, impl);
    auto loadedExistingImpl = builder.create<POP::LoadOp>(existingImpl);
    auto funcPtrPtr = builder.create<StructGEPOp>(copySelf, copy);
    auto ptrToImpl = builder.create<StructGEPOp>(copySelf, impl);
    auto loadedFuncPtr = builder.create<POP::LoadOp>(funcPtrPtr);
    builder.create<CallSignatureOp>(noneType, loadedFuncPtr,
                                    ValueRange({
                                        ptrToImpl,
                                        loadedExistingImpl,
                                    }));
  }
  // Populate move constructor.
  {
    Block *initialBlock = builder.getBlock();
    Block::iterator initialInsertionPoint = builder.getInsertionPoint();
    Location initialLocation = builder.getLoc();
    builder.setLoc(moveCtr.getLoc());
    builder.setInsertionPoint(moveCtr.getBody(), moveCtr.getBody()->begin());
    Value copySelf = moveCtr.getBody()->getArgument(0);
    Value copyExisting = moveCtr.getBody()->getArgument(1);
    Value existingImpl = builder.create<StructGEPOp>(copyExisting, impl);
    Value ptrToSelfImpl = builder.create<StructGEPOp>(copySelf, impl);
    auto loadedExistingImpl = builder.create<POP::LoadOp>(existingImpl);

    // Take the impl from the existing.
    builder.create<POP::StoreOp>(loadedExistingImpl, ptrToSelfImpl);
    auto opaquePointerTypeAttr =
        M::PointerAttr::get(builder.getContext(), 0, opaquePointer);
    Value nullPtr =
        builder.create<ParamConstantOp>(opaquePointer, opaquePointerTypeAttr);
    builder.create<POP::StoreOp>(nullPtr, existingImpl);

    builder.setInsertionPoint(initialBlock, initialInsertionPoint);
    builder.setLoc(initialLocation);
  }

  // Add the __call__ Method.
  SignatureType closureMethodSignatureType =
      addClosureSelfArgToFunctionSignature(ptrToSelfType, signatureType);
  LIT::FuncOp callMethod = structEmitter.synthesizeMethodInStruct(
      "__call__", closureMethodSignatureType.getValueInputs(),
      closureMethodSignatureType.getValueInputConventions(),
      callMemberSignatureType.getArgNames(),
      signatureType.getValueResults().front(), declOp,
      SpecialFunctionKind::kNormal, astDecl.getLoc(),
      closureMethodSignatureType.getFnEffects());

  // Populate the body of ClosureWrapper::__call__.
  builder = ImplicitLocOpBuilder::atBlockBegin(callMethod.getLoc(),
                                               callMethod.getBody());
  Value callSelf = hasResultSlot ? callMethod.getBody()->getArgument(1)
                                 : callMethod.getBody()->getArgument(0);
  SmallVector<Value> arguments;
  if (hasResultSlot)
    arguments.push_back(callMethod.getBody()->getArgument(0));

  arguments.push_back(
      builder.create<POP::LoadOp>(builder.create<StructGEPOp>(callSelf, impl)));

  for (unsigned i = hasResultSlot ? 2 : 1, e = callMethod.getNumArguments();
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
  return declOp;
}

static bool isSLValue(ASTDecl *astDecl, SMLoc loc, SharedState &shared) {
  if (astDecl->getIfSLValue())
    return true;
  if (Operation *op = astDecl->getIfOperation()) {
    if (auto varlet = dyn_cast<VarLetDeclOp>(op)) {
      if (ASTType(varlet.getType().getElementAsType())
              .isRegisterPassable(loc, shared))
        return true;
    }
  }
  return false;
}

StructDeclOp ClosureEmitter::replaceNestedFunctionWithClosureImplStructDecl(
    SMLoc location, ASTDecl &nestedFunctionDecl, ClosureCache &cache) {
  MLIRContext *ctx = shared.getContext();

  FuncOp nestedFunction = dyn_cast<LIT::FuncOp>(nestedFunctionDecl);
  assert(nestedFunction && "a function must back the nestedFunctionDecl");

  SignatureType closureWrapperSignature = nestedFunction.getSignature();
  size_t wrapperNumArgs = closureWrapperSignature.getNumInputs();

  auto captureRange = shared.getCaptureRangeInScope(nestedFunctionDecl);
  unsigned captureCount = std::distance(
      captureRange.begin(), captureRange.end()); // TODO: use llvm::size

  SmallVector<Type> closureImplSigTypes;
  closureImplSigTypes.reserve(captureCount + wrapperNumArgs);
  SmallVector<ValueInputConvention> closureImplSigConventions;
  closureImplSigConventions.reserve(captureCount + wrapperNumArgs);
  SmallVector<StringAttr> closureImplSigArgNames;
  closureImplSigArgNames.reserve(captureCount + wrapperNumArgs);

  SmallVector<Type> fieldTypes;
  fieldTypes.reserve(captureCount);
  // TODO: Enable expression of how to capture.
  for (const auto &[i, declCaptureIter] : llvm::enumerate(captureRange)) {
    Capture capture = declCaptureIter.second;
    closureImplSigTypes.push_back(capture.getInitType());

    Type fieldType = capture.getFieldType();
    if (ASTType(fieldType).isRegisterPassable(location, shared))
      closureImplSigConventions.push_back(ValueInputConvention::OwnedInReg);
    else
      closureImplSigConventions.push_back(ValueInputConvention::OwnedInMem);

    if (auto signatureType = dyn_cast<SignatureType>(fieldType)) {
      if (signatureType.isCapturing())
        shared.emitError(location,
                         "TODO: Cannot capture a signature type that "
                         "captures until new closures are turned on.");
      if (signatureType.isEscaping())
        shared.emitError(location,
                         "TODO: Cannot capture a signature type that escapes "
                         "until new closures are turned on.");
    }
    fieldTypes.push_back(fieldType);

    closureImplSigArgNames.push_back(
        StringAttr::get(ctx, "field" + std::to_string(i)));
  }

  // Create the closure impl signature from the captures and the wrapper
  // signature.
  llvm::append_range(closureImplSigTypes,
                     closureWrapperSignature.getValueInputs());
  llvm::append_range(
      closureImplSigConventions,
      closureWrapperSignature.getMetadata().getInputConventions());
  llvm::append_range(closureImplSigArgNames,
                     closureWrapperSignature.getArgNames());

  SignatureType closureImplSignature = SignatureType::get(
      closureWrapperSignature.getInputParamTypes(),
      closureWrapperSignature.getResultParamTypes(),
      FunctionType::get(ctx, closureImplSigTypes,
                        closureWrapperSignature.getValueResults()),
      FnMetadataAttr::get(ctx,
                          StringArrayAttr::get(ctx, closureImplSigArgNames),
                          closureImplSigConventions, {},
                          closureWrapperSignature.getFnEffects()));

  std::pair<SignatureType, StringAttr> key(closureImplSignature,
                                           fileModuleOp.getSymNameAttrName());
  if (auto existing = cache.getExisting(key)) {
    nestedFunction->erase();
    return existing;
  }

  bool hasByRefReturn = closureWrapperSignature.hasMemoryOnlyResult();

  StringAttr name =
      getClosureNameFromType("_CI_", fileModuleOp, closureImplSignature);

  ASTDecl &parent = shared.declResolver->getDeclForTypeSymbol(
      SymbolRefAttr::get(fileModuleOp.getDeclName()));
  StructDeclOp declOp =
      createStruct(fileModuleOp, name, fieldTypes, fileModuleOp.getLoc());
  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), declOp.getDeclName(), parent.getLoc(), &parent);

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

  SmallVector<StringAttr> initSigNames{StringAttr::get(ctx, "self")};
  llvm::append_range(initSigNames,
                     llvm::drop_end(closureImplSigArgNames, wrapperNumArgs));

  GeneratedStubs stubs = structEmitter.addMissingValueMemberStubsToStruct(
      declOp, astDecl.getLoc(), astDecl, /*generateFieldwiseInit=*/false);
  structEmitter.synthesizeMemberwiseInit(astDecl.getLoc(), declOp, initSigTypes,
                                         initSigConventions, initSigNames);

  LIT::FuncOp copyCtr = stubs.getCopyConstrucotr();
  LIT::FuncOp moveCtr = stubs.getMoveConstructor();

  if (failed(structEmitter.populateMoveCopy(copyCtr, declOp, astDecl,
                                            astDecl.getLoc(), false)))
    shared.emitError(copyCtr.getLoc(), "Cannot copy captured value because")
        << declOp.getSymName() << "` does not implement copy constructor.";

  // It is permissible for a closure implementation to not have a move
  // constructor.
  if (failed(structEmitter.populateMoveCopy(moveCtr, declOp, astDecl,
                                            astDecl.getLoc(), true)))
    moveCtr.erase();
  else
    declOp.setMoveInitAttr(moveCtr.getBoundReference());

  declOp.setCopyInitAttr(copyCtr.getBoundReference());

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
  assert(closureImplSignature.getValueResults().size() == 1 &&
         "Multiple outputs are not supported.");
  Type closureResultType = closureImplSignature.getValueResults().front();

  // Move by ref result argument to front before self argument.
  if (hasByRefReturn) {
    callInputTypes.push_back(closureWrapperSignature.getValueInputs()[0]);
    callConventions.push_back(closureWrapperSignature.getInputConvention(0));
    callNames.push_back(StringAttr::get(ctx, "__result__"));
  }

  // Currently Closure Impls are not register passable, so use BorrowInMem
  // convention.
  callInputTypes.push_back(ptrToClosureImplType);
  callConventions.push_back(ValueInputConvention::BorrowedInMem);
  callNames.push_back(StringAttr::get(ctx, "self"));
  for (unsigned i = hasByRefReturn, e = closureWrapperSignature.getNumInputs();
       i < e; ++i) {
    callInputTypes.push_back(closureWrapperSignature.getValueInputs()[i]);
    callConventions.push_back(closureWrapperSignature.getInputConvention(i));
    callNames.push_back(closureWrapperSignature.getArgName(i));
  }
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      declOp.getLoc(), &declOp.getFields().front());
  LIT::FuncOp callFunc = structEmitter.createFunction(
      "__call__", callInputTypes, callConventions, callNames, closureResultType,
      SpecialFunctionKind::kNormal, location, builder);
  declOp->setAttr(callMethodAttr, callFunc.getBoundReference());
  // Populate the body of the call op.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = callFunc.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Clone the body of the nested function. When closures are turned on, we can
  // take the body and remove the nested function op from the IR. For now though
  // to preserve the old pipeline we leave the original nested function intact.
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
      else
        return sp;
    });
    replacer.recursivelyReplaceElementsIn(callFunc, true, true);
  }

  builder =
      ImplicitLocOpBuilder::atBlockBegin(callFunc.getLoc(), callFunc.getBody());
  SmallVector<Value> fieldValues;
  Value selfArg = callFunc.getBodyRegion().insertArgument(
      hasByRefReturn, ptrToClosureImplType, callFuncLocation);
  for (auto [declAndCapture, fieldOp] :
       llvm::zip(captureRange, declOp.getFieldDecls())) {
    Capture capture = declAndCapture.second;
    auto ptrToField = builder.create<StructGEPOp>(selfArg, fieldOp);
    bool isRegisterPassable =
        ASTType(fieldOp.getType()).isRegisterPassable(location, shared);
    bool expectsPointer = !isRegisterPassable;
    if (isSLValue(declAndCapture.first, location, shared))
      expectsPointer = true;
    Value target = expectsPointer
                       ? ptrToField
                       : builder.create<POP::LoadOp>(ptrToField).getResult();
    replaceAllUsesInRegionWith(capture.getMlirValue(), target,
                               callFunc.getBodyRegion());
  }
  cache.storeClosure(key, declOp);
  nestedFunction->erase();
  return declOp;
}

LIT::FuncOp ClosureEmitter::createWrapperInitWithImpl(
    StructDeclOp closureWrapper, StructDeclOp closureImpl, SMLoc location) {
  auto ptrToClosureImplType =
      PointerType::get(ASTDecl::computeSelfTypeForStruct(closureImpl));
  if (auto init =
          structEmitter.findInitInStruct(closureWrapper, ptrToClosureImplType))
    return init;

  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = PointerType::get(emptyList);

  SmallVector<Type> argTypes;
  SmallVector<ValueInputConvention> argConventions;
  SmallVector<StringAttr> argNames;

  // Add the self to the closure init.
  StringAttr selfName = StringAttr::get(closureWrapper.getContext(), "self");
  Type closureSelfType =
      PointerType::get(ASTDecl::computeSelfTypeForStruct(closureWrapper));
  argTypes.push_back(closureSelfType);
  argConventions.push_back(ValueInputConvention::InitSelf);
  argNames.push_back(selfName);

  // Add the injected ClosureImpl argument to the initializer.
  argTypes.push_back(ptrToClosureImplType);
  argConventions.push_back(ValueInputConvention::BorrowedInMem);
  argNames.push_back(StringAttr::get(closureWrapper->getContext(), "impl"));
  FuncOp init = structEmitter.addVoidMethod(
      closureWrapper, "__init__", argTypes, argConventions, argNames,
      SpecialFunctionKind::kInit, location);

  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());

  // Allocate memory on heap and copy argument into allocated memory.

  auto allocateHeapMemory = [](PointerType ptrToClosureImplType,
                               ImplicitLocOpBuilder &builder) {
    Type elementType = ptrToClosureImplType.getElementAsType();
    Type indexType = builder.getIndexType();
    TypedAttr targetAttr = ParamOperatorAttr::get(
        POC::CurrentTarget, {}, builder.getType<TargetType>());
    TypedAttr sizeOfAttr = ParamOperatorAttr::get(
        POC::GetSizeOf,
        {ParameterizedTypeConstantAttr::get(elementType), targetAttr},
        builder.getType<TargetType>());
    Value sizeOf = builder.create<ParamConstantOp>(indexType, sizeOfAttr);
    TypedAttr alignOfAttr = ParamOperatorAttr::get(
        POC::GetAlignOf,
        {ParameterizedTypeConstantAttr::get(elementType), targetAttr},
        indexType);
    Value alignOf = builder.create<ParamConstantOp>(indexType, alignOfAttr);
    return builder.create<POP::AlignedAllocOp>(
        ptrToClosureImplType, ArrayRef<Value>{alignOf, sizeOf});
  };
  Value target = allocateHeapMemory(ptrToClosureImplType, builder);

  // Copy the contents of the injected impl into the heap memory.
  SymbolConstantAttr copySym;
  if (closureImpl.getMoveInit().has_value()) {
    copySym = cast<SymbolConstantAttr>(closureImpl.getMoveInit().value());
  } else {
    assert(closureImpl.getCopyInit().has_value() &&
           "All closure Implementations should have a generated copy "
           "constructor.");
    copySym = cast<SymbolConstantAttr>(closureImpl.getCopyInit().value());
  }
  ArrayRef<ParamDeclAttr> params;
  Value source = init.getBody()->getArgument(1);
  builder.create<CallOp>(
      copySym.getType().getValueResults(), copySym,
      ParamDeclArrayAttr::get(closureImpl.getContext(), params),
      ValueRange({target, source}));

  StructFieldOp implField = *closureWrapper.getFieldDecls().begin();
  Value self = init.getBody()->getArgument(0);
  Value ptrToImpl = builder.create<LIT::StructGEPOp>(
      PointerType::get(opaquePointer), implField.getNameAttr(), self);
  Value erasedType =
      builder.create<POP::PointerBitcastOp>(opaquePointer, target);
  builder.create<POP::StoreOp>(erasedType, ptrToImpl);
  auto generateName = [&](StringRef prefix) {
    return (closureWrapper.getDeclName().str() + prefix +
            closureImpl.getDeclName().str())
        .str();
  };
  auto setMember = [&](LIT::FuncOp topLevelFunc, StringAttr fieldName) {
    builder = ImplicitLocOpBuilder::atBlockBegin(init.getLoc(), init.getBody());
    auto dtorMember = builder.create<StructGEPOp>(
        PointerType::get(topLevelFunc.getBoundReference().getType()), fieldName,
        init.getBody()->getArgument(0));
    auto funcSymbol = builder.create<CreateClosureOp>(
        topLevelFunc.getBoundReference(), ValueRange());
    builder.create<POP::StoreOp>(funcSymbol, dtorMember);
  };

  // Create the top level copy constructor.
  // The copy constructor takes the Wrapper instance and the impl of the
  // existing.
  StringAttr existingName =
      StringAttr::get(closureWrapper.getContext(), "other");
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCopyInit = structEmitter.createFunction(
      generateName("_copyinit_"),
      {PointerType::get(opaquePointer), opaquePointer},
      {ValueInputConvention::BorrowedInReg,
       ValueInputConvention::BorrowedInMem},
      {StringAttr::get(closureWrapper.getContext(), "ptrToImpl"), existingName},
      shared.getNoneType(), SpecialFunctionKind::kNormal, location, builder);
  // Populate init body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(),
                                               topLevelCopyInit.getBody());
    Block *body = topLevelCopyInit.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelCopyInit.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
    // Allocate memory on heap and call copy constructor
    Value target = allocateHeapMemory(ptrToClosureImplType, builder);
    Value existingPtr = builder.create<POP::PointerBitcastOp>(
        ptrToClosureImplType, body->getArgument(1));
    if (TypedAttr symbol = closureImpl.getCopyInitAttr()) {
      builder.create<CallOp>(
          copySym.getType().getValueResults(), cast<SymbolConstantAttr>(symbol),
          ParamDeclArrayAttr::get(closureImpl.getContext(), params),
          ValueRange({target, existingPtr}));
    }
    // Store the allocated and populated impl into the closure wrapper.
    Value ptrToImpl = topLevelCopyInit.getBody()->getArgument(0);
    Value erasedType =
        builder.create<POP::PointerBitcastOp>(opaquePointer, target);
    builder.create<POP::StoreOp>(erasedType, ptrToImpl);

    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelCopyInit.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder,
        builder.create<ParamConstantOp>(builder.getAttr<LIT::NoneAttr>()),
        topLevelCopyInit);
    builder.create<LIT::EndFuncOp>();
    setMember(topLevelCopyInit, copyFieldAttr);
  }

  // Create top level destructor.
  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());

  LIT::FuncOp topLevelDtor = structEmitter.createFunction(
      generateName("_dtor_"), ArrayRef<Type>{opaquePointer},
      ArrayRef<ValueInputConvention>{ValueInputConvention::OwnedInReg},
      ArrayRef<StringAttr>{selfName}, shared.getNoneType(),
      SpecialFunctionKind::kNormal, location, builder);

  // Populate destructor body.
  {
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(),
                                               topLevelDtor.getBody());
    Block *body = topLevelDtor.getBody();
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (DebugInfo::DIScopeAttr spAttr = topLevelDtor.getLocScope())
      diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

    // Cast the opaque pointer back to the closure impl type.
    Value implPtr = builder.create<POP::PointerBitcastOp>(ptrToClosureImplType,
                                                          body->getArgument(0));

    // Call the destructor on the closure wrapper if it has one.
    if (closureImpl.getDestructor().has_value()) {
      auto dtorSym =
          cast<SymbolConstantAttr>(closureImpl.getDestructor().value());
      builder.create<CallOp>(
          dtorSym.getType().getValueResults(), dtorSym,
          ParamDeclArrayAttr::get(closureImpl.getContext(), params),
          ValueRange({implPtr}));
    }
    // Free the memory we allocated on the heap to store the closure.
    builder.create<POP::AlignedFreeOp>(implPtr);
    builder = ImplicitLocOpBuilder::atBlockEnd(topLevelDtor.getLoc(), body);
    ExprEmitter::emitNormalReturn(
        builder,
        builder.create<ParamConstantOp>(builder.getAttr<LIT::NoneAttr>()),
        topLevelDtor);
    builder.create<LIT::EndFuncOp>();
  }

  // Set the member.
  setMember(topLevelDtor, dtorFieldAttr);

  // Create the __call__ function.
  assert(closureWrapper.getClosureSignature().has_value() &&
         "The closure signature should have been set at creation time");
  auto functionSignature =
      cast<SignatureType>(closureWrapper.getClosureSignatureAttr().getValue());
  SignatureType closureSignature =
      addClosureSelfArgToFunctionSignature(opaquePointer, functionSignature);
  assert(closureSignature.getValueResults().size() == 1);

  builder = ImplicitLocOpBuilder::atBlockEnd(
      fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
  LIT::FuncOp topLevelCall = structEmitter.createFunction(
      generateName("_call_"), closureSignature.getValueInputs(),
      closureSignature.getValueInputConventions(),
      closureSignature.getArgNames(),
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
    Value closureArg = closureSignature.hasMemoryOnlyResult()
                           ? body->getArgument(1)
                           : body->getArgument(0);
    Value implPtr =
        builder.create<POP::PointerBitcastOp>(ptrToClosureImplType, closureArg);
    // Call the __call__ on the closure impl.
    assert(closureImpl->hasAttr(callMethodAttr) &&
           "Closure Impls are generated with a __call__ method.");
    SymbolConstantAttr symbol =
        cast<SymbolConstantAttr>(closureImpl->getAttr(callMethodAttr));
    SmallVector<Value> args;
    if (closureSignature.hasMemoryOnlyResult())
      args.push_back(topLevelCall.getArgument(0));
    args.push_back(implPtr);
    for (unsigned i = closureSignature.hasMemoryOnlyResult() + 1,
                  e = closureSignature.getNumInputs();
         i < e; ++i)
      args.push_back(topLevelCall.getArgument(i));
    Value result =
        builder
            .create<CallOp>(
                symbol.getType().getValueResults(), symbol,
                ParamDeclArrayAttr::get(closureImpl.getContext(), params), args)
            .getResult(0);
    ExprEmitter::emitNormalReturn(builder, result, topLevelDtor);
    builder.create<LIT::EndFuncOp>();
  }
  setMember(topLevelCall, callFieldAttr);
  return init;
}

Capture::Capture(AnyValue value, Type fieldType, Type initType)
    : fieldType(fieldType), initType(initType), anyValue(value), init(true) {}

Type Capture::getFieldType() const { return fieldType; }

Type Capture::getInitType() const { return initType; }

Value Capture::getMlirValue() const {
  if (auto v = anyValue.getIfSLValue()) {
    return v;
  } else if (auto v = anyValue.getIfSBValue()) {
    return v;
  } else if (auto v = anyValue.getIfSLValue()) {
    return v;
  } else if (auto v = anyValue.getIfMRValue()) {
    return v;
  } else if (auto v = anyValue.getIfMBValue()) {
    return v;
  } else if (auto v = anyValue.getIfSRValue()) {
    return v;
  }
  return {};
}

AnyValue Capture::getAnyValue() const { return anyValue; }
