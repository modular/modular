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

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"

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

/// Given a function of the form "lit.func __initFromExisting__(%target:
/// !pop.pointer<@MyStruct>, %existing: !pop.pointer<@MyStruct>)", the opaque
/// pointer FieldOp member, and a FieldOp member that is of type
/// kgen.signature<(x,x) -> x>, where x is opaque pointer, populate the function
/// with the following:
///       %funcPtrPtr = lit.struct.gep %self[funcPtr]
///       %selfImplPtr = lit.struct.gep %self[impl]
///       %existingImplPtr = lit.struct.gep $existing[impl]
///       %funcPtr = pop.load %funcPtrPtr
///       %loadedSelfImpl = pop.load %selfImplPtr
///       %loadedExistingImpl = pop.load %existingImplPtr
///       kgen.call_signature %funcPtr(%loadedSelfImpl, %loadedExistingImpl)
static void populateMoveCopy(ImplicitLocOpBuilder &builder,
                             StructFieldOp fieldOp, LIT::FuncOp wrapperFunc,
                             StructFieldOp impl, Type noneType) {
  Block *initialBlock = builder.getBlock();
  Block::iterator initialInsertionPoint = builder.getInsertionPoint();
  Location initialLocation = builder.getLoc();
  builder.setLoc(wrapperFunc.getLoc());
  builder.setInsertionPoint(wrapperFunc.getBody(),
                            wrapperFunc.getBody()->begin());
  Value copySelf = wrapperFunc.getBody()->getArgument(0);
  Value copyExisting = wrapperFunc.getBody()->getArgument(1);
  Value selfImpl = builder.create<StructGEPOp>(copySelf, impl);
  Value existingImpl = builder.create<StructGEPOp>(copyExisting, impl);

  auto loadedSelfImpl = builder.create<POP::LoadOp>(selfImpl);
  auto loadedExistingImpl = builder.create<POP::LoadOp>(existingImpl);
  auto funcPtrPtr = builder.create<StructGEPOp>(copySelf, fieldOp);
  auto loadedFuncPtr = builder.create<POP::LoadOp>(funcPtrPtr);
  builder.create<CallSignatureOp>(noneType, loadedFuncPtr,
                                  ValueRange({
                                      loadedSelfImpl,
                                      loadedExistingImpl,
                                  }));
  builder.setInsertionPoint(initialBlock, initialInsertionPoint);
  builder.setLoc(initialLocation);
}

StructDeclOp
ClosureEmitter::createClosureWrapperStructDecl(StringAttr name,
                                               SignatureType signatureType) {
  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = POP::PointerType::get(emptyList);
  SmallVector<Type> fieldTypes;
  fieldTypes.push_back(opaquePointer);
  StructDeclOp declOp =
      createStruct(fileModuleOp, name, fieldTypes, fileModuleOp.getLoc());
  TypedAttr signatureAttr = SymbolConstantAttr::get(
      SymbolRefAttr::get(
          StringAttr::get(name.getContext(), name.str() + "_closureSignature")),
      signatureType);
  declOp.setClosureSignatureAttr(signatureAttr);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());
  auto dtor = b.create<StructFieldOp>(
      declOp.getLoc(), dtorFieldAttr,
      SignatureType::get(b.getContext(), opaquePointer, noneType), nullptr);
  SmallVector<Type> callInputTypes;
  callInputTypes.push_back(opaquePointer);
  llvm::append_range(callInputTypes, signatureType.getValueInputs());
  auto createCopyOrMoveMember = [&](bool isCopy) {
    SmallVector<ValueInputConvention> inputConventions;
    inputConventions.push_back(ValueInputConvention::InitSelf);
    if (isCopy)
      inputConventions.push_back(ValueInputConvention::BorrowedInMem);
    else
      inputConventions.push_back(ValueInputConvention::OwnedInMem);

    Type opaquePtrType = POP::PointerType::get(
        POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1)));
    SmallVector<Type> inputTypes({opaquePtrType, opaquePtrType});
    StringAttr fieldName = isCopy ? copyFieldAttr : moveFieldAttr;
    SignatureType cpySignatureType = SignatureType::get(
        {}, {},
        FunctionType::get(signatureType.getContext(), inputTypes, noneType),
        b.getAttr<FnMetadataAttr>(inputConventions, ArrayRef<TypedAttr>(),
                                  FnEffects()));
    return b.create<StructFieldOp>(declOp.getLoc(), fieldName, cpySignatureType,
                                   nullptr);
  };
  auto copy = createCopyOrMoveMember(true);
  auto move = createCopyOrMoveMember(false);

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
  LIT::FuncOp destructor = stubs.getDestructor();
  LIT::FuncOp copyCtr = stubs.getCopyConstrucotr();
  LIT::FuncOp moveCtr = stubs.getMoveConstructor();

  // Populate methods.
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
      destructor.getLoc(), destructor.getBody());
  Value dtorSelf = destructor.getBody()->getArgument(0);
  builder.create<CallSignatureOp>(
      noneType,
      builder.create<POP::LoadOp>(builder.create<StructGEPOp>(dtorSelf, dtor)),
      ValueRange({builder.create<POP::LoadOp>(
          builder.create<StructGEPOp>(dtorSelf, impl))}));

  populateMoveCopy(builder, copy, copyCtr, impl, noneType);
  populateMoveCopy(builder, move, moveCtr, impl, noneType);
  return declOp;
}

StructDeclOp ClosureEmitter::createClosureImplStructDecl(
    SMLoc location, ASTDecl &nestedFunctionDecl, ClosureCache &cache) {
  FuncOp nestedFunction = dyn_cast<LIT::FuncOp>(nestedFunctionDecl);
  assert(nestedFunction && "a function must back the nestedFunctionDecl");
  ArrayRef<Capture> captures = shared.getCapturesInScope(nestedFunctionDecl);
  SmallVector<Type> closureImplSigTypes;
  SmallVector<ValueInputConvention> closureImplSigConventions;

  unsigned captureCount = captures.size();
  unsigned initArgCount = captureCount + 1;
  SmallVector<Type> fieldTypes;
  SmallVector<Type> initSigTypes(initArgCount);
  SmallVector<ValueInputConvention> initSigConventions(initArgCount);
  SmallVector<StringAttr> initSigNames(initArgCount);
  ExprEmitter emitter(shared, nestedFunctionDecl, EC_Type);
  // TODO: Enable expression of how to capture.
  unsigned i = 0;
  for (Capture capture : captures) {
    Type fieldType = capture.getFieldType();
    Type initType = capture.getInitType();

    ValueInputConvention inputConvention;
    if (ASTType(fieldType).isRegisterPassable(location, shared))
      inputConvention = ValueInputConvention::OwnedInReg;
    else
      inputConvention = ValueInputConvention::OwnedInMem;

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
    closureImplSigConventions.push_back(inputConvention);
    closureImplSigTypes.push_back(initType);
    initSigTypes[i + 1] = initType;
    initSigConventions[i + 1] = inputConvention;
    initSigNames[i + 1] =
        StringAttr::get(shared.getContext(), "field" + std::to_string(i));
    i++;
  }
  // Create the closure impl signature from the captures and the wrapper
  // signature.
  SignatureType closureWrapperSignature = nestedFunction.getSignature();
  llvm::append_range(closureImplSigTypes,
                     closureWrapperSignature.getValueInputs());
  llvm::append_range(
      closureImplSigConventions,
      closureWrapperSignature.getMetadata().getInputConventions());
  SignatureType closureImplSignature = SignatureType::get(
      closureWrapperSignature.getInputParamTypes(),
      closureWrapperSignature.getResultParamTypes(),
      FunctionType::get(shared.getContext(), closureImplSigTypes,
                        closureWrapperSignature.getValueResults()),
      FnMetadataAttr::get(shared.getContext(), closureImplSigConventions, {},
                          closureWrapperSignature.getFnEffects()));

  std::pair<SignatureType, StringAttr> key(closureImplSignature,
                                           fileModuleOp.getSymNameAttrName());
  if (auto existing = cache.getExisting(key))
    return existing;

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

  auto ptrToClosureImplType =
      POP::PointerType::get(ASTDecl::computeSelfTypeForStruct(declOp));
  initSigTypes[0] = ptrToClosureImplType;
  initSigConventions[0] = ValueInputConvention::InitSelf;
  initSigNames[0] = StringAttr::get(shared.getContext(), "self");

  GeneratedStubs stubs = structEmitter.addMissingValueMemberStubsToStruct(
      declOp, astDecl.getLoc(), astDecl, /*generateFieldwiseInit*/ false);
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
  cache.storeClosure(key, declOp);
  return declOp;
}

LIT::FuncOp ClosureEmitter::createWrapperInitWithImpl(
    StructDeclOp closureWrapper, StructDeclOp closureImpl, SMLoc location) {
  auto ptrToClosureImplType =
      POP::PointerType::get(ASTDecl::computeSelfTypeForStruct(closureImpl));
  if (auto init =
          structEmitter.findInitInStruct(closureWrapper, ptrToClosureImplType))
    return init;

  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = POP::PointerType::get(emptyList);

  SmallVector<Type> argTypes;
  SmallVector<ValueInputConvention> argConventions;
  SmallVector<StringAttr> argNames;

  // Add the self to the closure init.
  StringAttr selfName = StringAttr::get(closureWrapper.getContext(), "self");
  Type closureSelfType =
      POP::PointerType::get(ASTDecl::computeSelfTypeForStruct(closureWrapper));
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
  Type elementType = ptrToClosureImplType.getElementAsType();
  Type indexType = builder.getIndexType();
  Attribute targetAttr = ParamOperatorAttr::get(POC::CurrentTarget, {},
                                                builder.getType<TargetType>());
  Attribute sizeOfAttr =
      ParamOperatorAttr::get(POC::GetSizeOf,
                             {ParameterizedTypeConstantAttr::get(elementType),
                              cast<TypedAttr>(targetAttr)},
                             builder.getType<TargetType>());
  Value sizeOf =
      builder.create<ParamConstantOp>(indexType, cast<TypedAttr>(sizeOfAttr));
  Attribute alignOfAttr = ParamOperatorAttr::get(
      POC::GetAlignOf,
      {cast<TypedAttr>(ParameterizedTypeConstantAttr::get(elementType)),
       cast<TypedAttr>(targetAttr)},
      indexType);
  Value alignOf =
      builder.create<ParamConstantOp>(indexType, cast<TypedAttr>(alignOfAttr));
  Value target = builder.create<POP::AlignedAllocOp>(
      ptrToClosureImplType, ArrayRef<Value>{alignOf, sizeOf});

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
      POP::PointerType::get(opaquePointer), implField.getNameAttr(), self);
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
        POP::PointerType::get(topLevelFunc.getBoundReference().getType()),
        fieldName, init.getBody()->getArgument(0));
    auto funcSymbol = builder.create<CreateClosureOp>(
        topLevelFunc.getBoundReference(), ValueRange());
    builder.create<POP::StoreOp>(funcSymbol, dtorMember);
  };

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

  // Create the copy constructors.
  auto makeCopyMoveConstructor = [&](bool isMove) {
    builder = ImplicitLocOpBuilder::atBlockEnd(
        fileModuleOp.getLoc(), &fileModuleOp.getBodyRegion().front());
    StringRef prefix = isMove ? "_moveinit_" : "_copyinit_";
    StringAttr existingName =
        StringAttr::get(closureWrapper.getContext(), "existing");
    LIT::FuncOp topLevelInit = structEmitter.createFunction(
        generateName(prefix), ArrayRef<Type>{opaquePointer, opaquePointer},
        ArrayRef<ValueInputConvention>{
            ValueInputConvention::InitSelf,
            isMove ? ValueInputConvention::OwnedInMem
                   : ValueInputConvention::BorrowedInMem},
        ArrayRef<StringAttr>{selfName, existingName}, shared.getNoneType(),
        SpecialFunctionKind::kNormal, location, builder);
    // Populate init body.
    {
      builder = ImplicitLocOpBuilder::atBlockEnd(topLevelInit.getLoc(),
                                                 topLevelInit.getBody());
      Block *body = topLevelInit.getBody();
      DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
      if (DebugInfo::DIScopeAttr spAttr = topLevelInit.getLocScope())
        diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
      Value selfPtr = builder.create<POP::PointerBitcastOp>(
          ptrToClosureImplType, body->getArgument(0));
      Value existingPtr = builder.create<POP::PointerBitcastOp>(
          ptrToClosureImplType, body->getArgument(1));
      TypedAttr symbol = isMove ? closureImpl.getMoveInitAttr()
                                : closureImpl.getCopyInitAttr();
      if (symbol) {
        auto copySym = cast<SymbolConstantAttr>(symbol);
        builder.create<CallOp>(
            copySym.getType().getValueResults(), copySym,
            ParamDeclArrayAttr::get(closureImpl.getContext(), params),
            ValueRange({selfPtr, existingPtr}));
      }
      if (isMove)
        builder.create<OwnershipMarkDestroyedOp>(existingPtr);
      builder = ImplicitLocOpBuilder::atBlockEnd(topLevelInit.getLoc(), body);
      ExprEmitter::emitNormalReturn(
          builder,
          builder.create<ParamConstantOp>(builder.getAttr<LIT::NoneAttr>()),
          topLevelInit);
      builder.create<LIT::EndFuncOp>();
      setMember(topLevelInit, isMove ? moveFieldAttr : copyFieldAttr);
    }
  };
  makeCopyMoveConstructor(false);
  makeCopyMoveConstructor(true);

  return init;
}

Type Capture::getFieldType() const { return fieldType; }

Type Capture::getInitType() const { return initType; }

Value Capture::getMlirValue() const { return mlirValue; }
