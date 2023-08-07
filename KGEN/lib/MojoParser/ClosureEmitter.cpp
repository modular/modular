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

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

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
      declOp.getLoc(), StringAttr::get(b.getContext(), "dtor"),
      SignatureType::get(b.getContext(), TypeRange({opaquePointer}), noneType),
      nullptr);
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
    std::string name = isCopy ? "copy" : "move";
    SignatureType cpySignatureType = SignatureType::get(
        {}, {},
        FunctionType::get(signatureType.getContext(), inputTypes, noneType),
        b.getAttr<FnMetadataAttr>(inputConventions, ArrayRef<TypedAttr>(),
                                  FnEffects()));
    return b.create<StructFieldOp>(declOp.getLoc(),
                                   StringAttr::get(b.getContext(), name),
                                   cpySignatureType, nullptr);
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

  auto [destructor, copyCtr, moveCtr] =
      structEmitter.addMissingValueMemberStubsToStruct(
          declOp, parent.getLoc(), astDecl, /*forceGenerateDestructor*/ true);

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

StructDeclOp
ClosureEmitter::createClosureImplStructDecl(StringAttr name,
                                            SignatureType closureImplSignature,
                                            unsigned captureCount) {
  SmallVector<Type> types;
  for (auto [i, type, convention] : llvm::enumerate(
           closureImplSignature.getValueInputs(),
           closureImplSignature.getMetadata().getInputConventions())) {
    if (i >= captureCount)
      break;
    // The convention defines a map from the closureImplSignature parameter type
    // to the field type. The parameter type in the ClosureImpl initializer that
    // corresponds to this field will match the type in the closureImplSignature
    // but the fieldType may differ.
    switch (convention) {
    case ValueInputConvention::InitSelf:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::ByRef: {
      assert(isa<POP::PointerType>(type) &&
             "convention does not match type requirement.");
      types.emplace_back(type);
      break;
    }
    case ValueInputConvention::OwnedInReg:
    case ValueInputConvention::BorrowedInReg:
      types.emplace_back(type);
      break;
    case ValueInputConvention::OwnedInMem:
    case ValueInputConvention::BorrowedInMem: {
      POP::PointerType ptr = cast<POP::PointerType>(type);
      types.emplace_back(ptr.getElementAsType());
      break;
    }
    }
  }
  ASTDecl &parent = shared.declResolver->getDeclForTypeSymbol(
      SymbolRefAttr::get(fileModuleOp.getDeclName()));
  StructDeclOp declOp =
      createStruct(fileModuleOp, name, types, fileModuleOp.getLoc());
  ASTDecl &astDecl = shared.declResolver->addFullyResolvedDecl(
      declOp.getOperation(), declOp.getDeclName(), parent.getLoc(), &parent);

  for (StructFieldOp field : declOp.getFieldDecls())
    shared.declResolver->addFullyResolvedDecl(
        field.getOperation(), field.getNameAttr(), astDecl.getLoc(), &astDecl);

  auto [_, copyCtr, moveCtr] = structEmitter.addMissingValueMemberStubsToStruct(
      declOp, astDecl.getLoc(), astDecl);

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
  return declOp;
}
