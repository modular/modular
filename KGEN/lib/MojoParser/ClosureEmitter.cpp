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
#include "DeclResolver.h"
#include "ExprEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a struct and a list of arguments, generate a function. For example,
/// given {MyStruct, "prefix", [ParamType1, ParamType2], [borrow_in_mem,
/// borrow_in_mem], ["x","b"]}, this function produces:
///       lit.func @prefixParam1Param2(%self: !pop.pointer<@MyStruct> init_self,
///       %x: ParamType1 borrow_in_mem, %b : ParamType2 borrow_in_mem) ->
///       !lit.none  {
///          %0 = kgen.param.constant: !lit.none = <#lit.none>
///          lit.return %0 : !lit.none
///          lit.end_func
///      }
static LIT::FuncOp addVoidMethod(StructDeclOp selfStruct, StringRef prefix,
                                 ArrayRef<Type> argTypes,
                                 ArrayRef<ValueInputConvention> argConventions,
                                 ArrayRef<StringAttr> argNames,
                                 SpecialFunctionKind kind,
                                 ClosureEmitter &emitter) {
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      selfStruct.getLoc(), &selfStruct.getFields().front());
  auto metadata = b.getAttr<MetadataAttr>(argConventions, ArrayRef<TypedAttr>(),
                                          FnEffects());
  auto funcType = FunctionType::get(selfStruct->getContext(), argTypes,
                                    emitter.getNoneType());
  SignatureType signature = SignatureType::get({}, {}, funcType, metadata);

  StringAttr nameAttr =
      DeclResolver::getMangledName(b.getStringAttr(prefix), signature);
  auto func = b.create<LIT::FuncOp>(nameAttr, signature, argNames, kind);
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  DeclResolver::setLocationDebugScope(*emitter.sharedState(), diScopeGuard,
                                      func, nameAttr);

  Block *body = func.getBody();
  for (Type inputVal : func.getArgumentTypes())
    body->addArgument(inputVal, selfStruct.getLoc());

  b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), body);
  ExprEmitter::emitNormalReturn(
      b, b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>()), func);
  b.create<LIT::EndFuncOp>();
  return func;
}

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 SmallVector<TypeAttr> const &fields,
                                 Location location) {
  OpBuilder b(module.getRegion());
  StructDeclOp declOp = b.create<StructDeclOp>(location, nameAttr);
  if (declOp.getFields().empty())
    declOp.getFields().push_back(new Block());
  b.setInsertionPointToStart(&declOp.getFields().front());
  unsigned i = 0;
  for (TypeAttr type : fields)
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

StructDeclOp ClosureEmitter::createClosureWrapperStructDecl(
    StringAttr name, Location location, SignatureType signatureType) {
  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = TypeAttr::get(POP::PointerType::get(emptyList));
  SmallVector<TypeAttr> fieldTypes;
  fieldTypes.push_back(opaquePointer);
  StructDeclOp declOp = createStruct(fileModuleOp, name, fieldTypes, location);
  TypedAttr signatureAttr = SymbolConstantAttr::get(
      SymbolRefAttr::get(
          StringAttr::get(name.getContext(), name.str() + "_closureSignature")),
      signatureType);
  declOp.setClosureSignatureAttr(signatureAttr);

  StructFieldOp impl = *declOp.getFieldDecls().begin();
  // function ptr fields
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());
  auto dtor = b.create<StructFieldOp>(
      location, StringAttr::get(b.getContext(), "dtor"),
      SignatureType::get(b.getContext(), TypeRange({opaquePointer.getValue()}),
                         noneType),
      nullptr);
  SmallVector<Type> callInputTypes;
  callInputTypes.push_back(opaquePointer.getValue());
  llvm::append_range(callInputTypes, signatureType.getValueInputs());
  auto createCopyOrMoveMember = [&](bool isCopy) {
    SmallVector<ValueInputConvention> inputConventions;
    inputConventions.push_back(ValueInputConvention::InitSelf);
    if (isCopy)
      inputConventions.push_back(ValueInputConvention::BorrowedInMem);
    else
      inputConventions.push_back(ValueInputConvention::OwnedInMem);

    TypeRange opaqueInputs(
        {opaquePointer.getValue(), opaquePointer.getValue()});
    std::string name = isCopy ? "copy" : "move";
    SignatureType cpySignatureType = SignatureType::get(
        {}, {},
        FunctionType::get(signatureType.getContext(), opaqueInputs, noneType),
        b.getAttr<MetadataAttr>(inputConventions, ArrayRef<TypedAttr>(),
                                FnEffects()));
    return b.create<StructFieldOp>(location,
                                   StringAttr::get(b.getContext(), name),
                                   cpySignatureType, nullptr);
  };
  auto copy = createCopyOrMoveMember(true);
  auto move = createCopyOrMoveMember(false);
  Type ptrToSelf =
      POP::PointerType::get(ASTDecl::computeSelfTypeForStruct(declOp));
  StringAttr selfName = b.getStringAttr("self");
  StringAttr existingName = b.getStringAttr("existing");

  // Create the member methods.
  LIT::FuncOp destructorFunc = addVoidMethod(
      declOp, "__del__", SmallVector<Type>({ptrToSelf}),
      SmallVector<ValueInputConvention>({ValueInputConvention::OwnedInMem}),
      SmallVector<StringAttr>({selfName}), SpecialFunctionKind::kDel, *this);
  LIT::FuncOp moveFunc = addVoidMethod(
      declOp, "__moveinit__", SmallVector<Type>({ptrToSelf, ptrToSelf}),
      SmallVector<ValueInputConvention>(
          {ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem}),
      SmallVector<StringAttr>({selfName, existingName}),
      SpecialFunctionKind::kMoveInit, *this);
  LIT::FuncOp copyFunc = addVoidMethod(
      declOp, "__copyinit__", SmallVector<Type>({ptrToSelf, ptrToSelf}),
      SmallVector<ValueInputConvention>({ValueInputConvention::InitSelf,
                                         ValueInputConvention::BorrowedInMem}),
      SmallVector<StringAttr>({selfName, existingName}),
      SpecialFunctionKind::kCopyInit, *this);

  // Populate methods.
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
      destructorFunc.getLoc(), destructorFunc.getBody());
  Value dtorSelf = destructorFunc.getBody()->getArgument(0);
  builder.create<CallSignatureOp>(
      noneType,
      builder.create<POP::LoadOp>(builder.create<StructGEPOp>(dtorSelf, dtor)),
      ValueRange({builder.create<POP::LoadOp>(
          builder.create<StructGEPOp>(dtorSelf, impl))}));

  populateMoveCopy(builder, copy, copyFunc, impl, noneType);
  populateMoveCopy(builder, move, moveFunc, impl, noneType);
  return declOp;
}
