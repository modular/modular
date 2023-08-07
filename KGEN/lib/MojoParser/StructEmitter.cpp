//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the StructEmitter class.
//
//===----------------------------------------------------------------------===//

#include "StructEmitter.h"
#include "ASTDecl.h"
#include "DeclResolver.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"
#include <bitset>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

LIT::FuncOp StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, Type resultType, StructDeclOp structOp,
    SpecialFunctionKind specialFnID, SMLoc loc) {
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  // Get the signature for the function.
  auto fnType = builder.getFunctionType(argTypes, resultType);

  FnEffects fnEffects = FnEffects();
  // If the result of the function is a non-trivial type, mark the function
  // effect as having an owned result so ownership tracking will notice it.
  if (!ASTType(resultType).isRegisterPassable(loc, shared))
    fnEffects = fnEffects | FnEffects::OwnedResult;

  // TODO: Should raise if anything we invoke raises.
  auto metadata = builder.getAttr<FnMetadataAttr>(
      argConventions, /*no default args=*/ArrayRef<TypedAttr>(), fnEffects);
  auto signature = SignatureType::get({}, {}, fnType, metadata);

  // Create the empty function.
  StringAttr nameAttr =
      DeclResolver::getMangledName(builder.getStringAttr(name), signature);
  auto funcOp =
      builder.create<LIT::FuncOp>(nameAttr, signature, argNames, specialFnID);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  DeclResolver::setLocationDebugScope(shared, diScopeGuard, funcOp, name);

  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getRegisterPassable() ==
      StructDeclOp::RP_RegisterPassableTrivial)
    funcOp.setAlwaysInlineLevel(AlwaysInlineLevel::EnabledNoDebug);

  return funcOp;
}

/// Given a function of the form
/// "lit.func __copyinit__(%target: !pop.pointer<@MyStruct>, %existing:
/// !pop.pointer<@MyStruct>), populate the method with the following:
/// %targetField0Ptr = lit.struct.get %self[field0]
/// %sourceField0Ptr = lit.struct.get %existing[field0]
/// copyinit_of_type_of_field0(%targetField0, %field
LogicalResult StructEmitter::populateMoveCopy(LIT::FuncOp func,
                                              StructDeclOp declOp,
                                              ASTDecl &declScope,
                                              SMLoc location, bool isMove) {
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = func.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());
  bool isMemoryOnly = !declOp.isRegisterPassable();
  ExprEmitter emitter(shared, declScope, b);
  if (isMemoryOnly) {
    assert(func.getNumArguments() == 2 &&
           "copy functions should have two arguments");
    Value copySelf = func.getBody()->getArgument(0);
    Value copyExisting = func.getBody()->getArgument(1);
    for (StructFieldOp fieldOp : declOp.getFieldDecls()) {
      auto targetFieldOp = b.create<StructGEPOp>(copySelf, fieldOp);
      auto srcFieldOp = b.create<StructGEPOp>(copyExisting, fieldOp);
      CValue src =
          isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));
      SLValue destination(targetFieldOp);
      SyntheticNode srcExpr(location);
      emitter.emitStoreToLValue({src, &srcExpr}, destination,
                                EC_AttributeRefBase);
    }
    return success();
  } else {
    assert(
        func.getNumArguments() == 1 &&
        "copy functions of register passable types should have one argument");
    func.setIsStatic(true);
    // Otherwise, extract all the values and finish with a struct create.  We
    // know all the subfields must be register passable.
    BlockArgument existingArg = func.getBody()->getArgument(0);
    SmallVector<Value> fieldVals;
    SmallVector<StringAttr> fieldNames;
    for (StructFieldOp fieldOp : declOp.getFieldDecls()) {
      SyntheticNode srcExpr(location);
      Value fieldValue = b.create<StructExtractOp>(existingArg, fieldOp);
      // Emit an SBValue -> SRValue conversion to get ownership of the value.
      Value copiedVal =
          emitter.emitSRValue({SBValue(fieldValue), &srcExpr}, EC_CallArgValue);
      if (!copiedVal)
        return failure();
      fieldVals.push_back(copiedVal);
      fieldNames.push_back(fieldOp.getNameAttr());
    }
    Type selfType = ASTDecl::computeSelfTypeForStruct(declOp);
    auto result = SRValue(b.create<StructCreateOp>(
        selfType, fieldVals,
        StringArrayAttr::get(func.getContext(), fieldNames)));

    ExprEmitter::emitNormalReturn(b, result, func);
    b.create<LIT::EndFuncOp>();
    return success();
  }
}

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
LIT::FuncOp StructEmitter::addVoidMethod(
    StructDeclOp selfStruct, StringRef prefix, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, SpecialFunctionKind kind, SMLoc loc) {
  M::KGEN::LIT::FuncOp func =
      synthesizeMethodInStruct(prefix, argTypes, argConventions, argNames,
                               shared.getNoneType(), selfStruct, kind, loc);
  Block *body = func.getBody();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = func.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
  for (Type inputVal : func.getArgumentTypes())
    body->addArgument(inputVal, selfStruct.getLoc());

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), body);
  ExprEmitter::emitNormalReturn(
      b, b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>()), func);
  b.create<LIT::EndFuncOp>();
  return func;
}

struct ValueInfo {
  enum FuncIndex { Destruct = 0, Move = 1, Copy = 2 };
  ValueInfo(StructDeclOp structDeclOp, SharedState &shared) {
    existingFunctions.reset();
    Type selfType = ASTDecl::computeSelfTypeForStruct(structDeclOp);
    ASTDecl &astDecl = shared.declResolver->getDeclForTypeSymbol(
        cast<DeclRefType>(selfType).getSymbol());
    for (auto &childDecl : astDecl.getDeclsInScope()) {
      for (auto iter = childDecl.second.begin(); iter != childDecl.second.end();
           iter++) {
        ASTDecl *declaration = *iter;
        if (auto func = dyn_cast<LIT::FuncOp>(*declaration)) {
          // Resolving the signature will guarantee the special kind is set.
          shared.declResolver->resolveSignature(*declaration,
                                                declaration->getLoc());
          switch ((SpecialFunctionKind)func.getSpecialFnKind()) {
          case SpecialFunctionKind::kDel:
            existingFunctions[FuncIndex::Destruct].flip();
            break;
          case SpecialFunctionKind::kCopyInitReg:
          case SpecialFunctionKind::kCopyInit:
            existingFunctions[FuncIndex::Copy].flip();
            break;
          case SpecialFunctionKind::kMoveInit:
            existingFunctions[FuncIndex::Move].flip();
            break;
          default:
            break;
          }
        }
      }
    }
  }
  bool hasDestructor() const { return existingFunctions[FuncIndex::Destruct]; }
  bool hasMove() const { return existingFunctions[FuncIndex::Move]; }
  bool hasCopy() const { return existingFunctions[FuncIndex::Copy]; }

private:
  std::bitset<3> existingFunctions;
};

GeneratedStubs StructEmitter::addMissingValueMemberStubsToStruct(
    StructDeclOp declOp, SMLoc loc, ASTDecl &parent,
    bool forceGenerateDestructor) {
  ValueInfo valueInfo(declOp, shared);
  bool isMemoryOnly = !declOp.isRegisterPassable();
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());
  Type selfType = ASTDecl::computeSelfTypeForStruct(declOp);
  Type ptrToSelf = POP::PointerType::get(selfType);
  StringAttr selfName = b.getStringAttr("self");
  StringAttr existingName = b.getStringAttr("existing");
  LIT::FuncOp destructorFunc;

  if (!valueInfo.hasDestructor()) {
    ASTDecl &structDecl = shared.declResolver->getDeclForTypeSymbol(
        cast<DeclRefType>(selfType).getSymbol());
    bool needsDtor = forceGenerateDestructor;
    if (!needsDtor) {
      for (auto field : declOp.getFieldDecls()) {
        auto fieldEntries =
            structDecl.lookupInCurrentScope(field.getNameAttr());
        assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
        ASTDecl &fieldASTDecl = *fieldEntries[0];
        if (failed(shared.declResolver->resolveSignature(
                fieldASTDecl, fieldASTDecl.getLoc())))
          continue;
        if (ASTType(field.getType())
                .hasDestructor(fieldASTDecl.getLoc(), shared)) {
          needsDtor = true;
          break;
        }
      }
    }

    if (needsDtor) {
      std::string name = "__del__";
      destructorFunc = addVoidMethod(
          declOp, name, SmallVector<Type>({ptrToSelf}),
          SmallVector<ValueInputConvention>({ValueInputConvention::OwnedInMem}),
          SmallVector<StringAttr>({selfName}), SpecialFunctionKind::kDel, loc);
      shared.declResolver->addFullyResolvedDecl(
          destructorFunc.getOperation(),
          StringAttr::get(shared.getContext(), name), loc, &parent);
    }
  }
  LIT::FuncOp copyFunc;
  if (!valueInfo.hasCopy() && declOp.getRegisterPassable() !=
                                  StructDeclOp::RP_RegisterPassableTrivial) {
    std::string name = "__copyinit__";
    if (isMemoryOnly) {
      copyFunc =
          addVoidMethod(declOp, name, SmallVector<Type>({ptrToSelf, ptrToSelf}),
                        SmallVector<ValueInputConvention>(
                            {ValueInputConvention::InitSelf,
                             ValueInputConvention::BorrowedInMem}),
                        SmallVector<StringAttr>({selfName, existingName}),
                        SpecialFunctionKind::kCopyInit, loc);
    } else {
      copyFunc = synthesizeMethodInStruct(
          name, SmallVector<Type>({selfType}),
          SmallVector<ValueInputConvention>(
              {ValueInputConvention::BorrowedInReg}),
          SmallVector<StringAttr>({existingName}), selfType, declOp,
          SpecialFunctionKind::kCopyInitReg, loc);
      if (!copyFunc.getBody())
        copyFunc.addBlock();
      for (Type argument : copyFunc.getSignature().getValueInputs())
        copyFunc.getBody()->addArgument(argument, copyFunc.getLoc());
    }
    shared.declResolver->addFullyResolvedDecl(
        copyFunc.getOperation(), StringAttr::get(shared.getContext(), name),
        loc, &parent);
  }
  LIT::FuncOp moveFunc;
  if (!valueInfo.hasMove() && isMemoryOnly) {
    std::string name = "__moveinit__";
    moveFunc = addVoidMethod(
        declOp, name, SmallVector<Type>({ptrToSelf, ptrToSelf}),
        SmallVector<ValueInputConvention>(
            {ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem}),
        SmallVector<StringAttr>({selfName, existingName}),
        SpecialFunctionKind::kMoveInit, loc);
    shared.declResolver->addFullyResolvedDecl(
        moveFunc.getOperation(), StringAttr::get(shared.getContext(), name),
        loc, &parent);
  }
  return GeneratedStubs{destructorFunc, copyFunc, moveFunc};
}
