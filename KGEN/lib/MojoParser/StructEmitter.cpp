//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the StructEmitter class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/StructEmitter.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"
#include <bitset>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

LIT::FuncOp StructEmitter::createFunction(
    StringRef name, ArrayRef<ParamDeclAttr> inputParameters,
    ArrayRef<PassingKind> paramPassingKinds, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects) {
  // If the result of the function is a non-trivial type, mark the function
  // effect as having an owned result so ownership tracking will notice it.
  if (ASTType(resultType).getRegisterPassability(loc, shared) !=
      StructDeclOp::RP_RegisterPassableTrivial)
    fnEffects.setOwnedRegisterResult();

  SmallVector<StringAttr> parameterNames;
  for (ParamDeclAttr p : inputParameters)
    parameterNames.push_back(p.getName());

  auto metadata =
      FnMetadataAttr::get(builder.getContext(), argNames, argPassingKinds,
                          parameterNames, paramPassingKinds,
                          /*defaultArguments=*/{}, /*defaultParameters=*/{});
  FunctionType functionType = builder.getFunctionType(argTypes, {resultType});
  Location location = shared.translateLocation(loc);
  LITSignatureType signature = IndexRefRemapper::remapToSignature(
      inputParameters, /*resultParams=*/{}, functionType, argConventions,
      fnEffects, metadata, [&] { return mlir::emitError(location); });
  StringAttr nameAttr =
      DeclResolver::getMangledName(builder.getStringAttr(name), signature);
  auto funcOp = builder.create<LIT::FuncOp>(nameAttr, signature, specialFnID);

  // Set the attributes on the FuncOp in bulk.
  NamedAttrList attrs = funcOp->getAttrDictionary();
  if (!inputParameters.empty())
    attrs.set(funcOp.getInputParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(inputParameters));
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  DeclResolver::setLocationDebugScope(shared, diScopeGuard, funcOp, name);
  if (!funcOp.getBody())
    funcOp.getBodyRegion().push_back(new Block);
  for (Type param : argTypes)
    funcOp.getBody()->addArgument(param, funcOp.getLoc());

  return funcOp;
}

std::pair<LIT::FuncOp, ASTDecl &> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> inputParameters,
    ArrayRef<PassingKind> paramPassingKinds, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    Type resultType, ASTDecl &structDecl, SpecialFunctionKind specialFnID,
    FnEffects effects) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  LIT::FuncOp funcOp =
      createFunction(name, inputParameters, paramPassingKinds, argTypes,
                     argConventions, argNames, argPassingKinds, resultType,
                     specialFnID, structDecl.getLoc(), builder, effects);

  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getRegisterPassable() ==
      StructDeclOp::RP_RegisterPassableTrivial)
    funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Register the method in the struct.
  ASTDecl &funcDecl = shared.declResolver->addFullyResolvedDecl(
      funcOp.getOperation(), StringAttr::get(shared.getContext(), name),
      structDecl.getLoc(), &structDecl);

  // Set the symbol and notice if we are redeclaring something.
  if (shared.declResolver->finalizeFuncSignature(funcOp, funcDecl)) {
    shared.emitError(structDecl.getLoc(),
                     "Duplicate definition of " + funcOp.getSymName().str());
  }

  return {funcOp, funcDecl};
}

LIT::FuncOp StructEmitter::synthesizeMemberwiseInit(
    ASTDecl &structDecl, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds) {
  auto structOp = cast<StructDeclOp>(structDecl);
  ASTType selfType = structDecl.getSelfType();
  bool isMemoryOnly =
      structOp.getRegisterPassable() == StructDeclOp::RP_MemoryOnly;

  // Figure out the type of the 'self' argument/result.
  Type resultType = isMemoryOnly ? shared.getNoneType() : selfType;

  auto specialFnId =
      isMemoryOnly ? SpecialFunctionKind::kInit : SpecialFunctionKind::kInitReg;

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", /*inputParameters=*/{}, /*paramPassingKinds=*/{}, argTypes,
      argConventions, argNames, argPassingKinds, resultType, structDecl,
      specialFnId);

  // Set up the body.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  builder.setLoc(funcOp->getLoc());
  ASTDecl *funcDecl = shared.declResolver->getDeclForFuncSymbol(
      cast<SymbolConstantAttr>(funcOp.getBoundReference()).getSymbol());
  ExprEmitter emitter(shared, *funcDecl, builder);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = funcOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // For a memory-only initializer, we emit a bunch of stores to fields indexing
  // self.
  if (isMemoryOnly) {
    BlockArgument selfArg = body->getArgument(0);
    assert(selfArg.getType().isa<PointerType>());
    size_t idx = 1;
    for (StructFieldOp field : structOp.getFieldDecls()) {
      // Add the block argument, get it as an RValue since it is owned.
      BlockArgument arg = body->getArgument(idx);
      CValue argVal;
      switch (argConventions[idx]) {
      default:
        llvm_unreachable("unknown convention");
      case ValueInputConvention::OwnedInReg:
        argVal = SRValue(arg);
        break;
      case ValueInputConvention::BorrowedInReg:
        argVal = SBValue(arg);
        break;
      case ValueInputConvention::OwnedInMem:
        // FIXME(references): This won't be right for first-class references.
        if (isa<RefType>(arg.getType()))
          argVal = XRValue(arg);
        else
          argVal = MRValue(arg);
        break;
      case ValueInputConvention::BorrowedInMem:
        // FIXME(references): This won't be right for first-class references.
        if (isa<RefType>(arg.getType()))
          argVal = XBValue(arg);
        else
          argVal = MBValue(arg);
        break;
      }

      // Project self to the right field and store the RValue.
      auto fieldPtr = builder.create<StructGEPOp>(selfArg, field);
      SyntheticNode srcExpr(structDecl.getLoc());
      emitter.emitStoreToLValue({argVal, &srcExpr}, MLValue(fieldPtr),
                                EC_AttributeRefBase);
      ++idx;
    }

    // Finish off the function with a return + lit.endfunc.
    ExprEmitter::emitNormalReturn(
        builder, builder.create<ParamConstantOp>(noneAttr), funcOp);
    builder.create<LIT::EndFuncOp>();
    return funcOp;
  }

  funcOp.setIsStatic(true);

  // Otherwise, emit all the values and finish with a struct create.  We know
  // all the subfields must be register passable.
  SmallVector<Value> fieldVals;
  for (size_t idx = 0, e = argTypes.size(); idx != e; ++idx) {
    // Add the block argument, get it as an RValue since it is owned.
    BlockArgument arg = body->getArgument(idx);
    fieldVals.push_back(arg);
  }

  auto result = SRValue(builder.create<StructCreateOp>(
      selfType.mlirType, fieldVals,
      StringArrayAttr::get(emitter.getContext(), argNames)));

  ExprEmitter::emitNormalReturn(builder, result, funcOp);
  builder.create<LIT::EndFuncOp>();
  return funcOp;
}

/// Given a function of the form
/// "lit.func __copyinit__(%target: !kgen.pointer<@MyStruct>, %existing:
/// !kgen.pointer<@MyStruct>), populate the method with the following:
/// %targetField0Ptr = lit.struct.get %self[field0]
/// %sourceField0Ptr = lit.struct.get %existing[field0]
/// copyinit_of_type_of_field0(%targetField0, %field
LogicalResult StructEmitter::populateMoveCopy(ASTDecl &functionDecl,
                                              bool isMove) {
  auto func = dyn_cast<LIT::FuncOp>(functionDecl);
  if (!func)
    return failure();
  ASTDecl *declScope = functionDecl.getParentDecl();
  if (!declScope)
    return failure();
  StructDeclOp declOp = dyn_cast<StructDeclOp>(declScope);
  // We want to populate a move but the move/copy should be a method!
  if (!declOp)
    return failure();
  SMLoc location = functionDecl.getLoc();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = func.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
      shared.translateLocation(location), func.getBody());
  bool isMemoryOnly = !declOp.isRegisterPassable();
  ExprEmitter emitter(shared, *declScope, b);
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
      MLValue destination(targetFieldOp);
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
/// given {
///  MyStruct, "prefix", [ParamType1, ParamType2],
///  [borrow_in_mem, borrow_in_mem], ["x","b"]
/// }, this function produces:
///
/// ```
/// lit.func @prefixParam1Param2(%self: !kgen.pointer<@MyStruct>
///     init_self, %x: ParamType1 borrow_in_mem, %b : ParamType2 borrow_in_mem
/// ) -> !kgen.none  {
///   %0 = kgen.param.constant: none = <#kgen.none>
///   lit.return %0 : !kgen.none
///   lit.end_func
/// }
/// ```
LIT::FuncOp StructEmitter::addVoidMethod(
    ASTDecl &structDecl, StringRef prefix,
    ArrayRef<ParamDeclAttr> inputParameters,
    ArrayRef<PassingKind> paramPassingKinds, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    SpecialFunctionKind kind) {
  auto [func, _] = synthesizeMethodInStruct(
      prefix, inputParameters, paramPassingKinds, argTypes, argConventions,
      argNames, argPassingKinds, shared.getNoneType(), structDecl, kind);
  Block *body = func.getBody();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = func.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), body);
  ExprEmitter::emitNormalReturn(b, b.create<ParamConstantOp>(noneAttr), func);
  b.create<LIT::EndFuncOp>();
  return func;
}

struct ValueInfo {
  enum FuncIndex { Destruct = 0, Move = 1, Copy = 2, FieldwiseInit = 3 };

  static ValueInfo createValueInfo(StructDeclOp structDeclOp,
                                   SharedState &shared) {
    std::bitset<4> existingFunctions;
    existingFunctions.reset();
    Type selfType = ASTDecl::computeSelfTypeForStruct(structDeclOp);
    ASTDecl &astDecl = shared.declResolver->getDeclForTypeSymbol(
        cast<DeclRefType>(selfType).getSymbol());

    auto setBit = [&](StringRef name, SpecialFunctionKind kind,
                      unsigned index) -> LogicalResult {
      LookupResult lookupResult =
          shared.lookupAndResolveDecl(name, astDecl.getLoc(), astDecl,
                                      /*searchParentScopes=*/false);
      if (!lookupResult.isSuccess())
        return success();
      if (lookupResult.getIfSuccess().size() > 1)
        return shared.emitError(structDeclOp.getLoc())
               << "multiple overloaded methods named '" << name << "'";

      if (lookupResult.getIfSuccess().size() == 1) {
        ASTDecl *result = lookupResult.getIfSuccess().front();
        if (auto func = dyn_cast<LIT::FuncOp>(result))
          if ((SpecialFunctionKind)func.getSpecialFnKind() == kind)
            existingFunctions[index].flip();
      }

      return success();
    };
    bool isMemoryOnly =
        structDeclOp.getRegisterPassable() == StructDeclOp::RP_MemoryOnly;
    if (failed(
            setBit("__del__", SpecialFunctionKind::kDel, FuncIndex::Destruct)))
      return {};
    if (failed(setBit("__copyinit__",
                      isMemoryOnly ? SpecialFunctionKind::kCopyInit
                                   : SpecialFunctionKind::kCopyInitReg,
                      FuncIndex::Copy)))
      return {};
    if (isMemoryOnly) {
      if (failed(setBit("__moveinit__", SpecialFunctionKind::kMoveInit,
                        FuncIndex::Move)))
        return {};
    }
    LookupResult inits =
        shared.lookupAndResolveDecl("__init__", astDecl.getLoc(), astDecl,
                                    /*searchParentScopes=*/false);
    if (inits.isErroneous())
      return {};

    unsigned numFields = std::distance(structDeclOp.getFieldDecls().begin(),
                                       structDeclOp.getFieldDecls().end());
    for (ASTDecl *declaration : inits.getIfSuccess()) {
      auto func = dyn_cast<LIT::FuncOp>(declaration);
      if (!func)
        continue;
      auto signature = func.getSignature();
      ArrayRef<Type> inputTypes = signature.getValueInputs();
      ArrayRef<ValueInputConvention> convs = signature.getInputConventions();
      if (isMemoryOnly) {
        inputTypes = inputTypes.drop_front();
        convs = convs.drop_front();
      }
      // TODO: Handle default arguments.
      if (inputTypes.size() != numFields)
        continue;

      bool isMatch = true;
      for (auto [type, conv, field] :
           llvm::zip(inputTypes, convs, structDeclOp.getFieldDecls())) {
        // Strip the pointer type if present.
        Type argType = type;
        if (conv != ValueInputConvention::OwnedInReg &&
            conv != ValueInputConvention::BorrowedInReg)
          argType = ASTType(argType).getReferenceElementType();
        StructFieldOp op = field;
        if (argType != op.getType()) {
          isMatch = false;
          break;
        }
      }
      if (isMatch)
        existingFunctions[FuncIndex::FieldwiseInit].flip();
    }
    return ValueInfo(existingFunctions);
  }
  bool hasDestructor() const { return existingFunctions[FuncIndex::Destruct]; }
  bool hasMove() const { return existingFunctions[FuncIndex::Move]; }
  bool hasCopy() const { return existingFunctions[FuncIndex::Copy]; }
  bool hasFieldwiseInit() const {
    return existingFunctions[FuncIndex::FieldwiseInit];
  }
  ValueInfo() : initialized(false) {}
  operator bool() const { return initialized; }

private:
  ValueInfo(std::bitset<4> const &existingFunctions)
      : existingFunctions(existingFunctions), initialized(true) {}
  std::bitset<4> existingFunctions;
  bool initialized;
};

std::optional<GeneratedStubs> StructEmitter::addMissingValueMemberStubsToStruct(
    ASTDecl &structDecl, bool generateFieldwiseInit,
    bool forceGenerateDestructor) {
  auto declOp = cast<StructDeclOp>(structDecl);
  ValueInfo valueInfo = ValueInfo::createValueInfo(declOp, shared);
  if (!valueInfo)
    return {};

  bool isMemoryOnly = !declOp.isRegisterPassable();
  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());
  Type selfType = ASTDecl::computeSelfTypeForStruct(declOp);
  Type ptrToSelf = PointerType::get(selfType);
  StringAttr selfName = b.getStringAttr("self");
  StringAttr existingName = b.getStringAttr("other");
  LIT::FuncOp destructorFunc;
  LIT::FuncOp init;
  if (!valueInfo.hasFieldwiseInit() && generateFieldwiseInit) {
    SmallVector<Type> argTypes;
    SmallVector<ValueInputConvention> argConventions;
    SmallVector<StringAttr> argNames;
    if (isMemoryOnly) {
      argTypes.push_back(PointerType::get(selfType));
      argConventions.push_back(ValueInputConvention::InitSelf);
      argNames.push_back(StringAttr::get(shared.getContext(), "self"));
    }
    // We declare all of the operands to the init constructor as owned.  This
    // enables it to work with move-only fields, and, for copyable types, forces
    // the copy into the caller, which can then be elided with a consume or
    // RValue.
    for (auto fieldOp : declOp.getFieldDecls()) {
      ASTType fieldType = fieldOp.getType();
      ValueInputConvention conv;
      switch (fieldType.getRegisterPassability(structDecl.getLoc(), shared)) {
      default:
        llvm_unreachable("unknown case");
      case StructDeclOp::RP_MemoryOnly:
        fieldType = PointerType::get(fieldType);
        conv = ValueInputConvention::OwnedInMem;
        break;
      case StructDeclOp::RP_RegisterPassable:
        conv = ValueInputConvention::OwnedInReg;
        break;
      case StructDeclOp::RP_RegisterPassableTrivial:
        conv = ValueInputConvention::BorrowedInReg;
        break;
      }
      argTypes.push_back(fieldType);
      argConventions.push_back(conv);
      argNames.push_back(fieldOp.getNameAttr());
    }
    SmallVector<PassingKind> argPassingKinds(argNames.size(),
                                             PassingKind::PosOnly);
    init = synthesizeMemberwiseInit(structDecl, argTypes, argConventions,
                                    argNames, argPassingKinds);
  }
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
      destructorFunc = addVoidMethod(
          structDecl, "__del__", /*inputParameters=*/{},
          /*paramPassingKinds=*/{}, ptrToSelf, ValueInputConvention::OwnedInMem,
          selfName, PassingKind::PosOnly, SpecialFunctionKind::kDel);
    }
  }
  LIT::FuncOp copyFunc;
  if (!valueInfo.hasCopy() && declOp.getRegisterPassable() !=
                                  StructDeclOp::RP_RegisterPassableTrivial) {
    if (isMemoryOnly) {
      copyFunc = addVoidMethod(
          structDecl, "__copyinit__", /*inputParameters=*/{},
          /*paramPassingKinds=*/{}, {ptrToSelf, ptrToSelf},
          {ValueInputConvention::InitSelf, ValueInputConvention::BorrowedInMem},
          {selfName, existingName},
          {PassingKind::PosOnly, PassingKind::PosOnly},
          SpecialFunctionKind::kCopyInit);
    } else {
      copyFunc =
          synthesizeMethodInStruct(
              "__copyinit__", /*inputParameters=*/{}, /*paramPassingKinds=*/{},
              selfType, ValueInputConvention::BorrowedInReg, existingName,
              PassingKind::PosOnly, selfType, structDecl,
              SpecialFunctionKind::kCopyInitReg)
              .first;
    }
  }
  LIT::FuncOp moveFunc;
  if (!valueInfo.hasMove() && isMemoryOnly) {
    moveFunc = addVoidMethod(
        structDecl, "__moveinit__", /*inputParameters=*/{},
        /*paramPassingKinds=*/{}, {ptrToSelf, ptrToSelf},
        {ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem},
        {selfName, existingName}, {PassingKind::PosOnly, PassingKind::PosOnly},
        SpecialFunctionKind::kMoveInit);
  }

  return GeneratedStubs(destructorFunc, copyFunc, moveFunc, init);
}

LIT::FuncOp StructEmitter::findInitInStruct(StructDeclOp structOp,
                                            ArrayRef<Type> operands) {
  SpecialFunctionKind initKind;
  unsigned expectedNumInputs;
  if (structOp.isRegisterPassable()) {
    initKind = SpecialFunctionKind::kInitReg;
    expectedNumInputs = operands.size();
  } else {
    initKind = SpecialFunctionKind::kInit;
    expectedNumInputs = operands.size() + 1;
  }

  for (auto candidate : structOp.getOps<LIT::FuncOp>()) {
    SpecialFunctionKind kind =
        (SpecialFunctionKind)candidate.getSpecialFnKind();
    if (kind == initKind &&
        candidate.getBody()->getArguments().size() == expectedNumInputs) {
      bool isMatch = true;
      for (auto [existing, proposed] :
           llvm::zip(structOp.isRegisterPassable()
                         ? candidate.getSignature().getValueInputs()
                         : candidate.getSignature().getValueInputs().slice(1),
                     operands)) {
        if (existing != proposed) {
          isMatch = false;
          break;
        }
      }
      if (isMatch)
        return candidate;
    }
  }
  return {};
}
