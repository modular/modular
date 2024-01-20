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
#include "KGEN/LITDialect/LITUtils.h"
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
    StringRef name, ArrayRef<ParamDeclAttr> inputParams,
    ArrayRef<PassingKind> paramPassingKinds, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef prefix) {
  // If the result of the function is a non-trivial type, mark the function
  // effect as having an owned result so ownership tracking will notice it.
  if (ASTType(resultType).getRegisterPassability(loc, shared) !=
      TypeConvention::RegisterPassableTrivial)
    fnEffects.setOwnedRegisterResult();

  // This starts with implicit lifetimes and then gets explicitly declared input
  // params.
  SmallVector<ParamDeclAttr> fullInputParams;

  // The caller specifies all the input types, which means that all the input
  // reference types that carry implicit lifetimes will already have them
  // specified with names, so dig those out and use them as input parameters.
  // If the caller provided indexed inputs, rewrite them to named inputs as our
  // body will expect.
  SmallVector<Type> adjustedArgTypes;
  for (auto [argNo, argType, argConv] :
       llvm::enumerate(argTypes, argConventions)) {
    adjustedArgTypes.push_back(argType);
    if (!SignatureType::hasAddress(argConv))
      continue;

    // Dig out the lifetime decl.
    auto refArgType = cast<RefType>(argType);
    auto lifetimeAttr = refArgType.getLifetime();
    // If this has an indexed value, synthesize a decl.
    ParamDeclAttr decl;
    if (isa<ImplicitLifetimeRefAttr>(lifetimeAttr)) {
      auto lifetimeName = StringAttr::get(
          shared.getContext(), llvm::utostr(argNo) + "_unnamed" + "`");
      decl = ParamDeclAttr::get(lifetimeName, shared.getLifetimeType());

      // Replace the argument type with a named reference.
      auto newLifetime = ParamDeclRefAttr::get(lifetimeName, decl.getType());
      adjustedArgTypes.back() = refArgType.getWithLifetime(newLifetime);
    } else {
      // If this is a reference to a named one already, just reuse the name.
      auto lifetimeRef = cast<ParamDeclRefAttr>(lifetimeAttr);
      assert(isa<LifetimeType>(lifetimeRef.getType()) &&
             "lifetimes should have LifetimeType");
      decl = ParamDeclAttr::get(lifetimeRef);
    }
    fullInputParams.push_back(decl);
  }
  size_t numImplicitLifetimeDecls = fullInputParams.size();

  SmallVector<StringAttr> parameterNames;
  for (ParamDeclAttr p : inputParams) {
    parameterNames.push_back(
        StringAttr::get(getContext(), demangleParameterName(p.getName())));
  }

  auto metadata = FnMetadataAttr::get(
      builder.getContext(), argNames, argPassingKinds, parameterNames,
      paramPassingKinds, /*defaultPosArgs=*/{}, /*defaultPosParams=*/{},
      /*defaultKwOnlyArgs=*/{}, /*defaultKwOnlyParams=*/{},
      numImplicitLifetimeDecls);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  LITSignatureType signature = SignatureType::remapToSignature(
      inputParams, {}, functionType, argConventions, fnEffects, metadata,
      [&] { return mlir::emitError(location); });
  // Strip off the named lifetime decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FuncOp and
  // in the BBArgs.
  signature = signature.replaceImplicitLifetimesWithIndexes(fullInputParams);

  StringAttr sourceName = builder.getStringAttr(name);
  StringAttr mangledName = builder.getStringAttr(
      prefix + DeclResolver::getMangledName(sourceName, signature).getValue());
  auto funcOp = builder.create<LIT::FuncOp>(mangledName, sourceName, signature,
                                            specialFnID);
  funcOp.setIsSynthetic(true);

  // Set the attributes on the FuncOp in bulk.
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Figure out the full set of parameter declarations, this is the implicit
  // lifetimes + explicit parameter declarations.
  fullInputParams.append(inputParams.begin(), inputParams.end());
  if (!fullInputParams.empty()) {
    attrs.set(funcOp.getInputParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullInputParams));
  }
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  shared.setLocationDebugScope(diScopeGuard, funcOp);
  if (!funcOp.getBody())
    funcOp.getBodyRegion().push_back(new Block());
  for (Type argType : adjustedArgTypes)
    funcOp.getBody()->addArgument(argType, funcOp.getLoc());

  return funcOp;
}

std::pair<LIT::FuncOp, ASTDecl &> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    Type resultType, ASTDecl &structDecl, SpecialFunctionKind specialFnID,
    FnEffects effects, StringRef prefix) {
  return synthesizeMethodInStruct(
      name, /*inputParams=*/{}, /*paramPassingKinds=*/{}, argTypes,
      argConventions, argNames, argPassingKinds, resultType, structDecl,
      specialFnID, effects, prefix);
}

std::pair<LIT::FuncOp, ASTDecl &> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> inputParams,
    ArrayRef<PassingKind> paramPassingKinds, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    Type resultType, ASTDecl &structDecl, SpecialFunctionKind specialFnID,
    FnEffects effects, StringRef prefix) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  LIT::FuncOp funcOp = createFunction(
      name, inputParams, paramPassingKinds, argTypes, argConventions, argNames,
      argPassingKinds, resultType, specialFnID, structDecl.getLoc(), builder,
      effects.setParamVarArgs(effects.hasParamVarArgs() ||
                              structOp.getSignature().getParamVarArg()),
      prefix);

  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getConvention() == TypeConvention::RegisterPassableTrivial)
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

void StructEmitter::addTraitParent(StructDeclOp structOp, ASTDecl *traitDecl) {
  SmallVector<TypeLineageAttr> parentTypes =
      llvm::to_vector(structOp.getParentTypes());

  // Helper to detect if a parent type is already in the list.
  auto hasParent = [&](TypeLineageAttr parent) {
    return llvm::find_if(parentTypes, [&](TypeLineageAttr type) {
             return type.getType() == parent.getType();
           }) != parentTypes.end();
  };

  auto targetTrait = cast<TraitDeclOp>(traitDecl);
  TypeLineageAttr traitType = TypeLineageAttr::get(targetTrait.bindReference());

  // Add the parent type and all transitive parents. Typical use case is
  // Movable, which has only 1 parent.
  if (hasParent(traitType))
    return;
  parentTypes.push_back(traitType);
  // If trait was added, add parents as well.
  for (TypeLineageAttr parentTrait : targetTrait.getParentTypes())
    if (!hasParent(parentTrait))
      parentTypes.push_back(parentTrait);

  structOp.setParentTypes(parentTypes);
}

LIT::FuncOp StructEmitter::synthesizeMemberwiseInit(
    ASTDecl &structDecl, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds) {
  auto structOp = cast<StructDeclOp>(structDecl);
  ASTType selfType = structDecl.getSelfType();
  bool isMemoryOnly = !structOp.isRegisterPassable();

  // Figure out the type of the 'self' argument/result.
  Type resultType = isMemoryOnly ? shared.getNoneType() : selfType;

  auto specialFnId =
      isMemoryOnly ? SpecialFunctionKind::kInit : SpecialFunctionKind::kInitReg;

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argNames, argPassingKinds,
      resultType, structDecl, specialFnId);

  // Set up the body.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  builder.setLoc(funcOp->getLoc());
  ASTDecl *funcDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(funcOp));
  ExprEmitter emitter(shared, *funcDecl, builder);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = funcOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // For a memory-only initializer, we emit a bunch of stores to fields indexing
  // self.
  if (isMemoryOnly) {
    BlockArgument selfArg = body->getArgument(0);
    assert(selfArg.getType().isa<RefType>());
    for (auto [idx, field] : llvm::enumerate(structOp.getFieldDecls())) {
      // Add the block argument, get it as an RValue since it is owned. Skip the
      // self argument.
      BlockArgument arg = body->getArgument(idx + 1);
      CValue argVal;
      switch (argConventions[idx + 1]) {
      default:
        llvm_unreachable("unknown convention");
      case ValueInputConvention::OwnedInReg:
        argVal = SRValue(arg);
        break;
      case ValueInputConvention::BorrowedInReg:
        argVal = SBValue(arg);
        break;
      case ValueInputConvention::OwnedInMem:
        argVal = MRValue(arg);
        break;
      case ValueInputConvention::BorrowedInMem:
        argVal = MBValue(arg);
        break;
      }

      // Project self to the right field and store the RValue.
      auto fieldRef = builder.create<RefStructGEROp>(selfArg, field);
      emitter.emitStoreToLValue({argVal, SyntheticNode(structDecl.getLoc())},
                                MLValue(fieldRef), EC_AttributeRefBase);
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
/// "lit.func __copyinit__(%target: !lit.ref<mut @MyStruct, ...>, %existing:
/// !lit.ref<@MyStruct, ...>), populate the method with the following:
/// %targetField0Ptr = lit.ref.struct.ger %self[field0]
/// %sourceField0Ptr = lit.ref.struct.ger %existing[field0]
/// copyinit_of_type_of_field0(%targetField0, %field
LogicalResult StructEmitter::populateMoveCopy(ASTDecl &functionDecl,
                                              bool isMove) {
  auto func = cast<LIT::FuncOp>(functionDecl);
  ASTDecl *declScope = functionDecl.getParentDecl();
  StructDeclOp declOp = cast<StructDeclOp>(declScope);

  // We want to populate a move but the move/copy should be a method!
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
      auto targetFieldOp = b.create<RefStructGEROp>(copySelf, fieldOp);
      Value srcFieldOp = b.create<RefStructGEROp>(copyExisting, fieldOp);
      CValue src =
          isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));
      emitter.emitStoreToLValue({src, SyntheticNode(location)},
                                MLValue(targetFieldOp), EC_AttributeRefBase);
    }
    return success();
  }
  assert(func.getNumArguments() == 1 &&
         "copy functions of register passable types should have one argument");
  // Otherwise, extract all the values and finish with a struct create.  We
  // know all the subfields must be register passable.
  BlockArgument existingArg = func.getBody()->getArgument(0);
  SmallVector<Value> fieldVals;
  SmallVector<StringAttr> fieldNames;
  for (StructFieldOp fieldOp : declOp.getFieldDecls()) {
    Value fieldValue = b.create<StructExtractOp>(existingArg, fieldOp);
    // Emit an SBValue -> SRValue conversion to get ownership of the value.
    Value copiedVal = emitter.emitSRValue(
        {SBValue(fieldValue), SyntheticNode(location)}, EC_CallArgValue);
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
    ASTDecl &structDecl, StringRef prefix, ArrayRef<Type> argTypes,
    ArrayRef<ValueInputConvention> argConventions,
    ArrayRef<StringAttr> argNames, ArrayRef<PassingKind> argPassingKinds,
    SpecialFunctionKind kind, ArrayRef<ParamDeclAttr> inputParams,
    ArrayRef<PassingKind> paramPassingKinds) {
  auto [func, _] = synthesizeMethodInStruct(
      prefix, inputParams, paramPassingKinds, argTypes, argConventions,
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
    bool isMemoryOnly = !structDeclOp.isRegisterPassable();
    if (failed(
            setBit("__del__", SpecialFunctionKind::kDel, FuncIndex::Destruct)))
      return {};
    if (failed(setBit("__copyinit__",
                      isMemoryOnly ? SpecialFunctionKind::kCopyInit
                                   : SpecialFunctionKind::kCopyInitReg,
                      FuncIndex::Copy)))
      return {};
    if (failed(setBit("__moveinit__",
                      isMemoryOnly ? SpecialFunctionKind::kMoveInit
                                   : SpecialFunctionKind::kMoveInitReg,
                      FuncIndex::Move)))
      return {};
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
      // Skip any kind of var-args.
      FnEffects effects = signature.getFnEffects();
      if (effects.hasAnyVarArgs() || effects.hasKWVarArgs())
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
  Type refToSelf = ASTType(selfType).getRefForArgument("self", /*isMut=*/true);
  StringAttr selfName = b.getStringAttr("self");
  StringAttr existingName = b.getStringAttr("other");
  LIT::FuncOp destructorFunc;
  LIT::FuncOp init;
  if (!valueInfo.hasFieldwiseInit() && generateFieldwiseInit) {
    SmallVector<Type> argTypes;
    SmallVector<ValueInputConvention> argConventions;
    SmallVector<StringAttr> argNames;
    SmallVector<PassingKind> argPassingKinds;
    if (isMemoryOnly) {
      argTypes.push_back(refToSelf);
      argConventions.push_back(ValueInputConvention::InitSelf);
      argNames.push_back(StringAttr::get(shared.getContext()));
      argPassingKinds.push_back(PassingKind::PosOnly);
    }
    // We declare all of the operands to the init constructor as owned.  This
    // enables it to work with move-only fields, and, for copyable types, forces
    // the copy into the caller, which can then be elided with a consume or
    // RValue.
    for (auto fieldOp : declOp.getFieldDecls()) {
      ASTType fieldType = fieldOp.getType();
      ValueInputConvention conv;
      switch (fieldType.getRegisterPassability(structDecl.getLoc(), shared)) {
      case TypeConvention::MemoryOnly:
        fieldType = fieldType.getRefForArgument(fieldOp.getName().str(),
                                                /*isMut=*/true);
        conv = ValueInputConvention::OwnedInMem;
        break;
      case TypeConvention::RegisterPassable:
        conv = ValueInputConvention::OwnedInReg;
        break;
      case TypeConvention::RegisterPassableTrivial:
        conv = ValueInputConvention::BorrowedInReg;
        break;
      }
      argTypes.push_back(fieldType);
      argConventions.push_back(conv);
      argNames.push_back(fieldOp.getNameAttr());
      argPassingKinds.push_back(PassingKind::PosOrKw);
    }
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

    // FIXME: This isn't using the logic for synthesizeEmptyDtor, not handling
    // register passable correctly.  We just decline to generate this for now,
    // which threads the needle between closure emission (which expects us to
    // synthesize all members but only generates memory only members) and
    // @value generation which doesn't need a del method because struct type
    // checking will add it.
    //
    // We should probably move synthesizeEmptyDtor into this code and use it
    // from the type checking logic.
    if (needsDtor && isMemoryOnly) {
      destructorFunc = addVoidMethod(
          structDecl, "__del__", refToSelf, ValueInputConvention::OwnedInMem,
          selfName, PassingKind::PosOnly, SpecialFunctionKind::kDel);
    }
  }

  auto addCopyOrMoveBuiltinTrait = [&](bool isCopy) {
    ASTDecl *traitDecl;
    if (isCopy) {
      traitDecl = shared.lookupCopyableTrait(structDecl.getLoc(),
                                             structDecl.getParentDecl());
    } else {
      traitDecl = shared.lookupMovableTrait(structDecl.getLoc(),
                                            structDecl.getParentDecl());
    }
    if (traitDecl)
      addTraitParent(declOp, traitDecl);
  };

  LIT::FuncOp copyFunc;
  if (!valueInfo.hasCopy()) {
    addCopyOrMoveBuiltinTrait(/*isCopy=*/true);
    if (isMemoryOnly) {
      Type refToExisting =
          ASTType(selfType).getRefForArgument("existing", /*isMut=*/false);
      copyFunc = addVoidMethod(
          structDecl, "__copyinit__", {refToSelf, refToExisting},
          {ValueInputConvention::InitSelf, ValueInputConvention::BorrowedInMem},
          {selfName, existingName},
          {PassingKind::PosOnly, PassingKind::PosOnly},
          SpecialFunctionKind::kCopyInit);
    } else {
      copyFunc = synthesizeMethodInStruct("__copyinit__", selfType,
                                          ValueInputConvention::BorrowedInReg,
                                          existingName, PassingKind::PosOnly,
                                          selfType, structDecl,
                                          SpecialFunctionKind::kCopyInitReg)
                     .first;
    }
  }
  addCopyOrMoveBuiltinTrait(/*isCopy=*/true);
  LIT::FuncOp moveFunc;
  addCopyOrMoveBuiltinTrait(/*isCopy=*/false);
  if (!valueInfo.hasMove()) {
    if (isMemoryOnly) {
      addCopyOrMoveBuiltinTrait(/*isCopy=*/false);
      Type refToExisting =
          ASTType(selfType).getRefForArgument("existing", /*isMut=*/true);
      moveFunc = addVoidMethod(
          structDecl, "__moveinit__", {refToSelf, refToExisting},
          {ValueInputConvention::InitSelf, ValueInputConvention::OwnedInMem},
          {selfName, existingName},
          {PassingKind::PosOnly, PassingKind::PosOnly},
          SpecialFunctionKind::kMoveInit);
    } else {
      addCopyOrMoveBuiltinTrait(/*isCopy=*/false);
      moveFunc = synthesizeMethodInStruct(
                     "__moveinit__", selfType, ValueInputConvention::OwnedInReg,
                     existingName, PassingKind::PosOnly, selfType, structDecl,
                     SpecialFunctionKind::kMoveInitReg)
                     .first;
    }
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
