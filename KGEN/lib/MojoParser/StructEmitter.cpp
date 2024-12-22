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
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

LIT::FuncOp StructEmitter::createFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel) {
  MLIRContext *ctx = getContext();

  // Figure out the implicit origins.
  SmallVector<ParamDeclAttr> implOriginParams;

  // The caller specifies all the input types, which means that all the input
  // reference types that carry implicit origins will already have them
  // specified with names, so dig those out and use them as parameters.
  // If the caller provided indexed inputs, rewrite them to named inputs as our
  // body will expect.
  SmallVector<Type> adjustedArgTypes;
  for (auto [argNo, argType, argConv] :
       llvm::enumerate(argTypes, argConventions)) {
    adjustedArgTypes.push_back(argType);
    if (!SignatureType::hasImplicitOrigin(argConv))
      continue;

    // Dig out the origin decl.
    auto refArgType = cast<RefType>(argType);
    auto originAttr = refArgType.getOrigin();
    ParamDeclAttr decl;
    // If this is a reference to a named one already, just reuse the name.
    if (auto originRef =
            dyn_cast<ParamDeclRefAttr>(OriginMutCastAttr::strip(originAttr))) {
      assert(isa<OriginType>(originRef.getType()) &&
             "origins should have OriginType");
      // Look through a cast to get the name, but use the expected mutability of
      // the origin type.
      decl = ParamDeclAttr::get(originRef.getName(), originAttr.getType());
    } else {
      // If this has an indexed value or something else, synthesize a decl.
      auto originName =
          StringAttr::get(ctx, llvm::utostr(argNo) + "_unnamed" + "`");
      decl = ParamDeclAttr::get(originName, originAttr.getType());

      // Replace the argument type with a named reference.
      auto newOrigin = ParamDeclRefAttr::get(originName, decl.getType());
      adjustedArgTypes.back() = refArgType.getWithOrigin(newOrigin);
    }
    implOriginParams.push_back(decl);
  }
  size_t numImplicitOriginDecls = implOriginParams.size();

  auto metadata = FnMetadataAttr::get(
      argListAttrs, paramListAttrs, numImplicitOriginDecls,
      getOriginsAccessibleByParams(paramListAttrs, params, shared,
                                   /*captureOrigins=*/nullptr),
      /*isNestedOriginExclusivityCheckingDisabled=*/false);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  LITSignatureType signature = SignatureType::remapToSignature(
      params, {}, functionType, argConventions, fnEffects, metadata,
      [&] { return mlir::emitError(location); });
  // Strip off the named origin decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FuncOp and
  // in the BBArgs.
  signature = signature.replaceImplicitOriginsWithIndexes(implOriginParams);

  StringAttr sourceName = builder.getStringAttr(name);
  StringAttr mangledName = builder.getStringAttr(
      DeclResolver::getMangledName(sourceName, parent, signature).getValue() +
      suffix);

  // If a function with this signature already exists in the struct, don't
  // create a new one. Return null to indicate that there was an existing
  // method.
  if (shared.lookupSymbolIn(&parent, mangledName))
    return nullptr;

  auto funcOp = builder.create<LIT::FuncOp>(mangledName, sourceName, signature,
                                            specialFnID);
  funcOp.setIsSynthetic(true);

  if (funcOp.getSpecialFunctionInfo().isImplicitlyStatic())
    funcOp.setIsStatic(true);

  // Set the attributes on the FuncOp in bulk.
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Figure out the full set of parameter declarations, this is the explicit
  // parameter declarations + implicit origins.
  SmallVector<ParamDeclAttr> fullParams;
  llvm::append_range(fullParams, params);
  llvm::append_range(fullParams, implOriginParams);
  if (!fullParams.empty()) {
    attrs.set(funcOp.getParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullParams));
  }
  attrs.set(funcOp.getIsSyntheticAttrName(), UnitAttr::get(ctx));
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  attrs.set(funcOp.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, inlineLevel));
  funcOp->setAttrs(attrs.getDictionary(ctx));

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(funcOp);
  if (!funcOp.getBody())
    funcOp.getBodyRegion().push_back(new Block());
  for (Type argType : adjustedArgTypes)
    funcOp.getBody()->addArgument(argType, funcOp.getLoc());

  return funcOp;
}

std::pair<LIT::FuncOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, ASTDecl &structDecl, SMLoc loc,
    SpecialFunctionKind specialFnID, FnEffects fnEffects, StringRef suffix,
    bool synthetic) {
  return synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      argTypes, argConventions, argListAttrs, resultType, structDecl, loc,
      specialFnID, fnEffects, suffix, synthetic);
}

std::pair<LIT::FuncOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> params, PogListAttr paramListAttrs,
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs, Type resultType, ASTDecl &structDecl, SMLoc loc,
    SpecialFunctionKind specialFnID, FnEffects fnEffects, StringRef suffix,
    bool synthetic) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  InlineLevel inlineLevel = InlineLevel::Automatic;
  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;
  return synthesizeFunction(structDecl, name, params, paramListAttrs, argTypes,
                            argConventions, argListAttrs, resultType,
                            specialFnID, loc, builder, fnEffects, suffix,
                            synthetic, inlineLevel);
}

std::pair<LIT::FuncOp, ASTDecl *> StructEmitter::synthesizeFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel) {
  LIT::FuncOp funcOp =
      createFunction(parent, name, params, paramListAttrs, argTypes,
                     argConventions, argListAttrs, resultType, specialFnID, loc,
                     builder, fnEffects, suffix, synthetic, inlineLevel);

  // Return null if the function already exists with the same signature.
  if (!funcOp)
    return {nullptr, nullptr};

  // Register the method in the struct.
  ASTDecl &funcDecl = shared.declResolver->addFullyResolvedDecl(
      funcOp.getOperation(), StringAttr::get(shared.getContext(), name), loc,
      &parent);

  // Set the symbol and notice if we are redeclaring something.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, funcDecl);
  assert(!existing && "unexpected redefinition of synthesized method");

  return {funcOp, &funcDecl};
}

void StructEmitter::appendTraits(SmallVectorImpl<TypeLineageAttr> &parentTypes,
                                 ASTDecl *traitDecl) {
  llvm::MapVector<Type, TypeLineageAttr> parentTypeSet;
  for (TypeLineageAttr parent : parentTypes)
    parentTypeSet.insert({parent.getType(), parent});

  auto targetTrait = cast<TraitDeclOp>(traitDecl);
  Type type = targetTrait.bindReference();

  // Add the trait parent if it isn't already there.
  if (!parentTypeSet.insert({type, TypeLineageAttr::get(type)}).second)
    return;

  // Inherit all parent types.
  for (TypeLineageAttr inherited : targetTrait.getParentTypes()) {
    if (auto it = parentTypeSet.find(inherited.getType());
        it != parentTypeSet.end())
      continue;
    SmallVector<Type> lineage = llvm::to_vector(inherited.getInheritedFrom());
    lineage.push_back(type);
    Type parent = inherited.getType();
    parentTypeSet.insert({parent, TypeLineageAttr::get(parent, lineage)});
  }

  for (auto [_, type] : llvm::drop_begin(parentTypeSet, parentTypes.size()))
    parentTypes.push_back(type);
}

void StructEmitter::addTraitParent(StructDeclOp structOp, ASTDecl *traitDecl) {
  SmallVector<TypeLineageAttr> parentTypes =
      llvm::to_vector(structOp.getParentTypes());
  appendTraits(parentTypes, traitDecl);
  structOp.setParentTypes(parentTypes);
}

void StructEmitter::appendDefaultReturnAndEndOp(ASTDecl &funcDecl) {
  auto func = cast<LIT::FuncOp>(funcDecl);
  LITSignatureType sig = func.getSignature();
  Block &body = *func.getBody();
  auto loc = func.getLoc();

  ExprEmitter emitter(funcDecl, OpBuilder::atBlockEnd(&body));

  // If the function had an explicit return, just append the default end
  // terminator.
  if (!body.empty() && isa<LIT::ReturnOp, LIT::RaiseOp>(body.back())) {
    emitter.builder->create<LIT::EndFuncOp>(loc);
    return;
  }

  auto makeNoneReturn = [&] {
    emitter.emitNormalReturn(loc, /*none*/ Value(), /*emitEndFunc=*/true);
  };

  // Functions with named results get a default return.
  // FIXME: This should use register results when possible.
  if (func.getNamedResultAttr())
    return makeNoneReturn();

  ASTType resultType = func.getUserResultType();
  if (resultType.isNoneType()) {
    if (!sig.hasMemoryOnlyResult())
      return makeNoneReturn();

    // Handle functions with memory-only results, which are returned through the
    // result slot.
    ValueDest resultDest(MLValue(func.getArguments().back()), EC_ReturnValue);
    emitter.emitResult(PValue(shared.getNoneAttr()),
                       SyntheticNode(funcDecl.getLoc()), resultDest);
    return makeNoneReturn();
  }

  // `def foo():` will return a None object by default.
  if (func.isDef()) {
    ASTType objType = shared.lookupObjectType(funcDecl, funcDecl.getLoc());
    if (objType.isEqualCanon(resultType) && func.getNumArguments()) {
      // Emit `object()` into the memory type return slot.
      // TODO: Use an expr form of result emission.
      ValueDest resultDest(MLValue(func.getArguments().back()), EC_ReturnValue);
      emitter.emitConstructorCall(objType, {}, SyntheticNode(funcDecl.getLoc()),
                                  CallSyntax::kTypeCall, resultDest);
      return makeNoneReturn();
    }
  }

  // Otherwise, just fall off the end.
  emitter.builder->create<LIT::EndFuncOp>(loc);
}

LIT::FuncOp StructEmitter::synthesizeMemberwiseInit(
    ASTDecl &structDecl, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs) {
  auto structOp = cast<StructDeclOp>(structDecl);

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argListAttrs, shared.getNoneType(),
      structDecl, structDecl.getLoc(), SpecialFunctionKind::kInit);
  assert(funcOp && "couldn't synthesize method or had a conflict?");
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Set up the body.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  builder.setLoc(funcOp->getLoc());
  ASTDecl *funcDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(funcOp));
  ExprEmitter emitter(*funcDecl, builder);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Emit a bunch of stores to fields indexing our 'out self' result.
  BlockArgument selfArg = body->getArgument(body->getNumArguments() - 1);
  assert(isa<RefType>(selfArg.getType()));
  for (auto [idx, field] : llvm::enumerate(structOp.getFieldDecls())) {
    // Add the block argument, get it as an RValue since it is owned. Skip the
    // self argument.
    BlockArgument arg = body->getArgument(idx);
    CValue argVal;
    switch (argConventions[idx]) {
    default:
      llvm_unreachable("unknown convention");
    case ArgConvention::ReadReg:
      argVal = SRValue(arg);
      break;
    case ArgConvention::OwnedMem:
      argVal = MRValue(arg);
      break;
    case ArgConvention::ReadMem:
      argVal = MBValue(arg);
      break;
    }

    // Project self to the right field and store the RValue.
    auto fieldRef = builder.create<RefStructGEROp>(selfArg, field);
    emitter.emitStoreToLValue({argVal, SyntheticNode(structDecl.getLoc())},
                              MLValue(fieldRef), EC_AttributeRefBase);
  }

  // Finish off the function with a return + lit.endfunc.
  emitter.emitNormalReturn(funcOp.getLoc(), /*none*/ Value(),
                           /*emitEndFunc=*/true);
  return funcOp;
}

/// Given a function of the form
///    fn __copyinit__(existing: MyStruct, out self: MyStruct)
/// populate the method with the following:
///   %targetField0Ptr = lit.ref.struct.ger %self[field0]
///   %sourceField0Ptr = lit.ref.struct.ger %existing[field0]
///   copyinit_of_type_of_field0(%targetField0, %field)
LogicalResult StructEmitter::populateMoveCopy(ASTDecl &functionDecl,
                                              bool isMove) {
  auto func = cast<LIT::FuncOp>(functionDecl);
  ASTDecl *declScope = functionDecl.getParentDecl();
  StructDeclOp declOp = cast<StructDeclOp>(declScope);

  // We want to populate a move but the move/copy should be a method!
  SMLoc location = functionDecl.getLoc();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(func.getLocScope());
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
      shared.translateLocation(location), func.getBody());
  ExprEmitter emitter(*declScope, b);

  assert(func.getNumArguments() == 2 &&
         "copy and move functions should have two arguments");
  Value existingArg = func.getBody()->getArgument(0);
  Value selfArg = func.getBody()->getArgument(1);

  // copyinit/moveinit of a register passable value will pass the value as a
  // register, not a reference.
  bool isMemory = !declOp.isRegisterPassableTrivial();
  for (StructFieldOp fieldOp : declOp.getFieldDecls()) {
    auto targetFieldOp = b.create<RefStructGEROp>(selfArg, fieldOp);
    CValue src;
    if (isMemory) {
      Value srcFieldOp = b.create<RefStructGEROp>(existingArg, fieldOp);
      src = isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));
    } else {
      // The value is trivial, so no copy ctor is needed.
      src = SRValue(b.create<StructExtractOp>(existingArg, fieldOp));
    }
    emitter.emitStoreToLValue({src, SyntheticNode(location)},
                              MLValue(targetFieldOp), EC_AttributeRefBase);
  }
  return success();
}

/// Given a struct and a list of arguments, generate a function. For example,
/// given {
///  MyStruct, "prefix", [ParamType1, ParamType2],
///  [read_mem, read_mem], ["x","b"]
/// }, this function produces:
///
/// ```
/// lit.func @prefixParam1Param2(%x: ParamType1 read_mem,
///    %b : ParamType2 read_mem, %self: !kgen.pointer<@MyStruct> byref_result
/// ) -> !kgen.none  {
///   %0 = kgen.param.constant: none = <#kgen.none>
///   lit.return %0 : !kgen.none
///   lit.end_func
/// }
/// ```
LIT::FuncOp StructEmitter::addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                                         ArrayRef<Type> argTypes,
                                         ArrayRef<ArgConvention> argConventions,
                                         PogListAttr argListAttrs,
                                         SpecialFunctionKind kind,
                                         ArrayRef<ParamDeclAttr> params,
                                         PogListAttr paramListAttrs) {
  auto [func, _] = synthesizeMethodInStruct(
      prefix, params, paramListAttrs, argTypes, argConventions, argListAttrs,
      shared.getNoneType(), structDecl, structDecl.getLoc(), kind);
  Block *body = func.getBody();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(func.getLocScope());

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), body);
  ExprEmitter::emitNormalReturn(b, b.create<ParamConstantOp>(noneAttr), func);
  b.create<LIT::EndFuncOp>();
  return func;
}

LIT::FuncOp StructEmitter::addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                                         ArrayRef<Type> argTypes,
                                         ArrayRef<ArgConvention> argConventions,
                                         PogListAttr argListAttrs,
                                         SpecialFunctionKind kind) {

  return addVoidMethod(structDecl, prefix, argTypes, argConventions,
                       argListAttrs, kind, /*params=*/{},
                       /*paramListAttrs=*/PogListAttr::get(getContext()));
}

LIT::FuncOp StructEmitter::synthesizeEmptyDtor(ASTDecl &structDecl) {
  auto structOp = cast<StructDeclOp>(structDecl);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());

  // Figure out the type of the 'self' argument, which is always indirect since
  // it is owned.
  ASTType selfType = structDecl.getTypeDeclSelf();
  selfType = selfType.getRefForArgument("self", /*isMut*/ true);
  StringAttr selfName = builder.getStringAttr("self");

  // Create the FuncOp and ASTDecl for the method.
  StructEmitter emitter(shared);
  auto [funcOp, funcDecl] = emitter.synthesizeMethodInStruct(
      "__del__", selfType.mlirType, ArgConvention::OwnedMem,
      PogListAttr::get(emitter.getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), structDecl, structDecl.getLoc(),
      SpecialFunctionKind::kDel);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Finish off the function with a return + lit.endfunc.
  appendDefaultReturnAndEndOp(*funcDecl);
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
  return funcOp;
}

static LIT::FuncOp synthesizeEmptyMoveOrCopyInit(StructEmitter &emitter,
                                                 ASTDecl &structDecl,
                                                 bool isMove) {
  ASTType selfType = structDecl.getTypeDeclSelf();
  StringRef name = isMove ? "__moveinit__" : "__copyinit__";
  MLIRContext *ctx = emitter.shared.getContext();
  Builder b(ctx);
  StringAttr existingName = b.getStringAttr("other");

  // If the type is register passable trivial, the 'existing' value will be
  // passed as a register, otherwise a reference.
  Type existingArgType;
  ArgConvention existingConv;
  if (cast<StructDeclOp>(structDecl).isRegisterPassableTrivial() && !isMove) {
    existingArgType = selfType;
    existingConv = ArgConvention::ReadReg;
  } else {
    existingArgType = selfType.getRefForArgument("existing", isMove);
    existingConv = isMove ? ArgConvention::OwnedMem : ArgConvention::ReadMem;
  }

  Type selfArgType = selfType.getRefForArgument("self", /*isMut=*/true);
  auto argListAttrs =
      PogListAttr::get(ctx, {existingName, b.getStringAttr("self")},
                       {PassingKind::PosOnly, PassingKind::Implicit});
  return emitter.addVoidMethod(
      structDecl, name, {existingArgType, selfArgType},
      {existingConv, ArgConvention::ByRefResult}, argListAttrs,
      isMove ? SpecialFunctionKind::kMoveInit : SpecialFunctionKind::kCopyInit);
}

LIT::FuncOp StructEmitter::synthesizeEmptyMoveInit(ASTDecl &structDecl) {
  return synthesizeEmptyMoveOrCopyInit(*this, structDecl, /*isMove=*/true);
}

LIT::FuncOp StructEmitter::synthesizeEmptyCopyInit(ASTDecl &structDecl) {
  return synthesizeEmptyMoveOrCopyInit(*this, structDecl, /*isMove=*/false);
}

std::optional<ValueInfo> ValueInfo::createValueInfo(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();
  std::bitset<4> existingFunctions;
  existingFunctions.reset();
  auto structOp = cast<StructDeclOp>(structDecl);

  auto setBit = [&](StringRef name, SpecialFunctionKind kind,
                    unsigned index) -> LogicalResult {
    LookupResult lookupResult =
        shared.lookupAndResolveDecl(name, structDecl.getLoc(), structDecl,
                                    /*searchParentScopes=*/false);
    if (!lookupResult.isSuccess())
      return success();
    if (lookupResult.getIfSuccess().size() > 1)
      return shared.emitError(structOp.getLoc())
             << "multiple overloaded methods named '" << name << "'";

    if (lookupResult.getIfSuccess().size() == 1) {
      ASTDecl *result = lookupResult.getIfSuccess().front();
      if (auto func = dyn_cast<LIT::FuncOp>(result))
        if ((SpecialFunctionKind)func.getSpecialFnKind() == kind)
          existingFunctions[index].flip();
    }

    return success();
  };
  if (failed(setBit("__del__", SpecialFunctionKind::kDel, FuncIndex::Destruct)))
    return {};
  if (failed(setBit("__copyinit__", SpecialFunctionKind::kCopyInit,
                    FuncIndex::Copy)))
    return {};
  if (failed(setBit("__moveinit__", SpecialFunctionKind::kMoveInit,
                    FuncIndex::Move)))
    return {};
  LookupResult inits =
      shared.lookupAndResolveDecl("__init__", structDecl.getLoc(), structDecl,
                                  /*searchParentScopes=*/false);
  if (inits.isErroneous())
    return {};

  unsigned numFields = std::distance(structOp.getFieldDecls().begin(),
                                     structOp.getFieldDecls().end());
  for (ASTDecl *declaration : inits.getIfSuccess()) {
    auto func = dyn_cast<LIT::FuncOp>(declaration);
    if (!func)
      continue;
    auto signature = func.getSignature();
    ArrayRef<Type> inputTypes = signature.getArguments();
    ArrayRef<ArgConvention> convs = signature.getArgConventions();
    // Ignore the result slot and error result.
    while (!convs.empty() && SignatureType::isResultSlot(convs.back())) {
      inputTypes = inputTypes.drop_back();
      convs = convs.drop_back();
    }
    // TODO: Handle default arguments.
    if (inputTypes.size() != numFields)
      continue;
    // Skip any kind of var-args.
    FnMetadataAttr metadata = signature.getMetadata();
    if (metadata.hasVarArgs() || metadata.hasPackVarArgs())
      continue;

    bool isMatch = true;
    for (auto [type, conv, field] :
         llvm::zip(inputTypes, convs, structOp.getFieldDecls())) {
      // Strip the pointer type if present.
      Type argType = type;
      if (SignatureType::hasImplicitOrigin(conv))
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

std::optional<GeneratedStubs> StructEmitter::addMissingValueMemberStubsToStruct(
    ASTDecl &structDecl, bool generateFieldwiseInit,
    bool forceGenerateDestructor) {
  auto declOp = cast<StructDeclOp>(structDecl);
  std::optional<ValueInfo> valueInfo = ValueInfo::createValueInfo(structDecl);
  if (!valueInfo)
    return {};

  OpBuilder b(&declOp.getFields().front(), declOp.getFields().front().end());

  ASTType selfType = structDecl.getTypeDeclSelf();
  Type refToSelf = selfType.getRefForArgument("self", /*isMut=*/true);

  LIT::FuncOp destructorFunc;
  LIT::FuncOp init;
  if (!valueInfo->hasFieldwiseInit() && generateFieldwiseInit) {
    SmallVector<Type> argTypes;
    SmallVector<ArgConvention> argConventions;
    SmallVector<StringAttr> argNames;
    SmallVector<PassingKind> argPassingKinds;

    // We declare all of the operands to the init constructor as owned.  This
    // enables it to work with move-only fields, and, for copyable types, forces
    // the copy into the caller, which can then be elided with a consume or
    // RValue.
    for (auto fieldOp : declOp.getFieldDecls()) {
      ASTType fieldType = fieldOp.getType();
      ArgConvention conv;
      switch (fieldType.getRegisterPassability(structDecl.getLoc(), shared)) {
      case TypeConvention::MemoryOnly:
      case TypeConvention::Unspecified:
      case TypeConvention::RegisterPassable:
        fieldType = fieldType.getRefForArgument(fieldOp.getName().str(),
                                                /*isMut=*/true);
        conv = ArgConvention::OwnedMem;
        break;
      case TypeConvention::RegisterPassableTrivial:
        conv = ArgConvention::ReadReg;
        break;
      }
      argTypes.push_back(fieldType);
      argConventions.push_back(conv);
      argNames.push_back(fieldOp.getNameAttr());
      argPassingKinds.push_back(PassingKind::PosOrKw);
    }

    // Add the 'out self' argument.
    argTypes.push_back(refToSelf);
    argConventions.push_back(ArgConvention::ByRefResult);
    argNames.push_back(StringAttr::get(shared.getContext(), "self"));
    argPassingKinds.push_back(PassingKind::Implicit);

    init = synthesizeMemberwiseInit(
        structDecl, argTypes, argConventions,
        PogListAttr::get(getContext(), argNames, argPassingKinds));
  }

  if (!valueInfo->hasDestructor() && forceGenerateDestructor)
    destructorFunc = synthesizeEmptyDtor(structDecl);

  auto addCopyOrMoveBuiltinTrait = [&](bool isCopy) {
    ASTDecl *traitDecl = shared.lookupBuiltinTrait(
        isCopy ? "Copyable" : "Movable", structDecl.getParentDecl(),
        structDecl.getLoc());
    if (traitDecl)
      addTraitParent(declOp, traitDecl);
  };

  LIT::FuncOp copyFunc;
  if (!valueInfo->hasCopy() && !declOp.isRegisterPassableTrivial())
    copyFunc = synthesizeEmptyCopyInit(structDecl);
  addCopyOrMoveBuiltinTrait(/*isCopy=*/true);

  LIT::FuncOp moveFunc;
  if (!valueInfo->hasMove() && !declOp.isRegisterPassable())
    moveFunc = synthesizeEmptyMoveInit(structDecl);
  addCopyOrMoveBuiltinTrait(/*isCopy=*/false);

  return GeneratedStubs{destructorFunc, copyFunc, moveFunc, init};
}

LIT::FuncOp StructEmitter::findInitInStruct(StructDeclOp structOp,
                                            ArrayRef<Type> operands) {
  size_t expectedNumInputs = operands.size() + 1;

  for (auto candidate : structOp.getOps<LIT::FuncOp>()) {
    SpecialFunctionKind kind = candidate.getSpecialFunctionKind();
    if (kind != SpecialFunctionKind::kInit ||
        candidate.getBody()->getArguments().size() != expectedNumInputs)
      continue;

    bool isMatch = true;
    for (auto [existing, proposed] : llvm::zip(
             candidate.getSignature().getArguments().slice(1), operands)) {
      if (existing != proposed) {
        isMatch = false;
        break;
      }
    }
    if (isMatch)
      return candidate;
  }
  return {};
}
