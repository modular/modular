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
    bool synthetic) {

  // This starts with implicit lifetimes and then gets explicitly declared input
  // params.
  SmallVector<ParamDeclAttr> fullParams;

  // The caller specifies all the input types, which means that all the input
  // reference types that carry implicit lifetimes will already have them
  // specified with names, so dig those out and use them as parameters.
  // If the caller provided indexed inputs, rewrite them to named inputs as our
  // body will expect.
  SmallVector<Type> adjustedArgTypes;
  for (auto [argNo, argType, argConv] :
       llvm::enumerate(argTypes, argConventions)) {
    adjustedArgTypes.push_back(argType);
    if (!SignatureType::hasImplicitLifetime(argConv))
      continue;

    // Dig out the lifetime decl.
    auto refArgType = cast<RefType>(argType);
    auto lifetimeAttr = refArgType.getLifetime();
    ParamDeclAttr decl;
    // If this is a reference to a named one already, just reuse the name.
    if (auto lifetimeRef = dyn_cast<ParamDeclRefAttr>(
            LifetimeMutCastAttr::strip(lifetimeAttr))) {
      assert(isa<LifetimeType>(lifetimeRef.getType()) &&
             "lifetimes should have LifetimeType");
      // Look through a cast to get the name, but use the expected mutability of
      // the lifetime type.
      decl = ParamDeclAttr::get(lifetimeRef.getName(), lifetimeAttr.getType());
    } else {
      // If this has an indexed value or something else, synthesize a decl.
      auto lifetimeName = StringAttr::get(
          shared.getContext(), llvm::utostr(argNo) + "_unnamed" + "`");
      decl = ParamDeclAttr::get(lifetimeName, lifetimeAttr.getType());

      // Replace the argument type with a named reference.
      auto newLifetime = ParamDeclRefAttr::get(lifetimeName, decl.getType());
      adjustedArgTypes.back() = refArgType.getWithLifetime(newLifetime);
    }
    fullParams.push_back(decl);
  }
  size_t numImplicitLifetimeDecls = fullParams.size();

  auto metadata = FnMetadataAttr::get(argListAttrs, paramListAttrs,
                                      numImplicitLifetimeDecls);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  LITSignatureType signature = SignatureType::remapToSignature(
      params, {}, functionType, argConventions, fnEffects, metadata,
      [&] { return mlir::emitError(location); });
  // Strip off the named lifetime decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FuncOp and
  // in the BBArgs.
  signature = signature.replaceImplicitLifetimesWithIndexes(fullParams);

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

  // Set the attributes on the FuncOp in bulk.
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Figure out the full set of parameter declarations, this is the implicit
  // lifetimes + explicit parameter declarations.
  fullParams.append(params.begin(), params.end());
  if (!fullParams.empty()) {
    attrs.set(funcOp.getParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullParams));
  }
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

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
    Type resultType, ASTDecl &structDecl, SpecialFunctionKind specialFnID,
    FnEffects fnEffects, StringRef suffix, bool synthetic) {
  return synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      argTypes, argConventions, argListAttrs, resultType, structDecl,
      specialFnID, fnEffects, suffix, synthetic);
}

std::pair<LIT::FuncOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> params, PogListAttr paramListAttrs,
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs, Type resultType, ASTDecl &structDecl,
    SpecialFunctionKind specialFnID, FnEffects fnEffects, StringRef suffix,
    bool synthetic) {
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  LIT::FuncOp funcOp = createFunction(
      structDecl, name, params, paramListAttrs, argTypes, argConventions,
      argListAttrs, resultType, specialFnID, structDecl.getLoc(), builder,
      fnEffects, suffix, synthetic);

  // Return null if the function already exists with the same signature.
  if (!funcOp)
    return {nullptr, nullptr};

  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Register the method in the struct.
  ASTDecl &funcDecl = shared.declResolver->addFullyResolvedDecl(
      funcOp.getOperation(), StringAttr::get(shared.getContext(), name),
      structDecl.getLoc(), &structDecl);

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
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), &body);

  // Insert the default end terminator.
  auto terminate = llvm::make_scope_exit([&] { b.create<LIT::EndFuncOp>(); });

  // If the function had an explicit return, just append the default end
  // terminator.
  if (!body.empty() && isa<LIT::ReturnOp, LIT::RaiseOp>(body.back()))
    return;

  auto makeNoneReturn = [&] {
    // A none return either returns None through the SSA output or, in a
    // throwing function, returns 0 as the error state.
    if (sig.isThrows()) {
      ExprEmitter::emitNormalReturn(
          b, b.create<ParamConstantOp>(b.getBoolAttr(false)), funcDecl);
    } else {
      ExprEmitter::emitNormalReturn(
          b, b.create<ParamConstantOp>(shared.getNoneAttr()), funcDecl);
    }
  };

  // Initializers and functions with named results get a default return.
  if (sig.hasInitSelfArg() || func.getNamedResultAttr())
    return makeNoneReturn();

  ASTType resultType = func.getUserResultType();
  ExprEmitter emitter(shared, funcDecl, EC_ReturnValue);
  emitter.builder = b;
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
      ValueDest resultDest(MLValue(func.getArguments().back()), EC_ReturnValue);
      emitter.emitConstructorCall(objType, {}, SyntheticNode(funcDecl.getLoc()),
                                  CallSyntax::kImplicitConvert, resultDest);
      return makeNoneReturn();
    }
  }
}

LIT::FuncOp StructEmitter::synthesizeMemberwiseInit(
    ASTDecl &structDecl, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs) {
  auto structOp = cast<StructDeclOp>(structDecl);

  // Figure out the type of the 'self' argument/result.
  Type resultType = shared.getNoneType();

  // Create the FuncOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argListAttrs, resultType,
      structDecl, SpecialFunctionKind::kInit);
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

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
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Emit a bunch of stores to fields indexing our 'inout self'.
  BlockArgument selfArg = body->getArgument(0);
  assert(isa<RefType>(selfArg.getType()));
  for (auto [idx, field] : llvm::enumerate(structOp.getFieldDecls())) {
    // Add the block argument, get it as an RValue since it is owned. Skip the
    // self argument.
    BlockArgument arg = body->getArgument(idx + 1);
    CValue argVal;
    switch (argConventions[idx + 1]) {
    default:
      llvm_unreachable("unknown convention");
    case ArgConvention::OwnedInReg:
      argVal = SRValue(arg);
      break;
    case ArgConvention::BorrowedInReg:
      argVal = SBValue(arg);
      break;
    case ArgConvention::OwnedInMem:
      argVal = MRValue(arg);
      break;
    case ArgConvention::BorrowedInMem:
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

/// Given a function of the form
/// "lit.func __copyinit__(%target: !lit.ref<@MyStruct, mut ...>, %existing:
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
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(func.getLocScope());
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
      shared.translateLocation(location), func.getBody());
  ExprEmitter emitter(shared, *declScope, b);

  assert(func.getNumArguments() == 2 &&
         "copy functions should have two arguments");
  Value selfArg = func.getBody()->getArgument(0);
  Value existingArg = func.getBody()->getArgument(1);

  // copyinit/moveinit of a register passable value will pass the value as a
  // register, not a reference.
  bool isMemoryOnly = !declOp.isRegisterPassable();
  for (StructFieldOp fieldOp : declOp.getFieldDecls()) {
    auto targetFieldOp = b.create<RefStructGEROp>(selfArg, fieldOp);
    CValue src;
    if (isMemoryOnly) {
      Value srcFieldOp = b.create<RefStructGEROp>(existingArg, fieldOp);
      src = isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));
    } else {
      Value fieldValue = b.create<StructExtractOp>(existingArg, fieldOp);
      // Emit an SBValue -> SRValue conversion to get ownership of the value.
      src = emitter.emitSRValue({SBValue(fieldValue), SyntheticNode(location)},
                                EC_CallArgValue);
      if (!src)
        return failure();
    }
    emitter.emitStoreToLValue({src, SyntheticNode(location)},
                              MLValue(targetFieldOp), EC_AttributeRefBase);
  }
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
LIT::FuncOp StructEmitter::addVoidMethod(ASTDecl &structDecl, StringRef prefix,
                                         ArrayRef<Type> argTypes,
                                         ArrayRef<ArgConvention> argConventions,
                                         PogListAttr argListAttrs,
                                         SpecialFunctionKind kind,
                                         ArrayRef<ParamDeclAttr> params,
                                         PogListAttr paramListAttrs) {
  auto [func, _] = synthesizeMethodInStruct(
      prefix, params, paramListAttrs, argTypes, argConventions, argListAttrs,
      shared.getNoneType(), structDecl, kind);
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

  // Figure out the type of the 'self' argument.  It is the struct's `Self`
  // type for register passable things, or indirect for a memory-only type.
  ASTType selfType = structDecl.getTypeDeclSelf();
  // The argument is always owned.
  ArgConvention convention = ArgConvention::OwnedInReg;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), shared)) {
    selfType = selfType.getRefForArgument("self", /*isMut*/ true);
    convention = ArgConvention::OwnedInMem;
  }

  StringAttr selfName = builder.getStringAttr("self");

  // Create the FuncOp and ASTDecl for the method.
  StructEmitter emitter(shared);
  auto [funcOp, funcDecl] = emitter.synthesizeMethodInStruct(
      "__del__", selfType.mlirType, convention,
      PogListAttr::get(emitter.getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), structDecl, SpecialFunctionKind::kDel);

  // Set up the body.
  Block *body = funcOp.getBody();
  BlockArgument arg = body->getArgument(0);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // We need to make a var box + store for register_passable values since that
  // is what lifetime tracking expects.  It does not track the individual
  // fields of register passable values since they cannot be transferred and
  // cannot be lit.ownership.mark_destroyed.
  if (convention == ArgConvention::OwnedInReg) {
    builder.setInsertionPointToStart(body);
    ExprEmitter emitter(shared, *funcDecl, builder);
    (void)emitter.makeArgLValueVarSlot(SRValue(arg), selfName,
                                       structDecl.getLoc());
  }

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

  // If the type is register passable, the 'existing' value will be passed as
  // a register, otherwise a reference.
  Type existingArgType;
  ArgConvention existingConv;

  if (cast<StructDeclOp>(structDecl).isRegisterPassable()) {
    existingArgType = selfType;
    existingConv =
        isMove ? ArgConvention::OwnedInReg : ArgConvention::BorrowedInReg;
  } else {
    existingArgType = selfType.getRefForArgument("existing", isMove);
    existingConv =
        isMove ? ArgConvention::OwnedInMem : ArgConvention::BorrowedInMem;
  }

  Type selfArgType = selfType.getRefForArgument("self", /*isMut=*/true);
  auto argListAttrs =
      PogListAttr::get(ctx, {b.getStringAttr("self"), existingName},
                       {PassingKind::PosOnly, PassingKind::PosOnly});
  return emitter.addVoidMethod(
      structDecl, name, {selfArgType, existingArgType},
      {ArgConvention::InitSelf, existingConv}, argListAttrs,
      isMove ? SpecialFunctionKind::kMoveInit : SpecialFunctionKind::kCopyInit);
}

LIT::FuncOp StructEmitter::synthesizeEmptyMoveInit(ASTDecl &structDecl) {
  return synthesizeEmptyMoveOrCopyInit(*this, structDecl, /*isMove=*/true);
}

LIT::FuncOp StructEmitter::synthesizeEmptyCopyInit(ASTDecl &structDecl) {
  return synthesizeEmptyMoveOrCopyInit(*this, structDecl, /*isMove=*/false);
}

std::optional<ValueInfo> ValueInfo::createValueInfo(ASTDecl &structDecl,
                                                    SharedState &shared) {
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
    // Drop the 'inout self' argument.
    if (!convs.empty() && convs.front() == ArgConvention::InitSelf) {
      inputTypes = inputTypes.drop_front();
      convs = convs.drop_front();
    }
    if (!convs.empty() && convs.back() == ArgConvention::ByRefError) {
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
      if (SignatureType::hasImplicitLifetime(conv))
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
  std::optional<ValueInfo> valueInfo =
      ValueInfo::createValueInfo(structDecl, shared);
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

    // Add the 'inout self' argument.
    argTypes.push_back(refToSelf);
    argConventions.push_back(ArgConvention::InitSelf);
    argNames.push_back(StringAttr::get(shared.getContext()));
    argPassingKinds.push_back(PassingKind::PosOnly);

    // We declare all of the operands to the init constructor as owned.  This
    // enables it to work with move-only fields, and, for copyable types, forces
    // the copy into the caller, which can then be elided with a consume or
    // RValue.
    for (auto fieldOp : declOp.getFieldDecls()) {
      ASTType fieldType = fieldOp.getType();
      ArgConvention conv;
      switch (fieldType.getRegisterPassability(structDecl.getLoc(), shared)) {
      case TypeConvention::MemoryOnly:
        fieldType = fieldType.getRefForArgument(fieldOp.getName().str(),
                                                /*isMut=*/true);
        conv = ArgConvention::OwnedInMem;
        break;
      case TypeConvention::RegisterPassable:
        conv = ArgConvention::OwnedInReg;
        break;
      case TypeConvention::RegisterPassableTrivial:
        conv = ArgConvention::BorrowedInReg;
        break;
      }
      argTypes.push_back(fieldType);
      argConventions.push_back(conv);
      argNames.push_back(fieldOp.getNameAttr());
      argPassingKinds.push_back(PassingKind::PosOrKw);
    }
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
