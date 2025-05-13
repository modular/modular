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
#include "MojoUtils.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

FnOp StructEmitter::createFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel) {
  MLIRContext *ctx = getContext();

  // Figure out the implicit origins we'll need to add.
  std::vector<ParamDeclAttr> newOriginParamDecls;
  llvm::MapVector<ImplicitOriginRefAttr, ParamDeclRefAttr>
      implicitOriginToNewParamRef;

  struct ImplicitOriginRefAttrReplacer
      : IndexParameterReplacer<ImplicitOriginRefAttrReplacer> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto implicitOriginRef = ::dyn_cast<ImplicitOriginRefAttr>(attr);
          implicitOriginRef && implicitOriginRef.getDepth() == depth) {

        auto iter = implicitOriginToNewParamRef->find(implicitOriginRef);
        if (iter != implicitOriginToNewParamRef->end()) {
          return iter->second;
        }
        auto newOriginNum = newOriginParamDecls->size();
        auto newOriginNameStr = llvm::utostr(newOriginNum) + "_unnamed" + "`";
        auto newOriginName = StringAttr::get(ctx, newOriginNameStr);
        auto newOriginDecl =
            ParamDeclAttr::get(newOriginName, implicitOriginRef.getType());
        newOriginParamDecls->push_back(newOriginDecl);
        // Replace the implicit origin ref with a named param decl ref.
        auto originParamRef =
            ParamDeclRefAttr::get(newOriginName, implicitOriginRef.getType());
        implicitOriginToNewParamRef->insert(
            {implicitOriginRef, originParamRef});
        return originParamRef;
      }
      return nullptr;
    }

    MLIRContext *ctx;
    std::vector<ParamDeclAttr> *newOriginParamDecls;
    llvm::MapVector<ImplicitOriginRefAttr, ParamDeclRefAttr>
        *implicitOriginToNewParamRef;
  } indexReplacer;
  indexReplacer.ctx = ctx;
  indexReplacer.newOriginParamDecls = &newOriginParamDecls;
  indexReplacer.implicitOriginToNewParamRef = &implicitOriginToNewParamRef;

  // The caller specifies all the input types, which means that all the input
  // reference types that carry implicit origins will already have them
  // specified with names, so dig those out and use them as parameters.
  // If the caller provided indexed inputs, rewrite them to named inputs as our
  // body will expect.
  SmallVector<Type> adjustedArgTypes;
  for (auto [argNo, argTypeMaybeWithIndices, argConv] :
       llvm::enumerate(argTypes, argConventions)) {
    auto argTypeNoIndices = indexReplacer.replace(argTypeMaybeWithIndices);
    adjustedArgTypes.push_back(argTypeNoIndices);
    if (!hasImplicitOrigin(argConv))
      continue;

    // Dig out the origin decl.
    auto refArgType = cast<RefType>(argTypeNoIndices);
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
    }
    // Only add a param decl if it doesn't already exist (and we don't already
    // plan to add it).
    bool foundExisting = false;
    for (auto &existingParam : params) {
      if (existingParam.getName() == decl.getName() &&
          existingParam.getType() == decl.getType()) {
        foundExisting = true;
      }
    }
    for (auto &existingParam : newOriginParamDecls) {
      if (existingParam.getName() == decl.getName() &&
          existingParam.getType() == decl.getType()) {
        foundExisting = true;
      }
    }
    if (!foundExisting)
      newOriginParamDecls.push_back(decl);
  }
  size_t numImplicitOriginDecls = newOriginParamDecls.size();

  auto metadata = FnMetadataAttr::get(
      argListAttrs, numImplicitOriginDecls,
      getOriginsAccessibleByParams(paramListAttrs, params, shared,
                                   /*captureOrigins=*/nullptr),
      /*isNestedOriginExclusivityCheckingDisabled=*/false);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  FnTypeGeneratorType sigGen = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      params, functionType, argConventions, fnEffects, metadata, paramListAttrs,
      [&] { return mlir::emitError(location); });
  // Strip off the named origin decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FnOp and
  // in the BBArgs.
  sigGen = sigGen.replaceImplicitOriginsWithIndexes(newOriginParamDecls);

  StringAttr sourceName = builder.getStringAttr(name);
  StringAttr mangledName = builder.getStringAttr(
      DeclResolver::getMangledName(sourceName, parent, sigGen).getValue() +
      suffix);

  // If a function with this signature already exists in the struct, don't
  // create a new one. Return null to indicate that there was an existing
  // method.
  if (shared.lookupSymbolIn(&parent, mangledName))
    return nullptr;

  FnOp fnOp = builder.create<FnOp>(mangledName, sourceName, sigGen);

  // Set the attributes on the FnOp in bulk.
  NamedAttrList attrs = fnOp->getAttrDictionary();

  if (SpecialFunctionInfo::get(specialFnID).isImplicitlyStatic())
    attrs.set(fnOp.getIsStaticAttrName(), UnitAttr::get(ctx)); // True.

  // Figure out the full set of parameter declarations, this is the explicit
  // parameter declarations + implicit origins.
  SmallVector<ParamDeclAttr> fullParams;
  llvm::append_range(fullParams, params);
  llvm::append_range(fullParams, newOriginParamDecls);
  if (!fullParams.empty()) {
    attrs.set(fnOp.getParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullParams));
  }

  attrs.set(fnOp.getSpecialFnKindAttrName(),
            builder.getI8IntegerAttr(uint8_t(specialFnID)));
  attrs.set(fnOp.getIsSyntheticAttrName(), UnitAttr::get(ctx)); // True.
  attrs.set(fnOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  attrs.set(fnOp.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, inlineLevel));
  fnOp->setAttrs(attrs.getDictionary(ctx));

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(fnOp);
  if (!fnOp.getBody())
    fnOp.getBodyRegion().push_back(new Block());
  for (Type argType : adjustedArgTypes)
    fnOp.getBody()->addArgument(argType, fnOp.getLoc());
  return fnOp;
}

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
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

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
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

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel) {
  FnOp funcOp =
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

/// Given a struct and a trait declaration, make the struct inherit from the
/// trait if it does not already.
static void addTraitParent(StructDeclOp structOp, ASTDecl *traitDecl) {
  // Pull in the entire ancestor chain of the new symbol.
  SmallVector<SymbolRefAttr> newSymbols = {traitDecl->getSymbolRef()};
  canonicalizeTraitCompositionSymbols(traitDecl->getShared(), newSymbols);
  // Merge the new canonical symbols with the existing canonical trait symbols.
  TraitType trait = structOp.getCanonicalTrait();
  llvm::append_range(newSymbols, trait.getSymbols());
  // No need to pull in any ancestors now. Just sort and deduplicate.
  sortAndDeduplicateSymbols(newSymbols);
  structOp.setCanonicalTrait(TraitType::get(structOp.getContext(), newSymbols));
}

/// Add a attribute initializer method for this struct with a body.
FnOp StructEmitter::synthesizeFieldwiseInit(ASTDecl &structDecl) {
  auto declOp = cast<StructDeclOp>(structDecl);
  ASTType selfType = structDecl.getTypeDeclSelf();

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

  // Add the 'out self' argument if memory-only.
  Type litResultType = selfType;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), shared)) {
    litResultType = shared.getNoneType();
    argTypes.push_back(selfType.getRefForArgument("self", /*isMut=*/true));
    argConventions.push_back(ArgConvention::ByRefResult);
    argNames.push_back(StringAttr::get(shared.getContext(), "self"));
    argPassingKinds.push_back(PassingKind::Implicit);
  }

  return synthesizeFieldwiseInit(
      structDecl, argTypes, argConventions,
      PogListAttr::get(getContext(), argNames, argPassingKinds), litResultType);
}

FnOp StructEmitter::synthesizeFieldwiseInit(
    ASTDecl &structDecl, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    // None or Self if register passable.
    ASTType litReturnType) {
  auto structOp = cast<StructDeclOp>(structDecl);

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argListAttrs, litReturnType,
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

  Value selfValue;
  bool hasResultTemp = false;
  if (!argConventions.empty() &&
      argConventions.back() == ArgConvention::ByRefResult) {
    selfValue = body->getArgument(body->getNumArguments() - 1);
  } else {
    // Register result needs a temporary.
    hasResultTemp = true;
    selfValue = emitter.emitVarDecl("self", litReturnType, funcOp.getLoc(),
                                    VarDeclKind::InitOutArg);
  }

  // Emit a bunch of stores to fields indexing our 'out self' result.
  for (auto [idx, fieldOp] : llvm::enumerate(structOp.getFieldDecls())) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];

    // Verify that this will work so we get a tailored error message.
    if (!fieldType.isCopyable(fieldASTDecl.getLoc(), shared) &&
        !fieldType.isMovable(fieldASTDecl.getLoc(), shared)) {
      auto diag = emitError(fieldASTDecl.getLoc())
                  << "cannot synthesize fieldwise init because field '"
                  << fieldOp.getName()
                  << "' has non-copyable and non-movable type " << fieldType;
      return {};
    }

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
    auto fieldRef = builder.create<RefStructGEROp>(selfValue, fieldOp);
    emitter.emitStoreToLValue({argVal, SyntheticNode(structDecl.getLoc())},
                              MLValue(fieldRef), EC_AttributeRefBase);
  }

  // For a register-passable result, load the result from the temporary.
  Value returnVal;
  if (hasResultTemp) {
    SyntheticNode exprTmp(funcDecl->getLoc());
    returnVal =
        emitter.emitSRValue({MRValue(selfValue), &exprTmp}, EC_ReturnValue);
  }

  // Finish off the function with a return + lit.endfunc.
  emitter.emitNormalReturn(funcOp.getLoc(), returnVal);
  return funcOp;
}

/// Given a function of the form
///    fn __copyinit__(existing: MyStruct, out self: MyStruct)
/// populate the method with the following:
///   %targetField0Ptr = lit.ref.struct.ger %self[field0]
///   %sourceField0Ptr = lit.ref.struct.ger %existing[field0]
///   copyinit_of_type_of_field0(%targetField0, %field)
LogicalResult StructEmitter::populateMoveCopy(ASTDecl &fnDecl, bool isMove) {
  // This method body resolves the decl.
  // TODO: This is because clients are directly calling this instead of having
  // declresolution do it.
  fnDecl.resolvedness = DeclResolvedness::body;

  auto fn = cast<FnOp>(fnDecl);
  ASTDecl *structDecl = fnDecl.getParentDecl();
  StructDeclOp structOp = cast<StructDeclOp>(structDecl);

  // We want to populate a move but the move/copy should be a method!
  SMLoc location = fnDecl.getLoc();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(fn.getLocScope());
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockBegin(
      shared.translateLocation(location), fn.getBody());
  ExprEmitter emitter(*structDecl, b);

  assert(fn.getNumArguments() == 2 &&
         "copy and move functions should have two arguments");
  Value existingArg = fn.getBody()->getArgument(0);
  Value selfArg = fn.getBody()->getArgument(1);

  // If the value is RP trivial then we can just load and store the whole thing
  // in one shot instead of breaking it down into fields because we know all the
  // underlying copy/move operations are trivial.
  // TODO: Use memcpy for memory trivial types when we have them.
  if (structOp.isRegisterPassableTrivial()) {
    Value value = b.create<LIT::RefLoadOp>(existingArg);
    b.create<RefStoreOp>(value, selfArg);
    return success();
  }

  // Otherwise, memory and register passable values are both passed by-reference
  // so we need to copy/move them fieldwise, invoking the copy/move ctors as
  // appropriate.
  for (StructFieldOp fieldOp : structOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl->lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return failure();

    // Verify that this will work so we get a tailored error message.
    if (!fieldType.isCopyable(fieldASTDecl.getLoc(), shared) &&
        !fieldType.isMovable(fieldASTDecl.getLoc(), shared)) {
      auto diag = emitError(fieldASTDecl.getLoc())
                  << "cannot synthesize " << fn.getSpecialFunctionInfo().name
                  << " because field '" << fieldOp.getName()
                  << "' has non-copyable and non-movable type " << fieldType;
      return failure();
    }

    auto targetFieldOp = b.create<RefStructGEROp>(selfArg, fieldOp);
    Value srcFieldOp = b.create<RefStructGEROp>(existingArg, fieldOp);
    CValue src =
        isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));
    emitter.emitStoreToLValue({src, SyntheticNode(location)},
                              MLValue(targetFieldOp), EC_AttributeRefBase);
  }
  SymbolConstantAttr ref = fn.getBoundSymbolRef(shared.getEvaluationContext());
  if (isMove)
    structOp.setMoveInitAttr(ref);
  else
    structOp.setCopyInitAttr(ref);
  return success();
}

/// Given a struct and a list of arguments, generate a function. For example,
/// given {
///  MyStruct, "prefix", [ParamType1, ParamType2],
///  [read_mem, read_mem], ["x","b"]
/// }, this function produces:
///
/// ```
/// lit.fn @prefixParam1Param2(%x: ParamType1 read_mem,
///    %b : ParamType2 read_mem, %self: !kgen.pointer<@MyStruct> byref_result
/// ) -> !kgen.none  {
///   %0 = kgen.param.constant: none = <#kgen.none>
///   lit.return %0 : !kgen.none
///   lit.end_fn
/// }
/// ```
std::pair<FnOp, ASTDecl *> StructEmitter::addVoidMethod(
    ASTDecl &structDecl, StringRef prefix, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    SpecialFunctionKind kind, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs) {
  auto [func, funcDecl] = synthesizeMethodInStruct(
      prefix, params, paramListAttrs, argTypes, argConventions, argListAttrs,
      shared.getNoneType(), structDecl, structDecl.getLoc(), kind);
  if (!func)
    return {};
  Block *body = func.getBody();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(func.getLocScope());

  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), body);
  ExprEmitter::emitNormalReturn(b);
  return {func, funcDecl};
}

std::pair<FnOp, ASTDecl *> StructEmitter::addVoidMethod(
    ASTDecl &structDecl, StringRef prefix, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    SpecialFunctionKind kind) {

  return addVoidMethod(structDecl, prefix, argTypes, argConventions,
                       argListAttrs, kind, /*params=*/{},
                       /*paramListAttrs=*/PogListAttr::get(getContext()));
}

FnOp StructEmitter::synthesizeEmptyDtor(ASTDecl &structDecl) {
  auto structOp = cast<StructDeclOp>(structDecl);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());

  // Figure out the type of the 'self' argument, which is always indirect since
  // it is owned.
  ASTType selfType = structDecl.getTypeDeclSelf();
  if (!selfType)
    return {};

  selfType = selfType.getRefForArgument("self", /*isMut*/ true);
  StringAttr selfName = builder.getStringAttr("self");

  // Create the FnOp and ASTDecl for the method.
  StructEmitter emitter(shared);
  auto [funcOp, funcDecl] = emitter.synthesizeMethodInStruct(
      "__del__", selfType.mlirType, ArgConvention::OwnedMem,
      PogListAttr::get(emitter.getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), structDecl, structDecl.getLoc(),
      SpecialFunctionKind::kDel);
  if (!funcOp)
    return {};
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Finish off the function with a return + lit.endfunc.
  builder = ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  ExprEmitter::emitNormalReturn(builder);

  // Remember this as the destructor for the struct.
  structOp.setDestructorAttr(
      funcOp.getBoundSymbolRef(shared.getEvaluationContext()));
  return funcOp;
}

FnOp StructEmitter::synthesizeEmptyMoveOrCopyInit(ASTDecl &structDecl,
                                                  bool isMove) {
  ASTType selfType = structDecl.getTypeDeclSelf();
  StringRef name = isMove ? "__moveinit__" : "__copyinit__";
  MLIRContext *ctx = shared.getContext();
  Builder b(ctx);
  StringAttr existingName = b.getStringAttr("other");

  // If the type is register passable trivial, the 'existing' value will be
  // passed as a register, otherwise a reference.
  Type existingArgType = selfType.getRefForArgument("existing", isMove);
  ArgConvention existingConv =
      isMove ? ArgConvention::OwnedMem : ArgConvention::ReadMem;

  Type selfArgType = selfType.getRefForArgument("self", /*isMut=*/true);
  auto argListAttrs =
      PogListAttr::get(ctx, {existingName, b.getStringAttr("self")},
                       {PassingKind::PosOnly, PassingKind::Implicit});
  auto [resultFn, resultDecl] = addVoidMethod(
      structDecl, name, {existingArgType, selfArgType},
      {existingConv, ArgConvention::ByRefResult}, argListAttrs,
      isMove ? SpecialFunctionKind::kMoveInit : SpecialFunctionKind::kCopyInit);
  if (!resultFn)
    return {};
  resultDecl->resolvedness = DeclResolvedness::signature;

  // TODO: Should only do this if the type is RP or small?
  resultFn.setInlineLevel(InlineLevel::AlwaysNoDebug);
  return resultFn;
}

FnOp StructEmitter::synthesizeExplicitCopy(ASTDecl &structDecl) {
  ASTType selfType = structDecl.getTypeDeclSelf();
  MLIRContext *ctx = this->shared.getContext();

  ExprEmitter emitter(structDecl, EC_Decorator);

  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl);

  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;

  // Add the `existing` argument
  //
  // If the type is register passable trivial, the 'existing' value will
  // be passed as a register, otherwise a reference.
  if (selfType.isTrivial(structDecl.getLoc(), shared)) {
    // Self is register trivial
    argTypes.push_back(selfType);
    argConventions.push_back(ArgConvention::ReadReg);
  } else {
    argTypes.push_back(selfType.getRefForArgument("existing", /*isMut=*/false));
    argConventions.push_back(ArgConvention::ReadMem);
  }
  argNames.push_back(StringAttr::get(ctx, "existing"));
  argPassingKinds.push_back(PassingKind::PosOnly);

  // Add result slot / return type
  //
  // If the type is register passable (trivial or not), the low-level function
  // return result type is Self. Otherwise, the low-level return type is None,
  // and the result is returned through a memory output `__result__` slot arg.
  Type mlirReturnType;

  if (selfType.isRegisterPassable(structDecl.getLoc(), shared)) {
    // The return type is register passable, so return it directly via the
    // low-level MLIR-level return type (not via a result slot argument).
    mlirReturnType = selfType;
  } else {
    argNames.push_back(StringAttr::get(ctx, "__result__"));
    argPassingKinds.push_back(PassingKind::Implicit);
    argTypes.push_back(
        selfType.getRefForArgument("__result__", /*isMut=*/true));
    argConventions.push_back(ArgConvention::ByRefResult);
    mlirReturnType = shared.getNoneType();
  }

  assert(mlirReturnType &&
         "failed to compute return type for synthesized copy()");

  // Construct an empty FnOp for copy() method
  auto argListAttrs = PogListAttr::get(ctx, argNames, argPassingKinds);
  auto [copyFunc, funcDecl] = this->synthesizeMethodInStruct(
      "copy", argTypes, argConventions, argListAttrs,
      /*resultType=*/mlirReturnType, structDecl, structDecl.getLoc());

  // Point a `builder` at the end of the new copy() FnOp
  emitter.builder = OpBuilder::atBlockEnd(copyFunc.getBody());

  // Now generate the body of the copy() method
  SyntheticNode synthNode(structDecl.getLoc());

  // FIXME: Should be invoking the recursive copy() method here if the element
  // is ExplictlyCopyable but not Copyable.
  Value resultToReturn;
  if (structDeclOp.isRegisterPassableTrivial()) {
    resultToReturn = copyFunc.getArgument(0);
  } else if (structDeclOp.isRegisterPassable()) {
    MBValue selfArg = MBValue(copyFunc.getArgument(0));
    resultToReturn = emitter.emitSRValue({selfArg, synthNode}, EC_ReturnValue);
  } else {
    MBValue selfArg = MBValue(copyFunc.getArgument(0));
    ValueDest resultSlotDest(MLValue(copyFunc.getArgument(1)), EC_ReturnValue);
    emitter.emitCopyOfValue({selfArg, synthNode}, resultSlotDest);
    // resultToReturn remains null.
  }
  emitter.emitNormalReturn(structDeclOp.getLoc(), resultToReturn);
  return copyFunc;
}

std::optional<ValueInfo> ValueInfo::lookupExisting(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();
  auto structOp = cast<StructDeclOp>(structDecl);

  ValueInfo result;
  auto find = [&](StringRef name, SpecialFunctionKind kind,
                  FnOp &member) -> LogicalResult {
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
      if (auto func = dyn_cast<FnOp>(result))
        if (SpecialFunctionKind(func.getSpecialFnKind()) == kind)
          member = func;
    }

    return success();
  };
  if (failed(find("__del__", SpecialFunctionKind::kDel, result.del)) ||
      failed(find("__copyinit__", SpecialFunctionKind::kCopyInit,
                  result.copyinit)) ||
      failed(find("__moveinit__", SpecialFunctionKind::kMoveInit,
                  result.moveinit)) ||
      failed(find("copy", SpecialFunctionKind::kNormal, result.copy)))
    return {};

  return result;
}

std::optional<ValueInfo> StructEmitter::addMissingValueMemberStubsToStruct(
    ASTDecl &structDecl, bool forceGenerateDestructor) {
  auto declOp = cast<StructDeclOp>(structDecl);
  std::optional<ValueInfo> valueInfo = ValueInfo::lookupExisting(structDecl);
  if (!valueInfo)
    return {};

  if (!valueInfo->del && forceGenerateDestructor)
    valueInfo->del = synthesizeEmptyDtor(structDecl);

  auto addCopyOrMoveBuiltinTrait = [&](StringRef traitName) {
    ASTDecl *traitDecl = shared.lookupBuiltinTrait(
        traitName, structDecl.getParentDecl(), structDecl.getLoc());
    if (traitDecl) // Don't crash if the builtin trait is not found.
      addTraitParent(declOp, traitDecl);
  };

  if (!valueInfo->copyinit && !declOp.isRegisterPassableTrivial())
    valueInfo->copyinit =
        synthesizeEmptyMoveOrCopyInit(structDecl, /*isMove=*/false);
  addCopyOrMoveBuiltinTrait("Copyable");

  if (!valueInfo->moveinit && !declOp.isRegisterPassable())
    valueInfo->moveinit =
        synthesizeEmptyMoveOrCopyInit(structDecl, /*isMove=*/true);
  addCopyOrMoveBuiltinTrait("Movable");

  // NOTE: The  behavior of this is scary: if there is no method named "copy"
  // with any signature, then this will get called to synthesize the copy()
  // method and get ExplicitlyCopyable.  If there is some method with this name
  // then it doesn't get added, even if it has nothing to do with
  // ExplicitlyCopyable.
  // We should just remove @value.
  if (!valueInfo->copy) {
    valueInfo->copy = synthesizeExplicitCopy(structDecl);
    addCopyOrMoveBuiltinTrait("ExplicitlyCopyable");
  }

  return valueInfo;
}

FnOp StructEmitter::findInitInStruct(StructDeclOp structOp,
                                     ArrayRef<Type> operands) {
  size_t expectedNumInputs = operands.size() + 1;

  for (auto candidate : structOp.getOps<FnOp>()) {
    SpecialFunctionKind kind = candidate.getSpecialFunctionKind();
    if (kind != SpecialFunctionKind::kInit ||
        candidate.getBody()->getArguments().size() != expectedNumInputs)
      continue;

    bool isMatch = true;
    for (auto [existing, proposed] :
         llvm::zip(candidate.getFuncTypeGenerator().getArguments().slice(1),
                   operands)) {
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
