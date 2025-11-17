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
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// FunctionEmitter
//===----------------------------------------------------------------------===//

static FnOp
createFunction(ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
               PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
               ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
               Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
               ImplicitLocOpBuilder &builder, FnEffects fnEffects,
               StringRef suffix, bool synthetic, InlineLevel inlineLevel) {
  MLIRContext *ctx = parent.getContext();
  SharedState &shared = parent.getShared();

  // Figure out the implicit origins we'll need to add.
  std::vector<ParamDeclAttr> newOriginParamDecls;
  llvm::MapVector<ImplicitOriginRefAttr, ParamDeclRefAttr>
      implicitOriginToNewParamRef;

  // Replace all `ImplicitOriginRefAttr` with `ParamRefDeclAttr`s that point to
  // explicitly *named* parameter-decls.
  struct ImplicitOriginRefAttrReplacer
      : IndexParameterReplacer<ImplicitOriginRefAttrReplacer> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // Check if we found an ImplicitOriginRefAttr that's pointing all the way
      // up the original function's root scope, see PSTIAIRAID.
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
      /*isNestedOriginExclusivityCheckingDisabled=*/false,
      /*constraints=*/{});
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

  FnOp fnOp = FnOp::create(builder, mangledName, sourceName, sigGen);

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
  attrs.set(fnOp.getSyntheticAttrName(), UnitAttr::get(ctx)); // True.
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

std::pair<FnOp, ASTDecl *> FunctionEmitter::synthesizeFunction(
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

FnOp StructEmitter::synthesizeDefaultTraitMethodWrapper(
    ASTDecl &existingDecl, StringRef name, FnTypeGeneratorType wrapperSignature,
    FnOp traitFn, ASTDecl *traitFnDecl, bool structDefinesMethod,
    ImplicitLocOpBuilder &builder, StringRef suffix) {

  assert(existingDecl.resolvedness <= DeclResolvedness::signature &&
         "synthesizeMethodInStruct is only valid on non-body resolved Fn "
         "ASTDecls");

  // Extract signature components from the high-level types
  FnType fnType = wrapperSignature.getBody();
  ArrayRef<Type> inputTypes = fnType.getArguments();

  PogListAttr traitArgListAttrs =
      traitFn.getFuncTypeGenerator().getArgListAttrs();

  ArrayRef<ParamDeclAttr> params = traitFn.getParams().drop_back(
      wrapperSignature.getNumImplicitOriginDecls());

  SmallVector<ParamDeclAttr> mangledParams;
  for (ParamDeclAttr param : params) {
    // Mangle the param name if a conflict exists -- this is needed for cases
    // where the struct we're creating the wrapper function in has a param with
    // the same name as one defined by the default trait method, for example:
    //
    // trait Foo:
    //   fn foo[x: Int](): ...
    //
    // struct Bar[x: Int](Foo): ...
    StringAttr mangledName =
        structDecl.mangleUserDefinedParamName(param.getName());
    ParamDeclAttr newParamDecl =
        ParamDeclAttr::get(mangledName, param.getType());
    mangledParams.push_back(newParamDecl);
  }

  // 'wrapperSignature' is generated by 'getTraitFunctionSignature' in
  // Traits.cpp and remaps param decl ref attrs to index ref attrs (for the sake
  // of comparing struct methods to trait methods to check conformances).
  //
  // Since we're also using it in this function to help synthesize a
  // default trait method wrapper function we must map ParamIndexRefAttrs back
  // to ParamDeclRefAttrs. (Otherwise we'd have a mismatch in expected types in
  // the lit.call op we later materialize in populateDefaultedTraitFunction).
  //
  // See DCRTODS in arcana/Generics.md for more details.
  class IndexRefReplacer : public IndexParameterReplacer<IndexRefReplacer> {
  public:
    IndexRefReplacer(ArrayRef<ParamDeclAttr> params) : params(params) {}

    Attribute tryReplace(Attribute value, size_t depth) {
      if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value)) {
        if (indexRef.getDepth() == depth)
          return ParamDeclRefAttr::get(params[indexRef.getIndex()]);
      }
      return {};
    }

    Type tryReplace(Type t, size_t depth) { return {}; }

  private:
    ArrayRef<ParamDeclAttr> params;
  } indexRefReplacer(mangledParams);

  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  for (auto [idx, argType] : llvm::enumerate(inputTypes)) {
    argTypes.push_back(indexRefReplacer.replace(argType));
    argConventions.push_back(fnType.getArgConvention(idx));
  }

  // Make sure we remap any IndexRefAttrs in the result type as well
  Type resultType = indexRefReplacer.replace(wrapperSignature.getResultType());

  InlineLevel inlineLevel = InlineLevel::Automatic;
  if (structDeclOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;

  FnOp funcOp = createFunction(
      structDecl, name, mangledParams, wrapperSignature.getParamListAttrs(),
      argTypes, argConventions, traitArgListAttrs, resultType,
      SpecialFunctionKind::kNormal, structDecl.getLoc(), builder,
      traitFn.getFuncTypeGenerator().getFnEffects(), suffix, /*synthetic=*/true,
      inlineLevel);

  if (!funcOp)
    return nullptr;

  // createFunction first calls FuncTypeGeneratorType::remapToFuncTypeGenerator
  // on the passed in arg/result types and constructs the FnOp with that. Part
  // of what that function does is remap ParamDeclRefAttrs to IndexRefAttrs of
  // arg/result types. Set the proper FunctionType here.
  //
  // TODO: Should this logic just go right into createFunction?
  funcOp.setFunctionType(FunctionType::get(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(), {resultType}));

  // Attach the new operation to the provided declaration.
  existingDecl.setIRValue(funcOp.getOperation());
  existingDecl.resolvedness = DeclResolvedness::signature;

  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, existingDecl);
  assert(!existing &&
         "unexpected redefinition when synthesizing method into existing decl");

  assert(funcOp && "Couldn't synthesize default trait wrapper in body");

  // Annotate with metadata linking back to trait default implementation.
  funcOp.setInheritedFromAttr(traitFnDecl->getSymbolRef());

  // Right now there's not really a great way to re-apply the decorators that
  // were on the defaulted trait method to the struct's wrapper lit.fn op, but
  // fortunately all the behavior for our current set of decorators is limited
  // changing the fn op's signature or attribute values.
  if (traitFn.getIsStatic())
    funcOp.setIsStatic(true);

  if (traitFn.isDef())
    funcOp.setDef(true);

  if (traitFn.isImplicitConversion())
    funcOp.setImplicitConversion(traitFn.getImplicitConversion());

  if (traitFn.isExternal())
    funcOp.setExternal(true);

  funcOp.setExportKind(traitFn.getExportKind());

  if (!traitFn.getLLVMMetadataArray().empty())
    funcOp.setLLVMMetadataArrayAttr(traitFn.getLLVMMetadataArrayAttr());

  if (!traitFn.getLLVMArgMetadataArray().empty())
    funcOp.setLLVMArgMetadataArrayAttr(traitFn.getLLVMArgMetadataArrayAttr());

  funcOp.setInlineLevel(KGEN::InlineLevel::AlwaysNoDebug);

  // When we're in the LSP we may not fully body resolve the wrapper
  // functions. Add a EndFnOp with unresolved=True so we can still verify
  // cleanly in passes run by the check LIT pipeline.
  auto atBlockEndBuilder = OpBuilder::atBlockEnd(funcOp.getBody());
  EndFnOp::create(atBlockEndBuilder, funcOp.getLoc(), /*unresolved=*/true);

  if (structDefinesMethod)
    funcOp.setDisabled(true);

  return funcOp;
}

/// Populates a struct's default trait method wrapper with the IR to actually
/// call the the trait method its wrapping. Takes the stub function that was
/// created during synthesizeDefaultTraitMethodWrapper and forwards all the
/// arguments of the FnOp created there to the call op on the actual defaulted
/// trait method.
LogicalResult StructEmitter::populateDefaultedTraitFunction(ASTDecl &fnDecl) {
  auto fn = cast<FnOp>(fnDecl.getIfOperation());
  ASTDecl &structDecl = *fnDecl.getParentDecl();
  ASTType structSelfType = structDecl.getTypeDeclSelf();

  IREmitter emitter(structDecl, EC_Trait);

  fn.getBody()->clear();

  emitter.builder = OpBuilder::atBlockBegin(fn.getBody());

  auto inheritedFromAttr = fn.getInheritedFrom();
  assert(inheritedFromAttr &&
         "inherited_from attribute should always be present on a"
         " default-method stub");

  // Look up the trait's default implementation function
  ASTDecl *traitDefaultMethodDecl =
      shared.declResolver->getDeclForFuncSymbol(*inheritedFromAttr);
  assert(traitDefaultMethodDecl &&
         "Could not find trait default method implementation");

  auto parentTraitRef = traitDefaultMethodDecl->getParentDecl()->getSymbolRef();

  TraitType parentTrait = TraitType::get(parentTraitRef);
  SyntheticNode synthNode(structDecl.getLoc());
  CValue selfTypeCValue(structSelfType.mlirType);
  PValue selfAsTrait = emitter.emitMetaTypeToTraitConversion(
      {selfTypeCValue, synthNode}, parentTrait);

  // emitMetaTypeToTraitConversion can fail if the struct holding the defaulted
  // trait function wrapper didn't conform to the trait due to an unimplemented
  // function.
  // Simply bail early without worrying about the body of the lit.fn op we're
  // currently working on as compilation will fail anyways.
  if (selfAsTrait.isNull()) {
    fnDecl.setErroneous();
    return failure();
  }

  FnOp traitDefaultMethodOp =
      cast<FnOp>(traitDefaultMethodDecl->getIfOperation());
  auto fnTypeGen = traitDefaultMethodOp.getFullSignature();

  auto &builder = *emitter.builder;

  // Collect the bindings needed to call the trait method in this.
  SmallVector<TypedAttr> callParamBindings;

  callParamBindings.push_back(selfAsTrait.get());

  auto fnParams =
      fn.getParams().drop_back(fnTypeGen.getNumImplicitOriginDecls());

  for (ParamDeclAttr param : fnParams)
    callParamBindings.push_back(KGEN::ParamDeclRefAttr::get(param));

  // create a specialized generator from the fnTypeGen
  FuncTypeGeneratorType specializedGenerator =
      fnTypeGen.getSpecializedGenerator(
          callParamBindings, &fnDecl.getShared().getEvaluationContext(),
          fn.getLoc());

  SymbolRefAttr calleeSym =
      LIT::getFullyResolvedSymbolRef(traitDefaultMethodOp);
  TypedAttr typedSymbol = KGEN::SymbolConstantAttr::get(
      calleeSym, specializedGenerator, callParamBindings);

  SmallVector<Value> operands(fn.getArguments().begin(),
                              fn.getArguments().end());

  SmallVector<TypedAttr> implicitOrigins;
  auto argConvs = fnTypeGen.getArgConventions();
  for (auto [val, conv] : llvm::zip(operands, argConvs))
    if (KGEN::hasImplicitOrigin(conv))
      implicitOrigins.push_back(cast<LIT::RefType>(val.getType()).getOrigin());

  ArrayRef<Type> resultTypes = specializedGenerator.getBody().getResults();
  auto callOp = LIT::CallOp::create(builder, fn.getLoc(), resultTypes,
                                    typedSymbol, implicitOrigins, operands);

  emitter.emitNormalReturn(fn.getLoc(), callOp->getResult(0));

  fnDecl.resolvedness = DeclResolvedness::body;
  return success();
}

//===----------------------------------------------------------------------===//
// StructEmitter
//===----------------------------------------------------------------------===//

StructEmitter::StructEmitter(ASTDecl &structDecl)
    : FunctionEmitter(structDecl.getShared()), structDecl(structDecl) {
  structDeclOp = cast<StructDeclOp>(*structDecl.getIfOperation());
}

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, FnEffects fnEffects,
    StringRef suffix, bool synthetic) {
  return synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      argTypes, argConventions, argListAttrs, resultType, specialFnID,
      fnEffects, suffix, synthetic);
}

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> params, PogListAttr paramListAttrs,
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs, Type resultType, SpecialFunctionKind specialFnID,
    FnEffects fnEffects, StringRef suffix, bool synthetic) {
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  InlineLevel inlineLevel = InlineLevel::Automatic;
  // If the struct is register_passable("trivial"), make this
  // @always_inline("nodebug").
  if (structDeclOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;
  return synthesizeFunction(structDecl, name, params, paramListAttrs, argTypes,
                            argConventions, argListAttrs, resultType,
                            specialFnID, structDecl.getLoc(), builder,
                            fnEffects, suffix, synthetic, inlineLevel);
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
FnOp StructEmitter::synthesizeFieldwiseInit() {
  ASTType selfType = structDecl.getTypeDeclSelf();

  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;

  // We declare all of the operands to the init constructor as owned.  This
  // enables it to work with move-only fields, and, for copyable types, forces
  // the copy into the caller, which can then be elided with a consume or
  // RValue.
  for (auto fieldOp : structDeclOp.getFieldDecls()) {
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
      argTypes, argConventions,
      PogListAttr::get(getContext(), argNames, argPassingKinds), litResultType);
}

FnOp StructEmitter::synthesizeFieldwiseInit(
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs,
    // None or Self if register passable.
    ASTType litReturnType) {

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argListAttrs, litReturnType,
      SpecialFunctionKind::kInit);
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
  IREmitter emitter(*funcDecl, builder);

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
  for (auto [idx, fieldOp] : llvm::enumerate(structDeclOp.getFieldDecls())) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];

    // Verify that this will work so we get a tailored error message.
    if (!fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared) &&
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
    auto fieldRef = RefStructGEROp::create(builder, selfValue, fieldOp);
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

FnOp StructEmitter::synthesizeEmptyDtor() {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());

  // Figure out the type of the 'self' argument, which is always indirect since
  // it is owned.
  ASTType selfType = structDecl.getTypeDeclSelf();
  if (!selfType)
    return {};

  selfType = selfType.getRefForArgument("self", /*isMut*/ true);
  StringAttr selfName = builder.getStringAttr("self");

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, funcDecl] = synthesizeMethodInStruct(
      "__del__", selfType.mlirType, ArgConvention::OwnedMem,
      PogListAttr::get(getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), SpecialFunctionKind::kDel);
  if (!funcOp)
    return {};
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Destructors consume their 'self' arg.
  funcOp.setSelfDeinit(true);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Finish off the function with a return + lit.endfunc.
  builder = ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  IREmitter::emitNormalReturn(builder);

  // Remember this as the destructor for the struct.
  structDeclOp.setDestructorAttr(
      funcOp.getBoundSymbolRef(shared.getEvaluationContext()));
  return funcOp;
}

FnOp StructEmitter::synthesizeEmptyMoveOrCopyInit(bool isMove) {
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
  auto [resultFn, resultDecl] = synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      /*argTypes*/ {existingArgType, selfArgType},
      /*argConvs*/ {existingConv, ArgConvention::ByRefResult}, argListAttrs,
      shared.getNoneType(),
      isMove ? SpecialFunctionKind::kMoveInit : SpecialFunctionKind::kCopyInit);
  if (!resultFn)
    return {};
  resultDecl->resolvedness = DeclResolvedness::signature;

  // Add a unresolved EndFnOp to the end of the function. This makes the
  // function able to verify clean, even if we don't body or signature resolve
  // it.
  auto resultAtBlockEndBuilder = OpBuilder::atBlockEnd(resultFn.getBody());
  EndFnOp::create(resultAtBlockEndBuilder, resultFn.getLoc(),
                  /*unresolved=*/true);
  if (isMove) // Move constructors consume their 'self' arg.
    resultFn.setSelfDeinit(true);

  // TODO: Should only do this if the type is RP or small?
  resultFn.setInlineLevel(InlineLevel::AlwaysNoDebug);
  return resultFn;
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

  auto fn = cast<FnOp>(fnDecl.getIfOperation());

  // We want to populate a move but the move/copy should be a method!
  SMLoc location = fnDecl.getLoc();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(fn.getLocScope());

  // Start by emitting the return at the end of the function.  Closure emission
  // may have emitted stuff into the body of one of these functions and the
  // return needs to come at the end.
  auto endFn = cast<EndFnOp>(fn.getBody()->getTerminator());
  endFn.setUnresolved(false); // Body is resolved now.
  ImplicitLocOpBuilder b(fn.getLoc(), endFn);
  IREmitter::emitNormalReturn(b, Value(), /*emitEndFunc=*/false);

  // Generate the copy/moves of all of the elements, emit this at the start of
  // the function so it is ahead of whatever closure emission might generate.
  b = ImplicitLocOpBuilder::atBlockBegin(fn.getLoc(), fn.getBody());
  IREmitter emitter(structDecl, b);

  assert(fn.getNumArguments() == 2 &&
         "copy and move functions should have two arguments");
  Value existingArg = fn.getBody()->getArgument(0);
  Value selfArg = fn.getBody()->getArgument(1);

  // If the value is RP trivial then we can just load and store the whole thing
  // in one shot instead of breaking it down into fields because we know all the
  // underlying copy/move operations are trivial.
  // TODO: Use memcpy for memory trivial types when we have them.
  if (structDeclOp.isRegisterPassableTrivial()) {
    Value value = LIT::RefLoadOp::create(b, existingArg);
    RefStoreOp::create(b, value, selfArg);
    return success();
  }

  // Otherwise, invoke the copy/move ctors fieldwise as appropriate.
  bool isImplicitlyCopyableStruct =
      structDecl.getTypeDeclSelf().isImplicitlyCopyable(structDecl.getLoc(),
                                                        shared);
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return failure();

    auto targetFieldOp = RefStructGEROp::create(b, selfArg, fieldOp);
    Value srcFieldOp = RefStructGEROp::create(b, existingArg, fieldOp);
    CValue src =
        isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));

    // Verify that this will work so we get a tailored error message.
    if (isMove) {
      // The move constructor can work with movable (preferably) or implicitly
      // copyable (as a fallback) types.
      if (!fieldType.isMovable(fieldASTDecl.getLoc(), shared) &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared)) {
        return emitError(fieldASTDecl.getLoc())
               << "cannot synthesize " << fn.getSpecialFunctionInfo().name
               << " because field '" << fieldOp.getName()
               << "' has non-copyable and non-movable type " << fieldType;
      }
    } else {
      // We only synthesize __copyinit__ for `ImplicitlyCopyable` object iff all
      // its fields are `ImplicitlyCopyable`. That is, we won't synthesize for
      // the following struct:
      // ```
      // struct T(ImplicitlyCopyable):
      //   var f: some Copyable
      // ```
      if (!fieldType.isCopyable(fieldASTDecl.getLoc(), shared,
                                isImplicitlyCopyableStruct)) {
        return emitError(fieldASTDecl.getLoc())
               << "cannot synthesize " << fn.getSpecialFunctionInfo().name
               << " because field '" << fieldOp.getName()
               << "' has non-copyable type " << fieldType;
      }

      // If this a copy constructor and the field is only `Copyable` but not
      // implicitly copyable, generate the explicit call to `__copyinit__` so
      // the rest of the compiler doesn't have to know about explicit copying.
      // We only do this when not-implicitly copyable so we don't have to deal
      // with the MLIR types.
      if (!isImplicitlyCopyableStruct &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared)) {

        ValueDest dest(MLValue(targetFieldOp), EC_SynthesizedMethod);
        SyntheticNode expr(location);
        (void)emitter.emitNamedMethodCall("__copyinit__", {{{src, &expr}}},
                                          dest, CallSyntax::kImplicitCopyInit,
                                          &expr);
        continue;
      }
    }

    emitter.emitStoreToLValue({src, SyntheticNode(location)},
                              MLValue(targetFieldOp), EC_SynthesizedMethod);
  }

  SymbolConstantAttr ref = fn.getBoundSymbolRef(shared.getEvaluationContext());
  if (isMove)
    structDeclOp.setMoveInitAttr(ref);
  else
    structDeclOp.setCopyInitAttr(ref);
  return success();
}

FnOp StructEmitter::synthesizeEmptyExplicitCopy(ASTDecl &structDecl) {
  IREmitter emitter(structDecl, EC_Decorator);

  ASTType selfType = structDecl.getTypeDeclSelf();
  MLIRContext *ctx = this->shared.getContext();
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
      /*resultType=*/mlirReturnType);

  funcDecl->resolvedness = DeclResolvedness::signature;

  // FIXME(MOCO-2288):
  //  This is a hack to get `ImplicitlyCopyable(Copyable)` inheritance
  //  to work. We should not have to early populate `copy()` here.
  populateExplicitCopy(*funcDecl);

  return copyFunc;
}

void StructEmitter::populateExplicitCopy(ASTDecl &fnDecl) {
  auto fn = cast<FnOp>(fnDecl.getIfOperation());
  IREmitter emitter(structDecl, EC_Decorator);

  // Point a `builder` at the end of the new copy() FnOp
  emitter.builder = OpBuilder::atBlockEnd(fn.getBody());

  // Now generate the body of the copy() method
  SyntheticNode synthNode(structDecl.getLoc());

  // If the struct is copyable, then just generate a call to the copy
  // constructor to reduce code size.
  Value resultToReturn;
  if (structDecl.getTypeDeclSelf().isImplicitlyCopyable(fnDecl.getLoc(),
                                                        shared)) {
    if (structDeclOp.isRegisterPassableTrivial()) {
      resultToReturn = fn.getArgument(0);
    } else if (structDeclOp.isRegisterPassable()) {
      MBValue selfArg = MBValue(fn.getArgument(0));
      resultToReturn =
          emitter.emitSRValue({selfArg, synthNode}, EC_ReturnValue);
    } else {
      MBValue selfArg = MBValue(fn.getArgument(0));
      ValueDest resultSlotDest(MLValue(fn.getArgument(1)), EC_ReturnValue);
      emitter.emitCopyOfValue({selfArg, synthNode}, resultSlotDest);
      // resultToReturn remains null.
    }
  } else {
    emitError(structDecl.getLoc())
        << "cannot synthesize explicit 'copy()'"
        << " for non-copyable struct " << structDecl.getTypeDeclSelf()
        << "; declare 'copy()' manually";
  }

  emitter.emitNormalReturn(structDeclOp.getLoc(), resultToReturn);
  fnDecl.resolvedness = DeclResolvedness::body;
}

std::optional<ValueInfo> ValueInfo::lookupExisting(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();
  auto structOp = cast<StructDeclOp>(*structDecl.getIfOperation());

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
      if (auto func = dyn_cast_or_null<FnOp>(result->getIfOperation()))
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
    bool forceGenerateDestructor) {
  std::optional<ValueInfo> valueInfo = ValueInfo::lookupExisting(structDecl);
  if (!valueInfo)
    return {};

  if (!valueInfo->del && forceGenerateDestructor)
    valueInfo->del = synthesizeEmptyDtor();

  auto addCopyOrMoveBuiltinTrait = [&](StringRef traitName) {
    ASTDecl *traitDecl = shared.lookupBuiltinTrait(
        traitName, structDecl.getParentDecl(), structDecl.getLoc());
    if (traitDecl) // Don't crash if the builtin trait is not found.
      addTraitParent(structDeclOp, traitDecl);
  };

  if (!valueInfo->copyinit && !structDeclOp.isRegisterPassableTrivial())
    valueInfo->copyinit = synthesizeEmptyMoveOrCopyInit(/*isMove=*/false);
  addCopyOrMoveBuiltinTrait("ImplicitlyCopyable");

  if (!valueInfo->moveinit && !structDeclOp.isRegisterPassable())
    valueInfo->moveinit = synthesizeEmptyMoveOrCopyInit(/*isMove=*/true);
  addCopyOrMoveBuiltinTrait("Movable");

  // NOTE: The  behavior of this is scary: if there is no method named "copy"
  // with any signature, then this will get called to synthesize the copy()
  // method and get Copyable.  If there is some method with this name
  // then it doesn't get added, even if it has nothing to do with
  // Copyable.
  // We should just remove @value.
  if (!valueInfo->copy)
    valueInfo->copy = synthesizeEmptyExplicitCopy(structDecl);

  return valueInfo;
}

/// Synthesize an unresolved alias into the struct with the specified name .
ASTDecl *StructEmitter::synthesizeUnresolvedAlias(StringRef name) {
  auto paramDecl =
      ParamDeclAttr::get(name, LIT::UnresolvedType::get(getContext()));

  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  auto declOp = AliasDeclOp::create(builder, paramDecl);

  // Create an ASTDecl so it can be resolved with name lookup.
  ASTDecl &aliasDecl = getDeclResolver().addDecl(
      declOp, structDecl.getLoc(), StringAttr::get(getContext(), name),
      &structDecl, LexerCursor(), LexerCursor(), /*indentation=*/0);
  aliasDecl.resolvedness = DeclResolvedness::unparsed;
  return &aliasDecl;
}

TypedAttr StructEmitter::populateSpecialFnIsTrivial(SpecialFunctionKind kind) {
  assert((kind == SpecialFunctionKind::kDel ||
          kind == SpecialFunctionKind::kCopyInit ||
          kind == SpecialFunctionKind::kMoveInit) &&
         "unknown synthesized alias");

  IREmitter emitter(structDecl, EC_AliasValue);
  // NOTE: we have to first synthesize the bit to `i1` (instead of `Bool`) to
  // avoid signature resolving `Bool::__init__`s, the implicit conversion will
  // be taken care of when body resolve conformanceOp.
  auto emitBoolAttr = [&](BoolAttr v) -> TypedAttr {
    SyntheticNode synthNode(structDecl.getLoc());
    return emitter.emitBool({v, synthNode}, EC_OperatorOperandValue)
        .getIfPValue();
  };

  // This emits an "and" as a PValue expression, maintaining the type of lhs/rhs
  // (which are Bool) instead of turning them into i1.
  auto emitAnd = [this, &emitter](PValue lhs, PValue rhs) {
    SyntheticNode synthNode(structDecl.getLoc());
    PValue lhsI1Val =
        emitter.emitI1({lhs, synthNode}, EC_OperatorOperandValue).getIfPValue();
    return ParamOperatorAttr::get(POC::Cond, {lhsI1Val, rhs, lhs});
  };

  auto spFnInfo = SpecialFunctionInfo::get(kind);
  LookupResult spDecls = shared.lookupAndResolveDecl(
      spFnInfo.name, structDecl.getLoc(), structDecl,
      /*searchParentScope=*/false);
  if (spDecls.isErroneous())
    return nullptr;
  ArrayRef<ASTDecl *> decls = spDecls.getIfSuccess();
  assert(decls.size() == 1 && "special fn decls cannot be overloaded");

  // If has a user provided implementation, consider them as non-trivial.
  if (!decls.front()->getCursor().isInvalid())
    return emitBoolAttr(BoolAttr::get(emitter.getContext(), false));

  // We have a synthesize __del__/__moveinit__/__copyinit__ function, in this
  // case all the fields have to conform to AnyType/Movable/Copyable or it will
  // fail to synthesize the special function during `populateMoveCopy`.
  StringRef traitName = [kind] {
    if (kind == SpecialFunctionKind::kDel)
      return "AnyType";
    else if (kind == SpecialFunctionKind::kCopyInit)
      return "Copyable";
    return "Movable";
  }();

  ASTDecl *traitDecl = shared.lookupBuiltinTrait(
      traitName, structDecl.getParentDecl(), structDecl.getLoc());
  auto witnessName =
      StringAttr::get(getContext(), Twine(spFnInfo.name) + "is_trivial");
  auto witnessSymbolName = getFlattenedSymbolName(traitDecl->getSymbolRef());

  TypedAttr ret = emitBoolAttr(BoolAttr::get(emitter.getContext(), true));
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return nullptr;

    if (!ASTType(fieldOp.getType()).getMetaType())
      continue; // skip simple mlir type

    TypedAttr fieldIsTrivial = shared.getEvaluationContext().getGetWitnessAttr(
        PValue(fieldOp.getType()),
        StringAttr::get(getContext(), witnessSymbolName), witnessName,
        ret.getType());

    ret = emitAnd(ret, fieldIsTrivial);
  }

  return ret;
}
