//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the StructEmitter class.
//
//===----------------------------------------------------------------------===//

#include "StructEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Check if a parameterized field type conditionally conforms to a builtin
/// trait based on the function's where-clause constraints.  Constructs a
/// conforms_to proposition and delegates to isImplicationProven.
///
/// Returns the resolved TraitDeclOp on success (for reuse in code-gen),
/// or a null TraitDeclOp if no conditional conformance was found.
static TraitDeclOp
fieldConditionallyConformsToBuiltin(ASTType fieldType, StringRef traitName,
                                    SharedState &shared, ASTDecl &structDecl,
                                    ArrayRef<ConstraintAttr> bodyConstraints) {
  auto paramType = dyn_cast<ParamType>(fieldType);
  if (!paramType)
    return {};

  TraitASTDecl *traitDecl =
      shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
  if (!traitDecl)
    return {};

  TraitDeclOp traitOp = traitDecl->getOp();

  TypedAttr fieldParam = getCanonicalAttr(paramType.getParam());
  SmallVector<TraitSymbolAttr> traitSymbols = {
      TraitSymbolAttr::get(traitDecl->getSymbolRef())};
  auto traitType = TraitType::get(fieldParam.getContext(), traitSymbols);
  auto conformsTo =
      TypeConformsToTraitAttr::get(fieldParam, traitType.getPValue());

  for (ConstraintAttr constraint : bodyConstraints)
    if (isImplicationProven(conformsTo, constraint.getProposition()))
      return traitOp;
  return {};
}

//===----------------------------------------------------------------------===//
// FunctionEmitter
//===----------------------------------------------------------------------===//

static FnOp
createFunction(ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
               PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
               ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
               Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
               ImplicitLocOpBuilder &builder, FnEffects fnEffects,
               StringRef suffix, bool synthetic, InlineLevel inlineLevel,
               bool isNestedOriginsReadOnly = false) {
  MLIRContext *ctx = parent.getContext();
  SharedState &shared = parent.getShared();

  // Replace all `ImplicitOriginRefAttr` with `ParamRefDeclAttr`s that point to
  // explicitly *named* parameter-decls.
  ImplicitOriginToNameRefAttrReplacer indexReplacer(ctx);

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
    for (auto &existingParam : indexReplacer.getNewOriginParamDecls()) {
      if (existingParam.getName() == decl.getName() &&
          existingParam.getType() == decl.getType()) {
        foundExisting = true;
      }
    }
    if (!foundExisting)
      indexReplacer.getNewOriginParamDecls().push_back(decl);
  }
  size_t numImplicitOriginDecls = indexReplacer.getNewOriginParamDecls().size();

  auto metadata = FnMetaOriginDataAttr::get(
      argListAttrs.getContext(), numImplicitOriginDecls,
      getOriginsAccessibleByParams(paramListAttrs, params, shared,
                                   /*captureOrigins=*/nullptr),
      isNestedOriginsReadOnly, /*definesInteriorOrigins=*/false);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  FnTypeGeneratorType sigGen = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      params, functionType, argConventions, fnEffects, metadata, paramListAttrs,
      [&] { return mlir::emitError(location); }, argListAttrs);
  // Strip off the named origin decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FnOp and
  // in the BBArgs.
  sigGen = sigGen.replaceImplicitOriginsWithIndexes(
      indexReplacer.getNewOriginParamDecls());

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
  llvm::append_range(fullParams, indexReplacer.getNewOriginParamDecls());
  if (!fullParams.empty()) {
    attrs.set(fnOp.getParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullParams));
  }

  attrs.set(fnOp.getSpecialFnKindAttrName(),
            builder.getI8IntegerAttr(uint8_t(specialFnID)));
  attrs.set(fnOp.getSyntheticAttrName(), UnitAttr::get(ctx)); // True.
  attrs.set(fnOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  attrs.set(fnOp.getInlineLevelAttrName(),
            getInlineLevelAttr(ctx, inlineLevel));
  fnOp->setAttrs(attrs.getDictionary(ctx));

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(fnOp);
  if (!fnOp.getBody())
    fnOp.getBodyRegion().push_back(new Block());
  for (Type argType : adjustedArgTypes)
    fnOp.getBody()->addArgument(argType, fnOp.getLoc());
  return fnOp;
}

std::pair<FnOp, DefASTDecl *> FunctionEmitter::synthesizeFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel, bool isNestedOriginsReadOnly) {
  FnOp funcOp = createFunction(parent, name, params, paramListAttrs, argTypes,
                               argConventions, argListAttrs, resultType,
                               specialFnID, loc, builder, fnEffects, suffix,
                               synthetic, inlineLevel, isNestedOriginsReadOnly);

  // Return null if the function already exists with the same signature.
  if (!funcOp)
    return {nullptr, nullptr};

  // Register the method in the struct.
  auto &funcDecl = cast<DefASTDecl>(shared.declResolver->addFullyResolvedDecl(
      funcOp.getOperation(), StringAttr::get(shared.getContext(), name), loc,
      &parent));
  funcDecl.createIRSignatureDecls(*shared.declResolver);

  // Synthesized functions skip DeclResolution's insertKnownAssumptions, so
  // inject the fn's where-clause constraints as known assumptions here,
  // when the ASTDecl is first created.
  funcDecl.insertKnownAssumptions(paramListAttrs.getBodyConstraints());

  // Set the symbol and notice if we are redeclaring something.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, funcDecl);
  assert(!existing && "unexpected redefinition of synthesized method");

  return {funcOp, &funcDecl};
}

FnOp StructEmitter::synthesizeDefaultTraitMethodWrapper(
    DefASTDecl &existingDecl, StringRef name,
    FnTypeGeneratorType wrapperSignature, FnOp traitFn, ASTDecl *traitFnDecl,
    ImplicitLocOpBuilder &builder, StringRef suffix,
    ConstraintAttr conformanceConstraint) {

  assert(existingDecl.resolvedness <= DeclResolvedness::signature &&
         "synthesizeMethodInStruct is only valid on non-body resolved Fn "
         "ASTDecls");

  // Extract signature components from the high-level types
  FuncType fnType = wrapperSignature.getBody();
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
    //   def foo[x: Int](): ...
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
  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  for (auto [idx, argType] : llvm::enumerate(inputTypes)) {
    argTypes.push_back(replaceIndexRefsWithNamedRefs(argType, mangledParams));
    argConventions.push_back(fnType.getArgConvention(idx));
  }

  // Make sure we remap any IndexRefAttrs in the result type as well
  Type resultType = replaceIndexRefsWithNamedRefs(
      wrapperSignature.getResultType(), mangledParams);

  InlineLevel inlineLevel = InlineLevel::Automatic;
  if (structDeclOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;

  SmallVector<ConstraintAttr> bodyConstraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    bodyConstraints.push_back(conformanceConstraint);

  PogListAttr wrapperParamListAttrs = wrapperSignature.getParamListAttrs();
  PogListAttr paramListAttrs = PogListAttr::get(
      wrapperParamListAttrs.getContext(), wrapperParamListAttrs.getPogs(),
      bodyConstraints, wrapperParamListAttrs.getOrigVariadicConvention());
  FnOp funcOp =
      createFunction(structDecl, name, mangledParams, paramListAttrs, argTypes,
                     argConventions, traitArgListAttrs, resultType,
                     SpecialFunctionKind::kNormal, structDecl.getLoc(), builder,
                     traitFn.getFuncTypeGenerator().getFnEffects(), suffix,
                     /*synthetic=*/true, inlineLevel);

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
  existingDecl.setOp(funcOp);
  existingDecl.resolvedness = DeclResolvedness::signature;
  if (existingDecl.getParamDecls().empty() &&
      existingDecl.getArgDecls().empty())
    existingDecl.createIRSignatureDecls(*shared.declResolver);

  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, existingDecl);
  assert(!existing &&
         "unexpected redefinition when synthesizing method into existing decl");

  existingDecl.insertKnownAssumptions(paramListAttrs.getBodyConstraints());

  assert(funcOp && "Couldn't synthesize default trait wrapper in body");

  funcOp.setInheritedFromAttr(
      TraitSymbolAttr::get(traitFnDecl->getParentDecl()->getSymbolRef()));
  // Annotate with metadata linking back to trait default implementation. The
  // trait names the conformance this wrapper satisfies; the function symbol is
  // what `populateDefaultedTraitFunction` forwards the call to.
  funcOp.setDefaultFnRefAttr(traitFnDecl->getSymbolRef());

  // Right now there's not really a great way to re-apply the decorators that
  // were on the defaulted trait method to the struct's wrapper lit.fn op, but
  // fortunately all the behavior for our current set of decorators is limited
  // changing the def op's signature or attribute values.
  if (traitFn.getIsStatic())
    funcOp.setIsStatic(true);

  if (traitFn.isImplicitConversion())
    funcOp.setImplicitConversion(traitFn.getImplicitConversion());

  if (traitFn.isExternal())
    funcOp.setExternal(true);

  funcOp.setExportKind(traitFn.getExportKind());

  if (!traitFn.getFnAttrs().empty())
    funcOp.setFnAttrsAttr(traitFn.getFnAttrsAttr());

  if (!traitFn.getFnArgAttrs().empty())
    funcOp.setFnArgAttrsAttr(traitFn.getFnArgAttrsAttr());

  funcOp.setInlineLevelAttr(getInlineLevelAttr(
      funcOp.getContext(), KGEN::InlineLevel::AlwaysNoDebug));

  // When we're in the LSP we may not fully body resolve the wrapper
  // functions. Add a EndFnOp with unresolved=True so we can still verify
  // cleanly in passes run by the check LIT pipeline.
  auto atBlockEndBuilder = OpBuilder::atBlockEnd(funcOp.getBody());
  EndFnOp::create(atBlockEndBuilder, funcOp.getLoc(), /*unresolved=*/true);

  return funcOp;
}

/// Populates a struct's default trait method wrapper with the IR to actually
/// call the the trait method its wrapping. Takes the stub function that was
/// created during synthesizeDefaultTraitMethodWrapper and forwards all the
/// arguments of the FnOp created there to the call op on the actual defaulted
/// trait method.
LogicalResult
StructEmitter::populateDefaultedTraitFunction(DefASTDecl &fnDecl) {
  auto fn = fnDecl.getOp();
  auto &structDecl = cast<StructASTDecl>(*fnDecl.getParentDecl());
  ASTType structSelfType = structDecl.getSelfType();

  IREmitter emitter(fnDecl, EC_Trait);
  fn.getBody()->clear(); // Remove the lit.end_fn

  emitter.builder = OpBuilder::atBlockBegin(fn.getBody());

  auto defaultFnRefAttr = fn.getDefaultFnRef();
  assert(defaultFnRefAttr &&
         "defaultFnRef attribute should always be present on a"
         " default-method stub");

  // Look up the trait's default implementation function
  DefASTDecl *traitDefaultMethodDecl =
      shared.declResolver->getDeclForFuncSymbol(*defaultFnRefAttr);
  assert(traitDefaultMethodDecl &&
         "Could not find trait default method implementation");

  ASTDecl *parentTraitDecl = traitDefaultMethodDecl->getParentDecl();
  [[maybe_unused]] LogicalResult result =
      fnDecl.getShared().getDeclResolver().resolveSignature(*parentTraitDecl,
                                                            fnDecl.getLoc());
  assert(result.succeeded() && "failed to resolve signature");

  TraitType parentTrait =
      cast<TraitASTDecl>(parentTraitDecl)->getOp().getCanonicalTrait();
  SyntheticNode synthNode(structDecl.getLoc());
  CValue selfTypeCValue(structSelfType.mlirType);
  PValue selfAsTrait = emitter.emitMetaTypeToTraitConversion(
      {selfTypeCValue, &synthNode}, parentTrait);

  // emitMetaTypeToTraitConversion can fail if the struct holding the defaulted
  // trait function wrapper didn't conform to the trait due to an unimplemented
  // function.
  // Simply bail early without worrying about the body of the lit.fn op we're
  // currently working on as compilation will fail anyways.
  if (selfAsTrait.isNull()) {
    fnDecl.setErroneous();
    return failure();
  }

  FnOp traitDefaultMethodOp = traitDefaultMethodDecl->getOp();
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

  // Bail if specialization failed (already diagnosed)
  if (!specializedGenerator) {
    fnDecl.setErroneous();
    return failure();
  }

  SymbolRefAttr calleeSym =
      LIT::getFullyResolvedSymbolRef(traitDefaultMethodOp);
  TypedAttr typedSymbol = KGEN::SymbolConstantAttr::get(
      calleeSym, specializedGenerator, callParamBindings);

  SmallVector<Value> operands(fn.getArguments().begin(),
                              fn.getArguments().end());

  // Determine the values for all of the implicit origins that need to be passed
  // to the calls.  These are the origins of any reference argument.
  SmallVector<TypedAttr> implicitOrigins;
  auto argConvs = fnTypeGen.getArgConventions();
  for (auto [idx, val, conv] : llvm::enumerate(operands, argConvs)) {
    if (hasImplicitOrigin(conv))
      implicitOrigins.push_back(cast<LIT::RefType>(val.getType()).getOrigin());
  }

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

StructEmitter::StructEmitter(StructASTDecl &structDecl)
    : FunctionEmitter(structDecl.getShared()), structDecl(structDecl) {
  structDeclOp = structDecl.getOp();
}

std::pair<FnOp, DefASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, FnEffects fnEffects,
    StringRef suffix, bool synthetic) {
  return synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      argTypes, argConventions, argListAttrs, resultType, specialFnID,
      fnEffects, suffix, synthetic);
}

std::pair<FnOp, DefASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> params, PogListAttr paramListAttrs,
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs, Type resultType, SpecialFunctionKind specialFnID,
    FnEffects fnEffects, StringRef suffix, bool synthetic) {
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  InlineLevel inlineLevel = InlineLevel::Automatic;
  // If the struct is TrivialRegisterPassable, make this
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
static void addTraitParent(StructDeclOp structOp, TraitASTDecl &traitDecl) {
  // Pull in the entire ancestor chain of the new symbol.
  SmallVector<TraitSymbolAttr> newSymbols = {
      TraitSymbolAttr::get(traitDecl.getSymbolRef())};
  canonicalizeTraitCompositionSymbols(traitDecl.getShared(), newSymbols);
  // Merge the new canonical symbols with the existing canonical trait symbols.
  TraitType trait = structOp.getCanonicalTrait();
  llvm::append_range(newSymbols, trait.getSymbols());
  // No need to pull in any ancestors now. Just sort and deduplicate.
  sortAndDeduplicateTraitSymbols(newSymbols);
  structOp.setCanonicalTrait(TraitType::get(structOp.getContext(), newSymbols));
}

/// Add a attribute initializer method for this struct with a body.
FnOp StructEmitter::synthesizeFieldwiseInit() {
  ASTType selfType = structDecl.getSelfType();

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
      conv = ArgConvention::ImmReg;
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
  funcOp.setInlineLevelAttr(
      getInlineLevelAttr(funcOp.getContext(), InlineLevel::AlwaysNoDebug));

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

    if (!fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                        *funcDecl) &&
        !fieldType.isMovable(fieldASTDecl.getLoc(), shared, *funcDecl)) {
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
    case ArgConvention::ImmReg:
      argVal = SRValue(arg);
      break;
    case ArgConvention::OwnedMem:
      argVal = MRValue(arg);
      break;
    case ArgConvention::ImmMem:
      argVal = MBValue(arg);
      break;
    }

    // Project self to the right field and store the RValue.
    auto fieldRef = RefStructGEROp::create(builder, selfValue, fieldOp);
    SyntheticNode node(structDecl.getLoc());
    emitter.emitStoreToLValue({argVal, &node}, MLValue(fieldRef),
                              EC_AttributeRefBase);
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

static SmallVector<ConstraintAttr>
getDtorConstraints(ConstraintAttr conformanceConstraint) {
  SmallVector<ConstraintAttr> constraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    constraints.push_back(conformanceConstraint);
  return constraints;
}

FnOp StructEmitter::synthesizeEmptyDtor(ConstraintAttr conformanceConstraint) {
  DefASTDecl *dtorDecl = synthesizeEmptyDtorStub(conformanceConstraint);
  if (!dtorDecl)
    return {};
  diagnoseNonDeinitableFields(*dtorDecl, conformanceConstraint);
  return dtorDecl->getOp();
}

DefASTDecl *
StructEmitter::synthesizeEmptyDtorStub(ConstraintAttr conformanceConstraint) {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());

  // Figure out the type of the 'self' argument, which is always indirect since
  // it is owned.
  ASTType selfType = structDecl.getSelfType();
  if (!selfType)
    return {};

  selfType = selfType.getRefForArgument("self", /*isMut*/ true);
  StringAttr selfName = builder.getStringAttr("self");

  SmallVector<ConstraintAttr> constraints =
      getDtorConstraints(conformanceConstraint);

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, funcDecl] = synthesizeMethodInStruct(
      "__deinit__", /*params=*/{},
      /*paramListAttrs=*/
      PogListAttr::get(getContext(), /*numPogs=*/0, constraints),
      /*argTypes=*/{selfType.mlirType},
      /*argConvs=*/{ArgConvention::DeinitMem},
      /*argListAttrs=*/
      PogListAttr::get(getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), SpecialFunctionKind::kDeinit);
  if (!funcOp)
    return {};
  funcOp.setInlineLevelAttr(
      getInlineLevelAttr(funcOp.getContext(), InlineLevel::AlwaysNoDebug));

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Finish off the function with a return + lit.endfunc.
  builder = ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  IREmitter::emitNormalReturn(builder);
  return funcDecl;
}

void StructEmitter::diagnoseNonDeinitableFields(
    DefASTDecl &dtorDecl, ConstraintAttr conformanceConstraint) {
  // Perform semantic analysis to make sure all the fields are implicitly
  // deletable. We would rather error in the parser than in check lifetimes.
  // We do this after creating the function so we don't emit a redundant error
  // complaining that a missing __deinit__ means the struct doesn't conform to
  // Deinitable.
  SmallVector<ConstraintAttr> assumptions;
  structDecl.getKnownAssumptionsIncludingParents(assumptions);
  assumptions.append(getDtorConstraints(conformanceConstraint));
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();
    TriBool confResult =
        fieldType
            .conformsToBuiltinTrait("Deinitable", structDecl.getLoc(), shared,
                                    assumptions)
            .first;
    if (!confResult.isTrue() &&
        !fieldConditionallyConformsToBuiltin(fieldType, "Deinitable", shared,
                                             structDecl, assumptions) &&
        !fieldType.isTrivialRegisterType(structDecl.getLoc(), shared)) {
      emitError(fieldOp.getLoc())
          << "field '" << fieldOp.getName() << "' has non-'Deinitable' type "
          << fieldType;
      dtorDecl.setErroneous();
      break;
    }
  }
}

FnOp StructEmitter::synthesizeEmptyMoveOrCopyInit(
    bool isMove, ConstraintAttr conformanceConstraint) {
  ASTType selfType = structDecl.getSelfType();
  MLIRContext *ctx = shared.getContext();
  Builder b(ctx);
  StringAttr srcName = b.getStringAttr(isMove ? "move" : "copy");

  // If the type is register passable trivial, the 'src' value will be
  // passed as a register, otherwise a reference.
  Type srcArgType = selfType.getRefForArgument(srcName.strref(), isMove);
  ArgConvention srcConv =
      isMove ? ArgConvention::DeinitMem : ArgConvention::ImmMem;

  SmallVector<ConstraintAttr> constraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    constraints.push_back(conformanceConstraint);

  Type selfArgType = selfType.getRefForArgument("self", /*isMut=*/true);
  auto argListAttrs =
      PogListAttr::get(ctx, {srcName, b.getStringAttr("self")},
                       {PassingKind::KwOnly, PassingKind::Implicit});
  auto [resultFn, resultDecl] = synthesizeMethodInStruct(
      "__init__", /*params=*/{},
      /*paramListAttrs=*/PogListAttr::get(ctx, /*numPogs=*/0, constraints),
      /*argTypes*/ {srcArgType, selfArgType},
      /*argConvs*/ {srcConv, ArgConvention::ByRefResult}, argListAttrs,
      shared.getNoneType(),
      isMove ? SpecialFunctionKind::kMoveCtor : SpecialFunctionKind::kCopyCtor);
  if (!resultFn)
    return {};
  resultDecl->resolvedness = DeclResolvedness::signature;

  // Add a unresolved EndFnOp to the end of the function. This makes the
  // function able to verify clean, even if we don't body or signature resolve
  // it.
  auto resultAtBlockEndBuilder = OpBuilder::atBlockEnd(resultFn.getBody());
  EndFnOp::create(resultAtBlockEndBuilder, resultFn.getLoc(),
                  /*unresolved=*/true);

  // TODO: Should only do this if the type is RP or small?
  resultFn.setInlineLevelAttr(
      getInlineLevelAttr(resultFn.getContext(), InlineLevel::AlwaysNoDebug));
  return resultFn;
}

/// Rebind a field reference to access it through a trait downcast.
static Value rebindRefForTrait(Value fieldRef, Type fieldMLIRType,
                               TraitDeclOp traitOp, ImplicitLocOpBuilder &b) {
  assert(isa<ParamType>(fieldMLIRType) &&
         "rebindRefForTrait requires a parameterized field type");
  auto paramType = cast<ParamType>(fieldMLIRType);
  TypedAttr downcast =
      DowncastAttr::get(traitOp.getCanonicalTrait(), paramType.getParam());
  Type reboundElementType = ParamType::get(downcast);
  RefType fieldRefType = cast<RefType>(fieldRef.getType());
  return RebindOp::create(b, fieldRefType.getWithElement(reboundElementType),
                          fieldRef);
}

/// Given a function of the form
///    def __init__(out self: MyStruct, *, copy: MyStruct)
/// populate the method with the following:
///   %targetField0Ptr = lit.ref.struct.ger %self[field0]
///   %sourceField0Ptr = lit.ref.struct.ger %copy[field0]
///   copyinit_of_type_of_field0(%targetField0, %field)
LogicalResult StructEmitter::populateMoveCopy(DefASTDecl &fnDecl, bool isMove) {
  // This method body resolves the decl.
  // TODO: This is because clients are directly calling this instead of having
  // declresolution do it.
  fnDecl.resolvedness = DeclResolvedness::body;

  auto fn = fnDecl.getOp();

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
  // Use fnDecl (not structDecl) so emitConstructorCall resolves overloads in
  // the fn's scope, where where-clause constraints are visible as assumptions.
  IREmitter emitter(fnDecl, b);

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
      structDecl.getSelfType().isImplicitlyCopyable(structDecl.getLoc(), shared,
                                                    fnDecl);
  ArrayRef<ConstraintAttr> bodyConstraints =
      fn.getFuncTypeGenerator().getParamListAttrs().getBodyConstraints();

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

    // Set to the resolved trait when the field requires a conditional
    // conformance (i.e. a where-clause constraint proves the field's type
    // parameter conforms to the required trait).  Null otherwise.
    TraitDeclOp conditionalTraitOp;

    if (isMove) {
      // The move constructor can work with movable (preferably) or implicitly
      // copyable (as a fallback) types.
      if (!fieldType.isMovable(fieldASTDecl.getLoc(), shared, fnDecl) &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                          fnDecl)) {
        conditionalTraitOp = fieldConditionallyConformsToBuiltin(
            fieldType, "Movable", shared, structDecl, bodyConstraints);
        if (!conditionalTraitOp)
          return emitError(fieldASTDecl.getLoc())
                 << "cannot synthesize move constructor because field '"
                 << fieldOp.getName()
                 << "' has non-movable and non-implicitly-copyable type "
                 << fieldType;
      }
    } else {
      // We only synthesize copy ctor for `ImplicitlyCopyable` object iff all
      // its fields are `ImplicitlyCopyable`. That is, we won't synthesize for
      // the following struct:
      // ```
      // struct T(ImplicitlyCopyable):
      //   var f: some Copyable
      // ```
      if (!fieldType.isCopyable(fieldASTDecl.getLoc(), shared,
                                isImplicitlyCopyableStruct, fnDecl)) {
        conditionalTraitOp = fieldConditionallyConformsToBuiltin(
            fieldType,
            isImplicitlyCopyableStruct ? "ImplicitlyCopyable" : "Copyable",
            shared, structDecl, bodyConstraints);
        if (!conditionalTraitOp) {
          return emitError(fieldASTDecl.getLoc())
                 << "cannot synthesize "
                 << (isImplicitlyCopyableStruct ? "implicit " : "")
                 << "copy constructor because field '" << fieldOp.getName()
                 << "' has non-"
                 << (isImplicitlyCopyableStruct ? "implicitly-" : "")
                 << "copyable type " << fieldType;
        }
      }
    }

    if (conditionalTraitOp) {
      Value reboundTarget = rebindRefForTrait(targetFieldOp, fieldType.mlirType,
                                              conditionalTraitOp, b);
      Value reboundSrc = rebindRefForTrait(srcFieldOp, fieldType.mlirType,
                                           conditionalTraitOp, b);

      SyntheticNode expr(location);
      if (isMove) {
        CValue reboundMoveSrc = CValue(MRValue(reboundSrc));
        emitter.emitStoreToLValue({reboundMoveSrc, &expr},
                                  MLValue(reboundTarget), EC_SynthesizedMethod);
      } else {
        CValue reboundCopySrc = CValue(MBValue(reboundSrc));
        ExprDest dest(MLValue(reboundTarget), EC_SynthesizedMethod);
        ASTType refinedFieldType(
            cast<RefType>(reboundTarget.getType()).getElementType());
        CallOperands operands(CallSyntax::kImplicitCopyCtor, &expr,
                              std::move(dest));
        operands.add(StringAttr::get(shared.getContext(), "copy"),
                     {reboundCopySrc, &expr}, ArgUnpackStyle::kKeyword);
        emitter.emitConstructorCall(refinedFieldType, std::move(operands));
      }
      continue;
    }

    CValue src =
        isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));

    SyntheticNode expr(location);
    if (!isMove) {
      // If this a copy constructor and the field is only `Copyable` but not
      // implicitly copyable, generate the explicit call to copy ctor so
      // the rest of the compiler doesn't have to know about explicit copying.
      // We only do this when not-implicitly copyable so we don't have to deal
      // with the MLIR types.
      if (!isImplicitlyCopyableStruct &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                          fnDecl)) {
        // Invoke `T(*, copy: Self)`.
        ExprDest dest(MLValue(targetFieldOp), EC_SynthesizedMethod);
        CallOperands operands(CallSyntax::kImplicitCopyCtor, &expr,
                              std::move(dest));
        operands.add(StringAttr::get(shared.getContext(), "copy"), {src, &expr},
                     ArgUnpackStyle::kKeyword);
        (void)emitter.emitConstructorCall(fieldType, std::move(operands));
        continue;
      }
    }
    emitter.emitStoreToLValue({src, &expr}, MLValue(targetFieldOp),
                              EC_SynthesizedMethod);
  }

  SymbolConstantAttr ref = fn.getBoundSymbolRef(shared.getEvaluationContext());
  if (isMove)
    structDeclOp.setMoveInitAttr(ref);
  else
    structDeclOp.setCopyInitAttr(ref);
  return success();
}

std::optional<ValueInfo> ValueInfo::lookupExisting(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();
  ValueInfo result;
  auto find = [&](StringRef name, SpecialFunctionKind kind,
                  FnOp &member) -> LogicalResult {
    LookupResult lookupResult =
        shared.lookupAndResolveDecl(name, structDecl.getLoc(), structDecl,
                                    /*searchParentScopes=*/false);
    if (!lookupResult.isSuccess())
      return success();
    for (ASTDecl *decl : lookupResult.getIfSuccess()) {
      if (auto *fnDecl = dyn_cast<DefASTDecl>(decl))
        if (SpecialFunctionKind(fnDecl->getOp().getSpecialFnKind()) == kind)
          member = fnDecl->getOp();
    }

    return success();
  };
  if (failed(find("__deinit__", SpecialFunctionKind::kDeinit, result.del)))
    return {};
  if (failed(
          find("__init__", SpecialFunctionKind::kCopyCtor, result.copyctor)) ||
      failed(find("__init__", SpecialFunctionKind::kMoveCtor, result.movector)))
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
    TraitASTDecl *traitDecl =
        shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
    if (traitDecl) // Don't crash if the builtin trait is not found.
      addTraitParent(structDeclOp, *traitDecl);
  };

  if (!valueInfo->copyctor && !structDeclOp.isRegisterPassableTrivial())
    valueInfo->copyctor = synthesizeEmptyMoveOrCopyInit(/*isMove=*/false);
  addCopyOrMoveBuiltinTrait("ImplicitlyCopyable");

  if (!valueInfo->movector && !structDeclOp.isRegisterPassable())
    valueInfo->movector = synthesizeEmptyMoveOrCopyInit(/*isMove=*/true);
  addCopyOrMoveBuiltinTrait("Movable");
  return valueInfo;
}

/// Synthesize an unresolved alias into the struct with the specified name .
AliasASTDecl *StructEmitter::synthesizeUnresolvedAlias(StringRef name) {
  auto paramDecl =
      ParamDeclAttr::get(name, LIT::UnresolvedType::get(getContext()));

  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  auto declOp = AliasDeclOp::create(builder, paramDecl);

  // Create an ASTDecl so it can be resolved with name lookup.
  auto &aliasDecl = cast<AliasASTDecl>(getDeclResolver().addDecl(
      declOp, structDecl.getLoc(), StringAttr::get(getContext(), name),
      &structDecl, LexerCursor(), LexerCursor(), /*indentation=*/0));
  aliasDecl.resolvedness = DeclResolvedness::unparsed;
  return &aliasDecl;
}

TypedAttr StructEmitter::populateSpecialFnIsTrivial(SpecialFunctionKind kind) {
  StringRef baseName;
  StringRef traitName;
  switch (kind) {
  case SpecialFunctionKind::kDeinit:
    baseName = "__del__";
    traitName = "Deinitable";
    break;
  case SpecialFunctionKind::kCopyCtor:
    baseName = "__copy_ctor_";
    traitName = "Copyable";
    break;
  case SpecialFunctionKind::kMoveCtor:
    baseName = "__move_ctor_";
    traitName = "Movable";
    break;
  default:
    llvm_unreachable("unknown synthesized alias");
  }

  IREmitter emitter(structDecl, EC_AliasValue);
  auto emitBoolAttr = [&](TypedAttr v) -> TypedAttr {
    SyntheticNode node(structDecl.getLoc());
    return getCanonicalAttr(
        emitter.emitBool({v, &node}, EC_OperatorOperandValue).getIfPValue());
  };

  StringRef spName =
      kind == SpecialFunctionKind::kDeinit ? "__deinit__" : "__init__";
  LookupResult spDecls =
      shared.lookupAndResolveDecl(spName, structDecl.getLoc(), structDecl,
                                  /*searchParentScope=*/false);
  if (spDecls.isErroneous())
    return nullptr;
  for (ASTDecl *decl : spDecls.getIfSuccess()) {
    auto *fnDecl = dyn_cast<DefASTDecl>(decl);
    if (!fnDecl)
      continue;
    auto fnOp = fnDecl->getOp();
    if (fnOp.getSpecialFunctionKind() == kind) {
      // If has a user provided implementation, consider them as non-trivial.
      if (!decl->getCursor().isInvalid())
        return emitBoolAttr(BoolAttr::get(emitter.getContext(), false));
    }
  }

  auto getScalarBool = [&](CValue value) -> PValue {
    SyntheticNode node(structDecl.getLoc());
    PValue i1 = emitter.emitScalarBool({value, &node}, EC_OperatorOperandValue)
                    .getIfPValue();
    // The expression is compiler synthesized, sugar is not very useful.
    return getCanonicalAttr(i1);
  };

  TraitASTDecl *traitDecl =
      shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
  auto witnessName =
      StringAttr::get(getContext(), Twine(baseName) + "is_trivial");
  auto traitSymbol = TraitSymbolAttr::get(traitDecl->getSymbolRef());

  ASTType boolType =
      shared.lookupBuiltinType("Bool", structDecl, structDecl.getLoc());
  PValue ret = SIMDAttr::getScalarBool(emitter.getContext(), true);
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return nullptr;

    if (ASTType(fieldOp.getType())
            .isTrivialRegisterType(fieldASTDecl.getLoc(), shared))
      continue; // skip MLIR types and TrivialRegisterPassable types.

    TypedAttr fieldIsTrivial =
        getScalarBool(shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
            PValue(fieldOp.getType()), traitSymbol, witnessName, boolType));
    // Relying on POC::And for folding.
    ret = ParamOperatorAttr::get(POC::And, ret, fieldIsTrivial);
  }
  // convert from scalar<bool> to Bool.
  return emitBoolAttr(ret);
}

//===----------------------------------------------------------------------===//
// Enum Synthesis
//===----------------------------------------------------------------------===//
//
// The compiler declares an `__enum`'s members; the actual implementations of
// their bodies are Mojo code in `std.builtin._enum`. The compiler finds that
// module's declarations by path (`lookupEnumSupportDecl`) and the enum's
// inherited `__Enum` members in the enum's own scope, whose names
// `checkEnumMemberNames` keeps free, so user code cannot shadow either. For
//
//     __enum Token(Copyable):
//         case eof
//         case ident(String)
//         case span(Int, Int)
//
// the generated members are, written as Mojo:
//
//     comptime _enum_case_names = ["eof", "ident", "span"]
//     comptime _enum_case_types = [NoneType, String, Tuple[Int, Int]]
//     var __enum_storage: __EnumStorage[NoneType, String, Tuple[Int, Int]]
//
//     comptime eof: Token = Token.__enum_init[0, NoneType]()
//
//     @staticmethod
//     def ident(var payload: String) -> Token:
//         return Token.__enum_init[1, String](payload^)
//
//     @staticmethod
//     def span(var payload0: Int, var payload1: Int) -> Token:
//         return Token.__enum_init[2, Tuple[Int, Int]](payload0^, payload1^)
//
// and `Token` conforms to `__Enum`, which supplies `__get_enum_storage`,
// `__enum_init`, `_get_enum_discriminant` and `_unsafe_get_enum_payload`.
// The parts:
//
// - Storage. `__enum_storage` is the enum's only field. `__EnumStorage` is a
//   `!kgen.variant` over the case types, addressed by case index, so it holds
//   the active case's payload and that case's index
//   (`__enum_discriminant_dtype` sizes the index to the case count). `NoneType`
//   stands for no payload, the marker `EnumLike` imposes (MOCO-5080).
//   `__EnumStorage` copies, moves and destroys the
//   active payload and carries the conditional conformances (`Copyable` where
//   every case type is, and so on). The enum's own copy, move and destructor
//   are nothing enum-specific: the struct machinery synthesizes them
//   field-wise from the enum's declared conformances, and there is one field.
//
// - Construction. `__enum_init[id, Case, *Ts](*payload)`, a static method of
//   `__Enum` with `id` the case index, `Case` its payload type and `Ts` the
//   payload's element types, builds the value of that case, writing each
//   element straight into the storage; it is one ordinary function, and whether
//   a call of it runs in the comptime interpreter or at run time depends only
//   on where the call is written. A case without a payload gets a `comptime`
//   member initialized by a call with no arguments, as `eof` above. A case with
//   a payload gets a static method that takes the payload's elements as
//   positional arguments and tail-calls it with them.
//
// - Access. `__get_enum_storage` returns the storage field. The enum's
//   `EnumLike` methods are the other two defaults of `__Enum`, a trait
//   refining `EnumLike` that every enum conforms to; they read the index and
//   the active payload from that field.
//
// - Case lists. `_enum_case_names` and `_enum_case_types` are `comptime`
//   members the compiler computes from the `lit.enum.case` ops. Everything
//   above, and `match`, reads them through `EnumLike`.

static constexpr StringLiteral kEnumStorageFieldName = "__enum_storage";
static constexpr StringLiteral kEnumCaseNamesName = "_enum_case_names";
static constexpr StringLiteral kEnumCaseTypesName = "_enum_case_types";
static constexpr StringLiteral kEnumDiscriminantName = "_get_enum_discriminant";
static constexpr StringLiteral kEnumPayloadName = "_unsafe_get_enum_payload";
static constexpr StringLiteral kEnumEltTypeName = "_enum_elt_type_for_case";
static constexpr StringLiteral kEnumInitName = "__enum_init";
static constexpr StringLiteral kEnumStorageGetterName = "__get_enum_storage";

ASTDecl *LIT::lookupEnumSupportDecl(SharedState &shared, StringRef name,
                                    SMLoc loc) {
  if (shared.hasBuiltinModule()) {
    ModuleASTDecl &module = shared.importModule(
        {"std", "builtin", "_enum"}, /*currentPackage=*/nullptr, loc);
    LookupResult result = shared.lookupAndResolveDecl(
        name, loc, module, /*searchParentScopes=*/false);
    if (result.isSuccess() && result.getIfSuccess().size() == 1)
      return result.getIfSuccess().front();
  }
  shared.emitError(loc) << "internal error: enums require '" << name
                        << "' from 'std.builtin._enum'";
  return nullptr;
}

/// Return the `EnumLike` trait, emitting an error if it is missing.
static TraitASTDecl *lookupEnumLikeTrait(SharedState &shared, SMLoc loc) {
  TraitASTDecl *traitDecl = shared.lookupBuiltinTrait("EnumLike", loc);
  if (!traitDecl)
    shared.emitError(loc) << "internal error: enums require 'EnumLike'";
  return traitDecl;
}

/// Return the cases of `structOp`, in declaration order.
static auto getEnumCases(StructDeclOp structOp) {
  return structOp.getFields().getOps<EnumCaseOp>();
}

/// Return the index of the case of `structOp` named `name` and its op, if any.
static std::optional<std::pair<size_t, EnumCaseOp>>
lookupEnumCase(StructDeclOp structOp, StringRef name) {
  for (auto [idx, caseOp] : llvm::enumerate(getEnumCases(structOp)))
    if (caseOp.getName() == name)
      return std::make_pair(idx, caseOp);
  return std::nullopt;
}

/// Return the types listed in the declaration of `caseOp`: `[Int, Int]` for
/// `case span(Int, Int)`, where `_enum_case_types` has `Tuple[Int, Int]`.
static SmallVector<Type> getEnumCasePayload(EnumCaseOp caseOp) {
  assert(caseOp.getPayloadTypesAttr() &&
         "an enum is synthesized only once every case has resolved");
  auto types = caseOp.getPayloadTypesAttr().getAsValueRange<TypeAttr>();
  return SmallVector<Type>(types.begin(), types.end());
}

LogicalResult StructEmitter::synthesizeEnumCaseConstructor(
    StringRef caseName, ArrayRef<Type> payload, SMLoc caseLoc,
    DocStringAttr docString) {
  MLIRContext *ctx = getContext();
  SMLoc loc = structDecl.getLoc();
  ASTType selfType = structDecl.getSelfType();
  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  for (auto [eltIndex, eltType] : llvm::enumerate(payload)) {
    std::string argName = payload.size() == 1
                              ? std::string("payload")
                              : ("payload" + Twine(eltIndex)).str();
    argTypes.push_back(
        ASTType(eltType).getRefForArgument(argName, /*isMut=*/true));
    argConventions.push_back(ArgConvention::OwnedMem);
    argNames.push_back(StringAttr::get(ctx, argName));
    argPassingKinds.push_back(PassingKind::PosOnly);
  }
  Type resultType = selfType;
  if (!selfType.isRegisterPassable(loc, shared)) {
    resultType = shared.getNoneType();
    argTypes.push_back(
        selfType.getRefForArgument("__result__", /*isMut=*/true));
    argConventions.push_back(ArgConvention::ByRefResult);
    argNames.push_back(StringAttr::get(ctx, "__result__"));
    argPassingKinds.push_back(PassingKind::Implicit);
  }
  // Located at the case, for diagnostics, debug info and doc strings.
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      shared.translateLocation(caseLoc), &structDeclOp.getFields().front());
  auto [fn, fnDecl] = synthesizeFunction(
      structDecl, caseName, /*params=*/{}, PogListAttr::get(ctx), argTypes,
      argConventions, PogListAttr::get(ctx, argNames, argPassingKinds),
      resultType, SpecialFunctionKind::kNormal, caseLoc, builder, FnEffects(),
      /*suffix=*/"", /*synthetic=*/true, InlineLevel::AlwaysNoDebug);
  if (!fn)
    return emitError(caseLoc)
           << "internal error: could not synthesize enum case '" << caseName
           << "'";
  fn.setIsStatic(true);
  fn.setDocStringAttr(docString);
  fnDecl->resolvedness = DeclResolvedness::signature;
  auto atEnd = OpBuilder::atBlockEnd(fn.getBody());
  EndFnOp::create(atEnd, fn.getLoc(), /*unresolved=*/true);
  return success();
}

/// Report every clash between the members an `__enum` gets and the members
/// written in it; failure if there was one.
static LogicalResult checkEnumMemberNames(ASTDecl &enumDecl,
                                          ArrayRef<EnumCaseOp> cases,
                                          ArrayRef<SMLoc> caseLocs,
                                          ASTDecl &enumLikeDecl) {
  SharedState &shared = enumDecl.getShared();
  // Every synthesized or `__Enum`-provided name must be free: the members a
  // user writes live in the same scope. `_enum_elt_type_for_case` is derived
  // from the case types; `_enum_is_exhaustive` may be declared to open the
  // enum.
  constexpr StringLiteral synthesizedNames[] = {
      kEnumStorageFieldName, kEnumCaseNamesName, kEnumCaseTypesName,
      kEnumDiscriminantName, kEnumPayloadName,   kEnumInitName,
      kEnumStorageGetterName};
  constexpr StringLiteral undeclarableNames[] = {
      kEnumStorageFieldName, kEnumCaseNamesName,    kEnumCaseTypesName,
      kEnumDiscriminantName, kEnumPayloadName,      kEnumEltTypeName,
      kEnumInitName,         kEnumStorageGetterName};
  bool hasConflict = false;
  for (StringRef name : undeclarableNames) {
    for (ASTDecl *existing : enumDecl.lookupInCurrentScope(name)) {
      shared.emitError(existing->getLoc())
          << "'" << name << "' is synthesized for enums and cannot be declared";
      hasConflict = true;
    }
  }
  for (auto [constCaseOp, caseLoc] : llvm::zip_equal(cases, caseLocs)) {
    EnumCaseOp caseOp = constCaseOp; // op accessors are not const
    StringRef caseName = caseOp.getName();
    // The member a case generates must not collide with a synthesized or
    // `EnumLike` member either.
    if (llvm::is_contained(synthesizedNames, caseName) ||
        !enumLikeDecl.lookupInCurrentScope(caseName).empty()) {
      shared.emitError(caseLoc)
          << "invalid redefinition of '" << caseName << "'";
      hasConflict = true;
      continue;
    }
    ArrayRef<ASTDecl *> existing = enumDecl.lookupInCurrentScope(caseName);
    if (existing.empty())
      continue;
    auto diag = shared.emitError(caseLoc)
                << "enum case '" << caseName
                << "' conflicts with a member of the same name";
    diag.attachNote(existing.front()->getLoc()) << "member declared here";
    hasConflict = true;
  }
  return failure(hasConflict);
}

/// Compute the values of `enumDecl`'s `_enum_case_names` and
/// `_enum_case_types`, in that order, from its case ops: `noneType` for a case
/// without a payload, a `Tuple` for a case with several payload types.
static FailureOr<std::pair<ParamListAttr, ParamListAttr>>
computeEnumCaseNamesAndTypes(ASTDecl &enumDecl, ArrayRef<EnumCaseOp> cases,
                             TraitType anyTypeTrait, ASTType noneType) {
  MLIRContext *ctx = enumDecl.getShared().getContext();
  SMLoc loc = enumDecl.getLoc();
  // `EnumLike` lists each case's name and payload type: `NoneType` for a case
  // without a payload, a `Tuple` for a case with several.
  // TODO(MOCO-5080): `NoneType` is the marker `EnumLike` imposes; it cannot
  // tell such a case from a payload of type `NoneType`. Every payload becomes
  // a `Tuple`, `Tuple[]` for none.
  IREmitter emitter(enumDecl, EC_Type);
  SyntheticNode node(loc);
  SmallVector<TypedAttr> nameValues, typeValues;
  for (EnumCaseOp caseOp : cases) {
    nameValues.push_back(
        StringAttr::get(caseOp.getName(), StringType::get(ctx)));
    SmallVector<Type> payload = getEnumCasePayload(caseOp);
    ASTType payloadType;
    if (payload.empty())
      payloadType = noneType;
    else if (payload.size() == 1)
      payloadType = payload.front();
    else
      payloadType = emitter.getBuiltinTupleInstantiation(loc, payload);
    if (!payloadType)
      return failure();
    PValue asAnyType = emitter.emitMetaTypeToTraitConversion(
        {CValue(payloadType.mlirType), &node}, anyTypeTrait);
    if (!asAnyType)
      return failure();
    typeValues.push_back(asAnyType.get());
  }
  return std::make_pair(
      ParamListAttr::get(nameValues, ParamListType::get(StringType::get(ctx))),
      ParamListAttr::get(typeValues, ParamListType::get(anyTypeTrait)));
}

void StructEmitter::synthesizeEnumCaseListsAndStorage(
    StructDeclOp enumStorageDeclOp, ParamListAttr caseNames,
    ParamListAttr caseTypes) {
  SMLoc loc = structDecl.getLoc();
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  auto addAlias = [&](StringRef name, TypedAttr value) {
    auto aliasOp =
        AliasDeclOp::create(builder, ParamDeclAttr::get(name, value.getType()));
    aliasOp.setValueAttr(value);
    aliasOp.setHasDocHiddenDecorator(true);
    getDeclResolver().addFullyResolvedDecl(
        aliasOp.getOperation(), StringAttr::get(structDecl.getContext(), name),
        loc, &structDecl);
  };
  addAlias(kEnumCaseNamesName, caseNames);
  addAlias(kEnumCaseTypesName, caseTypes);

  // The storage is the only field: `std.builtin._enum` reads it as field 0.
  StructType storageType = enumStorageDeclOp.bindReference({caseTypes});
  auto storageField =
      StructFieldOp::create(builder, kEnumStorageFieldName, storageType);
  storageField.setIsDocHidden(true);
  getDeclResolver().addFullyResolvedDecl(
      storageField.getOperation(),
      StringAttr::get(structDecl.getContext(), kEnumStorageFieldName), loc,
      &structDecl);
}

LogicalResult StructEmitter::synthesizeEnumMembers() {
  SMLoc loc = structDecl.getLoc();
  SmallVector<EnumCaseOp> cases = llvm::to_vector(getEnumCases(structDeclOp));
  // An enum with errors gets no members: a case that failed to resolve has no
  // payload types, and the members would only add follow-on errors.
  if (structDecl.isErroneous() || llvm::any_of(cases, [](EnumCaseOp caseOp) {
        return !caseOp.getPayloadTypesAttr();
      }))
    return failure();
  // A case's members are located at the case.
  SmallVector<SMLoc> caseLocs = llvm::map_to_vector(cases, [&](EnumCaseOp op) {
    return shared.diags.convertLocToSMLoc(op.getLoc());
  });

  ASTDecl *enumLikeDecl = lookupEnumLikeTrait(shared, loc);
  if (!enumLikeDecl ||
      failed(getDeclResolver().resolveBody(*enumLikeDecl, loc)))
    return failure();
  if (failed(checkEnumMemberNames(structDecl, cases, caseLocs, *enumLikeDecl)))
    return failure();

  ASTDecl *enumStorageDecl =
      lookupEnumSupportDecl(shared, "__EnumStorage", loc);
  TraitType anyTypeTrait = shared.lookupBuiltinTraitType("AnyType", loc);
  // TODO(MOCO-5080): `NoneType` is the "no payload" marker; it becomes
  // `Tuple[]`.
  ASTType noneType = shared.lookupBuiltinType("NoneType", structDecl, loc);
  if (!enumStorageDecl || !anyTypeTrait || noneType.isTypeCheckErrorType())
    return failure();
  auto *enumStorageStruct = dyn_cast<StructASTDecl>(enumStorageDecl);
  if (!enumStorageStruct) {
    emitError(loc) << "internal error: '__EnumStorage' in "
                      "'std.builtin._enum' is not a struct";
    return failure();
  }
  StructDeclOp enumStorageDeclOp = enumStorageStruct->getOp();

  FailureOr<std::pair<ParamListAttr, ParamListAttr>> lists =
      computeEnumCaseNamesAndTypes(structDecl, cases, anyTypeTrait, noneType);
  if (failed(lists))
    return failure();
  auto [caseNames, caseTypes] = *lists;
  synthesizeEnumCaseListsAndStorage(enumStorageDeclOp, caseNames, caseTypes);
  // The case op is unlisted, so its doc string reaches doc validation and doc
  // generation only through the member synthesized for it.
  for (auto [caseOp, caseLoc] : llvm::zip_equal(cases, caseLocs)) {
    StringRef caseName = caseOp.getName();
    SmallVector<Type> payload = getEnumCasePayload(caseOp);
    if (payload.empty()) {
      // TODO: the member is a `comptime` value, and one whose type is not
      // `ImplicitlyCopyable` does not materialize at run time, although a case
      // without a payload owns nothing.
      AliasASTDecl *aliasDecl = synthesizeUnresolvedAlias(caseName);
      aliasDecl->setLoc(caseLoc);
      AliasDeclOp aliasOp = aliasDecl->getOp();
      aliasOp->setLoc(caseOp.getLoc());
      aliasOp.setDocStringAttr(caseOp.getDocStringAttr());
      continue;
    }
    if (failed(synthesizeEnumCaseConstructor(caseName, payload, caseLoc,
                                             caseOp.getDocStringAttr())))
      return failure();
  }
  return success();
}

/// Return `caseIndex` as a compile-time `Int`, the value for the `id`
/// parameter of `__enum_init`.
static TypedAttr getEnumCaseIndexValue(IREmitter &emitter, size_t caseIndex,
                                       const ExprNode *node) {
  TypedAttr indexAttr =
      IntegerAttr::get(IndexType::get(emitter.getContext()), caseIndex);
  CValue index = emitter.emitInt(ASTExprAnd<PValue>{PValue(indexAttr), node},
                                 EC_CallParamValue);
  if (!index || !index.getIfPValue())
    return {};
  return index.getIfPValue().get();
}

/// Return the payload type of case `caseIndex` as the `AnyType` value the
/// enum's `_enum_case_types` member lists for it.
static TypedAttr getEnumCaseTypeValue(ASTDecl &enumDecl, size_t caseIndex) {
  ArrayRef<ASTDecl *> decls = enumDecl.lookupInCurrentScope(kEnumCaseTypesName);
  auto *aliasDecl =
      decls.size() == 1 ? dyn_cast<AliasASTDecl>(decls.front()) : nullptr;
  AliasDeclOp aliasOp = aliasDecl ? aliasDecl->getOp() : AliasDeclOp();
  auto types = aliasOp ? dyn_cast_or_null<ParamListAttr>(aliasOp.getValueAttr())
                       : ParamListAttr();
  if (!types || caseIndex >= types.getValues().size())
    return {};
  return types.getValues()[caseIndex];
}

/// Emit the call `Self.__enum_init[caseIndex, Case](operands...)`, the
/// constructor of case `caseIndex` taking the payload's elements, with `Case`
/// the case's payload type. As for the same call written in Mojo, the enum's
/// own parameters come from `selfType`, and the element types and the pack's
/// origins are inferred from the operands.
static CValue emitEnumInitCall(IREmitter &emitter, ASTDecl &enumDecl,
                               ASTType selfType, size_t caseIndex,
                               CallOperands &&operands, const ExprNode *node) {
  auto fail = [&]() -> CValue {
    operands.dest.resetForError(emitter);
    return {};
  };
  TypedAttr indexValue = getEnumCaseIndexValue(emitter, caseIndex, node);
  TypedAttr caseType = getEnumCaseTypeValue(enumDecl, caseIndex);
  if (!indexValue || !caseType)
    return fail();
  OverloadSet overloads =
      OverloadSet::lookup(enumDecl, selfType, kEnumInitName, node,
                          CallSyntax::kMethodCallSynthetic);
  if (!overloads) {
    emitter.shared.emitError(node->getLoc())
        << "internal error: enum has no '" << kEnumInitName
        << "' member from '__Enum'";
    return fail();
  }
  overloads.explicitParamBindings.add(node, PValue(indexValue));
  overloads.explicitParamBindings.add(node, PValue(caseType));
  auto callee = overloads.filterOverloadSet(
      operands, /*emitDiagnosticOnFailure=*/true, emitter);
  if (!callee.isYes())
    return fail();
  return emitter.emitIndirectCall(callee.getYes(), std::move(operands));
}

/// Whether a written `_enum_is_exhaustive` of `enumDecl` fails conformance
/// to `EnumLike` (e.g. it is not a `Bool`); it is the only member of that
/// conformance a user can write. The conformance reports the error.
static bool enumExhaustiveDeclFailedConformance(ASTDecl &enumDecl, SMLoc loc) {
  SharedState &shared = enumDecl.getShared();
  ASTDecl *enumLikeDecl = lookupEnumLikeTrait(shared, loc);
  if (!enumLikeDecl)
    return false;
  ArrayRef<ASTDecl *> conformanceDecls = enumDecl.lookupInCurrentScope(
      TraitSymbolAttr::get(enumLikeDecl->getSymbolRef()).getFlattenedName());
  if (conformanceDecls.empty())
    return false;
  // The enum's body resolution has checked the conformance by now, so this
  // reads the result.
  ASTDecl &conformanceDecl = *conformanceDecls.front();
  return failed(shared.getDeclResolver().resolveBody(conformanceDecl, loc)) ||
         conformanceDecl.isErroneous();
}

LogicalResult StructEmitter::populateEnumCaseConstructor(DefASTDecl &fnDecl) {
  FnOp fn = fnDecl.getOp();
  // The enum's own errors are already diagnosed, and the `EnumLike`
  // conformance reports a wrong `_enum_is_exhaustive`.
  if (structDecl.isErroneous() ||
      enumExhaustiveDeclFailedConformance(structDecl, fnDecl.getLoc()))
    return failure();
  StringRef name = fn.getDeclName().getValue();
  auto enumCase = lookupEnumCase(structDeclOp, name);
  if (!enumCase)
    return emitError(fn.getLoc())
           << "internal error: '" << name << "' is not an enum case";

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(fn.getLocScope());
  Block *body = fn.getBody();
  body->clear(); // Remove the unresolved lit.end_fn.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockBegin(fn.getLoc(), body);
  IREmitter emitter(fnDecl, builder);
  SyntheticNode node(fnDecl.getLoc());

  // The arguments are the payload's elements, then the result slot when the
  // enum is not register passable; the elements are passed on to
  // `Self.__enum_init[case, Case]`.
  auto argConventions = fn.getFuncTypeGenerator().getArgConventions();
  bool hasResultSlot = !argConventions.empty() &&
                       argConventions.back() == ArgConvention::ByRefResult;
  ExprDest dest = hasResultSlot ? ExprDest(MLValue(body->getArguments().back()),
                                           EC_ReturnValue)
                                : ExprDest(EC_ReturnValue);
  CallOperands operands(CallSyntax::kMethodCallSynthetic, &node,
                        std::move(dest));
  for (Value arg : body->getArguments().drop_back(hasResultSlot))
    operands.add({MRValue(arg), &node});
  CValue result =
      emitEnumInitCall(emitter, structDecl, structDecl.getSelfType(),
                       enumCase->first, std::move(operands), &node);
  if (!result)
    return failure();
  Value returnValue;
  if (!hasResultSlot) {
    SRValue regResult = emitter.emitSRValue({result, &node}, EC_ReturnValue);
    if (!regResult)
      return failure();
    returnValue = regResult;
  }
  emitter.emitNormalReturn(fn.getLoc(), returnValue);
  return success();
}

LogicalResult StructEmitter::resolveEnumCaseAlias(AliasASTDecl &aliasDecl) {
  AliasDeclOp aliasOp = aliasDecl.getOp();
  StringRef name = aliasOp.getDeclName();
  auto enumCase = lookupEnumCase(structDeclOp, name);
  if (!enumCase || !getEnumCasePayload(enumCase->second).empty())
    return emitError(aliasDecl.getLoc())
           << "internal error: '" << name
           << "' is not an enum case without a payload";

  aliasDecl.resolvedness = DeclResolvedness::body;
  if (structDecl.isErroneous() ||
      enumExhaustiveDeclFailedConformance(structDecl, aliasDecl.getLoc())) {
    aliasDecl.setErroneous();
    return success();
  }

  // The value is `Self.__enum_init[case, NoneType]()`, evaluated at compile
  // time.
  ASTType selfType = structDecl.getSelfType();
  IREmitter emitter(structDecl, EC_AliasValue);
  SyntheticNode node(aliasDecl.getLoc());
  CallOperands operands(CallSyntax::kMethodCallSynthetic, &node,
                        ExprDest(EC_AliasValue));
  CValue value = emitEnumInitCall(emitter, structDecl, selfType,
                                  enumCase->first, std::move(operands), &node);
  if (!value || !value.getIfPValue()) {
    aliasDecl.setErroneous();
    return success();
  }
  aliasOp.setParamDeclAttr(ParamDeclAttr::get(name, selfType.mlirType));
  aliasOp.setValueAttr(value.getIfPValue().get());
  return success();
}
