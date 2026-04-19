//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation details of IREmitter that are related to
// value conversions.
//
//===----------------------------------------------------------------------===//

#include "CallEmission.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"

#include "MojoUtils.h"
#include "ParamMatcher.h"
#include "ParserEvaluationContext.h"
#include "SpecializeInf.h"
#include "StructEmitter.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Function Conversions
//===----------------------------------------------------------------------===//

// Strips references from the expected and actual types, reconciling allowed
// differences and extracting the pointee types to compare.
bool checkConventionsConvertible(ArgConvention expectedConv,
                                 ArgConvention actualConv) {
  // DeinitMem is the same as OwnedMem, so we can convert between them.
  if (expectedConv == ArgConvention::DeinitMem)
    expectedConv = ArgConvention::OwnedMem;
  if (actualConv == ArgConvention::DeinitMem)
    actualConv = ArgConvention::OwnedMem;

  // Check the argument convention, reconciling allowed differences and
  // extracting the actual type to compare. This also doesn't check for
  // passing convention, since those are trivially convertible.
  switch (expectedConv) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::ByRefError:
    // We checked that the function effects line up, so if we see
    // `byref_error`, then the other function must have it as well.
    assert(actualConv == ArgConvention::ByRefError &&
           "both functions must be throwing");
    [[fallthrough]];
  case ArgConvention::OwnedMem:
  case ArgConvention::MutRef:
  case ArgConvention::Ref:
  case ArgConvention::Mut:
    if (actualConv == ArgConvention::ReadMem) {
      // If the actual function accepts a read reference, and we have an
      // owned/mutref/ref/mut, we can make a thunk to convert those nicely.
    } else if (actualConv == ArgConvention::ReadReg) {
      // If the actual function accepts a register-passable read, and we have
      // an owned/mutref/ref/mut, we can make a thunk to convert that nicely.
    } else if (actualConv == expectedConv) {
      // Exactly equal, so can convert easily.
    } else {
      return false; // Otherwise, we can't convert.
    }
    break;

  case ArgConvention::ReadMem:
  case ArgConvention::ReadReg:
    if (!llvm::is_contained({ArgConvention::ReadMem, ArgConvention::ReadReg},
                            actualConv))
      return false;
    break;

  case ArgConvention::DeinitMem:
  case ArgConvention::ByRefResult:
    llvm_unreachable("`byref_result` was already handled");
  }

  return true;
}

// TODO: Return more than a boolean, so we can have better error messages.
static bool canConvertFunctionTypes(FnTypeGeneratorType actualGen,
                                    FnTypeGeneratorType expectedGen,
                                    const ExprNode *expr, ASTDecl &declScope) {
  ParamBindings bindings(declScope, expr);
  SpecializeInf paramInf(declScope, expr, /*no params to infer*/ {},
                         PogListAttr::get(declScope.getContext(), {}),
                         expr->getLoc(), /*discardError=*/true);

  ParamMatcher matcher(expr, paramInf, /*allowImplicitConversions=*/true);
  return succeeded(matcher.matchFunctionTypes(actualGen, expectedGen));
}

static bool canConvertGeneratorTypes(ASTExprAnd<CValue> valueExpr,
                                     GeneratorType actual,
                                     GeneratorType expected,
                                     ASTDecl &declScope) {

  // Handle function conversions.
  if (auto actualFnType = sugarDynCast<FnTypeGeneratorType>(actual))
    if (auto expectedFnType = sugarDynCast<FnTypeGeneratorType>(expected)) {
      return canConvertFunctionTypes(actualFnType, expectedFnType,
                                     valueExpr.expr, declScope);
    }

  if (auto actualType = sugarDynCast<FnLiteralTypeGeneratorType>(actual)) {
    if (auto expectedType = sugarDynCast<FnTypeGeneratorType>(expected)) {
      // See if the literal itself has a compatible type.
      return canConvertFunctionTypes(
          actualType.getSymbolConstantAttr().getType(), expectedType,
          valueExpr.expr, declScope);
    }
  }

  // Generators with different parameterization cannot be converted between each
  // other. If the types are equal but the passing conventions are different,
  // then the conversion is allowed.
  // TODO: Consider default parameter values and enable parameter inference to
  // reconcile differences.
  if (actual.getInputParamTypes() != expected.getInputParamTypes())
    return false;

  // We are pulling out the body of the generator to test type convertibility.
  // To do it correctly, we need to replace index ref to name refs. Otherwise,
  // it confuses parameter inference (as index refs are to be inferred).
  ParamRefRemapper remapper;
  for (size_t i = 0, e = actual.getInputParamTypes().size(); i != e; ++i) {
    remapper.parameters.push_back(
        StringAttr::get(actual.getContext(), "Ctx#" + Twine(i)));
  }

  // Otherwise, the bodies must be convertible. This is possible if we can get
  // the body, meaning the value must be a GeneratorAttr.
  auto genAttr =
      sugarDynCastIfPresent<GeneratorAttr>(valueExpr.ir.getIfPValue().get());
  if (!genAttr)
    return false;

  return IREmitter::canImplicitlyConvertToType(
      {remapper.replace(genAttr.getBody()), valueExpr.expr},
      ASTType(remapper.replace(expected.getBody())), declScope);
}

// Strip out irrelevant details of a function that can be rebound away to make
// convertibility checking easier.
static FnType getReducedFnType(FnType sig) {
  MLIRContext *ctx = sig.getContext();

  auto origPogListAttr = sig.getArgListAttrs();
  ArrayRef<PogMetadataAttr> pogs = origPogListAttr.getPogs();

  SmallVector<PassingKind> passingKinds;
  SmallVector<StringAttr> names;
  SmallVector<SmallVector<ConstraintAttr>> constraints;
  SmallVector<VariadicKind> variadics;
  SmallVector<TypedAttr> defaults(sig.getNumArguments(), {});
  for (size_t i = 0, e = sig.getNumArguments(); i != e; ++i) {
    passingKinds.push_back(origPogListAttr.getPassingKind(i));
    names.push_back(pogs[i].getName());
    variadics.push_back(origPogListAttr.getVariadicKind(i));
    constraints.emplace_back(pogs[i].getConstraints());
  }

  auto newPogListAttr = PogListAttr::get(
      ctx, names, passingKinds, variadics, defaults,
      origPogListAttr.getOrigVariadicConvention(), constraints);

  auto metadata = FnMetadataAttr::get(
      newPogListAttr, sig.getNumImplicitOriginDecls(),
      // Don't keep the capture origins, thunks don't care about those. Only the
      // parameter-value passed in at the callsite cares about those.
      {}, sig.getIsNestedOriginExclusivityCheckingDisabled(),
      sig.getMetadata().getConstraints());
  return FuncType::get(sig.getValues(), sig.getArgConventions(),
                       sig.getFnEffects(), metadata);
}

static GeneratorType getReducedGeneratorType(GeneratorType gen) {
  // If the body is a function, we can further reduce it.
  Type bodyType = gen.getBody();
  if (auto fnType = sugarDynCast<FnType>(bodyType))
    bodyType = getReducedFnType(fnType);

  return GeneratorType::get(
      gen.getInputParamTypes(), bodyType,
      PogListAttr::get(gen.getContext(), gen.getInputParamTypes().size()));
}

static std::string generateThunkName(Type expected, Type actual) {
  std::string name;
  llvm::raw_string_ostream os(name);
  ASTType(expected).print(os, /*diags=*/nullptr);
  os << '|';
  ASTType(actual).print(os, /*diags=*/nullptr);

  // Mix in the full signatures to disambiguate.
  std::string sigHash;
  llvm::raw_string_ostream sigHashOs(sigHash);
  expected.print(sigHashOs);
  actual.print(sigHashOs);
  os << '|';
  os << llvm::utohexstr(llvm::xxh3_64bits(sigHash),
                        /*LowerCase=*/true, /*Width=*/16);
  return name;
}

static FnOp generateConversionThunk(Attribute key, ASTDecl &moduleDecl,
                                    SMLoc useLoc) {
  auto &shared = moduleDecl.getShared();
  // Don't generate any debuginfo for the thunk. Push a null scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(/*scope=*/nullptr);

  auto keyValues = cast<ArrayAttr>(key);
  // The actual signature may be wrapped in a GeneratorType that provides the
  // scope for clarifying parameter index references. Unwrap if needed.
  Type keyActualType = cast<TypeAttr>(keyValues[0]).getValue();
  auto actualSignature = dyn_cast<FnTypeGeneratorType>(keyActualType);
  if (!actualSignature)
    actualSignature =
        cast<FnTypeGeneratorType>(cast<GeneratorType>(keyActualType).getBody());
  auto thunkSignature =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[1]).getValue());

  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type. Add the parameters from the
  // expected signature. This contains the types of the clarifying parameters
  // (see TAPCPTTT) and the actual function's input parameters.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator = shared.getParameterEvaluator();
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] :
       llvm::enumerate(thunkSignature.getInputParamTypes())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(
        ParamDeclAttr::get(moduleDecl.mangleUserDefinedParamName(
                               b.getStringAttr("_" + Twine(idx))),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.appendIndexBinding(paramValues.back());
  }
  // Rebind the argument and result types into the scope of the body.
  FunctionType functionType =
      thunkSignature
          .getSpecializedGenerator(paramValues, &shared.getEvaluationContext())
          .getBody()
          .getValues();

  // Add an additional parameter, representing the actual callee. Rebind the
  // actual function type into the scope of the body.
  auto calleeDecl = ParamDeclAttr::get(
      moduleDecl.mangleUserDefinedParamName(b.getStringAttr("callee")),
      evaluator.getReboundType(actualSignature));
  paramDecls.push_back(calleeDecl);

  // Generate a mangled name.
  std::string name = generateThunkName(thunkSignature, actualSignature);

  // Extract the callee's where-clause constraints from the rebound callee
  // type. The evaluator has already remapped index-based parameter references
  // to named references using the thunk's parameter declarations, so these
  // constraints can be used as known assumptions in the thunk's scope.
  // This is needed for TrivialRegisterPassable types with conditional
  // conformance: the witness entry uses a conversion thunk to bridge calling
  // conventions, and the callee (struct method) may carry a where clause.
  auto reboundCalleeType =
      sugarCast<FnTypeGeneratorType>(evaluator.getReboundType(actualSignature));
  ArrayRef<ConstraintAttr> remappedConstraints =
      reboundCalleeType.getBody().getMetadata().getConstraints();

  // Declare the function at the bottom of the decl.
  b = ImplicitLocOpBuilder(mlirLoc, moduleDecl.getDeclEndBuilder());
  FunctionEmitter structEmitter(shared);
  auto [thunk, thunkDecl] = structEmitter.synthesizeFunction(
      moduleDecl, name, paramDecls,
      PogListAttr::get(ctx, thunkSignature.getInputParamTypes().size() + 1),
      functionType.getInputs(), thunkSignature.getArgConventions(),
      PogListAttr::get(ctx, thunkSignature.getNumArguments()),
      functionType.getResults().front(), SpecialFunctionKind::kNormal,
      moduleDecl.getLoc(), b, remappedConstraints,
      thunkSignature.getFnEffects(),
      /*suffix=*/"", /*synthetic=*/true, InlineLevel::Automatic);

  // Annotate the function as a thunk by adding the conversion types.
  NamedAttrList attrs = thunk->getAttrDictionary();
  attrs.set(thunk.getThunkKeyAttrName(), key);

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  attrs.set(thunk.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, InlineLevel::AlwaysNoDebug));

  // Set the attributes.
  thunk->setAttrs(attrs.getDictionary(ctx));

  // Now prepare to emit the call.
  b = ImplicitLocOpBuilder::atBlockBegin(mlirLoc, thunk.getBody());
  IREmitter emitter(*thunkDecl, b);

  // Construct the call operands from the function block arguments.
  SyntheticNode node(useLoc);
  CallOperands operands(CallSyntax::kMethodCall, &node);

  std::optional<size_t> thunkVariadicArgIndexOpt =
      thunkSignature.findPackVarArgIndex();

  bool actualMemResult = actualSignature.hasMemoryOnlyResult();
  ArrayRef<Type> actualArgTypes =
      actualSignature.getArguments().drop_back(actualMemResult);

  for (size_t actualArgIndex = 0; actualArgIndex < actualArgTypes.size();
       actualArgIndex++) {
    bool actualArgIsForVariadic =
        thunkVariadicArgIndexOpt.has_value() &&
        actualArgIndex >= thunkVariadicArgIndexOpt.value();

    Value argForActual;
    KGEN::ArgConvention convForActual;
    if (actualArgIsForVariadic) {
      size_t thunkVariadicArgIndex = thunkVariadicArgIndexOpt.value();
      size_t indexInVariadic = actualArgIndex - thunkVariadicArgIndex;

      MBValue packRefMBValue =
          MBValue(thunk.getArgument(thunkVariadicArgIndex));

      // Emit: the_pack[index]
      auto indexAttr = IntegerAttr::get(IndexType::get(ctx), indexInVariadic);
      CValue indexCValue = emitter.emitInt(
          ASTExprAnd<PValue>{PValue(indexAttr), &node}, EC_ConversionThunk);
      SyntheticNode indexSynthNode(useLoc, indexCValue);
      SyntheticNode packSynthNode(useLoc, packRefMBValue);
      Operand subscriptOperand(&indexSynthNode, useLoc,
                               Operand::PassKind::kKeyword,
                               StringAttr::get(ctx, "index"));
      SubscriptNode packSubscriptNode(&packSynthNode, useLoc, subscriptOperand,
                                      useLoc);
      CValue getItemResult =
          emitter.emitExprCValue(&packSubscriptNode, EC_ConversionThunk);
      if (!getItemResult)
        return {};
      argForActual = getItemResult.getMlirValue();
      convForActual = ArgConvention::ReadMem;
    } else {
      argForActual = thunk.getArgument(actualArgIndex);
      convForActual = thunkSignature.getArgConvention(actualArgIndex);
    }

    AnyValue value;
    switch (convForActual) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the mojo parser");
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::Mut:
    case ArgConvention::MutRef:
      value = MLValue(argForActual);
      break;
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      value = MRValue(argForActual);
      break;
    case ArgConvention::ReadReg:
      value = SRValue(argForActual);
      break;
    case ArgConvention::ReadMem:
      value = MBValue(argForActual);
      break;
    case ArgConvention::Ref:
      value = MBPValue(argForActual);
      break;
    }

    // Pass any required-keyword args with a name.
    StringAttr name;
    if (!actualArgIsForVariadic &&
        thunkSignature.getArgListAttrs().getPassingKind(actualArgIndex) ==
            PassingKind::KwOnly)
      name = thunkSignature.getArgName(actualArgIndex);
    operands.addKeyword(name, {value, node});
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_ConversionThunk);
  bool hasRegisterResult = false;
  if (thunkSignature.isAsync()) {
    // An async call returns a coroutine we have to await.
  } else if (thunkSignature.hasMemoryOnlyResult()) {
    dest = ValueDest(MLValue(thunk.getArguments().back()), EC_ConversionThunk);
  } else {
    hasRegisterResult = true;
  }

  // Bind the function parameters declared on the thunk to the callee. This does
  // NOT include the clarifying parameters -- the callee has already been
  // rebound to them when it was declared on the parameter list.
  //
  // In this example (from TAAMCE):
  //
  //     def ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: def[Y: Bool](read Ship[Z])->None
  //     ](mut s: Ship[Z, Y]):
  //         callee[Y](s) # implicit cast to imm
  //
  // notice how we're calling `callee[Y](s)` and the clarifying parameter Z
  // doesn't appear on that call line.
  TypedAttr calleeParam = BindParamsAttr::get(
      ParamDeclRefAttr::get(calleeDecl),
      ArrayRef(paramValues)
          .take_back(actualSignature.getInputParamTypes().size()),
      &shared.getEvaluationContext());
  assert(sugarCast<FnTypeGeneratorType>(calleeParam.getType())
             .getInputParamTypes()
             .size() == 0);

  // EXPLICIT-COPY-REF-RETURN: If the callee has a ref result and we expect a
  // value result, then we need to copy out of the ref into the value result. As
  // a (very sad) hack, we need to allow explicitly copyable types (not just
  // implicitly) for __next__ in iterators to work.
  // TODO: Eliminate this when __next__ can return references and we have
  // stronger ref result and corresponding iterator traits.  This isn't
  // something we want to support in general.
  bool needsExplicitCopyOut = false;
  ValueDest explicitCopyOutDest(dest.getContext());
  if (actualSignature.isRefResult() && !thunkSignature.isRefResult()) {
    explicitCopyOutDest = std::move(dest);
    needsExplicitCopyOut = true;
  }

  CValue callResult =
      emitter.emitIndirectCall(PValue(calleeParam), std::move(operands), dest);
  // If we need an explicit copy out, emit a call to T(copy=) on the result into
  // the ultimate dest.
  if (needsExplicitCopyOut) {
    CallOperands operands(CallSyntax::kImplicitCopyCtor, node);
    operands.addKeyword(StringAttr::get(shared.getContext(), "copy"),
                        {callResult, node});
    callResult = emitter.emitConstructorCall(
        callResult.getRValueType(), std::move(operands), explicitCopyOutDest);
  }

  // If the callee is async, we got a coroutine. Now await it into the result.
  if (thunkSignature.isAsync()) {
    ValueDest dest(MLValue(thunk.getArguments().back()), EC_ConversionThunk);
    if (!emitter.emitNamedMethodCall(
            "__await__",
            CallOperands(CallSyntax::kMethodCall, &node, {{callResult, node}}),
            dest))
      return {};
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  Value retVal;
  if (hasRegisterResult)
    retVal = emitter.emitSRValue({callResult, node}, EC_ConversionThunk);

  emitter.emitNormalReturn(mlirLoc, retVal);
  return thunk;
}

static CValue convertFunctionGeneratorValue(CValue value, const ExprNode *expr,
                                            FnTypeGeneratorType expected,
                                            IREmitter &emitter,
                                            ValueDest &dest) {
  PValue callee = value.getIfPValue();
  if (!callee) {
    emitter.emitError(
        expr->getLoc(),
        "TODO: function type conversions between closures not supported yet")
        << expr->getRange();
    dest.resetForError(emitter);
    return {};
  }

  if (auto fnLiteralType =
          sugarDynCast<FuncLiteralTypeGeneratorType>(callee.getType())) {
    // Simply convert the literal itself, call the top-most conversion API,
    // because this could be a zero cost conversion without needing to generate
    // a thunk.
    return emitter.emitImplicitConversionToType(
        {PValue(fnLiteralType.getSymbolConstantAttr()), expr}, expected, dest);
  }

  // Strip all sugar so we don't bind parameters wrong.
  // TODO: We could improve this to maintain sugar better.
  callee = getCanonicalAttr(callee.get());
  expected = cast<FnTypeGeneratorType>(getCanonicalType(expected));

  MLIRContext *ctx = expected.getContext();
  auto actual = sugarCast<FnTypeGeneratorType>(callee.getType());

  // Canonicalize the function types. This strips away unnecessary metadata that
  // does not affect the conversion semantics. In other words, a function type
  // and its reduced type can be trivially converted with a rebind.
  auto reducedActual =
      sugarCast<FnTypeGeneratorType>(getReducedGeneratorType(actual));
  auto reducedExpected =
      sugarCast<FnTypeGeneratorType>(getReducedGeneratorType(expected));

  // We need to specially handle when `actual` mentions any parameters in its
  // scope, like how `= read_ship[Z]` mentions the `Z` parameter here:
  //
  //     struct Ship[X: int, Y: Bool]:
  //         pass
  //
  //     def read_ship[X: int, Y: Bool](read s: Ship[X, Y]):
  //         pass
  //
  //     def foo():
  //         alias Z: int = 42
  //         alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None =
  //             read_ship[Z]
  //
  // `read_ship[Z]`s type is `def(read Ship[Y: Bool][Z])`. However, when our
  // thunk accepts that type as an input parameter, the thunk is malformed
  // because it has no idea what `ZC` is (see TAPRCT for more).
  //
  // So, we prepend a "clarifying" parameter to the thunk's input parameters,
  // like the `Z` here:
  //
  //     def ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: def[Y: Bool](read Ship[Z])->None
  //     ](mut s: Ship[Z, Y]):
  //         callee[Y](s) # implicit cast to imm
  //
  // See TAPCPTTT for more.

  SmallVector<Type> thunkParamTypes;
  // `mentionedParamRefs` contains all of `actual`'s mentions of parameters from
  // the containing scope, like the `Z` in the above `read_ship[Z]`.
  // This *only* refers to parameters declared in/by `foo`.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> mentionedParamRefs;
  // NOTE: The walk here to determine the parameter mentions only works if the
  // walk visits types in the same order as lexical parsing. This is because the
  // mentioned parameters can depend on each other, so the list has to have them
  // in an order that keeps the dependencies valid.
  // I *think* we don't need to walk `expected` too... I could be wrong though.
  getCanonicalType(actual).walk(
      [&](ParamDeclRefAttr ref) { mentionedParamRefs.insert(ref); });
  // This replacer will help us figure out the thunk's param types, so the thunk
  // signature has a correct:
  //     mut s: Ship[ship_func_thunk's Z]
  // instead of an incorrect:
  //     mut s: Ship[foo's Z]
  // It also helps us generate some more general signatures for the thunk keys.
  ParameterEvaluator paramRefsReplacer = emitter.shared.getParameterEvaluator();
  for (auto [i, ref] : llvm::enumerate(mentionedParamRefs)) {
    // Add these mentioned param refs as "clarifying" parameters to the thunk,
    // see TAPCPTTT.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(ref.getType()));
    paramRefsReplacer.setDeclBinding(
        ref.getName(), ParamIndexRefAttr::get(i, thunkParamTypes.back()));
  }
  auto reparamActualForThunkKey = sugarCast<FnTypeGeneratorType>(
      paramRefsReplacer.getReboundType(reducedActual));
  // Above, clarifying parameters were at the beginning (and were replaced with
  // `*(0,i) where i < N`).
  //
  // Now, we need to add `expected`'s input params, like the `[Y: Bool]` in:
  //
  //     alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None = ...
  //
  // Note that `expected` contains param refs to parameters declared in/by foo.
  // `expected` does NOT contain paramrefs referring to the callee's function
  // definition's parameters.
  for (auto [i, type] : llvm::enumerate(expected.getInputParamTypes())) {
    // Note that `type` might contain UnboundAttr at this point, that's fine.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(type));
    paramRefsReplacer.appendIndexBinding(ParamIndexRefAttr::get(
        i + mentionedParamRefs.size(), thunkParamTypes.back()));
  }
  // The thunk metadata and function type will mostly look like `expected`,
  // except for the thunk param types (which also includes clarifying
  // parameters, see TAPCPTTT).
  auto thunkMetadata = FnMetadataAttr::get(
      reducedExpected.getArgListAttrs(),
      reducedExpected.getNumImplicitOriginDecls(),
      reducedExpected.getCaptureOrigins(),
      reducedExpected.getIsNestedOriginExclusivityCheckingDisabled(),
      reducedExpected.getFnMetadata().getConstraints());
  auto thunkFuncType = sugarCast<FunctionType>(
      paramRefsReplacer.getReboundType(reducedExpected.getValues()));
  auto thunkSignature = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/thunkParamTypes,
      /*values=*/thunkFuncType,
      /*argConvs=*/reducedExpected.getArgConventions(),
      /*effects=*/reducedExpected.getFnEffects(),
      /*fnMetadata=*/thunkMetadata,
      /*genMetadata=*/PogListAttr::get(ctx, thunkParamTypes.size()));

  // There shouldn't be any ParamDeclRefAttr in the thunk signature, because
  // there's no parent scope param-decls for them to refer to.
#ifndef NDEBUG
  getCanonicalType(thunkSignature).walk([&](ParamDeclRefAttr ref) {
    assert(false);
  });
#endif

  // We can attempt to generate the thunk now.
  // When there are clarifying parameters, `reparamActualForThunkKey` contains
  // depth-1 index references that refer to those parameters. Wrap it in a
  // GeneratorType whose inputParamTypes are the clarifying types so that the
  // depth-1 refs have a valid enclosing scope and don't escape.
  Type keyActualType = reparamActualForThunkKey;
  if (!mentionedParamRefs.empty()) {
    keyActualType = GeneratorType::get(
        ArrayRef(thunkParamTypes).take_front(mentionedParamRefs.size()),
        reparamActualForThunkKey);
  }
  Attribute key = ArrayAttr::get(
      ctx, {TypeAttr::get(keyActualType), TypeAttr::get(thunkSignature)});
  FnOp thunk = emitter.shared.getOrCreateFunctionThunk(
      key, generateConversionThunk, expr->getLoc());
  if (!thunk) {
    dest.resetForError(emitter);
    return {};
  }

  // Now that we have the thunk defined somewhere, we're going to reference it.
  // In the above `foo` example, in this `alias` line:
  //
  //     alias my_func_alias: def(mut Ship[ZC]) -> None =
  //         ship_func_thunk[ZC, read_ship[ZC]]
  //
  // ...we'll now produce the `ship_func_thunk[ZC, read_ship[ZC]]`.

  // First, cast the callee to the reduced actual type.
  auto calleeParam = ParamOperatorAttr::getRebind(callee.get(), reducedActual);

  // Assemble the parameters (`ZC, read_ship[ZC]`) that we'll bind to the thunk.
  ParameterEvaluator evaluator = emitter.shared.getParameterEvaluator();
  for (ParamDeclRefAttr ref : mentionedParamRefs) {
    // Bind the clarifying parameter (see TAPCPTTT).
    evaluator.appendIndexBinding(ref);
  }
  for (Type type :
       ArrayRef(thunkParamTypes).drop_front(mentionedParamRefs.size())) {
    // If there are "remaining input parameters", like in:
    //
    //     alias my_func_alias: def[Y: Bool]() -> None = ...
    //
    // then we leave them unbound (see TARIPNBITM).
    evaluator.appendIndexBinding(
        UnboundAttr::get(evaluator.getReboundType(type)));
  }
  appendThunkCallee(evaluator, calleeParam);

  SymbolConstantAttr symbol = thunk.getBoundSymbolRef(
      emitter.shared.getEvaluationContext(),
      ParameterExprArrayAttr::get(ctx, evaluator.getIndexBindings()));

  // Finally, cast the result back to the expected type.
  return emitter.emitCResult(ParamOperatorAttr::getRebind(symbol, expected),
                             expr, dest);
}

static CValue convertGeneratorValue(CValue value, const ExprNode *expr,
                                    GeneratorType expected, IREmitter &emitter,
                                    ValueDest &dest) {
  // If this is a function generator value, defer to function conversion.
  if (auto expectedFnType = sugarDynCast<FnTypeGeneratorType>(expected)) {
    return convertFunctionGeneratorValue(value, expr, expectedFnType, emitter,
                                         dest);
  }

  // We do not have dynamic generators at all.
  PValue genAttr = value.getIfPValue();
  if (!genAttr) {
    emitter.emitError(expr->getLoc(),
                      "TODO: dynamic generator conversions not supported yet")
        << expr->getRange();
    dest.resetForError(emitter);
    return {};
  }

  // This must be a concrete generator attr, and it should have been ensured by
  // `canConvertGeneratorTypes`
  auto concreteGenAttr = sugarCast<GeneratorAttr>(genAttr.get());
  ValueDest tmpDest(dest.getContext());
  CValue convBody = emitter.emitImplicitConversionToType(
      {concreteGenAttr.getBody(), expr}, expected.getBody(), tmpDest);

  assert(convBody && convBody.getIfPValue());
  auto convGen =
      GeneratorAttr::get(expected.getInputParamTypes(),
                         convBody.getIfPValue().get(), expected.getMetadata());
  return emitter.emitCResult(convGen, expr, dest);
}

//===----------------------------------------------------------------------===//
// Zero Cost Conversions
//===----------------------------------------------------------------------===//

static TypedAttr stripTypeValueUpcast(TypedAttr typeValue) {
  if (auto upcast = sugarDynCast<UpcastAttr>(typeValue))
    return stripTypeValueUpcast(upcast.getInputTypeValue());
  if (auto typeParam = sugarDynCast<TypeParamAttr>(typeValue))
    if (auto paramType = sugarDynCast<ParamType>(typeParam.getTypeValue()))
      return stripTypeValueUpcast(paramType.getParam());
  return typeValue;
}

static TypedAttr stripTypeValueDowncast(TypedAttr typeValue) {
  if (auto downcast = sugarDynCast<DowncastAttr>(typeValue))
    return stripTypeValueDowncast(downcast.getInputTypeValue());
  if (auto typeParam = sugarDynCast<TypeParamAttr>(typeValue))
    if (auto paramType = sugarDynCast<ParamType>(typeParam.getTypeValue()))
      return stripTypeValueDowncast(paramType.getParam());
  return typeValue;
}

static bool canZeroCostConvertParamTypes(ParamType fromParamType,
                                         ParamType toParamType,
                                         SharedState &shared) {
  // If the source & target types are both get_witness on the same type-values,
  // we can zero-cost convert.
  if (auto fromGetWitness =
          sugarDynCast<GetWitnessAttr>(fromParamType.getParam())) {
    if (auto toGetWitness =
            sugarDynCast<GetWitnessAttr>(toParamType.getParam())) {
      if (fromGetWitness.getWitnessName() != toGetWitness.getWitnessName())
        return false;

      auto fromTypeValue = stripTypeValueUpcast(fromGetWitness.getTypeValue());
      auto toTypeValue = stripTypeValueUpcast(toGetWitness.getTypeValue());
      if (fromTypeValue != toTypeValue)
        return false;

      return true;
    }
  }

  // Handle downcast<X> -> X conversions.
  // A downcasted type is semantically equivalent to its underlying type for
  // rebind purposes - the downcast only adds compile-time trait constraints.
  auto fromParam = fromParamType.getParam();
  auto toParam = toParamType.getParam();

  auto strippedFrom = stripTypeValueDowncast(fromParam);
  auto strippedTo = stripTypeValueDowncast(toParam);

  // If either had downcast wrappers, compare the underlying types
  if (strippedFrom != fromParam || strippedTo != toParam) {
    if (strippedFrom == strippedTo)
      return true;
    // Handle combined upcast/downcast cases, e.g. when a downcasted type from
    // struct_field_types is compared against an upcasted metatype. This can
    // occur in generic serialization code that uses reflection to iterate over
    // fields while also using trait-based dispatch.
    if (stripTypeValueUpcast(strippedFrom) == stripTypeValueUpcast(strippedTo))
      return true;
  }

  return false;
}

static FailureOr<bool>
isValidUpCastToTypeType(SharedState &shared, ASTType fromType, ASTType toType) {
  // Trait metatypes/struct MetaMetaType are allowed to upcast to trivial types.
  if (sugarIsa<TypeType>(toType)) {
    // Allowing casting from any metatype to type of all types.
    return sugarIsa<StructMetaType, StructMetaMetaType, TraitType, AnyTraitType,
                    TypeType, NonStructTypeType,
                    FnLiteralTypeGeneratorMetaType>(fromType);
  }

  // Not applicable.
  return failure();
}

/// Returns if a value of the specified type can be coerced to the other type
/// with a zero-cost conversion like a rebind.  This means that values of the
/// two types have exactly the same representation post-elaboration.
bool IREmitter::canZeroCostConvert(ASTType fromType, ASTType toType,
                                   SharedState &shared) {
  if (fromType.isEqualCanon(toType))
    return true; // No rebind needed!
  toType = getCanonicalType(toType);
  fromType = getCanonicalType(fromType);

  FailureOr<bool> upCastable =
      isValidUpCastToTypeType(shared, fromType, toType);
  if (succeeded(upCastable))
    return upCastable.value();

  // fn type is non-struct type (but should it?)
  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(fromType) &&
      sugarIsa<NonStructTypeType>(toType))
    return true;

  // Check for param type conversions.
  if (auto fromParamType = sugarDynCast<ParamType>(fromType))
    if (auto toParamType = sugarDynCast<ParamType>(toType))
      return canZeroCostConvertParamTypes(fromParamType, toParamType, shared);

  // Check for closure structs and dig out their underlying signature types to
  // check whether the conversion can occur.
  auto fromDecl = fromType.getDecl(shared);
  auto toDecl = toType.getDecl(shared);
  if (fromDecl && toDecl) {
    auto fromDeclOp =
        dyn_cast_or_null<StructDeclOp>(fromDecl->getIfOperation());
    auto toDeclOp = dyn_cast_or_null<StructDeclOp>(toDecl->getIfOperation());
    if (fromDeclOp && toDeclOp) {
      FuncTypeGeneratorType fromSig =
          fromDeclOp.getClosureSignature().value_or(nullptr);
      FuncTypeGeneratorType toSig =
          toDeclOp.getClosureSignature().value_or(nullptr);
      if (fromSig && toSig) {
        // Compare the specialized signatures.
        fromSig = fromSig.getSpecializedGenerator(
            fromType.getParamBindings(), &shared.getEvaluationContext());
        toSig = toSig.getSpecializedGenerator(toType.getParamBindings(),
                                              &shared.getEvaluationContext());
        return canZeroCostConvert(fromSig, toSig, shared);
      }
      return false;
    }
  }

  // Check origin downcasting.  The safe conversions are:
  //   Origins with identical mutability will be uniqued and already handled.
  //   Conversion from any mutability to KNOWN immutable is fine.
  //   Conversion from KNOWN mutable to any mutability is fine.
  //   Conversion from with mutability "X" to "X&Y" is known to be fine.
  // We allow KGEN to fold the true and false cases for us.
  if (auto fromOrigin = sugarDynCast<OriginType>(fromType))
    if (auto toOrigin = sugarDynCast<OriginType>(toType)) {
      auto toMut = toOrigin.getIsMutable();
      auto result =
          ParamOperatorAttr::get(POC::And, toMut, fromOrigin.getIsMutable());
      if (result == toMut)
        return true;
    }

  // Check reference downcasting.  The only thing allowed to disagree is the
  // origin set / mutability.
  if (auto fromRef = sugarDynCast<RefType>(fromType)) {
    if (auto toRef = sugarDynCast<RefType>(toType)) {
      // Element types and address space have to be exactly equal.
      if (fromRef.getAddressSpace() != toRef.getAddressSpace() ||
          !ASTType(fromRef.getElementType())
               .isEqualCanon(toRef.getElementType()))
        return false;

      // Verify compatible OriginType(mutability).  This is checking the type
      // of the origin, which contains its mutability specifier.
      auto toOriginType = toRef.getOriginType();
      if (fromRef.getOriginType() != toOriginType &&
          !canZeroCostConvert(fromRef.getOriginType(), toOriginType, shared))
        return false;

      // We allow converting an "any" origin to anything concrete.
      // NOTE: This is not memory safe; we should make this an explicit
      // operation someday.
      if (sugarIsa<AnyOriginAttr>(fromRef.getOrigin()))
        return true;

      // FIXME: People are using things StaticString to refer to comptime
      // strings, even though StaticString is a runtime concept :-/.
      if (sugarIsa<ComptimeOriginAttr>(fromRef.getOrigin())) {
        if (auto originField =
                sugarDynCast<OriginFieldAttr>(toRef.getOrigin())) {
          if (isa<StaticOriginAttr>(originField.getBase()) &&
              originField.getField().str() == "__constants__" &&
              originField.getType().isMutableKnown(false)) {
            return true;
          }
        }
      }

      // We can convert origin subset to a origins superset.
      auto toOrigin = toRef.getOrigin();
      auto originUnion = OriginUnionAttr::get(
          {toOrigin, OriginMutCastAttr::get(fromRef.getOrigin(), toOriginType)},
          toOriginType);
      return toOrigin == originUnion;
    }
  }

  if (auto actual = sugarDynCast<FnLiteralTypeGeneratorType>(fromType))
    if (auto expected = sugarDynCast<FnTypeGeneratorType>(toType))
      return canZeroCostConvert(actual.getSymbolConstantAttr().getType(),
                                expected, shared);

  // Otherwise handle function conversions.
  auto from = sugarDynCast<FnTypeGeneratorType>(fromType);
  auto to = sugarDynCast<FnTypeGeneratorType>(toType);
  if (!from || !to)
    return false;

  // Allow signature types to be converted for free if they differ only in
  // argument names, parameter names, passing kinds, or implicit origins.
  size_t fromNumArgs = from.getNumArguments();
  if (fromNumArgs != to.getNumArguments())
    return false;
  if (from.getArgConventions() != to.getArgConventions())
    return false;

  // Result types, and input/result parameter types must match exactly.
  if (from.getResults() != to.getResults() ||
      from.getInputParamTypes() != to.getInputParamTypes() ||
      from.getFnEffects() != to.getFnEffects())
    return false;

  // The input argument types may have different implicit origins but otherwise
  // must match exactly.
  for (auto [idx, fromTy, toTy, conv] : llvm::enumerate(
           from.getArguments(), to.getArguments(), from.getArgConventions())) {
    Type fromTyCmp = RefType::stripRefConvention(fromTy, conv);
    Type toTyCmp = RefType::stripRefConvention(toTy, conv);
    if (!ASTType(fromTyCmp).isEqualCanon(toTyCmp))
      return false;

    // If the argument has a required keyword, then the two must match names.
    if (from.getArgListAttrs().getPassingKind(idx) == PassingKind::KwOnly ||
        to.getArgListAttrs().getPassingKind(idx) == PassingKind::KwOnly) {
      if (from.getArgName(idx) != to.getArgName(idx))
        return false;
    }
  }

  // Otherwise, everything seems compatible.
  return true;
}

/// If there is a common type shared between the two reference types, return
/// it. Otherwise return null.
RefType IREmitter::getCommonRefType(RefType ref1, RefType ref2) {
  if (ref1 == ref2)
    return ref1;
  // Element types and addr spaces have to be exactly equal.
  auto eltType = ref1.getElementType();
  if (!ASTType(eltType).isEqualCanon(ref2.getElementType()) ||
      ref1.getAddressSpace() != ref2.getAddressSpace())
    return {};

  // If so, we can form a common type with a subset of their mutability and
  // a union of their origins.
  auto isMutableAttr =
      ParamOperatorAttr::get(POC::And, ref1.isMutable(), ref2.isMutable());

  auto l1 = OriginMutCastAttr::get(ref1.getOrigin(), isMutableAttr);
  auto l2 = OriginMutCastAttr::get(ref2.getOrigin(), isMutableAttr);
  auto origin =
      OriginUnionAttr::get({l1, l2}, sugarCast<OriginType>(l1.getType()));
  return RefType::get(eltType, origin, ref1.getAddressSpace());
}

/// If there is a shared supertype for the two specified types, return it in
/// 'result' and return success.
///
/// For example, we may have two derived classes that have the same base class
/// even if neither is convertible to the other.
///
/// This function uses `__merge_with__` if available, otherwise it uses
/// implicit conversions to find a common match.  If a `__merge_with__` is
/// involved, the PValue for the function to invoke is returned.
enum CommonTypeResult {
  CTR_Success,
  CTR_Ambiguous,
  CTR_NoCommonType,
  CTR_MergeWithConflict,
  CTR_MergeWithConvertFail, // One __merge_with__ exists, but other doesn't work
};

static std::tuple<CommonTypeResult, PValue, PValue>
findCommonType(ASTExprAnd<CValue> val1, ASTExprAnd<CValue> val2,
               ASTType &result, IREmitter &emitter, ASTType contextualType) {

  // If the types already match, then we're done.
  ASTType type1 = val1.ir.getRValueType();
  ASTType type2 = val2.ir.getRValueType();

  auto succeed =
      [&](ASTType type, PValue lhsMWPV = {},
          PValue rhsMWPV = {}) -> std::tuple<CommonTypeResult, PValue, PValue> {
    result = type;
    return {CTR_Success, lhsMWPV, rhsMWPV};
  };

  if (type1.isEqualCanon(type2))
    return succeed(type1);

  // Ok, they are different types.  If either type has a __merge_with__ member,
  // then we use that in preference to anything else.

  // This checks to see if 'src' has a __merge_with__ member that unambiguously
  // takes 'other' as an parameter. If so it returns the PValue for the method
  // and the result type of calling the method.
  auto lookupMergeWith = [&](ASTExprAnd<CValue> srcValue, ASTType srcType,
                             ASTType otherType) -> std::pair<PValue, ASTType> {
    // Look up __merge_with__ and bind other_type.
    OverloadSet os =
        OverloadSet::lookup(emitter.declScope, srcType, "__merge_with__",
                            srcValue.expr, CallSyntax::kMethodCall);
    os.paramBindings.add(srcValue.expr, PValue(otherType),
                         StringAttr::get(emitter.getContext(), "other_type"));
    CallOperands operands(CallSyntax::kMethodCall, srcValue.expr, {srcValue});
    auto res = os.filterOverloadSet(
        operands, /*emitDiagnosticsOnFailure=*/false, emitter);
    if (!res)
      return {{}, {}};
    return {res, res.getType().getSignatureUserResultType()};
  };

  auto [lhsMWPV, lhsMPType] = lookupMergeWith(val1, type1, type2);
  auto [rhsMWPV, rhsMPType] = lookupMergeWith(val2, type2, type1);

  // Handle two __merge_with__ methods.
  if (lhsMWPV && rhsMWPV) {
    if (!lhsMPType.isEqualCanon(rhsMPType))
      return {CTR_MergeWithConflict, lhsMWPV, rhsMWPV};
    // If both convert to the same type, then we're good.
    return succeed(lhsMPType, lhsMWPV, rhsMWPV);
  }
  // If there is one __merge_with__ method, then we use that if the other type
  // converts to the result value.
  if (lhsMWPV) {
    if (IREmitter::canImplicitlyConvertToType(val2, lhsMPType,
                                              emitter.declScope))
      return succeed(lhsMPType, lhsMWPV, PValue());
    result = lhsMPType;
    return {CTR_MergeWithConvertFail, lhsMWPV, PValue()};
  }
  if (rhsMWPV) {
    if (IREmitter::canImplicitlyConvertToType(val1, rhsMPType,
                                              emitter.declScope))
      return succeed(rhsMPType, PValue(), rhsMWPV);
    result = rhsMPType;
    return {CTR_MergeWithConvertFail, PValue(), rhsMWPV};
  }

  // Otherwise, we have no __merge_with__ method, see if there is a contextual
  // type.  If so, convert to that.
  if (contextualType) {
    if (IREmitter::canImplicitlyConvertToType(val1, contextualType,
                                              emitter.declScope) &&
        IREmitter::canImplicitlyConvertToType(val2, contextualType,
                                              emitter.declScope))
      return succeed(contextualType);
  }

  // Otherwise, check out implicit conversions from one value to the other.

  // If one type implicit converts to the other, then the other is a common
  // type.  Don't do this if both convert to each other, this would be
  // ambiguous.
  bool isConvertibleToType2 =
      IREmitter::canImplicitlyConvertToType(val1, type2, emitter.declScope);
  bool isConvertibleToType1 =
      IREmitter::canImplicitlyConvertToType(val2, type1, emitter.declScope);
  if (isConvertibleToType2 && !isConvertibleToType1)
    return succeed(type2);
  if (isConvertibleToType1 && !isConvertibleToType2)
    return succeed(type1);
  if (isConvertibleToType1 && isConvertibleToType2)
    return {CTR_Ambiguous, PValue(), PValue()};

  // If one or the other type is nonmaterializable, the conversion is free,
  // so check to see if there is an unambiguous common type.
  bool type2ConvertsToType1Nonmat = false;
  bool type1ConvertsToType2Nonmat = false;
  auto type1Nonmat = type1.getNonmaterializableTarget(emitter.shared);
  auto type2Nonmat = type2.getNonmaterializableTarget(emitter.shared);
  if (type1Nonmat)
    type2ConvertsToType1Nonmat = IREmitter::canImplicitlyConvertToType(
        val2, type1Nonmat, emitter.declScope);
  if (type2Nonmat)
    type1ConvertsToType2Nonmat = IREmitter::canImplicitlyConvertToType(
        val1, type2Nonmat, emitter.declScope);

  if (type2ConvertsToType1Nonmat && !type1ConvertsToType2Nonmat)
    return succeed(type1Nonmat);
  if (type1ConvertsToType2Nonmat && !type2ConvertsToType1Nonmat)
    return succeed(type2Nonmat);
  if (type1ConvertsToType2Nonmat && type2ConvertsToType1Nonmat) {
    if (type1Nonmat.isEqualCanon(type2Nonmat))
      return succeed(type1Nonmat);
    return {CTR_Ambiguous, PValue(), PValue()};
  }

  // No common type found.
  return {CTR_NoCommonType, PValue(), PValue()};
}

/// Given two values that need to match, try to coerce one to the other if they
/// disagree on type.  This emits an error (when loc is non-null) and returns
/// failure if the request is ambiguous or impossible.
///
/// The 'configEmitter' function is called to set the insertion point of the
/// emitter for the true/false branches of the conditional.
ParseResult IREmitter::coerceTypesToEachOther(
    SMLoc loc, CValue &lhs, const ExprNode *lhsExpr, CValue &rhs,
    const ExprNode *rhsExpr, std::function<void(bool isLHS)> configEmitter,
    ASTType contextualType) {
  if (!configEmitter)
    configEmitter = [&](bool isLHS) {};

  if (!lhs || !rhs)
    return failure();

  // If they are the same or if there is a common type between these, convert
  // them to it.
  ASTType commonType;
  auto [commonTypeResult, lhsMWPV, rhsMWPV] = findCommonType(
      {lhs, lhsExpr}, {rhs, rhsExpr}, commonType, *this, contextualType);

  // If we failed and have no source location, we just return failure without
  // returning an error.
  if (commonTypeResult != CTR_Success && !loc.isValid())
    return failure();

  ASTType lhsType = lhs.getRValueType(), rhsType = rhs.getRValueType();
  switch (commonTypeResult) {
  case CTR_Success:
    break;
  case CTR_NoCommonType:
    emitError(loc, "value of type ")
        << lhsType << " is not compatible with value of type " << rhsType
        << lhsExpr->getRange() << rhsExpr->getRange();
    return failure();
  case CTR_Ambiguous: {
    auto diag = emitError(loc, "ambiguous merge: left value has type ")
                << lhsType << " and right value has type " << rhsType
                << ", and both convert to each other" << lhsExpr->getRange()
                << rhsExpr->getRange();
    diag.attachNote(loc)
        << "you could disambiguate by casting the left value to " << rhsType
        << lhsExpr->getRange();
    diag.attachNote(loc) << "or cast the right value to " << lhsType
                         << rhsExpr->getRange();
    return failure();
  }
  case CTR_MergeWithConflict: {
    auto diag = emitError(loc, "value of types ")
                << lhsType << " and " << rhsType
                << " have '__merge_with__' methods that disagree on common type"
                << lhsExpr->getRange() << rhsExpr->getRange();
    auto lhsDest = lhsMWPV.getType().getSignatureUserResultType();
    auto rhsDest = rhsMWPV.getType().getSignatureUserResultType();
    diag.attachNote(loc) << "one returns " << lhsDest
                         << " and the other returns " << rhsDest;
    return failure();
  }
  case CTR_MergeWithConvertFail: {
    auto diag = emitError(loc, "value of types ")
                << lhsType << " and " << rhsType << " cannot be merged to type "
                << commonType << lhsExpr->getRange() << rhsExpr->getRange();
    // One of lhsMWPV/rhsMWPV will be nonnull, indicating which mergewith.
    diag.attachNote(loc) << (lhsMWPV ? rhsType : lhsType)
                         << " does not implicitly convert to " << commonType;
    return failure();
  }
  }

  // Okay we found a successful conversion path.  See if we need to apply any
  // __merge_with__ methods first.
  if (lhsMWPV) {
    configEmitter(/*isLHS*/ true);
    ValueDest dest(EC_MergeWith);
    lhs = emitIndirectCall(
        lhsMWPV,
        CallOperands(CallSyntax::kMethodCall, lhsExpr, {{lhs, lhsExpr}}), dest);
  }
  if (rhsMWPV) {
    configEmitter(/*isLHS*/ false);
    ValueDest dest(EC_MergeWith);
    rhs = emitIndirectCall(
        rhsMWPV,
        CallOperands(CallSyntax::kMethodCall, rhsExpr, {{rhs, rhsExpr}}), dest);
  }

  // Next apply any implicit conversions that may be needed.
  if (!lhsType.isEqualCanon(commonType)) {
    configEmitter(/*isLHS*/ true);
    lhs = emitCValue({lhs, lhsExpr}, EC_OperatorOperandValue, commonType);
  }
  if (!rhsType.isEqualCanon(commonType)) {
    configEmitter(/*isLHS*/ false);
    rhs = emitCValue({rhs, rhsExpr}, EC_OperatorOperandValue, commonType);
  }

  // If we are in a dynamic context and the result is nonmaterializable, then
  // we need to emit the conversion in the parameter domain before the
  // conditional and decide what the result type should be based on that.
  if (builder) {
    if (auto mat = commonType.getNonmaterializableTarget(shared)) {
      configEmitter(/*isLHS*/ true);
      lhs = emitCValue({lhs, lhsExpr}, EC_CondExpr, mat);
      configEmitter(/*isLHS*/ false);
      rhs = emitCValue({rhs, rhsExpr}, EC_CondExpr, mat);
    }
  }

  // Ensure sugar types agree.
  if (lhs && rhs &&
      lhs.getRValueType().mlirType != rhs.getRValueType().mlirType) {
    configEmitter(/*isLHS*/ false);
    Type destType;
    // LHS and RHS may differ in MValue'ness.  The LHS might be an SRValue and
    // the RHS may be an MLValue for example.
    if (rhs.isMValue())
      destType = rhs.getMValueType().getWithElement(lhs.getRValueType());
    else
      destType = lhs.getRValueType();
    rhs = rebindValue({rhs, rhsExpr}, destType);
  }

  return success(lhs && rhs);
}

/// Given a value of a type that can be zero cost converted to another type,
/// emit a rebind or other operation to get it in the right type.
PValue IREmitter::emitZeroCostConvert(PValue value, ASTType toType,
                                      SharedState &shared) {
  assert(toType.mlirType != value.getType() && "Already the same");

  // PValues of origin type have a special conversion.
  if (sugarIsa<OriginType>(toType) && sugarIsa<OriginType>(value.getType()))
    value = OriginMutCastAttr::get(value, toType);

  if (sugarIsa<TypeType>(toType) && sugarIsa<TraitType>(value.getType()))
    return TypeParamAttr::get(ASTType(value), toType);

  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(value.getType()) &&
      sugarIsa<NonStructTypeType>(toType))
    return TypeParamAttr::get(ASTType(value), toType);

  if (auto actual = sugarDynCast<FnLiteralTypeGeneratorType>(value.getType()))
    if (auto expected = sugarDynCast<FnTypeGeneratorType>(toType))
      return ParamOperatorAttr::getRebind(actual.getSymbolConstantAttr(),
                                          expected);

  return ParamOperatorAttr::getRebind(value.get(), toType);
}

CValue IREmitter::emitZeroCostConvert(ASTExprAnd<CValue> value,
                                      ASTType toType) {
  assert(toType.mlirType != value.ir.getType() && "Already the same");

  // PValue handling has a helper.
  if (auto pv = value.ir.getIfPValue())
    return emitZeroCostConvert(pv, toType, shared);

  // The RValue types need to be rebound, but MValues have a level of
  // reference around them that we want to maintain.
  if (value.ir.isMValue())
    toType = value.ir.getMValueType().getWithElement(toType);

  // Rebind the value if we can.
  return rebindValue(value, toType);
}

//===----------------------------------------------------------------------===//
// Trait conversions
//===----------------------------------------------------------------------===//

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkMLIRTypeConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  // Use a special wrapper decl in the builtins as stubs.
  ASTType wrapperType = shared.getBuiltinStubsMLIRType(loc);
  return wrapperType.checkConformance(
             trait, shared,
             ASTDecl::getAssumptionsFromScope(wrapperType.getDecl(shared))) ==
         ConformanceResult::Yes;
}

/// Emit a conversion from an MLIR type to a trait type by materializing stubs
/// for the type's witness table.
PValue IREmitter::bindNonStructTypeToTrait(ASTExprAnd<CValue> value,
                                           TraitType trait) {
  // Only parameter-domain type-values are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    shared.emitError(value.expr->getLoc(),
                     "existentials are not supported yet!");
    return {};
  }

  // If the function generator type is upcastable to a non-struct type (but
  // should it?, esp. for parametric type. We can not easily disable the
  // conversion at the moment since many existing code relies on it).
  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(typeValue.getType()))
    typeValue =
        UpcastAttr::get(NonStructTypeType::get(getContext()), typeValue.get());

  ASTType mlirType = typeValue.getIfTypeValue();
  SMLoc loc = value.expr->getLoc();

  // Use a special wrapper decl in the builtins as stubs.
  ASTType wrapperType = shared.getBuiltinStubsMLIRType(loc);
  ASTDecl *wrapperDecl = wrapperType.getDecl(shared);
  if (!wrapperDecl ||
      !isa_and_nonnull<StructDeclOp>(wrapperDecl->getIfOperation())) {
    shared.emitError(loc, "malformed builtin._stubs.__MLIRType");
    return {};
  }

  // Explicitly check that the wrapper conforms to the trait so that
  // conformances & special functions may be generated.  __MLIRType has only
  // unconditional conformances, so no caller scope is needed.
  if (wrapperType.checkConformance(trait, shared, {}) !=
      ConformanceResult::Yes) {
    MojoInflightDiag diag =
        shared.emitError(value.expr->getLoc(), "cannot bind MLIR type ")
        << mlirType << " to trait " << ASTType(trait);
    return {};
  }

  // If the type is a param type, then we just need to upcast it to the trait.
  if (auto paramType = sugarDynCast<ParamType>(mlirType)) {
    return UpcastAttr::get(trait, PValue(paramType.getParam()));
  }

  // Otherwise, create a new type value whose witness table is provided by the
  // wrapper stub.
  ASTType boundWrapper = cast<StructDeclOp>(wrapperDecl->getIfOperation())
                             .bindReference({typeValue});
  return TypeParamAttr::get(boundWrapper, mlirType, trait);
}

//===----------------------------------------------------------------------===//
// Generalized Implicit Conversions
//===----------------------------------------------------------------------===//

static ASTDecl *getClosureTraitDecl(SharedState &shared,
                                    const TraitType &traitTy) {
  for (const auto &symbol : traitTy.getSymbols()) {
    auto &symbolDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
    if (symbolDecl.isErroneous())
      continue;

    if (auto traitDeclOp =
            dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
        traitDeclOp && traitDeclOp.getDefinesClosure())
      return &symbolDecl;
  }

  return nullptr;
}

/// Build the concrete closure-wrapper type instantiated with a function symbol.
static Type getConcreteClosureWrapperTypeForFnSymbol(SharedState &shared,
                                                     ASTDecl &declScope,
                                                     SMLoc loc,
                                                     PValue fnPValue) {
  auto fnSig = cast<FnTypeGeneratorType>(fnPValue.getType());
  ASTDecl &moduleDecl = *declScope.getNearestDeclOfType<FileModuleOp>();
  auto &closureEmitter = shared.getClosureEmitter();
  auto rvClosureTrait = shared.getOrCreateClosureTrait(loc, moduleDecl, fnSig);
  ASTDecl *wrapper = closureEmitter.createFnStructWrapper(
      moduleDecl, *rvClosureTrait, fnSig, loc);
  assert(wrapper && "createFnStructWrapper must return a declaration");

  auto structDeclOp = dyn_cast<StructDeclOp>(wrapper->getIfOperation());
  assert(structDeclOp && "createFnStructWrapper must return a StructDeclOp");
  assert(!structDeclOp.getInputParams().empty() &&
         "closure wrapper must have an implementation parameter");

  TypedAttr fnVal = ParamOperatorAttr::getRebind(
      fnPValue.get(), structDeclOp.getInputParams().front().getType());
  return structDeclOp.bindReference({fnVal});
}

// Returns true/false to indicate that whether a type value can be upcast to a
// trait.
// Returns failure when it is an non-applicable cases (i.e., `fromType` is not a
// typetype and/or `toType` is not a trait type).
FailureOr<bool> IREmitter::canMetaTypeUpCastTo(SharedState &shared, SMLoc loc,
                                               ASTType fromType, ASTType toType,
                                               ASTDecl *declScope) {
  if (isEqualCanon(fromType, toType))
    return true;

  // Trait metatypes/struct MetaMetaType are allowed to upcast to trivial
  // types.
  FailureOr<bool> upCastable =
      isValidUpCastToTypeType(shared, fromType, toType);
  if (succeeded(upCastable))
    return upCastable;

  // Values of known {struct/trait/mlir} type can convert to any trait type
  // they implement.
  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType.extractMetaType())) {
    TraitType trait = anyTrait.getTraitType();
    bool result = false;

    if (sugarIsa<NonStructTypeType>(fromType)) {
      // MLIR types can conform to traits that have limited requirements.
      // AnyTraitType (the type of all traits) conforms to traits with only a
      // destructor (e.g. AnyType) since all traits have that.
      result = checkMLIRTypeConformance(shared, loc, trait);
    } else if (sugarIsa<StructMetaMetaType>(fromType.extractMetaType()) ||
               sugarIsa<AnyTraitType>(fromType.extractMetaType())) {
      if (ASTType(fromType).getDecl(shared)) {
        // Check for closure rebindability.
        for (const auto &symbol : trait.getSymbols()) {
          auto &symbolDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
          if (auto traitDeclOp =
                  dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
              traitDeclOp && traitDeclOp.getDefinesClosure()) {
            if (succeeded(shared.closureEmitter->isCompatibleWith(
                    fromType, &symbolDecl))) {
              return true;
            }
          }
        }

        // Assumptions needed: e.g. `where AllWritable[*Ts]` proves
        // Tuple[*Ts]: Writable when binding to a Writable parameter.
        // Assumptions needed: implicit conversion of e.g. Tuple[*Ts] to
        // Writable
        // inside a fn with `where AllWritable[*Ts]`.
        auto assumptions = ASTDecl::getAssumptionsFromScope(declScope);
        return fromType.checkConformance(trait, shared, assumptions) ==
               ConformanceResult::Yes;
      }
    } else if (auto fnGen =
                   sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaType>(
                       fromType)) {
      TraitType closureTrait = anyTrait.getTraitType();
      if (auto traitDecl = getClosureTraitDecl(shared, closureTrait)) {
        TypedAttr fnPValue = fnGen.getType().getSymbolConstantAttr();
        Type concreteWrapperType = getConcreteClosureWrapperTypeForFnSymbol(
            shared, *declScope, loc, fnPValue);
        return succeeded(shared.getClosureEmitter().isCompatibleWith(
            concreteWrapperType, traitDecl));
      }
      // Maintain convertibility as a MLIR type ...
      return checkMLIRTypeConformance(shared, loc, closureTrait);
    } else {
      // This isn't relevant, e.g. in function pointer to closure case.
      return failure();
    }
    return result;
  }

  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType)) {
    ASTType concreteType;
    // 2 cases, e.g,:
    // 1st, AnyTraitType[Copyable] to AnyTraitType[AnyType].
    // 2nd, Meta[Meta[Int]] to AnyTraitType[Copyable]
    if (auto rvAnyTrait = sugarDynCast<AnyTraitType>(fromType)) {
      concreteType = ASTType(rvAnyTrait.getTraitType());
    } else if (auto mmType = sugarDynCast<StructMetaMetaType>(fromType)) {
      concreteType = ASTType(mmType.getType());
    }

    // Assumptions needed: e.g. AnyTraitType[Copyable] → AnyTraitType[Movable]
    // upcast when the Copyable conformance depends on caller assumptions.
    if (concreteType) {
      auto assumptions = ASTDecl::getAssumptionsFromScope(declScope);
      return concreteType.checkConformance(anyTrait.getTraitType(), shared,
                                           assumptions) ==
             ConformanceResult::Yes;
    }
  }

  // Not applicable.
  return failure();
}

//===----------------------------------------------------------------------===//
// Generalized Implicit Conversions
//===----------------------------------------------------------------------===//

static bool isClosureWrapperStruct(SharedState &shared, PValue value,
                                   LIT::StructType structTy) {
  if (!value)
    return false;
  ASTDecl &decl =
      shared.declResolver->getDeclForTypeSymbol(structTy.getSymbolRef());
  if (StructDeclOp structOp = dyn_cast<StructDeclOp>(decl.getIfOperation())) {
    return structOp.getDefinesClosure() && !structTy.getParamValues().empty() &&
           isEqualCanon(structTy.getParamValues().front(), value.get());
  }

  return false;
}

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
///
/// CAUTION: This method must line up with `emitImplicitConversionToType`!!!
bool IREmitter::canImplicitlyConvertToType(ASTExprAnd<CValue> value,
                                           ASTType requiredType,
                                           ASTDecl &declScope) {
  auto &shared = declScope.getShared();
  assert(value.ir && "Should only query valid values");
  ASTType rvType = value.ir.getRValueType();

  // If it already matches, then we're done.
  if (rvType.isEqualCanon(requiredType))
    return true;

  // If the types are identical after elaboration then they are implicitly
  // convertible.
  if (canZeroCostConvert(rvType, requiredType, shared))
    return true;

  // Origin values can convert into an OriginSet by becoming a member of the
  // set.  OriginSet is a singleton type, the value carries the origins.
  if (sugarIsa<OriginType>(rvType) && sugarIsa<OriginSetType>(requiredType))
    return true;

  // Check to see if we already cached this convertibility check.
  std::optional<bool> cache =
      shared.getCachedImplicitConvertibility(rvType, requiredType);
  if (cache.has_value())
    return cache.value();

  auto cacheAndReturnVal = [&shared](ASTType from, ASTType to,
                                     bool isConvertible) -> bool {
    // Cache the result of this convertibility check.
    shared.cacheImplicitConvertibility(from, to, isConvertible);
    return isConvertible;
  };

  FailureOr<bool> canUpCast = canMetaTypeUpCastTo(
      shared, value.expr->getLoc(), rvType, requiredType, &declScope);
  if (succeeded(canUpCast))
    return cacheAndReturnVal(rvType, requiredType, canUpCast.value());

  if (sugarIsa<ParamListType>(rvType) &&
      sugarIsa<ParamListType>(requiredType)) {
    // If the element types of the variadic is meta type (AnyStruct/AnyTrait),
    // we allow them to be implicitly converted.
    //
    // That is, we allow `VariadicOf[Copyable]       -> VariadicOf[AnyType]`
    // and,              `VariadicOf[AnyStruct[xxx]] -> VariadicOf[AnyType]`
    //
    // Notably, this does NOT support implicit conversion between from
    // `Variadic[Int]` to `Variadic[UInt]`
    ASTType toEltTp = sugarCast<ParamListType>(requiredType).getElementType();
    ASTType fromEltTp = sugarCast<ParamListType>(rvType).getElementType();
    // Reuse assumptions from above for variadic element upcast.
    FailureOr<bool> canUpCast = canMetaTypeUpCastTo(
        shared, value.expr->getLoc(), fromEltTp, toEltTp, &declScope);
    if (succeeded(canUpCast))
      return cacheAndReturnVal(rvType, requiredType, canUpCast.value());
  }

  // Support implicit conversions of generator types (incl. non-trivial function
  // generator conversions).
  if (auto requiredGenerator = sugarDynCast<GeneratorType>(requiredType)) {
    bool result = false;
    if (auto rvGeneratorType = sugarDynCast<GeneratorType>(rvType))
      result = canConvertGeneratorTypes(value, rvGeneratorType,
                                        requiredGenerator, declScope);
    return cacheAndReturnVal(rvType, requiredType, result);
  }

  // Functions can implicitly convert to their corresponding closure wrapper.
  // This is distinct from converting to a closure trait.
  if (sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(rvType)) {
    if (auto structMeta =
            sugarDynCast<StructMetaType>(requiredType.extractMetaType())) {
      StructType structTy = structMeta.getType();
      PValue target = value.ir.getIfPValue();
      if (auto fnLiteral = sugarDynCast<FnLiteralTypeGeneratorType>(rvType))
        target = PValue(fnLiteral.getSymbolConstantAttr());
      if (isClosureWrapperStruct(shared, target, structTy))
        return cacheAndReturnVal(rvType, requiredType, true);
    }
  }

  // We can implicitly convert to the specified type if we can construct it with
  // the value as an implicit conversion.
  //
  // TODO: can we make `canConstructType` working without passing in the ir
  // here? This is the only reason that prevent us from turning
  // `ASTExprAnd<CValue> value` into a `ASTType actualType` in the signature
  // (such that we can ensure type conversion not looking at the value itself
  // for future changes to guarantee referential transparency).
  FailureOr<PValue> result = OverloadSet::canConstructType(
      requiredType,
      CallOperands{CallSyntax::kImplicitConvert, value.expr, {value}},
      declScope);
  bool isConvertible = succeeded(result) && result.value();
  // Must cache the overall value type, not just its stripped down rvType.
  shared.cacheImplicitConvertibility(value.ir.getType(), requiredType,
                                     isConvertible);
  return isConvertible;
}

/// This emits an implicit conversion to the specified type if the types
/// differ, including emitting any implicit constructor calls as well as
/// implicit promotions like origin conversions.
///
/// CAUTION: This method must line up with `canImplicitlyConvertToType`!!!
CValue IREmitter::emitImplicitConversionToType(ASTExprAnd<CValue> valueExpr,
                                               ASTType requiredType,
                                               ValueDest &dest) {
  CValue value = valueExpr.ir;
  const ExprNode *expr = valueExpr.expr;

  // If converting to or from a TypeCheckError type, then there is an
  // already-diagnosed error about this expression.
  auto rvType = value.getRValueType();
  if (rvType.isTypeCheckErrorType() || requiredType.isTypeCheckErrorType()) {
    dest.resetForError(*this);
    return {};
  }

  // If the types are already identical, then we're done.
  if (requiredType.isEqualCanon(rvType))
    return emitCResult(value, expr, dest);

  // If we are dealing with types that differ only pre-elaboration,
  // we insert a rebind or equivalent
  if (canZeroCostConvert(rvType, requiredType, shared)) {
    value = emitZeroCostConvert({value, expr}, requiredType);
    return emitCResult(value, expr, dest);
  }

  // Handle conversions between origins and origin sets.
  if (sugarIsa<OriginType>(rvType) && sugarIsa<OriginSetType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetAttr::get(pv, sugarCast<OriginSetType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }
  if (sugarIsa<OriginSetType>(rvType) && sugarIsa<OriginType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetUnionAttr::get(pv, sugarCast<OriginType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }

  auto emitTypeValueUpCastToTrait =
      [this](ASTExprAnd<CValue> valueExpr, ASTType fromType,
             ASTType toType) -> FailureOr<PValue> {
    // Emit metatype conversions to trait types if the metatype implements
    // the specified trait.
    if (auto anyTrait = sugarDynCast<AnyTraitType>(toType.extractMetaType())) {
      TraitType trait = anyTrait.getTraitType();
      if (sugarIsa<NonStructTypeType>(fromType)) {
        // Conversions from MLIR types.
        return bindNonStructTypeToTrait(valueExpr, trait);
      }

      if (sugarIsa<StructMetaMetaType>(fromType.extractMetaType()) ||
          sugarIsa<AnyTraitType>(fromType.extractMetaType())) {
        // Augment the witness table of closure wrapper with rebind if
        // necessary. We do this for every closure trait in the type.
        for (const auto &symbol : trait.getSymbols()) {
          auto &symbolDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
          if (auto traitDeclOp =
                  dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
              traitDeclOp && traitDeclOp.getDefinesClosure()) {
            (void)shared.getClosureEmitter().augmentWitnessTablesToConformTo(
                fromType, &symbolDecl);
          }
        }
        // Conversions from structs or traits.
        return emitMetaTypeToTraitConversion(valueExpr, trait);
      }

      if (auto fnGen =
              sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaType>(fromType)) {
        if (auto traitDecl = getClosureTraitDecl(shared, trait)) {
          TypedAttr fnPValue = fnGen.getType().getSymbolConstantAttr();
          ASTType structWrapper = getConcreteClosureWrapperTypeForFnSymbol(
              shared, declScope, valueExpr.expr->getLoc(), fnPValue);
          (void)shared.getClosureEmitter().augmentWitnessTablesToConformTo(
              structWrapper, traitDecl);
          return emitMetaTypeToTraitConversion(
              {PValue(structWrapper), valueExpr.expr}, trait);
        }

        // FnTypeGeneratorType is still a non-struct type...
        return bindNonStructTypeToTrait(valueExpr, trait);
      }
    }

    // We can convert from AnyTraitType[Derived] to AnyTraitType[Base].
    // This is a conversion of things like "the Movable type" (which has
    // type "AnyTraitType[Movable]") to "AnyTraitType[AnyType]".
    if (auto anyTrait = sugarDynCast<AnyTraitType>(toType)) {
      PValue typePValue = valueExpr.ir.getIfPValue();
      if (!typePValue) {
        emitError(valueExpr.expr->getLoc(),
                  "existentials are not supported yet!");
        return PValue();
      }

      ASTType concreteType;
      if (auto rvAnyTrait = sugarDynCast<AnyTraitType>(fromType)) {
        concreteType = ASTType(rvAnyTrait.getTraitType());
      } else if (auto mmType = sugarDynCast<StructMetaMetaType>(fromType)) {
        concreteType = ASTType(mmType.getType());
      }

      if (concreteType &&
          concreteType.checkConformance(anyTrait.getTraitType(), shared, {}) ==
              ConformanceResult::Yes) {
        // This is just the trait itself, not a conformance, just upcast.
        return PValue(TypeParamAttr::get(ASTType(typePValue), anyTrait));
      }
    }

    // Not applicable
    return failure();
  };

  FailureOr<PValue> typeValueCast =
      emitTypeValueUpCastToTrait(valueExpr, rvType, requiredType);
  // This handles nullptr case too.
  if (succeeded(typeValueCast))
    return emitCResult(*typeValueCast, expr, dest);

  // Conversions from function pointers to closures.
  if (sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(rvType)) {
    // Functions can implicitly convert to their corresponding closure wrapper.
    if (auto structMeta =
            sugarDynCast<StructMetaType>(requiredType.extractMetaType())) {
      StructType structTy = structMeta.getType();
      auto target = valueExpr.ir.getIfPValue();
      if (auto fnLiteral = sugarDynCast<FnLiteralTypeGeneratorType>(rvType))
        target = PValue(fnLiteral.getSymbolConstantAttr());
      if (isClosureWrapperStruct(shared, target, structTy)) {
        return emitConstructorCall(
            structTy, CallOperands(CallSyntax::kTypeCall, expr, {}), dest);
      }
    }
  }

  if (sugarIsa<ParamListType>(rvType) &&
      sugarIsa<ParamListType>(requiredType)) {
    auto emitVariadicError = [&]() -> CValue {
      shared.emitError(valueExpr.expr->getLoc(), "can not convert ")
          << rvType << " to " << requiredType << valueExpr.expr->getRange();
      dest.resetForError(*this);
      return {};
    };

    auto dstVATp = sugarCast<ParamListType>(requiredType);
    ASTType fromEltTp = sugarCast<ParamListType>(rvType).getElementType();
    ASTType toEltTp = dstVATp.getElementType();
    TypedAttr srcVal = valueExpr.ir.getIfPValue().get();
    if (auto vVal = sugarDynCast<ParamListAttr>(srcVal)) {
      SmallVector<TypedAttr> converted;
      for (auto elt : vVal.getValues()) {
        if (!LIT::isTypeExpr(elt))
          return emitVariadicError();

        // TODO(MOCO-2742): overwriting the type below should not be necessary.
        fromEltTp = ASTType(elt).extractMetaType();
        if (fromEltTp.mlirType != toEltTp.mlirType) {
          FailureOr<PValue> castToOr = emitTypeValueUpCastToTrait(
              {elt, valueExpr.expr}, fromEltTp, toEltTp);
          if (failed(castToOr) || castToOr->isNull())
            return {};
          converted.push_back(*castToOr);
        } else {
          // Simple case such as: !Int : !AnyType -> !Int: !mt_Int
          converted.push_back(TypeParamAttr::get(ASTType(elt), fromEltTp));
        }
      }
      return emitCResult(ParamListAttr::get(converted, dstVATp), expr, dest);
    } else {
      // Must match the check in canImplicitlyConvertToType.
      FailureOr<bool> canUpCast =
          canMetaTypeUpCastTo(shared, valueExpr.expr->getLoc(), fromEltTp,
                              toEltTp, &getDeclScope());
      if (failed(canUpCast) || !canUpCast.value())
        return emitVariadicError();

      // The source is not resolved yet, this is a simple upcast.
      // For example, we upcast a variadic of `Copyable`s to `AnyTypes` by
      // `#upcast<:param_list<!Copyable> T> :!param_list<!AnyType>`
      return emitCResult(UpcastAttr::get(requiredType, srcVal), expr, dest);
    }
  }

  // Support implicit conversions of generator types (incl. function
  // generators).
  if (auto requiredGenerator = sugarDynCast<GeneratorType>(requiredType)) {
    if (auto rvGeneratorType = sugarDynCast<GeneratorType>(rvType))
      if (canConvertGeneratorTypes(valueExpr, rvGeneratorType,
                                   requiredGenerator, declScope))
        return convertGeneratorValue(value, expr, requiredGenerator, *this,
                                     dest);
  }

  // We disable implicit conversions to prevent converting T -> S -> U in
  // one step, and to avoid infinite conversion cycles.
  return emitConstructorCall(
      requiredType,
      CallOperands(CallSyntax::kImplicitConvert, expr, {valueExpr}), dest);
}
