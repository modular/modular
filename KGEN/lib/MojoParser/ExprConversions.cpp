//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation details of ExprEmitter that are related to
// value conversions.
//
//===----------------------------------------------------------------------===//

#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "StructEmitter.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/CallOperands.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Function Conversions
//===----------------------------------------------------------------------===//

static bool canConvertFunctionTypes(FnTypeGeneratorType actual,
                                    FnTypeGeneratorType expected) {
  // We should have already checked that the function types are not
  // trivially-convertible between each other.

  // If the function effects are different, then the conversion cannot be
  // performed.
  // TODO: Enable non-raising to raising conversions.
  if (actual.getFnEffects() != expected.getFnEffects())
    return false;

  // Functions with different parameterization cannot be converted between each
  // other. If the types are equal but the passing conventions are different,
  // then the conversion is allowed.
  // TODO: Consider default parameter values and enable parameter inference to
  // reconcile differences.
  if (actual.getInputParamTypes() != expected.getInputParamTypes())
    return false;

  // If the functions differ in return type conventions, check if the nominal
  // types are equal.
  bool actualMemResult = actual.hasMemoryOnlyResult();
  bool expectedMemResult = expected.hasMemoryOnlyResult();
  // TODO: We could allow implicit conversions here.
  if (actual.getUserResultType() != expected.getUserResultType())
    return false;

  ArrayRef<Type> actualArgTypes =
      actual.getArguments().drop_back(actualMemResult);
  ArrayRef<Type> expectedArgTypes =
      expected.getArguments().drop_back(expectedMemResult);

  // Functions with a different number of arguments cannot be converted between
  // each other.
  // TODO: Consider default argument values.
  if (actualArgTypes.size() != expectedArgTypes.size())
    return false;

  // Function types are compatible if the arguments only differ in passing
  // conventions due to register-passibility.
  // TODO: Handle variadics and packs.
  for (auto [actualConv, expectedConv, actualType, expectedType] :
       llvm::zip(actual.getArgConventions().drop_back(actualMemResult),
                 expected.getArgConventions().drop_back(expectedMemResult),
                 actualArgTypes, expectedArgTypes)) {
    ASTType lhs = actualType;
    ASTType rhs = expectedType;

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
      // These conventions do not vary based on the register-passibility of the
      // type. They must always match
      if (actualConv != expectedConv)
        return false;
      lhs = lhs.getReferenceElementType();
      rhs = rhs.getReferenceElementType();
      break;

    case ArgConvention::ReadMem:
    case ArgConvention::ReadReg:
      if (!llvm::is_contained({ArgConvention::ReadMem, ArgConvention::ReadReg},
                              actualConv))
        return false;
      lhs = getFunctionArgumentRValueType(actualType, actualConv);
      rhs = getFunctionArgumentRValueType(expectedType, expectedConv);
      break;

    case ArgConvention::ByRefResult:
      llvm_unreachable("`byref_result` was already handled");
    }

    // Now check that the argument types line up.
    if (!lhs.isEqualCanon(rhs))
      return false;
  }

  // The function types are convertible.
  return true;
}

static FnTypeGeneratorType getReducedFunctionType(FnTypeGeneratorType sig) {
  MLIRContext *ctx = sig.getContext();

  SmallVector<PassingKind> passingKinds(sig.getNumArguments(),
                                        PassingKind::PosOnly);
  SmallVector<StringAttr> names(sig.getNumArguments(), StringAttr::get(ctx));

  // The passing kinds for results slots must be implicit;
  if (sig.hasMemoryOnlyResult())
    passingKinds.back() = PassingKind::Implicit;
  if (sig.isThrows())
    passingKinds.end()[-2] = PassingKind::Implicit;

  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, names, passingKinds),
      sig.getNumImplicitOriginDecls(), sig.getCaptureOrigins(),
      sig.getIsNestedOriginExclusivityCheckingDisabled());
  return FuncTypeGeneratorType::get(
      sig.getInputParamTypes(), sig.getValues(), sig.getArgConventions(),
      sig.getFnEffects(), metadata,
      PogListAttr::get(ctx, sig.getInputParamTypes().size()));
}

static std::string generateThunkName(Type expected, Type actual) {
  std::string name;
  llvm::raw_string_ostream os(name);
  ASTType(expected).print(os, /*diags=*/nullptr, /*demangleParams=*/true);
  os << '|';
  ASTType(actual).print(os, /*diags=*/nullptr, /*demangleParams=*/true);

  // Mix in the full signatures to disambiguate.
  std::string sigHash;
  llvm::raw_string_ostream sigHashOs(sigHash);
  expected.print(sigHashOs);
  actual.print(sigHashOs);
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)sigHash.data(), sigHash.size()));

  os << '|';
  os << llvm::encodeBase64(hash);
  return name;
}

static FnOp generateConversionThunk(Attribute key, ASTDecl &moduleDecl) {
  auto &shared = moduleDecl.getShared();
  // Don't generate any debuginfo for the thunk. Push a null scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(/*scope=*/nullptr);

  auto keyValues = cast<ArrayAttr>(key);
  auto actual =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[0]).getValue());
  auto expected =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[1]).getValue());

  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type. Add the parameters from the
  // expected signature. This contains the types of the captured parameters and
  // the actual function parameters.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] : llvm::enumerate(expected.getInputParamTypes())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(
        ParamDeclAttr::get(moduleDecl.mangleUserDefinedParamName(
                               b.getStringAttr("_" + Twine(idx))),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }
  // Rebind the argument and result types into the scope of the body.
  FunctionType types =
      expected.getSpecializedGenerator(paramValues).getBody().getValues();

  // Add an additional parameter, representing the actual callee. Rebind the
  // actual function type into the scope of the body.
  auto calleeDecl = ParamDeclAttr::get(
      moduleDecl.mangleUserDefinedParamName(b.getStringAttr("callee")),
      evaluator.getReboundType(actual));
  paramDecls.push_back(calleeDecl);

  // Generate a mangled name.
  std::string name = generateThunkName(expected, actual);

  // Declare the function at the bottom of the decl.
  b = ImplicitLocOpBuilder(mlirLoc, moduleDecl.getDeclEndBuilder());
  StructEmitter structEmitter(shared);
  auto [thunk, thunkDecl] = structEmitter.synthesizeFunction(
      moduleDecl, name, paramDecls,
      PogListAttr::get(ctx, expected.getInputParamTypes().size() + 1),
      types.getInputs(), expected.getArgConventions(),
      PogListAttr::get(ctx, expected.getNumArguments()),
      types.getResults().front(), SpecialFunctionKind::kNormal,
      moduleDecl.getLoc(), b, expected.getFnEffects());

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
  ExprEmitter emitter(*thunkDecl, b);

  // Construct the call operands from the function block arguments.
  CallOperands operands;
  SyntheticNode node(thunkDecl->getLoc());

  for (auto [arg, conv] :
       llvm::zip(thunk.getArguments(), expected.getArgConventions())) {
    AnyValue value;
    switch (conv) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the mojo parser");
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::Mut:
    case ArgConvention::MutRef:
      value = MLValue(arg);
      break;
    case ArgConvention::OwnedMem:
      value = MRValue(arg);
      break;
    case ArgConvention::ReadReg:
      value = SRValue(arg);
      break;
    case ArgConvention::ReadMem:
    case ArgConvention::Ref:
      value = MBValue(arg);
      break;
    }
    operands.add({value, node});
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_Trait);
  bool hasRegisterResult = false;
  if (expected.isAsync()) {
    // An async call returns a coroutine we have to await.
  } else if (expected.hasMemoryOnlyResult()) {
    dest = ValueDest(MLValue(thunk.getArguments().back()), EC_Trait);
  } else {
    hasRegisterResult = true;
  }

  // Bind the function parameters declared on the thunk to the callee. This does
  // NOT include the capture parameters -- the callee has already been rebound
  // to them when it was declared on the parameter list.
  TypedAttr calleeParam = BindParamsAttr::get(
      ParamDeclRefAttr::get(calleeDecl),
      ArrayRef(paramValues).take_back(actual.getInputParamTypes().size()));
  assert(cast<FnTypeGeneratorType>(calleeParam.getType())
             .getInputParamTypes()
             .size() == 0);

  CValue callResult =
      emitter.emitIndirectCall(PValue(calleeParam), std::move(operands), dest,
                               CallSyntax::kMethodCall, node);
  if (!callResult) {
    dest.resetForError();
    return {};
  }

  // If the callee is async, we got a coroutine. Now await it into the result.
  if (expected.isAsync()) {
    ValueDest dest(MLValue(thunk.getArguments().back()), EC_Trait);
    if (!emitter.emitNamedMethodCall("__await__",
                                     CallOperands({{callResult, node}}), dest,
                                     CallSyntax::kMethodCall, node)) {
      dest.resetForError();
      return {};
    }
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  Value retVal;
  if (hasRegisterResult)
    retVal = emitter.emitSRValue({callResult, node}, EC_Trait);
  emitter.emitNormalReturn(mlirLoc, retVal);
  return thunk;
}

static CValue convertFunctionValue(CValue value, const ExprNode *expr,
                                   FnTypeGeneratorType expected,
                                   ExprEmitter &emitter, ValueDest &dest) {
  PValue callee = value.getIfPValue();
  if (!callee) {
    emitter.emitError(
        expr->getLoc(),
        "TODO: function type conversions between closures not supported yet")
        << expr->getRange();
    dest.resetForError();
    return {};
  }

  MLIRContext *ctx = expected.getContext();
  auto actual = cast<FnTypeGeneratorType>(callee.getType());

  // Canonicalize the function types. This strips away unnecessary metadata that
  // does not affect the conversion semantics. In other words, a function type
  // and its reduced type can be trivially converted with a rebind.
  FnTypeGeneratorType reducedActual = getReducedFunctionType(actual);
  FnTypeGeneratorType reducedExpected = getReducedFunctionType(expected);

  // Given a function of type
  //
  //  fn[*ParamTypes](*ArgTypes) -> ResultType
  //
  // That captures parameters from its scope, we are going to produce a function
  // that looks like
  //
  //  fn thunk[*CapturedParamTypes, *params: *ParamTypes,
  //           callee: fn[*ParamTypes](*ArgTypes) -> ResultType](
  //        *args: *ArgTypes):
  //      return callee[*params](*args)
  //
  // To achieve this, we are going to deparameterize `reducedActual` such that
  // its `N` parameter captures are replaced with `*(1,0), ... *(1,N-1)`.
  //
  // For example, given
  //
  //   fn[p: Foo[a]](a: Bar[p]) -> Foo[b]
  //
  // where `a` and `b` are captures, we obtain
  //
  //   fn[Foo[$1|0]](a: Bar[$0]) -> Foo[$1|1]
  //
  // We then transform `reducedExpected` to be
  //
  //   fn[Int, Int, Foo[$0], fn[Foo[$1|0]](a: Bar[$0]) -> Foo[$1|1]]
  //     (Bar[$2]) -> Foo[$1]

  // First produce the parameter-isolated `reducedActual`.
  // NOTE: The walk here to determine the parameter captures only works if the
  // walk visits types in the same order as lexical parsing. This is because the
  // captured parameters can depend on each other, so they have to be
  // reparameterized in a order that keeps the dependencies valid.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> paramRefs;
  actual.walk([&](ParamDeclRefAttr ref) { paramRefs.insert(ref); });
  ParameterEvaluator replacer;
  SmallVector<Type> paramTypes;
  for (auto [i, ref] : llvm::enumerate(paramRefs)) {
    paramTypes.push_back(replacer.getReboundType(ref.getType()));
    replacer.setParameterValue(ref.getName(),
                               ParamIndexRefAttr::get(i, paramTypes.back()));
  }
  auto reparamActual =
      cast<FnTypeGeneratorType>(replacer.getReboundType(reducedActual));

  // Now reparameterize `reducedExpected`. Captured parameters are replaced with
  // `*(0,i) where i < N` and actual parameters are replaced with `*(0,N+j)`.
  for (auto [i, type] : llvm::enumerate(actual.getInputParamTypes())) {
    paramTypes.push_back(replacer.getReboundType(type));
    replacer.addInputParam(
        ParamIndexRefAttr::get(i + paramRefs.size(), paramTypes.back()));
  }
  auto reparamMetadata = FnMetadataAttr::get(
      reducedExpected.getArgListAttrs(),
      reducedExpected.getNumImplicitOriginDecls(),
      reducedExpected.getCaptureOrigins(),
      reducedExpected.getIsNestedOriginExclusivityCheckingDisabled());
  auto reparamExpected = FuncTypeGeneratorType::get(
      paramTypes,
      cast<FunctionType>(replacer.getReboundType(reducedExpected.getValues())),
      reducedExpected.getArgConventions(), reducedExpected.getFnEffects(),
      reparamMetadata, PogListAttr::get(ctx, paramTypes.size()));

  // We can attempt to generate the thunk now.
  Attribute key = ArrayAttr::get(
      ctx, {TypeAttr::get(reparamActual), TypeAttr::get(reparamExpected)});
  FnOp thunk =
      emitter.shared.getOrCreateFunctionThunk(key, generateConversionThunk);
  if (!thunk) {
    dest.resetForError();
    return {};
  }

  // Cast the callee to the reduced actual type.
  TypedAttr calleeParam =
      ParamOperatorAttr::get(POC::Rebind, callee.get(), reducedActual);

  // Bind the thunk to the captured parameters, leaving the actual parameters
  // unbound. This binds the callee type into the current scope.
  ParameterEvaluator evaluator;
  for (ParamDeclRefAttr ref : paramRefs)
    evaluator.addInputParam(ref);
  for (Type type : expected.getInputParamTypes())
    evaluator.addInputParam(UnboundAttr::get(evaluator.getReboundType(type)));
  evaluator.addInputParam(calleeParam);

  SymbolConstantAttr symbol = thunk.getBoundSymbolRef(
      ParameterExprArrayAttr::get(ctx, evaluator.getInputParams()));

  // Finally, cast the result back to the expected type.
  return emitter.emitCResult(
      ParamOperatorAttr::get(POC::Rebind, {symbol}, expected), expr, dest);
}

//===----------------------------------------------------------------------===//
// Zero Cost Conversions
//===----------------------------------------------------------------------===//

/// Returns if a value of the specified type can be coerced to the other type
/// with a zero-cost conversion like a rebind.  This means that values of the
/// two types have exactly the same representation post-elaboration.
bool ExprEmitter::canZeroCostConvert(ASTType fromType, ASTType toType,
                                     SharedState &shared) {
  if (fromType.isEqualCanon(toType))
    return true; // No rebind needed!

  // Trait metatypes are allowed to upcast to trivial types.
  if (isa<TypeType>(toType)) {
    if (isa<AnyTraitType>(fromType))
      return true;
    if (auto structType = dyn_cast<StructMetaType>(fromType)) {
      return ASTType(structType.getType())
                 .getRegisterPassability(SMLoc(), shared) ==
             TypeConvention::RegisterPassableTrivial;
    }
  }

  // Check for closure structs and dig out their underlying signature types to
  // check whether the conversion can occur.
  auto fromDecl = dyn_cast_or_null<StructDeclOp>(fromType.getDecl(shared));
  auto toDecl = dyn_cast_or_null<StructDeclOp>(toType.getDecl(shared));
  if (fromDecl && toDecl) {
    FuncTypeGeneratorType fromSig =
        fromDecl.getClosureSignature().value_or(nullptr);
    FuncTypeGeneratorType toSig =
        toDecl.getClosureSignature().value_or(nullptr);
    if (fromSig && toSig) {
      // Compare the specialized signatures.
      fromSig = fromSig.getSpecializedGenerator(fromType.getParamBindings());
      toSig = toSig.getSpecializedGenerator(toType.getParamBindings());
      return canZeroCostConvert(fromSig, toSig, shared);
    }
    return false;
  }

  // Check origin downcasting.  The safe conversions are:
  //   Origins with identical mutability will be uniqued and already handled.
  //   Conversion from any mutability to KNOWN immutable is fine.
  //   Conversion from KNOWN mutable to any mutability is fine.
  //   Conversion from with mutability "X" to "X&Y" is known to be fine.
  // We allow KGEN to fold the true and false cases for us.
  if (auto fromLife = dyn_cast<OriginType>(fromType))
    if (auto toLife = dyn_cast<OriginType>(toType)) {
      auto toMut = toLife.getIsMutable();
      auto result =
          ParamOperatorAttr::get(POC::And, toMut, fromLife.getIsMutable());
      if (result == toMut)
        return true;
    }

  // Check reference downcasting.  The only thing allowed to disagree is the
  // origin set / mutability.
  if (auto fromRef = dyn_cast<RefType>(fromType)) {
    if (auto toRef = dyn_cast<RefType>(toType)) {
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
      if (isa<AnyOriginAttr>(fromRef.getOrigin()))
        return true;

      // We can convert origin subset to a origins superset.
      auto toOrigin = toRef.getOrigin();
      auto originUnion = OriginUnionAttr::get(
          {toOrigin, OriginMutCastAttr::get(fromRef.getOrigin(), toOriginType)},
          toOriginType);
      return toOrigin == originUnion;
    }
  }

  // Otherwise handle function conversions.
  auto from = dyn_cast<FnTypeGeneratorType>(fromType);
  auto to = dyn_cast<FnTypeGeneratorType>(toType);
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

  // The input argument types may have different implicit origins.
  for (auto [fromTy, toTy, conv] : llvm::zip(
           from.getArguments(), to.getArguments(), from.getArgConventions())) {
    Type fromTyCmp = fromTy;
    Type toTyCmp = toTy;
    if (hasImplicitOrigin(conv)) {
      fromTyCmp = ASTType(fromTyCmp).getReferenceElementType();
      toTyCmp = ASTType(toTyCmp).getReferenceElementType();
    }
    if (!ASTType(fromTyCmp).isEqualCanon(toTyCmp))
      return false;
  }

  // Otherwise, everything seems compatible.
  return true;
}

/// If there is a common type shared between the two reference types, return
/// it. Otherwise return null.
RefType ExprEmitter::getCommonRefType(RefType ref1, RefType ref2) {
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
  auto origin = OriginUnionAttr::get({l1, l2}, cast<OriginType>(l1.getType()));
  return RefType::get(eltType, origin, ref1.getAddressSpace());
}

/// Returns a type if there is a shared supertype for the two specified types,
/// e.g. two derived classes may have the same base class even if neither is
/// convertible to the other.  This returns null if there is no common type.
ExprEmitter::CommonTypeResult
ExprEmitter::getCommonType(ASTExprAnd<CValue> val1, ASTExprAnd<CValue> val2,
                           ASTType &result) {
  auto succeed = [&](ASTType type) {
    result = type;
    return CTR_Success;
  };

  // If the types already match, then we're done.
  ASTType type1 = val1.ir.getRValueType();
  ASTType type2 = val2.ir.getRValueType();
  if (type1.isEqualCanon(type2))
    return succeed(type1);

  // Check reference downcasting.
  if (auto type1Ref = dyn_cast<RefType>(type1))
    if (auto type2Ref = dyn_cast<RefType>(type2)) {
      result = getCommonRefType(type1Ref, type2Ref);
      return result ? CTR_Success : CTR_NoCommonType;
    }

  // If one type implicit converts to the other, then the other is a common
  // type.  Don't do this if both convert to each other, this would be
  // ambiguous.
  bool isConvertibleToType2 =
      canImplicitlyConvertToType(val1, type2, declScope);
  bool isConvertibleToType1 =
      canImplicitlyConvertToType(val2, type1, declScope);
  if (isConvertibleToType2 && !isConvertibleToType1)
    return succeed(type2);
  if (isConvertibleToType1 && !isConvertibleToType2)
    return succeed(type1);
  if (isConvertibleToType1 && isConvertibleToType2)
    return CTR_Ambiguous;

  // If one or the other type is non-materializable, the conversion is free,
  // so check to see if there is an unambiguous common type.
  bool type2ConvertsToType1Nonmat = false;
  bool type1ConvertsToType2Nonmat = false;
  auto type1Nonmat = type1.getNonmaterializableTarget(shared);
  auto type2Nonmat = type2.getNonmaterializableTarget(shared);
  if (type1Nonmat)
    type2ConvertsToType1Nonmat =
        canImplicitlyConvertToType(val2, type1Nonmat, declScope);
  if (type2Nonmat)
    type1ConvertsToType2Nonmat =
        canImplicitlyConvertToType(val1, type2Nonmat, declScope);

  if (type2ConvertsToType1Nonmat && !type1ConvertsToType2Nonmat)
    return succeed(type1Nonmat);
  if (type1ConvertsToType2Nonmat && !type2ConvertsToType1Nonmat)
    return succeed(type2Nonmat);
  if (type1ConvertsToType2Nonmat && type2ConvertsToType1Nonmat) {
    if (type1Nonmat.isEqualCanon(type2Nonmat))
      return succeed(type1Nonmat);
    return CTR_Ambiguous;
  }

  // No common type found.
  return CTR_NoCommonType;
}

/// Given a value of a type that can be zero cost converted to another type,
/// emit a rebind or other operation to get it in the right type.
PValue ExprEmitter::emitZeroCostConvert(PValue value, ASTType toType,
                                        SharedState &shared) {
  assert(toType.mlirType != value.getType() && "Already the same");

  // PValues of origin type have a special conversion.
  if (isa<OriginType>(toType) && isa<OriginType>(value.getType()))
    value = OriginMutCastAttr::get(value, toType);

  return ParamOperatorAttr::get(POC::Rebind, value.get(), toType);
}

CValue ExprEmitter::emitZeroCostConvert(ASTExprAnd<CValue> value,
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

namespace {
/// The signature for a trait requirement will have a Self parameter first whose
/// type is a TraitType for the trait it was found in.  We want to force
/// substitute a new parameter for the Self references even though it has a
/// different metatype.  This doesn't remove the parameter, that will be done
/// later.
struct TraitSelfBinder : public IndexParameterReplacer<TraitSelfBinder> {
  TypedAttr selfValue;

  TraitSelfBinder(TypedAttr selfValue) : selfValue(selfValue) {}

  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth) {
    // Replace a reference to $(0,0) with the new selfValue.
    auto paramRef = dyn_cast<ParamIndexRefAttr>(attr);
    if (!paramRef || paramRef.getIndex() != 0 ||
        paramRef.getDepth() + 1 != depth)
      return {};
    return selfValue;
  }
  Type tryReplace(Type type, size_t depth) { return {}; }
};
} // namespace

/// Given a method from a trait like 'Movable.__del__', rebind the method to
/// have a different self for a conforming type, e.g.
/// 'RefinedMovableTrait.__del__' or 'Int.__del__'.  'newSelfType' is the
/// struct or trait type to bind.  For example, AnyType.__del__'s signature
/// looks like:
///    !lit.generator<<trait<@AnyType>>[1]("self":
///        !lit.ref<:trait<@AnyType> *(0,0), mut *[0,0]> owned_in_mem) -> none>
/// When binding this down to some MTT conforming to Movable, this will give us
/// something like:
///    !lit.generator<[1]>("self":
///        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
/// Resolving the *(0,0) into the Movable type, as well as the first param type.
static FnTypeGeneratorType
createRequirementSignature(FnOp traitFn, ASTType newSelfType,
                           ParameterEvaluator &traitAliasReplacer,
                           const DenseMap<StringAttr, TypedAttr> &aliasValues,
                           DeclResolver &declResolver) {
  // Get the selfType as a TypedAttr since we'll be using it as a parameter
  // value below.
  TypedAttr newSelfValue = PValue(newSelfType).get();

  if (auto paramType = dyn_cast<ParamType>(newSelfType.getMetaType())) {
    auto simpleTraitType =
        cast<AnyTraitType>(paramType.getParam().getType()).getTraitType();
    // Upcast from a parametric type of trait metatype value (e.g. "some
    // type that conforms to Movable) to the simple trait type (Movable)
    // so we can substitute the value into the signature.
    newSelfValue =
        UpcastAttr::get(simpleTraitType, PValue(newSelfType),
                        VTableAttr::get(simpleTraitType.getContext(), {}));
  }

  // Start with the full signature for the trait requirement.
  FnTypeGeneratorType signature = traitFn.getFullSignature();

  // The requirement will have a Self parameter whose type will be of the
  // current trait.  In order to get types to line up, we need to force it
  // to the implementation type.  This changes the parameter value, but also
  // changes the metatype of the value.  To support this, we use a custom
  // replacer.
  TraitSelfBinder selfBinder(newSelfValue);
  signature = selfBinder.replace(signature);

  // At this point, the first parameter is gone:
  //    !lit.generator<[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>

  // Next we'll replace trait aliases that appear in the trait methods, such
  // as:
  //
  //     trait MyTrait:
  //         alias T: ATrait
  //         fn bork(self) -> Something[T]: ...
  //         fn zork(self) -> Something[Self.T]: ...
  //
  // We'll replace them with the struct's trait value, like the int in:
  //
  //     struct MyStruct(MyTrait):
  //         alias T: ATrait = int
  //         fn bork(self) -> SIMD[int]: ...

  // bork's `T` is a regular paramRef, we use traitAliasReplacer to replace it.
  signature = traitAliasReplacer.replace(signature);

  // However, zork's `Self.T` is different, like: get_vtable_entry(Self, "T").
  // And after the first step, that Self is actually the struct, so the
  // requirementFn is really more like: get_vtable_entry(MyStruct, "T").
  // We'll manually replace those entire get_vtable_entry calls.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
    if (paramOp.getOpcode() == POC::GetVTableEntry &&
        paramOp.getOperand(0) == PValue(newSelfType).get()) {
      auto aliasName = cast<StringAttr>(paramOp.getOperand(1));
      // The vtable entries have type !kgen.string, but the entries from the
      // trait decl have a StringAttr with no type.  Reunique them to look up.
      aliasName = StringAttr::get(aliasName.getContext(), aliasName.strref());
      auto iter = aliasValues.find(aliasName);
      if (iter != aliasValues.end())
        return iter->second;
    }
    return paramOp;
  });
  signature = cast<FnTypeGeneratorType>(replacer.replace(signature));

  // At this point, signature's `self` argument's type is the struct or
  // trait.  For example when binding Self down to some "MTT: Movable", we have:
  //    !lit.generator<<trait<@AnyType>>[1]("self":
  //        !lit.ref<:trait<@Movable> MTT>, mut *[0,0]> owned_in_mem) -> none>>
  // Now we need to drop the "<trait<@AnyType>" parameter, which we do by
  // specializing it away.  We know all references to it are already gone.

  // NOTE: This is an UnknownAttr (which is an arbitrary attr that is never
  // used) not an UnboundAttr which remains an unbound parameter.
  ParameterEvaluator evaluator;
  evaluator.addInputValue(UnknownAttr::get(signature.getInputParamTypes()[0]));
  // Use UnboundAttr for any other parameters so they remain in the result.
  for (Type type : signature.getInputParamTypes().drop_front())
    evaluator.addInputValue(UnboundAttr::get(evaluator.getReboundType(type)));
  signature = signature.getSpecializedGenerator(evaluator.getInputParams());

  return signature;
}

/// Emit a metatype conversion to a trait type by materializing the meta type
/// of the specified CValue into a witness table for the trait.  For example,
/// if 'value' has struct type, and the trait is Movable, then this forms a
/// TypeParamAttr PValue with a vtable containing the __del__ and
/// __moveinit__ methods from the struct.
///
/// If the input value has a derived trait type and the required type is a
/// base trait, then this remaps each of the requirements into the expected
/// format of the result vtable, e.g.:
///   fn take_any_type[ATT: AnyType](x: ATT): pass
///   fn pass_movable[MTT: Movable](x: MTT): take_any_type(x)
///
/// Yields something like:
///     #kgen.type<!kgen.param<:trait<@Movable> MTT>, {
///        "__del__" : !lit.generator<[1](
///                    "self": !lit.ref<:trait<@Movable> MTT, ...)>
///          = get_vtable_entry(:trait<@Movable> MTT, "__del__")
///     }> : !lit.trait<@AnyType>
///
/// This maps from the Movable trait metatype into the AnyType trait metatype.
PValue ExprEmitter::emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                                  TraitType trait) {
  // Only static vtables are supported right now.
  PValue typePValue = value.ir.getIfPValue();
  if (!typePValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }

  // Get the StructMetaType or the TraitType of the value that we're checking
  // for conversion to the trait type.  This can also bind empty variadic
  // parameter lists and default parameters.
  ASTType type = emitType({typePValue, value.expr}, /*allowUnbound*/ false);
  if (!type)
    return {};
  value.ir = PValue(type); // update value.ir if the type was rebound.

  // Check that the struct or super trait implements the trait.
  ASTDecl *metaTypeDecl = type.getDecl(shared);
  if (!metaTypeDecl) {
    emitError(value.expr->getLoc(), "cannot get metatype of ")
        << type << value.expr->getRange();
    return {};
  }

  std::optional<InflightDiag> checkDiag;
  if (!metaTypeDecl->doesNominalTypeConformTo(trait, checkDiag)) {
    InflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                        << type << " to trait " << ASTType(trait)
                        << value.expr->getRange();
    if (checkDiag)
      diag.attachNote(metaTypeDecl->getLoc()) << std::move(*checkDiag);
    return {};
  }

  // Synthesize the vtable required for the trait from the struct. Make sure the
  // trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(getDeclResolver().resolveFully(*traitDecl, value.expr->getLoc())))
    return {};

  ArrayRef<ParamDeclAttr> structParamDecls;

  // Determine if the conforming value is trivial or register passable.  If so,
  // this will affect the methods we can synthesize in conformance.  Values of
  // trait type will already have been erased to a memory type.
  bool rpTrivial = false;
  bool regPassable = false;
  bool implicitlyDestructible = false;
  if (auto structDeclOp = dyn_cast<StructDeclOp>(metaTypeDecl)) {
    rpTrivial = structDeclOp.isRegisterPassable();
    regPassable = structDeclOp.isRegisterPassableTrivial();
    structParamDecls = structDeclOp.getParams();
    // TODO(MOCO-1468): Pull out into a helper, or make a method like
    // isRegisterPassable that can go on the structDeclOp.
    for (auto parentAttr : structDeclOp.getParentTypes()) {
      ASTDecl &parentDecl = shared.declResolver->getDeclForTypeSymbol(
          cast<TraitType>(parentAttr.getType()).getSymbol());
      if (auto parentTrait = dyn_cast<TraitDeclOp>(parentDecl)) {
        if (parentTrait.getSymName() == "AnyType") {
          implicitlyDestructible = true;
          break;
        }
      }
    }
  }

  // When we're looking for a trait's method in a certain struct, like:
  //     trait TraitWithAliasMethod:
  //         alias T: ATrait
  //         fn bork(self) -> Something[T]: ...
  // and a struct overrides it:
  //     struct ExplicitStructWithAliasMethod(TraitWithAliasMethod):
  //         alias T: ATrait = int
  //         fn bork(self) -> SIMD[int]: ...
  // we don't want to look for a `fn bork(self) -> Something[T]` in the struct,
  // we want to look for a `fn bork(self) -> SIMD[int]`. This helps us do that.
  ParameterEvaluator traitAliasReplacer;
  DenseMap<StringAttr, TypedAttr> aliasValues;

  // If the struct (e.g. List[T]) has an alias that uses an input parameter,
  // (e.g. `alias element_type = T`), then this will help us interpret that
  // alias value while filling the above traitAliasReplacer.
  // FIXME: We need to reject accessing aliases of a partially bound type, until
  // ParameterizedType is a thing!
  ParameterEvaluator implGenericsReplacer(structParamDecls,
                                          type.getParamBindings());

  // Bind each trait requirement into vtable entries.
  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, requirementDecls] : traitDecl->getDeclsInScope()) {
    // Each entry can have multiple overloads in 'decls'.
    if (requirementDecls.empty())
      continue;

    // Find candidates in the implementing type (either a struct or trait) which
    // also may have multiple overloads.
    LookupResult result =
        shared.lookupAndResolveDecl(name, value.expr->getLoc(), *metaTypeDecl,
                                    /*searchParentScopes=*/false);
    ArrayRef<ASTDecl *> impls = result.getIfSuccess();

    if (auto traitAliasDecl = dyn_cast<AliasDeclOp>(requirementDecls.front())) {
      // These asserts should be safe because we already know it correctly
      // conforms because we called `doesNominalTypeConformTo` above.
      assert(impls.size() == 1);
      auto implAlias = cast<AliasDeclOp>(impls.front());
      assert(implAlias.getValueAttr() && "struct's alias should have value");

      TypedAttr newValue = implAlias.getValueAttr();
      newValue = implGenericsReplacer.replace(newValue);
      // If a decl has a parameter "T : Trait" where Trait defines an associated
      // type "U : Trait2", then when we emit vtable for T, we must also emit
      // vtable for T.U.  We perform this by implicitly converting to the alias'
      // declared type.
      newValue = emitPValue({newValue, value.expr}, EC_Trait,
                            traitAliasDecl.getType());
      if (!newValue)
        return {};

      vtable.push_back(VTableEntryAttr::get(name, newValue));
      traitAliasReplacer.setParameterValue(traitAliasDecl.getParamDecl(),
                                           newValue);
      aliasValues[name] = newValue;
      continue;
    }

    // Traits shouldn't have var decls or other things.
    if (!isa<FnOp>(requirementDecls.front()))
      continue;

    // Each requirement may be overloaded, resolve each individually.
    for (ASTDecl *expected : requirementDecls) {
      auto traitFn = dyn_cast<FnOp>(expected);
      assert(traitFn && "trait has an alias and a fn with the same name!");

      // For any given requirement, the implementing type may have multiple
      // overloads.  Resolve which one we're using by forming an overload set
      // and filtering it.  Start by finding a set of param bindings for the
      // implementing function that get bound, including:
      //
      //  * The self type if the conforming type is a trait.
      //  * The conforming struct's values for the trait's aliases.

      // The requirement will have a Self parameter whose type will be of the
      // current trait.  In order to get types to line up, we need to force it
      // to the implementation type.  This changes the parameter value, but also
      // changes the metatype of the value.  To support this, we use a custom
      // replacer.
      FnTypeGeneratorType requirementSig = createRequirementSignature(
          traitFn, type, traitAliasReplacer, aliasValues, getDeclResolver());

      // Form a set of bindings to plow into the impl signature by binding Self
      // to the appropriate Struct or derived Trait type.
      auto implBindings =
          ParamBindings::getForDeclaredType(getDeclScope(), type, value.expr);
      // Leave the rest of the the parameters Unbound.
      ParameterEvaluator evaluator;
      for (Type type : requirementSig.getInputParamTypes()) {
        auto unbound = UnboundAttr::get(evaluator.getReboundType(type));
        evaluator.addInputValue(unbound);
        implBindings.addPrechecked(value.expr, unbound);
      }

      // Grab the matching function.  If the input type is a StructType, this
      // will directly bind the method in question.  If the input type is
      // something like "T: Movable" and we're binding __del__ then this will
      // end up with `get_vtable_entry(T, "__del__")`.
      OverloadSet ov(name, impls, std::move(implBindings), value.expr,
                     CallSyntax::kMethodCallSynthetic);
      auto result = ov.filterOverloadSetForValueType(
          requirementSig, getDeclScope(), /*emitDiagnosticOnFailure=*/false);
      if (!result) {
        // Don't error out if name is for the thunk functions that will be
        // synthesized when conformance check happens.
        if (canSynthesizeIfMissing(name, rpTrivial, regPassable,
                                   implicitlyDestructible)) {
          continue;
        }

        // The struct does not have the specified member and we cannot
        // synthesize it. Re-emit the error to get a diagnostic.
        (void)ov.filterOverloadSetForValueType(
            requirementSig, getDeclScope(), /*emitDiagnosticOnFailure=*/true);
        return {};
      }
      assert(result.getType().mlirType == requirementSig &&
             "didn't form a fn with signature of expected type");
      vtable.push_back(VTableEntryAttr::get(name, result));
    }
  }

  // Create the new type value with the vtable and the trait metatype.
  return TypeParamAttr::get(type, trait, VTableAttr::get(getContext(), vtable));
}

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkMLIRTypeConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveFully(traitDecl, loc)))
    return false; // an error was emitted
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto traitFn = dyn_cast<FnOp>(*decl);
      // Skip any children that aren't methods or are inherited. This could be
      // an alias.
      if (!traitFn || traitFn.getIsInherited())
        continue;
      // MLIR types are movable, copyable, and destructible only.
      if (llvm::is_contained({SpecialFunctionKind::kMoveInit,
                              SpecialFunctionKind::kCopyInit,
                              SpecialFunctionKind::kDel},
                             SpecialFunctionInfo::getKind(name)))
        continue;
      return false;
    }
  }
  return true;
}

/// Emit a conversion from an MLIR type to a trait type by materializing stubs
/// for the type's witness table.
static PValue bindMLIRTypeToTrait(ASTExprAnd<CValue> value, TraitType trait,
                                  ExprEmitter &emitter) {
  SharedState &shared = emitter.shared;

  // Only static vtables are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    shared.emitError(value.expr->getLoc(),
                     "existentials are not supported yet!");
    return {};
  }
  ASTType mlirType = typeValue.getIfTypeValue();

  SMLoc loc = value.expr->getLoc();
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveFully(traitDecl, loc)))
    return {};

  // Use a special wrapper decl in the builtins as stubs.
  ASTDecl *wrapperDecl = shared.getBuiltinStubsMLIRType(loc).getDecl(shared);
  if (!wrapperDecl || !isa<StructDeclOp>(wrapperDecl)) {
    shared.emitError(loc, "malformed builtin._stubs.__MLIRType");
    return {};
  }
  ASTType boundWrapper =
      cast<StructDeclOp>(wrapperDecl).bindReference({typeValue});

  // NOTE: This substantially duplicates emitMetaTypeToTraitConversion because
  // it is doing some crazy manual binding of the type into the parameter list
  // so the vtable entries are specialized on the MLIR type.
  //
  // FIXME(MOCO-1146): Could we instead just synthesize the members required and
  // eliminate __mlir_type entirely?  This __mlir_type thing introduces other
  // bugs.  We already do this for rp-trivial types which MLIR types are.
  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    if (decls.empty() || !isa<FnOp>(decls.front())) {
      InflightDiag diag = shared.emitError(loc, "cannot bind MLIR type ")
                          << mlirType << " to trait " << ASTType(trait);
      diag.attachNote(decls.front()->getLoc())
          << "MLIR type cannot satisfy this requirement";
      return {};
    }

    for (ASTDecl *decl : decls) {
      // MLIR types are movable, copyable, and destructible only.
      switch (SpecialFunctionInfo::getKind(name)) {
      case SpecialFunctionKind::kMoveInit:
      case SpecialFunctionKind::kCopyInit:
      case SpecialFunctionKind::kDel:
        break;
      default:
        if (name != "copy") {
          InflightDiag diag = shared.emitError(loc, "cannot bind MLIR type ")
                              << mlirType << " to trait " << ASTType(trait);
          diag.attachNote(decl->getLoc())
              << "MLIR type cannot satisfy required trait function here";
          return {};
        }
      }
      // We know the stub will provide exactly one overload for each allowed
      // trait requirement.
      auto ovSet =
          OverloadSet::lookup(emitter.getDeclScope(), boundWrapper, name,
                              value.expr, CallSyntax::kMethodCall);
      // Manually bind the type into the parameter list so the vtable entries
      // are specialized on the MLIR type.
      ovSet.paramBindings = ParamBindings::getForDeclaredType(
          emitter.getDeclScope(), boundWrapper, value.expr);

      PValue callee = ovSet.getIfPValue();
      if (!callee) {
        shared.emitError(loc, "internal error: MLIR type stub didn't resolve ")
            << name;
        return {};
      }
      vtable.push_back(VTableEntryAttr::get(name, callee));
    }
  }

  return TypeParamAttr::get(mlirType, trait,
                            VTableAttr::get(shared.getContext(), vtable));
}

//===----------------------------------------------------------------------===//
// Generalized Implicit Conversions
//===----------------------------------------------------------------------===//

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
///
/// CAUTION: This method must line up with `emitImplicitConversionToType`!!!
bool ExprEmitter::canImplicitlyConvertToType(ASTExprAnd<CValue> value,
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

  // Origins and origin sets can convert between each other.
  // FIXME: This seems wrong, why isn't it checking for inclusion and
  // compatibility??
  if ((isa<OriginType>(rvType) && isa<OriginSetType>(requiredType)) ||
      (isa<OriginSetType>(rvType) && isa<OriginType>(requiredType)))
    return true;

  // Check to see if we already cached this convertibility check.
  std::optional<bool> cache =
      shared.getCachedImplicitConvertibility(rvType, requiredType);
  if (cache.has_value())
    return cache.value();

  auto cacheAndReturnVal = [&](bool isConvertible) -> bool {
    // Cache the result of this convertibility check.
    shared.cacheImplicitConvertibility(rvType, requiredType, isConvertible);
    return isConvertible;
  };

  // Values of known {struct/trait/mlir} type can convert to any trait type they
  // implement.
  if (auto anyTrait =
          dyn_cast_if_present<AnyTraitType>(requiredType.getMetaType())) {
    TraitType trait = anyTrait.getTraitType();
    bool result = false;

    // MLIR types can conform to traits that have limited requirements.
    // AnyTraitType (the type of all traits) conforms to traits with only a
    // destructor (e.g. AnyType) since all traits have that.
    if (isa<TypeType>(rvType)) {
      result = checkMLIRTypeConformance(shared, value.expr->getLoc(), trait);
    } else if (auto pval = value.ir.getIfPValue();
               pval && LIT::isTypeExpr(pval)) {
      // Can only convert static types to traits, not existentials.
      if (ASTDecl *decl = ASTType(pval).getDecl(shared))
        return cacheAndReturnVal(decl->doesNominalTypeConformTo(trait));
    }
    return cacheAndReturnVal(result);
  }

  // We can convert from AnyTraitType[Derived] to AnyTraitType[Base].
  // This is a conversion of things like "the Movable type" (which has type
  // "AnyTraitType[Movable]") to "AnyTraitType[AnyType]".
  if (auto anyTrait = dyn_cast<AnyTraitType>(requiredType)) {
    if (auto fromAnyTrait = dyn_cast<AnyTraitType>(rvType))
      if (auto *fromDecl = ASTType(fromAnyTrait.getTraitType()).getDecl(shared))
        return cacheAndReturnVal(
            fromDecl->doesNominalTypeConformTo(anyTrait.getTraitType()));
  }

  // Check for non-trivial function type conversions.
  if (auto requiredFunction = dyn_cast<FnTypeGeneratorType>(requiredType)) {
    bool result = false;
    if (auto rvFunctionType = dyn_cast<FnTypeGeneratorType>(rvType))
      result = canConvertFunctionTypes(rvFunctionType, requiredFunction);
    return cacheAndReturnVal(result);
  }

  // We can implicitly convert to the specified type if we can construct it with
  // the value as an implicit conversion.
  FailureOr<PValue> result = OverloadSet::canConstructType(
      requiredType, {{value}}, value.expr, declScope,
      /*isImplicitConversion=*/true);
  return cacheAndReturnVal(succeeded(result) && result.value());
}

/// This emits an implicit conversion to the specified type if the types
/// differ, including emitting any implicit constructor calls as well as
/// implicit promotions like origin conversions.
///
/// CAUTION: This method must line up with `canImplicitlyConvertToType`!!!
CValue ExprEmitter::emitImplicitConversionToType(ASTExprAnd<CValue> valueExpr,
                                                 ASTType requiredType,
                                                 ValueDest &dest) {
  CValue value = valueExpr.ir;
  const ExprNode *expr = valueExpr.expr;

  // If converting to or from a TypeCheckError type, then there is an
  // already-diagnosed error about this expression.
  auto rvType = value.getRValueType();
  if (rvType.isTypeCheckErrorType() || requiredType.isTypeCheckErrorType()) {
    dest.resetForError();
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
  if (isa<OriginType>(rvType) && isa<OriginSetType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetAttr::get(pv, cast<OriginSetType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }
  if (isa<OriginSetType>(rvType) && isa<OriginType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetUnionAttr::get(pv, cast<OriginType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }

  // Emit metatype conversions to trait types if the metatype implements the
  // specified trait.
  if (auto anyTrait =
          dyn_cast_if_present<AnyTraitType>(requiredType.getMetaType())) {
    TraitType trait = anyTrait.getTraitType();
    PValue result;
    if (isa<TypeType>(rvType)) // Conversions from MLIR types.
      result = bindMLIRTypeToTrait(valueExpr, trait, *this);
    else // Conversions from everything else.
      result = emitMetaTypeToTraitConversion(valueExpr, trait);

    return emitCResult(result, expr, dest);
  }

  // We can convert from AnyTraitType[Derived] to AnyTraitType[Base].
  // This is a conversion of things like "the Movable type" (which has type
  // "AnyTraitType[Movable]") to "AnyTraitType[AnyType]".
  if (auto anyTrait = dyn_cast<AnyTraitType>(requiredType)) {
    PValue typePValue = value.getIfPValue();
    if (!typePValue) {
      emitError(expr->getLoc(), "existentials are not supported yet!");
      return {};
    }

    // This is just the trait itself, not a conformance, so we can use an empty
    // vtable, just upcast.
    return TypeParamAttr::get(ASTType(typePValue), anyTrait);
  }

  // Support implicit conversions of function types.
  if (auto requiredFunction = dyn_cast<FnTypeGeneratorType>(requiredType)) {
    if (auto rvFunctionType = dyn_cast<FnTypeGeneratorType>(rvType))
      if (canConvertFunctionTypes(rvFunctionType, requiredFunction))
        return convertFunctionValue(value, expr, requiredFunction, *this, dest);
  }

  // We disable implicit conversions to prevent converting T -> S -> U in
  // one step, and to avoid infinite conversion cycles.
  return emitConstructorCall(requiredType, CallOperands({valueExpr}), expr,
                             CallSyntax::kImplicitConvert, dest);
}
