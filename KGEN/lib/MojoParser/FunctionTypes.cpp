//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "FunctionTypes.h"
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "StructEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/CallOperands.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

bool LIT::canConvertFunctionTypes(LITSignatureType actual,
                                  LITSignatureType expected) {
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
  if (actual.getParamTypes() != expected.getParamTypes())
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
    case ArgConvention::ByRefError:
      // We checked that the function effects line up, so if we see
      // `byref_error`, then the other function must have it as well.
      assert(actualConv == ArgConvention::ByRefError &&
             "both functions must be throwing");
      [[fallthrough]];

    case ArgConvention::InitSelf:
    case ArgConvention::MutRef:
    case ArgConvention::Ref:
    case ArgConvention::InOut:
      // These conventions do not vary based on the register-passibility of the
      // type. They must always match
      if (actualConv != expectedConv)
        return false;
      lhs = lhs.getReferenceElementType();
      rhs = rhs.getReferenceElementType();
      break;

    case ArgConvention::BorrowedInMem:
    case ArgConvention::BorrowedInReg:
      if (!llvm::is_contained(
              {ArgConvention::BorrowedInMem, ArgConvention::BorrowedInReg},
              actualConv))
        return false;
      lhs = getFunctionArgumentRValueType(actualType, actualConv);
      rhs = getFunctionArgumentRValueType(expectedType, expectedConv);
      break;

    case ArgConvention::OwnedInReg:
      llvm_unreachable("not used by the mojo parser");
    case ArgConvention::OwnedInMem:
      if (actualConv != ArgConvention::OwnedInMem)
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

static LITSignatureType getReducedFunctionType(LITSignatureType sig) {
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
      PogListAttr::get(ctx, sig.getNumParams()),
      sig.getNumImplicitOriginDecls(), sig.getCaptureOrigins(),
      sig.getIsNestedOriginExclusivityCheckingDisabled());
  return SignatureType::get(sig.getValues(), sig.getParamTypes(), {},
                            sig.getArgConventions(), sig.getFnEffects(),
                            metadata);
}

static std::string generateThunkName(Type expected, Type actual) {
  std::string name;
  llvm::raw_string_ostream os(name);
  ASTType(expected).print(os, /*forDiag=*/false, /*demangleParams=*/true);
  os << '|';
  ASTType(actual).print(os, /*forDiag=*/false, /*demangleParams=*/true);

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

static LIT::FuncOp generateConversionThunk(Attribute key, ASTDecl &moduleDecl) {
  auto &shared = moduleDecl.getShared();
  // Don't generate any debuginfo for the thunk. Push a null scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(/*scope=*/nullptr);

  auto keyValues = cast<ArrayAttr>(key);
  auto actual = cast<LITSignatureType>(cast<TypeAttr>(keyValues[0]).getValue());
  auto expected =
      cast<LITSignatureType>(cast<TypeAttr>(keyValues[1]).getValue());

  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type. Add the parameters from the
  // expected signature. This contains the types of the captured parameters and
  // the actual function parameters.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] : llvm::enumerate(expected.getParamTypes())) {
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
      expected.getSpecializedSignature(paramValues).getValues();

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
      PogListAttr::get(ctx, expected.getNumParams() + 1), types.getInputs(),
      expected.getArgConventions(),
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
    case ArgConvention::OwnedInReg:
      llvm_unreachable("not used by the mojo parser");
    case ArgConvention::InitSelf:
      value = MLValue(arg);
      break;
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::InOut:
    case ArgConvention::MutRef:
      value = MLValue(arg);
      break;
    case ArgConvention::OwnedInMem:
      value = MRValue(arg);
      break;
    case ArgConvention::BorrowedInReg:
      value = SRValue(arg);
      break;
    case ArgConvention::BorrowedInMem:
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
  } else if (expected.hasInitSelfArg()) {
    // If both the caller and callee take initself, we initialize it directly
    // above and need to return none.
  } else {
    hasRegisterResult = true;
  }

  // Bind the function parameters declared on the thunk to the callee. This does
  // NOT include the capture parameters -- the callee has already been rebound
  // to them when it was declared on the parameter list.
  SmallVector<TypedAttr> bindOperands{ParamDeclRefAttr::get(calleeDecl)};
  llvm::append_range(bindOperands,
                     ArrayRef(paramValues).take_back(actual.getNumParams()));
  TypedAttr calleeParam =
      ParamOperatorAttr::get(POC::BindSignature, bindOperands);
  assert(cast<LITSignatureType>(calleeParam.getType()).getNumParams() == 0);

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
  b = ImplicitLocOpBuilder(mlirLoc, *emitter.builder);
  Value retVal;
  if (hasRegisterResult)
    retVal = emitter.emitSRValue({callResult, node}, EC_Trait);
  else if (expected.isThrows())
    retVal = b.create<ParamConstantOp>(b.getBoolAttr(false));
  else
    retVal = b.create<ParamConstantOp>(shared.getNoneAttr());
  b.create<KGEN::ReturnOp>(retVal);

  return thunk;
}

CValue LIT::convertFunctionValue(CValue value, const ExprNode *expr,
                                 LITSignatureType expected,
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
  auto actual = cast<LITSignatureType>(callee.getType());

  // Canonicalize the function types. This strips away unnecessary metadata that
  // does not affect the conversion semantics. In other words, a function type
  // and its reduced type can be trivially converted with a rebind.
  LITSignatureType reducedActual = getReducedFunctionType(actual);
  LITSignatureType reducedExpected = getReducedFunctionType(expected);

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
      cast<LITSignatureType>(replacer.getReboundType(reducedActual));

  // Now reparameterize `reducedExpected`. Captured parameters are replaced with
  // `*(0,i) where i < N` and actual parameters are replaced with `*(0,N+j)`.
  for (auto [i, type] : llvm::enumerate(actual.getParamTypes())) {
    paramTypes.push_back(replacer.getReboundType(type));
    replacer.addInputParam(
        ParamIndexRefAttr::get(i + paramRefs.size(), paramTypes.back()));
  }
  auto reparamMetadata = FnMetadataAttr::get(
      reducedExpected.getArgListAttrs(),
      PogListAttr::get(ctx, paramTypes.size()),
      reducedExpected.getNumImplicitOriginDecls(),
      reducedExpected.getCaptureOrigins(),
      reducedExpected.getIsNestedOriginExclusivityCheckingDisabled());
  auto reparamExpected = SignatureType::get(
      cast<FunctionType>(replacer.getReboundType(reducedExpected.getValues())),
      paramTypes, {}, reducedExpected.getArgConventions(),
      reducedExpected.getFnEffects(), reparamMetadata);

  // We can attempt to generate the thunk now.
  Attribute key = ArrayAttr::get(
      ctx, {TypeAttr::get(reparamActual), TypeAttr::get(reparamExpected)});
  LIT::FuncOp thunk =
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
  for (Type type : expected.getParamTypes())
    evaluator.addInputParam(UnboundAttr::get(evaluator.getReboundType(type)));
  evaluator.addInputParam(calleeParam);

  SymbolConstantAttr symbol = thunk.getBoundSymbolRef(
      ParameterExprArrayAttr::get(ctx, evaluator.getInputParams()));

  // Finally, cast the result back to the expected type.
  return emitter.emitCResult(
      PValue(ParamOperatorAttr::get(POC::Rebind, {symbol}, expected)), expr,
      dest);
}
