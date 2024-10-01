//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "FunctionTypes.h"
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
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

using namespace M;
using namespace KGEN;
using namespace LIT;

bool LIT::canConvertFunctionTypes(LITSignatureType actual,
                                  LITSignatureType expected,
                                  const TypeCheckScopeInfo &scopeInfo) {
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
  auto getRValueType = [](ASTType type, ArgConvention conv) {
    if (llvm::is_contained(
            {ArgConvention::BorrowedInReg, ArgConvention::OwnedInReg}, conv))
      return type;
    return type.getReferenceElementType();
  };
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
      lhs = getRValueType(actualType, actualConv);
      rhs = getRValueType(expectedType, expectedConv);
      break;

    case ArgConvention::OwnedInMem:
    case ArgConvention::OwnedInReg:
      if (llvm::is_contained(
              {ArgConvention::OwnedInMem, ArgConvention::OwnedInReg},
              actualConv))
        return false;
      lhs = getRValueType(actualType, actualConv);
      rhs = getRValueType(expectedType, expectedConv);
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
      sig.getNumImplicitLifetimeDecls(), sig.getCaptureLifetimes(),
      sig.getIsNestedLifetimeExclusivityCheckingDisabled());
  return SignatureType::get(sig.getValues(), sig.getParamTypes(), {},
                            sig.getArgConventions(), sig.getFnEffects(),
                            metadata);
}

LIT::FuncOp LIT::generateConversionThunk(LITSignatureType actual,
                                         LITSignatureType expected,
                                         ASTDecl &moduleDecl,
                                         SharedState &shared) {
  // TODO: Deduplicate in shared state.
  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.diags.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] : llvm::enumerate(expected.getParamTypes())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(
        ParamDeclAttr::get(moduleDecl.mangleUserDefinedParamName(
                               b.getStringAttr("i" + Twine(idx))),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }
  FunctionType types =
      expected.getSpecializedSignature(paramValues).getValues();

  // Add an additional parameter, representing the actual callee.
  auto calleeDecl = ParamDeclAttr::get(
      moduleDecl.mangleUserDefinedParamName(b.getStringAttr("callee")), actual);
  paramDecls.push_back(calleeDecl);

  // Generate a mangled name.
  StructEmitter structEmitter(shared);
  std::string name;
  llvm::raw_string_ostream os(name);
  ASTType(expected).print(os, /*forDiag=*/true, /*demangleParams=*/true);
  os << '|';
  ASTType(actual).print(os, /*forDiag=*/true, /*demangleParams=*/true);

  // Declare the function at the bottom of the decl.
  b = ImplicitLocOpBuilder(mlirLoc, moduleDecl.getDeclEndBuilder());
  LIT::FuncOp thunk = structEmitter.createFunction(
      moduleDecl, name, paramDecls,
      PogListAttr::get(ctx, expected.getNumParams() + 1), types.getInputs(),
      expected.getArgConventions(),
      PogListAttr::get(ctx, expected.getNumArguments()),
      expected.getResultType(), SpecialFunctionKind::kNormal,
      moduleDecl.getLoc(), b, expected.getFnEffects());

  // Annotate the function as a thunk by adding the conversion types.
  NamedAttrList attrs = thunk->getAttrDictionary();
  attrs.set(thunk.getThunkFromTypeAttrName(), TypeAttr::get(actual));
  attrs.set(thunk.getThunkToTypeAttrName(), TypeAttr::get(expected));

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  attrs.set(thunk.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, InlineLevel::AlwaysNoDebug));

  // Set the attributes.
  thunk->setAttrs(attrs.getDictionary(ctx));

  // Register the function as an ASTDecl to emit code inside it.
  ASTDecl &thunkDecl = shared.declResolver->addFullyResolvedDecl(
      &*thunk, thunk.getSymNameAttr(), moduleDecl.getLoc(), &moduleDecl);

  // Now prepare to emit the call.
  b = ImplicitLocOpBuilder::atBlockBegin(mlirLoc, thunk.getBody());
  ExprEmitter emitter(shared, thunkDecl, b);

  // Construct the call operands from the function block arguments.
  CallOperands operands;
  SyntheticNode node(thunkDecl.getLoc());

  for (auto [arg, conv] :
       llvm::zip(thunk.getArguments(), expected.getArgConventions())) {
    AnyValue value;
    switch (conv) {
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
    case ArgConvention::OwnedInReg:
      value = SRValue(arg);
      break;
    case ArgConvention::BorrowedInReg:
      value = SBValue(arg);
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

  CValue callResult = emitter.emitIndirectCall(
      PValue(ParamDeclRefAttr::get(calleeDecl)), std::move(operands), dest,
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

  // Canonicalize the function types.
  LITSignatureType reducedActual = getReducedFunctionType(actual);
  LITSignatureType reducedExpected = getReducedFunctionType(expected);

  LIT::FuncOp thunk =
      emitter.shared.getOrCreateFunctionThunk(reducedActual, reducedExpected);
  if (!thunk) {
    dest.resetForError();
    return {};
  }

  // Cast to the callee reduced type.
  TypedAttr calleeParam =
      ParamOperatorAttr::get(POC::Rebind, callee.get(), reducedActual);

  // Bind it to the thunk.
  ParameterEvaluator evaluator;
  for (Type type : expected.getParamTypes())
    evaluator.addInputParam(UnboundAttr::get(evaluator.getReboundType(type)));
  evaluator.addInputParam(calleeParam);

  SymbolConstantAttr symbol = thunk.getBoundSymbolRef(
      ParameterExprArrayAttr::get(ctx, evaluator.getInputParams()));

  return emitter.emitCResult(
      PValue(ParamOperatorAttr::get(POC::Rebind, {symbol}, expected)), expr,
      dest);
}
