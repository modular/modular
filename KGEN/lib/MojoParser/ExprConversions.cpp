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
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

// TODO(MOCO-1106): Use the higher-level emitGetterSetterAccess instead of using
// this directly.
extern LogicalResult bindParamValuesToDirectCall(OverloadSet &overloadSet,
                                                 ArrayRef<Operand> operands,
                                                 ExprEmitter &emitter);

//===----------------------------------------------------------------------===//
// Function Conversions
//===----------------------------------------------------------------------===//

// Strips references from the expected and actual types, reconciling allowed
// differences and extracting the pointee types to compare.
bool checkConventionsConvertible(ArgConvention expectedConv,
                                 ArgConvention actualConv) {
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

  case ArgConvention::ByRefResult:
    llvm_unreachable("`byref_result` was already handled");
  }

  return true;
}

// TODO: Return more than a boolean, so we can have better error messages.
bool canConvertFunctionTypes(SharedState &shared, FnTypeGeneratorType actual,
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

  // Functions with an incompatible number of arguments cannot be converted
  // between each other. The number of arguments should be equal (unless the
  // expected function is variadic).
  // TODO: Consider default argument values.
  std::optional<size_t> expectedVariadicArgIndexOpt =
      expected.findPackVarArgIndex();
  if (expectedVariadicArgIndexOpt.has_value()) {
    size_t expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();
    if (actualArgTypes.size() < expectedVariadicArgIndex) {
      // Caller didn't supply enough arguments.
      return false;
    }
  } else { // No variadic
    if (actualArgTypes.size() != expectedArgTypes.size()) {
      // Caller didn't supply the expected number of arguments.
      return false;
    }
  }

  // "Normal" here means it won't be received by a variadic arg in the expected
  // function.
  size_t numNormalArgs = actualArgTypes.size();
  if (expectedVariadicArgIndexOpt.has_value()) {
    numNormalArgs = expectedVariadicArgIndexOpt.value();
  }

  // Check all the normal args (which aren't going into a variadic arg).
  for (size_t actualArgIndex = 0; actualArgIndex < numNormalArgs;
       actualArgIndex++) {
    auto actualConv = actual.getArgConvention(actualArgIndex);
    ASTType actualAstType = actualArgTypes[actualArgIndex];

    // These accesses should be okay because we checked the number of actual
    // arguments above.
    ASTType expectedAstType = expectedArgTypes[actualArgIndex];
    ArgConvention expectedConv = expected.getArgConvention(actualArgIndex);

    if (!checkConventionsConvertible(expectedConv, actualConv))
      return false;

    ASTType expectedAstValueType =
        getFunctionArgumentRValueType(expectedAstType, expectedConv);
    ASTType actualValueAstType =
        getFunctionArgumentRValueType(actualAstType, actualConv);
    // Now check that the argument types line up.
    if (actualValueAstType.isEqualCanon(expectedAstValueType))
      continue;

    return false;
  }

  // The type the actual argument will be compared against. If the actual
  // argument is going into a variadic, then this will be the variadic's
  // element type, not the variadic's type itself.
  if (expectedVariadicArgIndexOpt.has_value()) {
    auto expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();

    ArgConvention expectedConv =
        expected.getArgConvention(expectedVariadicArgIndex);

    // Get the variadic pack's element trait.
    ASTType expectedArgVariadicPackType =
        expected.getIfVariadicPack(expectedVariadicArgIndex);
    RefPackType refPackType =
        expectedArgVariadicPackType.getVariadicPackInfo(shared);
    ASTType variadicElType = refPackType.getVariadicElementType();

    // This works because VariadicPack's element type is always a trait.
    auto expectedTraitType = cast<TraitType>(variadicElType.mlirType);

    for (size_t actualArgIndex = numNormalArgs;
         actualArgIndex < actualArgTypes.size(); actualArgIndex++) {
      auto actualConv = actual.getArgConvention(actualArgIndex);
      if (!checkConventionsConvertible(expectedConv, actualConv))
        return false;

      ASTType actualAstType = actualArgTypes[actualArgIndex];

      // Now that we know the conventions are valid, check that the actual
      // argument conforms to the variadic pack's element trait.

      ASTType actualValueAstType =
          getFunctionArgumentRValueType(actualAstType, actualConv);

      // If the arguments are exactly equal, skip the more expensive checks.
      if (actualValueAstType.isEqualCanon(variadicElType))
        continue;

      // We can convert a more general `actual` function (that takes in a trait
      // argument) to a more specific `expected` function that takes in a struct
      // argument, as long as that struct conforms to that trait.
      // In other words, here we're handling function conversions with covariant
      // arguments (see TTSMFS).
      ASTDecl *actualDeclOp = actualValueAstType.getDecl(shared);
      assert(actualDeclOp);
      if (!actualDeclOp)
        return false;
      std::optional<InflightDiag> x = std::nullopt;
      if (actualDeclOp->doesNominalTypeConformTo(expectedTraitType, x)) {
        continue;
      }

      return false;
    }
  }

  // The function types are convertible.
  return true;
}

static FnTypeGeneratorType getReducedFunctionType(FnTypeGeneratorType sig) {
  MLIRContext *ctx = sig.getContext();

  auto origPogListAttr = sig.getArgListAttrs();

  SmallVector<PassingKind> passingKinds(sig.getNumArguments(),
                                        PassingKind::PosOnly);
  SmallVector<StringAttr> names(sig.getNumArguments(), StringAttr::get(ctx));
  SmallVector<VariadicKind> variadics;
  for (size_t i = 0; i < sig.getNumArguments(); i++)
    variadics.push_back(origPogListAttr.getVariadicKind(i));

  // The passing kinds for results slots must be implicit;
  if (sig.hasMemoryOnlyResult())
    passingKinds.back() = PassingKind::Implicit;
  if (sig.isThrows())
    passingKinds.end()[-2] = PassingKind::Implicit;

  auto newPogListAttr =
      PogListAttr::get(ctx, names, passingKinds, {}, {}, variadics,
                       origPogListAttr.getOrigPackConvention());

  auto metadata = FnMetadataAttr::get(
      newPogListAttr, sig.getNumImplicitOriginDecls(),
      // Don't keep the capture origins, thunks don't care about those. Only the
      // parameter-value passed in at the callsite cares about those.
      {}, sig.getIsNestedOriginExclusivityCheckingDisabled());
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
  auto actualSignature =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[0]).getValue());
  auto thunkSignature =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[1]).getValue());

  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type. Add the parameters from the
  // expected signature. This contains the types of the clarifying parameters
  // (see TAPCPTTT) and the actual function's input parameters.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] :
       llvm::enumerate(thunkSignature.getInputParamTypes())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(
        ParamDeclAttr::get(moduleDecl.mangleUserDefinedParamName(
                               b.getStringAttr("_" + Twine(idx))),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }
  // Rebind the argument and result types into the scope of the body.
  FunctionType functionType =
      thunkSignature.getSpecializedGenerator(paramValues).getBody().getValues();

  // Add an additional parameter, representing the actual callee. Rebind the
  // actual function type into the scope of the body.
  auto calleeDecl = ParamDeclAttr::get(
      moduleDecl.mangleUserDefinedParamName(b.getStringAttr("callee")),
      evaluator.getReboundType(actualSignature));
  paramDecls.push_back(calleeDecl);

  // Generate a mangled name.
  std::string name = generateThunkName(thunkSignature, actualSignature);

  // Declare the function at the bottom of the decl.
  b = ImplicitLocOpBuilder(mlirLoc, moduleDecl.getDeclEndBuilder());
  StructEmitter structEmitter(shared);
  auto [thunk, thunkDecl] = structEmitter.synthesizeFunction(
      moduleDecl, name, paramDecls,
      PogListAttr::get(ctx, thunkSignature.getInputParamTypes().size() + 1),
      functionType.getInputs(), thunkSignature.getArgConventions(),
      PogListAttr::get(ctx, thunkSignature.getNumArguments()),
      functionType.getResults().front(), SpecialFunctionKind::kNormal,
      moduleDecl.getLoc(), b, thunkSignature.getFnEffects());

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
          MBValue(thunk.getArguments()[thunkVariadicArgIndex]);

      auto index = IntegerAttr::get(IndexType::get(ctx), indexInVariadic);

      SyntheticNode indexSynthNode(moduleDecl.getLoc(), PValue(index));

      auto variadicTypeFromFunctionType =
          functionType.getInputs()[thunkVariadicArgIndex];

      ValueDest eltDest(EC_VarArgArgument);

      // TODO(MOCO-1106): Use the higher-level emitGetterSetterAccess instead of
      // the below OverloadSet/emitCall directly. It'll require refactoring
      // emitGetterSetterAccess to avoid some index mismatch bugs.

      // Look up VariadicPack.__getitem__
      OverloadSet getItemOv = OverloadSet::lookup(
          emitter.getDeclScope(),
          ASTType(variadicTypeFromFunctionType).getReferenceElementType(),
          "__getitem__", node, CallSyntax::kDirectCall);
      if (failed(bindParamValuesToDirectCall(
              getItemOv,
              {Operand(&indexSynthNode, moduleDecl.getLoc(),
                       Operand::PassKind::kKeyword,
                       StringAttr::get(ctx, "index"))},
              emitter))) {
        // This should theoretically never happen, because we own VariadicPack.
        emitter.emitError(moduleDecl.getLoc(),
                          "Internal error: Couldn't find VariadicPack's "
                          "__getitem__ method for the "
                          "generated variadic thunk.");
        return {};
      }

      // Call the_pack.__getitem__[index]()
      CallOperands getItemOperands(
          {ASTExprAnd<MBValue>{packRefMBValue, &node}});
      CValue getItemResult =
          getItemOv.emitCall(std::move(getItemOperands), eltDest, emitter);
      if (!getItemResult) {
        // This should theoretically never happen, because we own VariadicPack.
        emitter.emitError(moduleDecl.getLoc(),
                          "Internal error: Couldn't call "
                          "VariadicPack.__getitem__[index] in the "
                          "generated variadic thunk.");
        return {};
      }
      argForActual = getItemResult.getMlirValue();

      // Thunks can only receive
      convForActual = ArgConvention::ReadMem;
    } else {
      argForActual = thunk.getArguments()[actualArgIndex];
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
      value = MRValue(argForActual);
      break;
    case ArgConvention::ReadReg:
      value = SRValue(argForActual);
      break;
    case ArgConvention::ReadMem:
    case ArgConvention::Ref:
      value = MBValue(argForActual);
      break;
    }
    operands.add({value, node});
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_Trait);
  bool hasRegisterResult = false;
  if (thunkSignature.isAsync()) {
    // An async call returns a coroutine we have to await.
  } else if (thunkSignature.hasMemoryOnlyResult()) {
    dest = ValueDest(MLValue(thunk.getArguments().back()), EC_Trait);
  } else {
    hasRegisterResult = true;
  }

  // Bind the function parameters declared on the thunk to the callee. This does
  // NOT include the clarifying parameters -- the callee has already been
  // rebound to them when it was declared on the parameter list.
  //
  // In this example (from TAAMCE):
  //
  //     fn ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: fn[Y: Bool](read Ship[Z])->None
  //     ](mut s: Ship[Z, Y]):
  //         callee[Y](s) # implicit cast to imm
  //
  // notice how we're calling `callee[Y](s)` and the clarifying parameter Z
  // doesn't appear on that call line.
  TypedAttr calleeParam = BindParamsAttr::get(
      ParamDeclRefAttr::get(calleeDecl),
      ArrayRef(paramValues)
          .take_back(actualSignature.getInputParamTypes().size()));
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
  if (thunkSignature.isAsync()) {
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

  // We need to specially handle when `actual` mentions any parameters in its
  // scope, like how `= read_ship[Z]` mentions the `Z` parameter here:
  //
  //     struct Ship[X: int, Y: Bool]:
  //         pass
  //
  //     fn read_ship[X: int, Y: Bool](read s: Ship[X, Y]):
  //         pass
  //
  //     fn foo():
  //         alias Z: int = 42
  //         alias my_func_alias: fn[Y: Bool](mut Ship[Z, Y]) -> None =
  //             read_ship[Z]
  //
  // `read_ship[Z]`s type is `fn(read Ship[Y: Bool][Z])`. However, when our
  // thunk accepts that type as an input parameter, the thunk is malformed
  // because it has no idea what `ZC` is (see TAPRCT for more).
  //
  // So, we prepend a "clarifying" parameter to the thunk's input parameters,
  // like the `Z` here:
  //
  //     fn ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: fn[Y: Bool](read Ship[Z])->None
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
  actual.walk([&](ParamDeclRefAttr ref) { mentionedParamRefs.insert(ref); });
  // This replacer will help us figure out the thunk's param types, so the thunk
  // signature has a correct:
  //     mut s: Ship[ship_func_thunk's Z]
  // instead of an incorrect:
  //     mut s: Ship[foo's Z]
  // It also helps us generate some more general signatures for the thunk keys.
  ParameterEvaluator paramRefsReplacer;
  for (auto [i, ref] : llvm::enumerate(mentionedParamRefs)) {
    // Add these mentioned param refs as "clarifying" parameters to the thunk,
    // see TAPCPTTT.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(ref.getType()));
    paramRefsReplacer.setParameterValue(
        ref.getName(), ParamIndexRefAttr::get(i, thunkParamTypes.back()));
  }
  auto reparamActualForThunkKey = cast<FnTypeGeneratorType>(
      paramRefsReplacer.getReboundType(reducedActual));
  // Above, clarifying parameters were at the beginning (and were replaced with
  // `*(0,i) where i < N`).
  //
  // Now, we need to add `expected`'s input params, like the `[Y: Bool]` in:
  //
  //     alias my_func_alias: fn[Y: Bool](mut Ship[Z, Y]) -> None = ...
  //
  // Note that `expected` contains param refs to parameters declared in/by foo.
  // `expected` does NOT contain paramrefs referring to the callee's function
  // definition's parameters.
  for (auto [i, type] : llvm::enumerate(expected.getInputParamTypes())) {
    // Note that `type` might contain UnboundAttr at this point, that's fine.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(type));
    paramRefsReplacer.addInputParam(ParamIndexRefAttr::get(
        i + mentionedParamRefs.size(), thunkParamTypes.back()));
  }
  // The thunk metadata and function type will mostly look like `expected`,
  // except for the thunk param types (which also includes clarifying
  // parameters, see TAPCPTTT).
  auto thunkMetadata = FnMetadataAttr::get(
      reducedExpected.getArgListAttrs(),
      reducedExpected.getNumImplicitOriginDecls(),
      reducedExpected.getCaptureOrigins(),
      reducedExpected.getIsNestedOriginExclusivityCheckingDisabled());
  auto thunkFuncType = cast<FunctionType>(
      paramRefsReplacer.getReboundType(reducedExpected.getValues()));
  auto thunkSignature = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/thunkParamTypes,
      /*values=*/thunkFuncType,
      /*argConvs=*/reducedExpected.getArgConventions(),
      /*effects=*/reducedExpected.getFnEffects(),
      /*fnMetadata=*/thunkMetadata,
      /*genMetadata=*/PogListAttr::get(ctx, thunkParamTypes.size()));

  thunkSignature.walk([&](ParamDeclRefAttr ref) {
    // There shouldn't be any ParamDeclRefAttr in the thunk signature, because
    // there's no parent scope param-decls for them to refer to.
    assert(false);
  });

  // We can attempt to generate the thunk now.
  Attribute key = ArrayAttr::get(ctx, {TypeAttr::get(reparamActualForThunkKey),
                                       TypeAttr::get(thunkSignature)});
  FnOp thunk =
      emitter.shared.getOrCreateFunctionThunk(key, generateConversionThunk);
  if (!thunk) {
    dest.resetForError();
    return {};
  }

  // Now that we have the thunk defined somewhere, we're going to reference it.
  // In the above `foo` example, in this `alias` line:
  //
  //     alias my_func_alias: fn(mut Ship[ZC]) -> None =
  //         ship_func_thunk[ZC, read_ship[ZC]]
  //
  // ...we'll now produce the `ship_func_thunk[ZC, read_ship[ZC]]`.

  // First, cast the callee to the reduced actual type.
  TypedAttr calleeParam =
      ParamOperatorAttr::get(POC::Rebind, callee.get(), reducedActual);

  // Assemble the parameters (`ZC, read_ship[ZC]`) that we'll bind to the thunk.
  ParameterEvaluator evaluator;
  for (ParamDeclRefAttr ref : mentionedParamRefs) {
    // Bind the clarifying parameter (see TAPCPTTT).
    evaluator.addInputParam(ref);
  }
  for (Type type : expected.getInputParamTypes()) {
    // If there are "remaining input parameters", like in:
    //
    //     alias my_func_alias: fn[Y: Bool]() -> None = ...
    //
    // then we leave them unbound (see TARIPNBITM).
    evaluator.addInputParam(UnboundAttr::get(evaluator.getReboundType(type)));
  }
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
  if (auto fromOrigin = dyn_cast<OriginType>(fromType))
    if (auto toOrigin = dyn_cast<OriginType>(toType)) {
      auto toMut = toOrigin.getIsMutable();
      auto result =
          ParamOperatorAttr::get(POC::And, toMut, fromOrigin.getIsMutable());
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
               ASTType &result, ExprEmitter &emitter) {

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
    CallOperands operands({srcValue});
    auto res = os.filterOverloadSet(operands, /*emitDiag*/ false, emitter);
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
    if (ExprEmitter::canImplicitlyConvertToType(val2, lhsMPType,
                                                emitter.declScope))
      return succeed(lhsMPType, lhsMWPV, PValue());
    result = lhsMPType;
    return {CTR_MergeWithConvertFail, lhsMWPV, PValue()};
  }
  if (rhsMWPV) {
    if (ExprEmitter::canImplicitlyConvertToType(val1, rhsMPType,
                                                emitter.declScope))
      return succeed(rhsMPType, PValue(), rhsMWPV);
    result = rhsMPType;
    return {CTR_MergeWithConvertFail, PValue(), rhsMWPV};
  }

  // Otherwise, we have no __merge_with__ method, check out implicit
  // conversions.

  // If one type implicit converts to the other, then the other is a common
  // type.  Don't do this if both convert to each other, this would be
  // ambiguous.
  bool isConvertibleToType2 =
      ExprEmitter::canImplicitlyConvertToType(val1, type2, emitter.declScope);
  bool isConvertibleToType1 =
      ExprEmitter::canImplicitlyConvertToType(val2, type1, emitter.declScope);
  if (isConvertibleToType2 && !isConvertibleToType1)
    return succeed(type2);
  if (isConvertibleToType1 && !isConvertibleToType2)
    return succeed(type1);
  if (isConvertibleToType1 && isConvertibleToType2)
    return {CTR_Ambiguous, PValue(), PValue()};

  // If one or the other type is non-materializable, the conversion is free,
  // so check to see if there is an unambiguous common type.
  bool type2ConvertsToType1Nonmat = false;
  bool type1ConvertsToType2Nonmat = false;
  auto type1Nonmat = type1.getNonmaterializableTarget(emitter.shared);
  auto type2Nonmat = type2.getNonmaterializableTarget(emitter.shared);
  if (type1Nonmat)
    type2ConvertsToType1Nonmat = ExprEmitter::canImplicitlyConvertToType(
        val2, type1Nonmat, emitter.declScope);
  if (type2Nonmat)
    type1ConvertsToType2Nonmat = ExprEmitter::canImplicitlyConvertToType(
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
ParseResult ExprEmitter::coerceTypesToEachOther(
    SMLoc loc, CValue &lhs, const ExprNode *lhsExpr, CValue &rhs,
    const ExprNode *rhsExpr, std::function<void(bool isLHS)> configEmitter) {
  if (!configEmitter)
    configEmitter = [&](bool isLHS) {};

  if (!lhs || !rhs)
    return failure();

  // If they are the same or if there is a common type between these, convert
  // them to it.
  ASTType commonType;
  auto [commonTypeResult, lhsMWPV, rhsMWPV] =
      findCommonType({lhs, lhsExpr}, {rhs, rhsExpr}, commonType, *this);

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
    lhs = emitIndirectCall(lhsMWPV, CallOperands({{lhs, lhsExpr}}), dest,
                           CallSyntax::kMethodCall, lhsExpr);
  }
  if (rhsMWPV) {
    configEmitter(/*isLHS*/ false);
    ValueDest dest(EC_MergeWith);
    rhs = emitIndirectCall(rhsMWPV, CallOperands({{rhs, rhsExpr}}), dest,
                           CallSyntax::kMethodCall, rhsExpr);
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

  return success(lhs && rhs);
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

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkMLIRTypeConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveBody(traitDecl, loc)))
    return false; // an error was emitted
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto traitFn = dyn_cast<FnOp>(*decl);
      // Skip any children that aren't methods or are inherited. This could be
      // an alias.
      if (!traitFn || traitFn.getInheritedFrom())
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
  if (failed(shared.declResolver->resolveBody(traitDecl, loc)))
    return {};

  // Use a special wrapper decl in the builtins as stubs.
  ASTDecl *wrapperDecl = shared.getBuiltinStubsMLIRType(loc).getDecl(shared);
  if (!wrapperDecl || !isa<StructDeclOp>(wrapperDecl)) {
    shared.emitError(loc, "malformed builtin._stubs.__MLIRType");
    return {};
  }
  ASTType boundWrapper =
      cast<StructDeclOp>(wrapperDecl).bindReference({typeValue});

  // Explicitly check that the wrapper conforms to the trait so that
  // conformances & special functions may be generated.
  std::optional<InflightDiag> checkDiag;
  if (!wrapperDecl->doesNominalTypeConformTo(trait, checkDiag)) {
    InflightDiag diag =
        shared.emitError(value.expr->getLoc(), "cannot bind MLIR type ")
        << mlirType << " to trait " << ASTType(trait)
        << " as it is unable to satisfy the following requirements";
    if (checkDiag)
      diag.attachNote(wrapperDecl->getLoc()) << std::move(*checkDiag);
    return {};
  }

  // NOTE: This substantially duplicates emitMetaTypeToTraitConversion because
  // it is doing some crazy manual binding of the type into the parameter list
  // so the vtable entries are specialized on the MLIR type.
  //
  // FIXME(MOCO-1146): Could we instead just synthesize the members required and
  // eliminate __mlir_type entirely?  This __mlir_type thing introduces other
  // bugs.  We already do this for rp-trivial types which MLIR types are.
  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    assert(!decls.empty() && isa<FnOp>(decls.front()));

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
      result =
          canConvertFunctionTypes(shared, rvFunctionType, requiredFunction);
    return cacheAndReturnVal(result);
  }

  // We can implicitly convert to the specified type if we can construct it with
  // the value as an implicit conversion.
  FailureOr<PValue> result = OverloadSet::canConstructType(
      requiredType, {{value}}, value.expr, declScope,
      /*isImplicitConversion=*/true);
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
      if (canConvertFunctionTypes(shared, rvFunctionType, requiredFunction))
        return convertFunctionValue(value, expr, requiredFunction, *this, dest);
  }

  // We disable implicit conversions to prevent converting T -> S -> U in
  // one step, and to avoid infinite conversion cycles.
  return emitConstructorCall(requiredType, CallOperands({valueExpr}), expr,
                             CallSyntax::kImplicitConvert, dest);
}
