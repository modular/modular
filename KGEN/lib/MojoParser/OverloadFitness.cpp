//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#include "OverloadFitness.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamInf.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Diagnostic emission implementation
//===----------------------------------------------------------------------===//

namespace {
/// Helper class to emit errors without cluttering the evaluation logic.
struct DiagEmitter : public SharedStateUser {
  DiagEmitter(SharedState &shared, SMLoc callLoc)
      : SharedStateUser(shared), callLoc(callLoc) {}

  MojoInflightDiag
  unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const;
  MojoInflightDiag wrongParamCount(size_t expectedNumParams,
                                   size_t actualNumParams) const;
  MojoInflightDiag wrongArgCountWithPack(size_t minRequiredArgs,
                                         size_t maxAllowedArgs,
                                         size_t numOperands) const;
  MojoInflightDiag unresolvedPackCount(size_t numOperands) const;
  MojoInflightDiag wrongPosOnlyCount(size_t minRequiredArgs,
                                     size_t maxAllowedArgs, size_t numOperands,
                                     const Twine &argOrParam) const;
  MojoInflightDiag resultGenericMemType(Type outputType) const;
  MojoInflightDiag argGenericMemType(size_t expectedArgIdx, Type expectedType,
                                     PogListAttr argListAttr) const;
  MojoInflightDiag missingArgs(ArrayRef<StringAttr> missingArgs,
                               const Twine &kindStr) const;
  MojoInflightDiag
  posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const;
  MojoInflightDiag tooManyPosArgs(size_t maxAllowedArgs,
                                  size_t numPosOperands) const;
  MojoInflightDiag byPosAndKw(ArrayRef<StringAttr> names) const;
  MojoInflightDiag badImplicitConversion(ASTType fromType,
                                         ASTType toType) const;

private:
  SMLoc callLoc;

  MojoInflightDiag initDiag() const { return shared.emitError(callLoc); }
};
} // namespace

MojoInflightDiag
DiagEmitter::unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const {
  auto diag = initDiag();
  emitUnknownKeywords(diag, unknownKwOperands, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::wrongParamCount(size_t expectedNumParams,
                                              size_t actualNumParams) const {
  auto diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                           /*maxAllowed=*/expectedNumParams, actualNumParams,
                           "parameter");
  return diag;
}

MojoInflightDiag DiagEmitter::wrongArgCountWithPack(size_t minRequiredArgs,
                                                    size_t maxAllowedArgs,
                                                    size_t numOperands) const {
  auto diag = initDiag() << "callee with non-empty variadic pack argument";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional operand");
  return diag;
}

MojoInflightDiag DiagEmitter ::unresolvedPackCount(size_t numOperands) const {
  return initDiag() << "assigning " << numOperands << " operand"
                    << plural(numOperands)
                    << " to an unresolvable variadic pack argument";
}

MojoInflightDiag DiagEmitter::wrongPosOnlyCount(size_t minRequiredArgs,
                                                size_t maxAllowedArgs,
                                                size_t numOperands,
                                                const Twine &argOrParam) const {
  auto diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional " + argOrParam);
  return diag;
}

MojoInflightDiag DiagEmitter::resultGenericMemType(Type outputType) const {
  return initDiag()
         << "result cannot bind __TypeOfAllTypes type to memory-only type "
         << ASTType(outputType);
}

MojoInflightDiag DiagEmitter::argGenericMemType(size_t argIdx,
                                                Type expectedType,
                                                PogListAttr argListAttr) const {
  return initDiag() << "cannot bind __TypeOfAllTypes type to memory-only type "
                    << ASTType(expectedType) << " expected by argument "
                    << argListAttr.getPogs()[argIdx].getName();
}

/// Attach extra type conversion error detail or hints to the user when
/// reporting an error passing `operand` to an argument of type `argType`.
static void addTypeConversionDetail(MojoInflightDiag &diag,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType argType, SharedState &shared) {
  auto loc = operand.expr->getLoc();
  ASTType operandType = operand.ir.getRValueTypeIfResolvable();
  if (!operandType) {
    diag.attachNote(loc) << "try resolving the overloaded function first";
    return;
  }
  // Try to detect mismatched memory result type.
  auto lhsSig = sugarDynCast<FnTypeGeneratorType>(operandType);
  auto rhsSig = sugarDynCast<FnTypeGeneratorType>(argType);
  if (lhsSig && rhsSig) {
    auto getByRefResult = [](FnTypeGeneratorType sig) -> std::pair<bool, Type> {
      return {sig.getBody().hasMemoryOnlyResult(),
              ASTType(sig.getUserResultType())};
    };
    auto [lhsByRef, lhsRetType] = getByRefResult(lhsSig);
    auto [rhsByRef, rhsRetType] = getByRefResult(rhsSig);
    if (lhsByRef == rhsByRef || lhsRetType != rhsRetType)
      return;
    // Different result semantics but same result type.
    diag.attachNote(loc) << "memory-only type bound to generic result type: "
                         << (lhsByRef ? "payload" : "argument") << " returns "
                         << ASTType(lhsRetType) << " by reference";
    return;
  }
}

/// Emit a tailored diagnostic when failing to convert a value to type !lit.ref.
/// This happens when the user is forming a Reference incorrectly which happens
/// when confusion and details run the highest.
static void diagnoseFailedRefTypeConversion(MojoInflightDiag &diag,
                                            ASTExprAnd<AnyValue> operand,
                                            RefType argType,
                                            SharedState &shared) {
  diag << "ref " << ASTType(argType.getElementType());

  auto loc = operand.expr->getLoc();
  if (operand.ir.getIfRValue()) {
    diag.attachNote(loc) << "cannot bind an RValue to a reference";
    return;
  }
  if (!operand.ir.isMValue()) {
    diag.attachNote(loc) << "operand does not have a memory representation";
    return;
  }

  auto operandRefTy = operand.ir.getMValueType();
  if (!ASTType(argType.getElementType())
           .isEqualCanon(operandRefTy.getElementType())) {
    diag.attachNote(loc) << "operand element type "
                         << ASTType(operandRefTy.getElementType())
                         << " doesn't match expected element type "
                         << ASTType(argType.getElementType());
  } else if (argType.getAddressSpace() != operandRefTy.getAddressSpace()) {
    diag.attachNote(loc) << "operand address space "
                         << operandRefTy.getAddressSpace()
                         << " doesn't match expected address space "
                         << argType.getAddressSpace();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy.getOriginType(),
                                            argType.getOriginType(), shared)) {
    diag.attachNote(loc) << "operand mutability " << operandRefTy.isMutable()
                         << " doesn't match expected mutability "
                         << argType.isMutable();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy, argType, shared)) {
    auto operandO = operandRefTy.getOrigin();
    auto argO = argType.getOrigin();
    // Strip off mutcasts etc - if the origins still differ we can complain
    // about the simpler thing.
    auto operandOS = OriginType::stripMutCastAndRebind(operandO);
    auto argOS = OriginType::stripMutCastAndRebind(argO);
    if (operandOS != argOS) {
      argO = argOS;
      operandO = operandOS;
    }

    diag.attachNote(loc) << "operand origin '"
                         << ASTType::getOriginAsString(operandO, &shared)
                         << "' doesn't match expected origin '"
                         << ASTType::getOriginAsString(argO, &shared) << "'";
  }
}

namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag) {
  if (auto initList = value.getIfInitializer()) {
    switch (initList->syntax) {
    case InitializerUValue::kSliceLiteral:
      diag << "slice literal";
      break;
    case InitializerUValue::kListLiteral:
      diag << "list literal";
      break;
    case InitializerUValue::kDictLiteral:
      diag << "dictionary literal";
      break;
    case InitializerUValue::kSetInitLiteral:
      diag << "initializer list or set literal";
      break;
    }
  } else
    diag << "unknown overload";
}

void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared) {
  // Special case implicit conversions with a custom message.
  if (syntax == CallSyntax::kImplicitConvert) {
    if (ASTType type = operand.ir.getRValueTypeIfResolvable())
      diag << type;
    else
      printUValueTypeInfo(operand.ir, diag);
    diag << " value to " << expectedType;
    return;
  }

  // When the function has no source-level pog metadata (an empty PogList,
  // e.g. a `FuncType` constructed outside the Mojo parser), fall back to a
  // generic diagnostic without the argument name.
  if (StringAttr name = argListAttr.getName(argIdx); !name.empty()) {
    diag << "value passed to " << name << " cannot be converted from "
         << operand.expr->getRange();
  } else {
    diag << "value cannot be converted from " << operand.expr->getRange();
  }
  ASTType rValueType = operand.ir.getRValueTypeIfResolvable();
  bool isConvertingTypeValue = expectedType.extractMetaType() == rValueType;
  if (rValueType) {
    if (isConvertingTypeValue)
      diag << "type value " << expectedType;
    else
      diag << rValueType;
  } else {
    printUValueTypeInfo(operand.ir, diag);
  }
  diag << " to ";

  if (auto refType = sugarDynCast<RefType>(expectedType)) {
    diagnoseFailedRefTypeConversion(diag, operand, refType, shared);
    return;
  }

  diag << (isConvertingTypeValue ? "an instance of " : "") << expectedType;
  if (isConvertingTypeValue)
    diag << "; did you mean to instantiate " << expectedType << "?";
  addTypeConversionDetail(diag, operand, expectedType, shared);
}
} // namespace M::KGEN::LIT

MojoInflightDiag DiagEmitter::missingArgs(ArrayRef<StringAttr> missingArgs,
                                          const Twine &kindStr) const {
  MojoInflightDiag diag = initDiag();
  emitMissing(diag, missingArgs, kindStr + " argument");
  return diag;
}

MojoInflightDiag
DiagEmitter::posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const {
  MojoInflightDiag diag = initDiag();
  emitPosOnlyPassedByKw(diag, posOnlyPassedByKw, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::tooManyPosArgs(size_t maxAllowedArgs,
                                             size_t numPosOperands) const {
  MojoInflightDiag diag = initDiag();
  diag << "expected at most " << maxAllowedArgs << " positional argument"
       << plural(maxAllowedArgs) << ", got " << numPosOperands;
  return diag;
}

MojoInflightDiag DiagEmitter::byPosAndKw(ArrayRef<StringAttr> names) const {
  MojoInflightDiag diag = initDiag();
  emitByPosAndKw(diag, names, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::badImplicitConversion(ASTType fromType,
                                                    ASTType toType) const {
  MojoInflightDiag diag = initDiag();
  diag << "cannot implicitly convert";
  if (fromType)
    diag << " " << fromType;

  // Add target type to diag
  if (toType)
    diag << " to " << toType << ": add an explicit cast";
  return diag;
}

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

bool OverloadFitness::isBetter(const OverloadFitness &other) const {
  // Neither operand should be invalid or have param constraint issues
  // (they should be filtered out before comparison).
  assert(getValidity() >= Validity::kFunctionConstraintInconclusive &&
         other.getValidity() >= Validity::kFunctionConstraintInconclusive);

  // We first compare the number of implicit conversions.
  size_t numConversions = getNumImplicitConversions();
  size_t otherNumConversions = other.getNumImplicitConversions();
  if (numConversions != otherNumConversions)
    return numConversions < otherNumConversions;

  // We consider exact matches of concrete types to be more specific than
  // those needing nonmaterializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work).
  if (payload.passesVarArgArgument != other.payload.passesVarArgArgument)
    return payload.passesVarArgArgument < other.payload.passesVarArgArgument;

  // Otherwise these candidates are almost identical, so we try to decide based
  // on the number of input conventions mismatches (e.g. register-passable
  // passed in memory).
  if (payload.numMismatchedConventions !=
      other.payload.numMismatchedConventions)
    return payload.numMismatchedConventions <
           other.payload.numMismatchedConventions;

  // If still ambiguous, we compare the number of bindings. This allows us to
  // treat a function with more parameters as worse than a parameter-less
  // function, so things like:
  //    def foo(a: Int):   and  def foo[T: AnyType](a: T):
  // resolve to the more specific function.
  return paramBindings.size() < other.paramBindings.size();
}

OverloadFitness OverloadFitness::evaluate(ASTDecl *candidate,
                                          const OverloadSet &callable,
                                          PValue selfPValue) {
  DeclResolver &resolver = *callable.getShared().declResolver;
  auto func = cast_or_null<FnOp>(candidate->getIfOperation());
  FnTypeGeneratorType signature = func.getFullSignature();

  if (selfPValue) {
    // TODO(MOCO-1259): Support static methods with associated aliases
    auto parentDecl = candidate->getParentDecl();
    if (dyn_cast_or_null<TraitDeclOp>(parentDecl->getIfOperation())) {
      signature = substituteTraitAliasesIntoSignature(
          resolver, *parentDecl, func, signature, selfPValue);
    }
  }

  ParamInf inference(callable.paramBindings, signature.getInputParamTypes(),
                     signature.getMetadata(),
                     /*allowImplicitConversions=*/true, candidate,
                     /*discardError=*/true);
  // Don't yield constraint failure for a single overload failure: we want a
  // better error message diagnosed for the entire set.
  ParameterExprArrayAttr bindings = inference.inferForStruct();

  if (!bindings) {
    // If no diagnostics were emitted, this must be an inconclusive fitness.
    if (!inference.diag.hasErrorEmitted()) {
      assert(!inference.unprovableConstraints.empty());
      return std::move(inference.unprovableConstraints);
    }
    // Otherwise, it's a real failure, so we return an invalid fitness.
    return std::move(*inference.diag.takeMojoDiag());
  }

  OverloadFitness fitness(
      bindings,
      /*noArgsNeedingOrigins*/ OperandsNeedingOriginsList());
  fitness.unprovableConstraints =
      std::move(inference.bodyUnprovableConstraints);
  return fitness;
}

/// Extract the closure name from a self operand. This handles two cases:
/// (1) Closure parameters: the type is a ParamType wrapping a ParamDeclRefAttr,
///     and the name is the parameter name (e.g. "C").
/// (2) Closure instances: the value is defined by a VarDeclOp (a nested
///     function materialized as a closure), and the name is the variable name
///     (e.g. "kernel").
static StringAttr closureNameFromSelfOperand(SharedState &shared,
                                             CValue selfCValue) {
  auto paramType = sugarDynCast<ParamType>(selfCValue.getRValueType().mlirType);
  if (paramType) {
    auto paramRef = dyn_cast<ParamDeclRefAttr>(paramType.getParam());
    if (paramRef) {
      if (ClosureEmitter::isClosureType(shared, paramRef.getType()))
        return paramRef.getName();
    }
  }
  Value mlirValue = selfCValue.getMlirValue();
  if (mlirValue) {
    if (auto varDecl = mlirValue.getDefiningOp<VarDeclOp>()) {
      if (ClosureEmitter::isClosureType(shared, varDecl.getType()))
        return varDecl.getNameAttr();
    }
  }
  return {};
}

static ArrayRef<ClosureParamCapture>
closureParamCapturesIfClosure(ASTDecl *funcIfDirect,
                              const CallOperands &operands,
                              const OverloadSet &callable) {
  if (!funcIfDirect)
    return {};
  if (operands.empty())
    return {};
  auto selfCValue = operands[0].ir.getIfCValue();
  if (!selfCValue)
    return {};
  StringAttr closureName =
      closureNameFromSelfOperand(funcIfDirect->getShared(), selfCValue);
  if (!closureName)
    return {};

  // Look up the captures on the operation that owns the closure definition.
  // For block arguments (closure parameters), this is the parent op of the
  // block. For VarDeclOps (closure instances), this is the parent op of the
  // block containing the VarDeclOp.
  Value mlirValue = selfCValue.getMlirValue();
  if (!mlirValue)
    return {};
  Operation *ownerOp = mlirValue.getParentBlock()->getParentOp();
  ClosureParamCaptures *captures =
      funcIfDirect->getShared().getClosureParamCapturesForOp(ownerOp);
  if (!captures)
    return {};
  auto ptr = captures->find(closureName);
  if (ptr != captures->end())
    return ptr->second;
  return {};
}

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `callOperands`.
///
/// The 'funcIfDirect' member is set if this is a direct call, or null if
/// indirect.  It can be used to tune diagnostics.
OverloadFitness OverloadFitness::evaluate(FnTypeGeneratorType signature,
                                          ASTDecl *funcIfDirect,
                                          const OverloadSet &callable,
                                          const CallOperands &operands,
                                          bool allowImplicitConversions) {
  // We set up diagnostics.
  size_t numPosOperands = operands.getNumPositional();

  SMLoc callLoc = callable.getExpr()->getLoc();
  SharedState &shared = callable.getShared();
  DiagEmitter emitDiagFor(shared, callLoc);

  if (!operands.empty()) {
    if (auto selfCValue = operands[0].ir.getIfCValue()) {
      if (auto selfPValue = PValue(selfCValue.getRValueType().mlirType)) {
        // TODO(MOCO-1259): Support static methods with associated aliases
        if (funcIfDirect) {
          if (auto func =
                  dyn_cast_or_null<FnOp>(funcIfDirect->getIfOperation())) {
            auto parentDecl = funcIfDirect->getParentDecl();
            if (dyn_cast_or_null<TraitDeclOp>(parentDecl->getIfOperation())) {
              signature = substituteTraitAliasesIntoSignature(
                  *shared.declResolver, *parentDecl, func, signature,
                  selfPValue);
            }
          }
        }
      }
    }
  }

  // If a variadic keyword arg is expected, we collect the unknown kw operands.
  PogListAttr argListAttr = signature.getArgListAttrs();
  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] =
      operands.diagnoseKeywordOperands(argListAttr, variadicKwOperands);
  switch (kwDiagRes) {
  case CallOperands::KwDiagResult::kMissingKwOnly:
    return emitDiagFor.missingArgs(kwDiagNames, "keyword-only");
  case CallOperands::KwDiagResult::kOutOfOrderInferredKw:
    llvm_unreachable("no inferred arguments");
  case CallOperands::KwDiagResult::kPosOnlyPassedByKw:
    return emitDiagFor.posOnlyPassedByKw(kwDiagNames);
  case CallOperands::KwDiagResult::kUnknownKeywords:
    return emitDiagFor.unexpectedKwArgs(kwDiagNames);
  default:
    break;
  }

  auto [posDiagRes, posDiagNames] = operands.diagnosePosOperands(argListAttr);
  // If any operand is a `*pack` splat and the count didn't match, attach
  // a note explaining the gap and pointing at the working pattern.
  // `CallNode::emitIR` eagerly expands splats whose `VariadicPack`
  // element list is statically resolved; this note primarily helps the
  // case where the element list is still symbolic (e.g. `Ts.values` in
  // a generic body), but it's also useful when a resolved splat happens
  // to have the wrong element count.
  auto attachPackSplatNote = [&](MojoInflightDiag &diag) {
    for (const OperandValue &op : operands.values) {
      if (!op.isUnpackedPositional())
        continue;
      diag.attachNote(op.expr->getLoc())
          << "'*' splat is only supported when the callee accepts a "
             "variadic pack argument at this position; to forward a "
             "runtime pack to a fixed-arity callee, route the call through "
             "a dispatcher whose argument is itself a variadic pack (e.g. "
             "`def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: "
             "def(*args: *Ts) thin](...): callee(*pack)`)";
      return;
    }
  };
  switch (posDiagRes) {
  case CallOperands::PosDiagResult::kMissingPos: {
    auto diag = emitDiagFor.missingArgs(posDiagNames, "positional");
    attachPackSplatNote(diag);
    return diag;
  }
  case CallOperands::PosDiagResult::kTooManyPos: {
    size_t numPosMaximum = countNumPositional(argListAttr);
    auto diag = emitDiagFor.tooManyPosArgs(numPosMaximum, numPosOperands);
    attachPackSplatNote(diag);
    return diag;
  }
  case CallOperands::PosDiagResult::kByPosAndKw:
    return emitDiagFor.byPosAndKw(posDiagNames);
  default:
    break;
  }

  // Check that the signature can be rebound with this set of bindings.
  OperandsNeedingOriginsList operandsNeedingOrigins;
  CallParamInf inference(callable.paramBindings, signature.getInputParamTypes(),
                         signature.getParamListAttrs(),
                         allowImplicitConversions, funcIfDirect,
                         /*discardError=*/false, signature, operands,
                         variadicKwOperands, operandsNeedingOrigins);
  // Check if we're calling a closure's __call__ method and need to set
  // captured closure parameters. Only applies to method call syntax on a
  // __call__ method — not direct calls that happen to pass a closure as an
  // argument.
  ArrayRef<ClosureParamCapture> implicitParams;
  if (funcIfDirect) {
    if (auto fnOp = dyn_cast_or_null<FnOp>(funcIfDirect->getIfOperation())) {
      if (fnOp.getSourceNameAttr() &&
          fnOp.getSourceNameAttr().getValue() == "__call__")
        implicitParams =
            closureParamCapturesIfClosure(funcIfDirect, operands, callable);
    }
  }
  if (!implicitParams.empty()) {
    // Capture slots are at the front of the method's own parameter list
    // (Inferred PassingKind), but getFullSignature() prepends contextual
    // params (e.g. the trait's _Self). Skip past them.
    auto fnOp = cast<FnOp>(funcIfDirect->getIfOperation());
    size_t contextualParams =
        signature.getInputParamTypes().size() -
        fnOp.getFuncTypeGenerator().getInputParamTypes().size();
    size_t paramIdx = contextualParams;
    for (const auto &[paramName, paramType] : implicitParams) {
      TypedAttr paramValue = ParamDeclRefAttr::get(paramName, paramType);
      if (failed(inference.setInitialInferredValue(paramIdx, paramValue))) {
        if (inference.diag.hasErrorEmitted())
          return std::move(*inference.diag.takeMojoDiag());
        assert(!inference.unprovableConstraints.empty());
        return std::move(inference.unprovableConstraints);
      }
      ++paramIdx;
    }
  }

  if (failed(inference.inferForCall())) {
    if (inference.diag.hasErrorEmitted())
      return std::move(*inference.diag.takeMojoDiag());
    // Then there must be unprovable per-parameter constraints. Report
    // immediately, as we cannot fully evaluate the fitness of this candidate.
    assert(!inference.unprovableConstraints.empty());
    return std::move(inference.unprovableConstraints);
    // If there were any unprovable body constraints, inferForCall would have
    // succeeded and populated its bodyUnprovableConstraints. These are taken as
    // part of the fitness result later.
  }

  ParameterExprArrayAttr newBindings = inference.getInferredValues();
  assert(inference.unprovableConstraints.empty() &&
         "expect no unprovable constraints on a successful inference.");
  assert(newBindings && "expected new bindings when no diagnostic was emitted");

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  std::tie(signature, newBindings) = getUnboundSpecializedSignature(
      signature, newBindings, &shared.getEvaluationContext());

  // This is the result we will return if we succeed.
  OverloadFitness result(newBindings, std::move(operandsNeedingOrigins));
  result.unprovableConstraints = std::move(inference.bodyUnprovableConstraints);

  // Get the overload ranking metrics.
  result.payload.numImplicitConversions = inference.numImplicitConversions;
  result.payload.numMismatchedConventions = inference.numMismatchedConventions;
  result.payload.passesVarArgArgument = inference.passesVarArgArgument;

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // Fail if this is a constructor call that returns the wrong result type. This
  // can happen with weird things like this:
  //     struct A[X: Int]: def __init__(out self: A[4]): pass
  //     var a = A[1]()  # Infers to A[4]; error!
  //     var b = A[4]()  # Ok!
  if (callable.syntax == CallSyntax::kTypeCall) {
    // Check to see if any of the parameter bound to the result type disagree
    // with the 'Self' parameters, which are bound into newBindings.
    auto resultType = ASTType(signature.getUserResultType());
    auto numBindings = resultType.getParamBindings().size();
    for (auto [paramIdx, actual, expected] :
         llvm::enumerate(resultType.getParamBindings(),
                         newBindings.getValue().take_front(numBindings))) {
      if (!isEqualCanon(actual, expected)) {
        DeclResolver::DiagnosticDeclContextChanger x(funcIfDirect);
        assert(resultType.getDecl(shared) &&
               "result type should have an associated decl");
        assert(resultType.getDecl(shared)->getIfOperation() &&
               "result type decl should have an operation");
        auto declOp =
            cast<StructDeclOp>(resultType.getDecl(shared)->getIfOperation());
        auto paramType = declOp.getSignature().getInputParamTypes()[paramIdx];
        MojoInflightDiag resultTypeDiag = shared.emitError(callLoc);
        resultTypeDiag << "return type " << resultType << " parameter "
                       << ParamDeclRefAttr::get(
                              declOp.getParams()[paramIdx].getName(), paramType)
                       << " has value " << actual
                       << " that doesn't match expected " << expected;
        return std::move(resultTypeDiag);
      }
    }
  }

  // Fail if this is an implicit conversion but the ctor is not marked @implicit
  if (funcIfDirect && callable.syntax == CallSyntax::kImplicitConvert &&
      !cast<FnOp>(funcIfDirect->getIfOperation()).isImplicitConversion()) {
    ASTType fromType = operands[0].ir.getRValueTypeIfResolvable();
    return emitDiagFor.badImplicitConversion(fromType,
                                             signature.getUserResultType());
  }

  // Otherwise we succeeded!
  return result;
}
