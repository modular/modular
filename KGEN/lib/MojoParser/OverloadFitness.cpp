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
#include "CallEmission.h"
#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
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

  diag << "value passed to " << argListAttr.getPogs()[argIdx].getName()
       << " cannot be converted from " << operand.expr->getRange();
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
  emitTooManyPositional(diag, maxAllowedArgs, numPosOperands, "argument");
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

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type.
std::optional<MojoInflightDiag> OverloadFitness::checkOneOperand(
    ASTExprAnd<AnyValue> operand, size_t argIdx,
    ArgConvention expectedConvention, ASTType expectedType,
    bool allowImplicitConversions, const OverloadSet &callable,
    PogListAttr argListAttr) {

  auto loc = operand.expr->getLoc();

  ASTType expectedRVType =
      RefType::stripRefConvention(expectedType, expectedConvention);

  // Allow overloading on "owned" vs "by-ref" arguments.
  // If the argument convention is owned but the operand is not an RValue then
  // we'll need to copy the value (or this is entirely invalid).  If the
  // argument convention is borrowed/ref but the value is an RValue then we have
  // an RValue decay.  Model these so that APIs can overload on owned vs
  // borrowed effectively.
  bool argTypesMatchOrIsUValue = !operand.ir.getIfCValue();
  if (!argTypesMatchOrIsUValue)
    argTypesMatchOrIsUValue =
        operand.ir.getIfCValue().getRValueType().isEqualCanon(expectedRVType);

  if (argTypesMatchOrIsUValue) {
    if (operand.ir.getIfBValue() || operand.ir.getIfLValue()) {
      // Heavily penalize implicit copies.
      if (expectedConvention == ArgConvention::OwnedMem ||
          expectedConvention == ArgConvention::DeinitMem)
        payload.numMismatchedConventions += 2;
    } else {
      assert((operand.ir.getIfUValue() || operand.ir.getIfRValue()) &&
             "UValue and RValue expressions are always owned");
      // Slightly penalize RValue->ref conversions.
      if (expectedConvention != ArgConvention::OwnedMem &&
          expectedConvention != ArgConvention::DeinitMem)
        ++payload.numMismatchedConventions;
    }
  }

  ASTDecl &declScope = callable.paramBindings.declScope;
  SharedState &shared = declScope.getShared();
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    assert(argVal && "Checked by param inference already");

    // If this is a wildcard type, we can match any operand.
    if (sugarIsa<NameLookupArgWildcardType>(argVal.getRValueType()))
      return {}; // Success.

    // ByRef argument types must exactly match, no conversions are allowed.
    assert(argVal.getRValueType().isEqualCanon(expectedRVType) &&
           "Checked by param inference already");

    // Notice if a register-passable type is being passed in-memory. This allows
    // 'mut' arguments overloads to be more expensive than borrowed.
    payload.numMismatchedConventions +=
        expectedRVType.isRegisterPassable(loc, shared);
    return {}; // Success.
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    // If we are binding to something that is already a reference, check for
    // compatibility of the references and we're done.
    if (operand.ir.isMValue())
      return {}; // Checked by param inference already.

    // Otherwise, we are binding something like a PValue or SRValue to a
    // reference argument, which doesn't have a origin.  This is a problem
    // because origins can be propagated through the type system of the
    // function call to other arguments and they all need to line up.  We
    // handle this in two phases: during overload resolution we bind this to
    // an immortal origin, and then after the candidate is selected, we
    // re-emit these arguments to memory and re-infer all the parameters.
    //
    // One detail is how we do this: we bind these arguments to immutable
    // temporaries, because we specifically do NOT want 'ref' arguments with
    // parametric mutability to treat these things as mutable.
    assert(!sugarCast<RefType>(expectedType).isMutableKnown(true) &&
           "Checked by param inference already");

    // Handle this like a normal memory argument, since the value can undergo
    // implicit conversions etc.
    [[fallthrough]];
  }
  case ArgConvention::ReadMem:
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    // If a register-passable type is being passed in-memory, remember this.
    payload.numMismatchedConventions +=
        expectedRVType.isRegisterPassable(loc, shared);
    [[fallthrough]];
  case ArgConvention::ReadReg:
    break;
  }

  // Get the argument if it has a concrete type.
  CValue argVal = operand.ir.getIfCValue();

  // If the argument is unresolved, see if we can resolve it with the expected
  // type.
  if (!argVal) {
    if (auto initValue = operand.ir.getIfInitializer()) {
      // Checked by param inference already.
      return {};
    }

    auto orValue = operand.ir.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");

    // Try to refine the OverloadSetUValue into a PValue.
    argVal = orValue->getDirectSymbol(expectedRVType, declScope);
    assert(argVal && !orValue->baseValue &&
           "Checked by param inference already");
  }

  ASTType argType = argVal.getRValueType();

  // If this is a wildcard type, we can match any operand.
  if (sugarIsa<NameLookupArgWildcardType>(argType))
    return {}; // Success.

  // Otherwise, we pass as an r-value.  If the argument types match, then
  // they are good.
  if (argType.isEqualCanon(expectedRVType))
    return {}; // Success.

  // Argument name mismatches don't count as implicit conversions.
  if (IREmitter::canZeroCostConvert(argType, expectedRVType, shared))
    return {}; // Success.

  if (auto nonmaterializableTarget =
          argType.getNonmaterializableTarget(shared)) {
    if (nonmaterializableTarget.isEqualCanon(expectedRVType)) {
      // Implicit conversion for nonmaterializable types to their target
      // type is allowed even if !allowImplicitConversions and count as half
      // as much of a mismatch as a normal implicit conversion.  This enables
      // exact matches to be more specific, and literals to be more compatible
      // than an actual conversion.
      ++payload.numImplicitConversions;
      return {}; // Success.
    }
  }

  // If implicit conversions are possible and one will work, then we succeed
  // with that conversion.
  if (allowImplicitConversions &&
      IREmitter::canImplicitlyConvertToType({argVal, operand.expr},
                                            expectedRVType, declScope)) {
    // If we had one, this bumps our # implicit conversions.
    payload.numImplicitConversions += 2;
    return {}; // Success.
  }

  llvm::errs() << "Mismatch between overload fitness and param inference\n";
  llvm::errs() << "Expected type: " << expectedRVType << "\n";
  llvm::errs() << "Actual type  : " << argType << "\n\n";
  llvm::errs() << "Expected type: " << expectedRVType.mlirType << "\n";
  llvm::errs() << "Actual type  : " << argType.mlirType << "\n\n";
  llvm_unreachable("Checked by param inference already");
}

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

  // If ambiguous, we compare the boolean metrics.
  int8_t mask = payload.getBoolMask();
  int8_t otherMask = other.payload.getBoolMask();
  if (mask != otherMask)
    return mask < otherMask;

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

int8_t OverloadFitness::Payload::getBoolMask() const {
  // We consider exact matches of concrete types to be more specific than
  // those needing nonmaterializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work), and
  // all of these more specific than matches with variadic parameters.
  return 2 * passesVarArgArgument + 1 * hasVariadicParams;
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
                     /*discardError=*/false);
  // Don't yield constraint failure for a single overload failure: we want a
  // better error message diagnosed for the entire set.
  ParameterExprArrayAttr bindings =
      inference.inferForStruct(/*emitConstraintFailure=*/false);

  if (!bindings) {
    // If no diagnostics were emitted, this must be an inconclusive fitness.
    if (!inference.diag.hasErrorEmitted()) {
      assert(!inference.unprovableConstraints.empty());
      return std::move(inference.unprovableConstraints);
    }
    // Otherwise, it's a real failure, so we return an invalid fitness.
    return std::move(*inference.diag.takeMojoDiag());
  }

  return OverloadFitness(bindings,
                         /*noArgsNeedingOrigins*/ OperandsNeedingOriginsList());
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

OverloadFitness::VisibleParamDeclBindings
OverloadFitness::collectVisibleParamDeclBindings(ASTDecl *callsiteScope) {
  VisibleParamDeclBindings bindings;
  if (!callsiteScope)
    return bindings;
  auto &index = callsiteScope->getShared()
                    .getClosureEmitter()
                    .getHoistedBindingsByScope();
  for (ASTDecl *scope = callsiteScope; scope; scope = scope->getParentDecl()) {
    auto it = index.find(scope);
    if (it != index.end())
      for (auto &[attr, paramDecl] : it->second)
        bindings.try_emplace(paramDecl.getName(), attr);
  }
  return bindings;
}

static void injectVisibleParamDeclBindings(
    const OverloadFitness::VisibleParamDeclBindings &bindings,
    ParameterEvaluator &evaluator) {
  for (const auto &[name, value] : bindings)
    evaluator.setDeclBinding(name, value);
}

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `callOperands`.
///
/// The 'funcIfDirect' member is set if this is a direct call, or null if
/// indirect.  It can be used to tune diagnostics.
OverloadFitness OverloadFitness::evaluate(
    FnTypeGeneratorType signature, ASTDecl *funcIfDirect,
    const OverloadSet &callable, const CallOperands &operands,
    bool allowImplicitConversions,
    const VisibleParamDeclBindings *visibleParamDeclBindings) {
  // We set up diagnostics.
  size_t numPosOperands = operands.getNumPositional();
  size_t numOperands = operands.size();

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
  switch (posDiagRes) {
  case CallOperands::PosDiagResult::kMissingPos:
    return emitDiagFor.missingArgs(posDiagNames, "positional");
  case CallOperands::PosDiagResult::kTooManyPos: {
    size_t numPosMaximum = countNumPositional(argListAttr);
    return emitDiagFor.tooManyPosArgs(numPosMaximum, numPosOperands);
  }
  case CallOperands::PosDiagResult::kByPosAndKw:
    return emitDiagFor.byPosAndKw(posDiagNames);
  default:
    break;
  }

  // Check that the signature can be rebound with this set of bindings.

  // Determine if this is an initializer that returns Self, which can be used
  // for inferring parameters on Self.
  bool returnsSelf = false;
  bool hasCTADParams = false;
  if (funcIfDirect) {
    auto fn = cast<FnOp>(funcIfDirect->getIfOperation());
    returnsSelf = fn.getSpecialFunctionInfo().hasSelfResult();
    hasCTADParams = !fn.getIsStatic() && isa<StructDeclOp>(fn->getParentOp());
  }

  ParamInf inference(callable.paramBindings, signature.getInputParamTypes(),
                     signature.getParamListAttrs(), allowImplicitConversions,
                     funcIfDirect, /*discardError=*/false);
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
    size_t paramIdx =
        signature.getInputParamTypes().size() - implicitParams.size();
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
  // TODO: inferForCall will eventually be separated. We will eventually blend
  // parameter inference into overload resolution have something like:
  //
  //  inference.inferFromParamBinding(...)
  //
  //  for (arg in callexpr) {
  //    case call_conv:
  //       inference.inferOneOperand(...)
  //    ...
  //  }
  OperandsNeedingOriginsList operandsNeedingOrigins;
  if (failed(inference.inferForCall(signature, operands, variadicKwOperands,
                                    returnsSelf, hasCTADParams,
                                    operandsNeedingOrigins))) {
    if (inference.diag.hasErrorEmitted())
      return std::move(*inference.diag.takeMojoDiag());
    // Then there must be unprovable constraints.
    assert(!inference.unprovableConstraints.empty());
    return std::move(inference.unprovableConstraints);
  }

  ParameterExprArrayAttr newBindings = inference.getInferredValues();
  assert(inference.unprovableConstraints.empty() &&
         "expect no unprovable constraints on a successful inference.");
  assert(newBindings && "expected new bindings when no diagnostic was emitted");

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  FnTypeGeneratorType originalSignature = signature;
  std::tie(signature, newBindings) = getUnboundSpecializedSignature(
      signature, newBindings, &shared.getEvaluationContext());

  // This is the result we will return if we succeed.
  OverloadFitness result(newBindings, std::move(operandsNeedingOrigins));

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // As we walk through the values provided as part of the argument list, we
  // match them up against arguments expected by the signature of the callee,
  // take note if variadic arguments are passed, and accumulate implicit
  // conversions required for a match.  This value indicates the next operand
  // to consider as a positional argument.
  size_t posOperandIdx = 0;

  argListAttr = signature.getArgListAttrs();
  // FiXME: This should be substituting implicit parameters as the arguments are
  // bound to be able to handle things like
  // https://github.com/modular/modular/issues/3855 correctly.
  for (auto [expectedArgIdx, expectedTypeX, expectedConvention] :
       llvm::enumerate(signature.getArguments(),
                       signature.getArgConventions())) {
    Type expectedType = expectedTypeX;
    assert(expectedType &&
           "specialized signature produced a null argument type");
    // Ignore the return slot if present.
    if (expectedConvention == ArgConvention::ByRefError)
      continue;
    if (expectedConvention == ArgConvention::ByRefResult) {
      result.payload.numMismatchedConventions +=
          ASTType(expectedType)
              .getReferenceElementType()
              .isRegisterPassable(callLoc, shared);
      continue;
    }

    if (signature.isKwVarArg(expectedArgIdx)) {
      expectedType = ASTType(expectedType).getKwargsDictRefValueType();
      auto refExpType = RefType::getAnyOrigin(expectedType, /*isMut=*/true);
      for (auto operand : variadicKwOperands) {
        // TODO: Passing OwnedMem is a hack that is needed because the value
        // type is not a reference type (and doesn't have a origin), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Reference` does not
        // (and in fact cannot) conform to `Copyable & Movable`.
        if (auto diag = result.checkOneOperand(
                operand, expectedArgIdx, ArgConvention::OwnedMem, refExpType,
                allowImplicitConversions, callable, argListAttr))
          return std::move(diag).value();
      }
      // This comes after all the positionals.
      posOperandIdx = numOperands;
      continue;
    }

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.
    // TODO: Don't let memory types bind to __TypeOfAllTypes.
    if (!ASTType(expectedType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType,
                                           argListAttr);

    // Figure out the next positional argument to process.
    while (posOperandIdx < numOperands && operands[posOperandIdx].keyword)
      ++posOperandIdx;

    // Handle case when there are no more provided positional operands.
    StringAttr argName = argListAttr.getName(expectedArgIdx);
    if (posOperandIdx == numOperands) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isPosVarArg(expectedArgIdx) ||
          signature.isPack(expectedArgIdx)) {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++result.payload.numImplicitConversions;
        continue;
      }

      // Check if the argument was passed as a keyword operand.
      if (const OperandValue *kwOperandOr = operands.findKwArg(argName)) {
        // If we found a keyword argument, we check it normally.
        if (auto diag = result.checkOneOperand(
                *kwOperandOr, expectedArgIdx, expectedConvention, expectedType,
                allowImplicitConversions, callable, argListAttr))
          return std::move(diag).value();
        continue;
      }

      // We ensured earlier that that can be no missing positional arguments.
      assert(argListAttr.getDefault(expectedArgIdx) &&
             "missing positional argument not caught by diagnostics");

      continue;
    }

    /// Check and process a single positional operand and advance the operand
    /// index.
    auto processPositionalOperand =
        [&](ASTType expectedType,
            ArgConvention conv) -> std::optional<MojoInflightDiag> {
      OperandValue operand = operands[posOperandIdx];
      if (operand.isUnpackedPositional()) {
        return shared.emitError(operand.expr->getLoc())
               << "concatenating unpacked positional arguments is not "
                  "supported";
      }
      if (auto diag = result.checkOneOperand(
              operand, expectedArgIdx, conv, expectedType,
              allowImplicitConversions, callable, argListAttr))
        return std::move(diag).value();
      ++posOperandIdx;
      return {};
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      if (operands[posOperandIdx].isUnpackedPositional()) {
        // Fully checked by ParamInf.
        ++posOperandIdx;
        result.payload.passesVarArgArgument = true;
        continue;
      }

      ASTType expectedRVType =
          RefType::stripRefConvention(expectedType, expectedConvention);
      auto varArgsEltType =
          expectedRVType.getVariadicListInfo().getElementRefType();
      auto actualArgConvention =
          signature.getVariadicConvention(expectedArgIdx);
      while (posOperandIdx != numPosOperands) {
        if (auto result =
                processPositionalOperand(varArgsEltType, actualArgConvention))
          return std::move(*result);
        result.payload.passesVarArgArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consume exactly that many positional operands.
    if (signature.isPack(expectedArgIdx)) {
      ASTType variadicPackType =
          RefType::stripRefConvention(expectedType, expectedConvention);
      if (operands[posOperandIdx].isUnpackedPositional()) {
        // Fully checked by ParamInf.
        ++posOperandIdx;
        result.payload.passesVarArgArgument = true;
        continue;
      }

      auto actualArgConvention =
          signature.getVariadicConvention(expectedArgIdx);
      RefPackType packType = variadicPackType.getVariadicPackInfo(shared);
      for (TypedAttr element : packType.getVariadicIfResolved().getValues()) {
        auto refType = packType.getElementRefTypeFor(ASTType(element).mlirType);
        if (auto result =
                processPositionalOperand(refType, actualArgConvention))
          return std::move(*result);
        result.payload.passesVarArgArgument = true;
      }
      continue;
    }

    if (operands[posOperandIdx].isUnpackedPositional()) {
      return shared.emitError(operands[posOperandIdx].expr->getLoc())
             << "unpacked positional arguments are only supported for callees "
                "that expect a variadic pack argument at this position";
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. We ensured earlier that it is not also passed as a keyword
    // operand, so we process it as usual.
    assert(
        (argListAttr.getPassingKind(expectedArgIdx) == PassingKind::PosOnly ||
         (!argName.empty() && !operands.findKwArg(argName))) &&
        "redundant argument not caught by diagnostics");
    if (auto result =
            processPositionalOperand(expectedType, expectedConvention))
      return std::move(*result);
  }

  assert(posOperandIdx == numOperands &&
         "should handle argument mismatch above");

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

  // Check that all def constraints are satisfied.
  SmallVector<ConstraintAttr> fnUnprovableConstraints;
  bool hasHoistedBindings =
      visibleParamDeclBindings && !visibleParamDeclBindings->empty();
  std::optional<ParameterEvaluator> fnConstraintEvaluator;
  if (hasHoistedBindings) {
    const ParameterEvaluator &inferenceEvaluator = inference.getEvaluator();
    fnConstraintEvaluator.emplace(inferenceEvaluator.getDeclBindings(),
                                  inferenceEvaluator.getIndexBindings(),
                                  inferenceEvaluator.getInputDepth());
    fnConstraintEvaluator->setEvaluationContext(
        inferenceEvaluator.getEvaluationContext());
    injectVisibleParamDeclBindings(*visibleParamDeclBindings,
                                   *fnConstraintEvaluator);
  }
  checkConstraints(callable.paramBindings.declScope,
                   originalSignature.getMetadata(),
                   signature.getFnMetadata().getConstraints(),
                   originalSignature.getFnMetadata().getConstraints(),
                   inference.diag.getDiag(), &fnUnprovableConstraints,
                   hasHoistedBindings ? &*fnConstraintEvaluator : nullptr);
  if (inference.diag.hasErrorEmitted())
    return std::move(*inference.diag.takeMojoDiag());
  if (!fnUnprovableConstraints.empty())
    result.unprovableConstraints = std::move(fnUnprovableConstraints);

  // Otherwise we succeeded!
  return result;
}
