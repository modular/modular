//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CallEmission.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/OverloadFitness.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "MojoUtils.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

#include <limits>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

StringRef LIT::stringifyCallSyntax(CallSyntax val) {
  switch (val) {
  case CallSyntax::kDirectCall:
    return "direct_call";
  case CallSyntax::kIndirectCall:
    return "indirect_call";
  case CallSyntax::kMethodCall:
    return "method_call";
  case CallSyntax::kTypeCall:
    return "type_call";
  case CallSyntax::kOperator:
    return "operator";
  case CallSyntax::kReversedOperator:
    return "reversed_operator";
  case CallSyntax::kSubscript:
    return "subscript";
  case CallSyntax::kAttribute:
    return "attribute";
  case CallSyntax::kImplicitConvert:
    return "implicit_convert";
  case CallSyntax::kDestructor:
    return "destructor";
  case CallSyntax::kTupleGetItem:
    return "tuple_get_item";
  case CallSyntax::kMethodCallSynthetic:
    return "method_call_synthetic";
  }
  llvm_unreachable("unknown CallSyntax");
  return "";
}

raw_ostream &LIT::operator<<(raw_ostream &os, CallSyntax val) {
  return os << stringifyCallSyntax(val);
}

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParamBindings &&paramBindings, const ExprNode *expr,
                         CallSyntax syntax, bool erroneous)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()),
      paramBindings(std::move(paramBindings)), expr(expr), syntax(syntax),
      erroneous(erroneous) {}

/// Resolve the callee into a single PValue callee.
static PValue getCallee(SharedState &shared, ASTDecl *fnDecl,
                        StringRef baseName, const ParamBindings &paramBindings,
                        const ExprNode *expr) {
  auto funcOp = cast<LIT::FuncOp>(*fnDecl);
  // Check if the function overload set resolved to a deprecated overload.
  if (StringAttr warning = funcOp.getDeprecationWarningAttr()) {
    auto diag = shared.emitWarning(expr->getLoc(), warning.getValue())
                << expr->getRange();
    diag.attachNote(fnDecl->getLoc())
        << "'" << *funcOp.getSourceName() << "' declared here";
  }
  return paramBindings.getBoundConstAttrFor(funcOp, baseName, expr);
}

/// Return if the given fitness is valid, and drop the diagnostics otherwise.
static bool isValid(OverloadFitness &eval) {
  if (eval.isValid())
    return true;
  eval.takeDiag().abandon();
  return false;
}

/// Assuming we have at least one valid candidate, filter the candidate list to
/// those with the best fitness. If there is more than one candidate with
/// maximal fitness, we filter for non-static methods.
///
/// To aid downstream diganostics, the function returns the fitness of the best
/// candidate. All diagnostics from erroneous candidates are dropped.
static OverloadFitness *
selectBestCandidates(ArrayRef<ASTDecl *> fnDecls,
                     MutableArrayRef<OverloadFitness> evaluations,
                     SmallVectorImpl<ASTDecl *> &newFnDecls) {
  assert(newFnDecls.empty());
  bool areTheBestCandidatesStatic = true;

  // Find the first valid candidate.
  evaluations = evaluations.drop_until(isValid);
  OverloadFitness *bestFitness = &evaluations.front();

  for (auto [candidate, eval] :
       llvm::zip(fnDecls.take_back(evaluations.size()), evaluations)) {
    // Ignore all subsequent failures and candidates that are definitely worse.
    if (!isValid(eval) || bestFitness->isBetter(eval))
      continue;

    // If we found a strictly better candidate, clear the list.
    if (eval.isBetter(*bestFitness)) {
      newFnDecls.clear();
      areTheBestCandidatesStatic = true;
    }

    // If the current best candidates are not static, we ignore new static
    // candidates.
    bool isStatic = cast<LIT::FuncOp>(*candidate).getIsStatic();
    if (!areTheBestCandidatesStatic && isStatic)
      continue;

    // If the current best candidates are static, and we just found a non-static
    // one, we clear the list.
    if (areTheBestCandidatesStatic && !isStatic) {
      newFnDecls.clear();
      areTheBestCandidatesStatic = false;
    }

    newFnDecls.push_back(candidate);
    bestFitness = &eval;
  }

  return bestFitness;
}

enum class CallKind { kMethod, kFunction, kIndirect };

static CallKind getCallKind(CallSyntax syntax) {
  switch (syntax) {
  case CallSyntax::kDirectCall:      //< f()
  case CallSyntax::kTypeCall:        //< T()
  case CallSyntax::kImplicitConvert: //< Conversion in an argument context
    return CallKind::kFunction;
  case CallSyntax::kIndirectCall: //< expr()
    return CallKind::kIndirect;
  case CallSyntax::kMethodCall:       //< x.f()
  case CallSyntax::kOperator:         //< -x and x + y
  case CallSyntax::kReversedOperator: //< y + x
  case CallSyntax::kSubscript:        // v[1, 2]
  case CallSyntax::kAttribute:        // v.x
  case CallSyntax::kDestructor:       //< Destructor due to a value definition.
  case CallSyntax::kTupleGetItem:     //< Call to getitem in a tuple assignment.
  case CallSyntax::kMethodCallSynthetic:
    return CallKind::kMethod;
  }
  llvm_unreachable("invalid call syntax");
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings on the overload
/// set. If so, return the single entry that works.  If not, generate a
/// diagnostic and return null.
PValue OverloadSet::filterOverloadSetForParamBindings(
    bool allowImplicitConversions) const {
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto func = cast<LIT::FuncOp>(*candidate);
    LITSignatureType sig = func.getFullSignature();
    evaluations.push_back(OverloadFitness::evaluate(
        sig.getParamTypes(), sig.getParamListAttrs(), *this,
        /*allowImplicitConversions=*/true));
    anyValid |= evaluations.back().isValid();
  }

  // If none are valid, emit an error.
  if (!anyValid) {
    if (isErroneous())
      return {};
    auto diag = getShared().emitError(
                    expr->getLoc(),
                    "cannot form a reference to overloaded declaration of '")
                << baseName << "'" << expr->getRange();
    for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
      diag.attachNote(candidate->getLoc())
          << "candidate not viable: " << eval.takeDiag();
      auto func = cast<LIT::FuncOp>(candidate);
      if (func.getIsSynthetic()) {
        diag.attachNote(candidate->getLoc())
            << "generated function with type "
            << ASTType(func.getFullSignature());
      }
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  SmallVector<ASTDecl *> newFnDecls;
  const OverloadFitness *bestFitness =
      selectBestCandidates(fnDecls, evaluations, newFnDecls);
  if (newFnDecls.size() == 1) {
    // On success, wrap things up into one callee.
    ParamBindings newBindings((const TypeCheckScopeInfo &)paramBindings);
    for (TypedAttr bind : bestFitness->getParamBindings())
      newBindings.addPrechecked(expr, bind);
    return getCallee(getShared(), newFnDecls[0], baseName, newBindings, expr);
  }
  if (isErroneous())
    return {};

  size_t minConversions = bestFitness->getNumImplicitConversions() / 2;
  auto diag = getShared().emitError(expr->getLoc(), "ambiguous reference to '")
              << baseName << "', each candidate requires " << minConversions
              << " implicit conversion" << plural(minConversions)
              << ", disambiguate with an explicit cast" << expr->getRange();
  for (ASTDecl *candidate : newFnDecls) {
    auto func = cast<LIT::FuncOp>(candidate);
    InflightDiag &note = diag.attachNote(candidate->getLoc());
    if (func.getIsSynthetic()) {
      note << "candidate generated with type "
           << ASTType(func.getFullSignature());
    } else {
      note << "candidate declared here";
    }
  }
  return {};
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings and provided
/// arguments.  If so, return the single entry that works.
///
/// NOTE: This can mutate the operand list, e.g. when calling a static method
/// that doesn't need a self value, and by pre-emitting PValues when not in an
/// parameter context. The actual emission needs to use the updated argument
/// list.
///
/// If not, generate a diagnostic (when `emitDiagnosticOnFailure` is true) and
/// return null.
PValue OverloadSet::filterOverloadSet(CallOperands &operands,
                                      bool allowImplicitConversions,
                                      bool emitDiagnosticOnFailure,
                                      ExprEmitter &emitter) const {
  CallOperands scratchOperands;
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto func = cast<LIT::FuncOp>(*candidate);

    // If we are dealing with a static method, we check if the operands include
    // a self operand and remove it, otherwise the signature might not match.
    const CallOperands *operandsToUse = &operands;
    if (operands.hasSelfOperand && func.getIsStatic()) {
      scratchOperands = CallOperands(operands);
      scratchOperands.values.erase(scratchOperands.values.begin());
      scratchOperands.hasSelfOperand = false;
      operandsToUse = &scratchOperands;
    }

    evaluations.push_back(
        OverloadFitness::evaluate(func.getFullSignature(), candidate, *this,
                                  *operandsToUse, allowImplicitConversions));
    anyValid |= evaluations.back().isValid();
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (!emitDiagnosticOnFailure || isErroneous())
      return {};

    auto diag = getShared().emitError(expr->getLoc()) << expr->getRange();

    // Diagnose the case when there are no candidates found by lookup.
    if (fnDecls.empty()) {
      diag << "invalid call to '" << baseName << "': no candidates found";
      return {};
    }

    // If we have one operand, get it to help tailor type conversion errors.
    ASTType selfOperandType, singleOperandType;
    if (operands.size() == 2 && !operands[0].keyword && !operands[1].keyword) {
      if (auto cValue = operands[0].ir.getIfCValue())
        if (auto selfRef = dyn_cast<RefType>(cValue.getRValueType()))
          selfOperandType = selfRef.getElementType();
      if (auto cValue = operands[1].ir.getIfCValue())
        singleOperandType = cValue.getRValueType();
    }

    // Reject Int(x) where x is already an Int with an error + fixit.
    if (syntax == CallSyntax::kTypeCall && singleOperandType &&
        selfOperandType && singleOperandType.isEqualCanon(selfOperandType) &&
        isa<CallNode>(expr)) {
      const CallNode &callNode = *cast<CallNode>(expr);
      // This removes the constructor call, but does not remove the parens
      // because we don't want to introduce precedence problems.
      diag << "cannot construct " << selfOperandType
           << " with itself, you can remove the constructor call"
           << operands[0].expr->getRange()
           << FixIt::remove(callNode.callee->getRange());
      return {};
    }

    // Diagnose implicit conversions with a custom message, unless this is
    // forming a Reference.
    if (syntax == CallSyntax::kImplicitConvert && selfOperandType &&
        singleOperandType) {
      // This is true if passing Int type to Int instead of Int() to Int.
      bool isConvertingTypeValue =
          selfOperandType.getMetaType() == singleOperandType;
      diag << "cannot implicitly convert ";
      if (isConvertingTypeValue)
        diag << selfOperandType << " type as a";
      else
        diag << singleOperandType;
      diag << " value to ";
      diag << (isConvertingTypeValue ? "an instance of " : "")
           << selfOperandType;

      if (isConvertingTypeValue)
        diag << "; did you mean to instantiate " << selfOperandType << "?";
      diag << expr->getRange();
      return {};
    }

    if (fnDecls.size() == 1)
      diag << "invalid ";
    else {
      diag << "no matching ";
      diag << (getCallKind(syntax) == CallKind::kMethod ? "method"
                                                        : "function");
      diag << " in ";
    }

    switch (syntax) {
    default:
      diag << "call to '" << baseName << "'";
      break;
    case CallSyntax::kTypeCall:
      diag << "initialization";
      break;
    case CallSyntax::kImplicitConvert:
      diag << "implicit conversion";
      break;
    }

    // If there is a single callee, emit a specific error about the call.
    if (fnDecls.size() == 1) {
      auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
      diag << ": " << evaluations[0].takeDiag();
      diag.attachNote(fnDecl.getLoc()) << "function declared here";
      return {};
    }

    // Add a note for what is wrong with each candidate.
    for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
      diag.attachNote(candidate->getLoc())
          << "candidate not viable: " << eval.takeDiag();
      auto func = cast<LIT::FuncOp>(candidate);
      if (func.getIsSynthetic()) {
        diag.attachNote(candidate->getLoc())
            << "generated function with type "
            << ASTType(func.getFullSignature());
      }
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  SmallVector<ASTDecl *> newFnDecls;
  OverloadFitness *bestFitness =
      selectBestCandidates(fnDecls, evaluations, newFnDecls);

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!newFnDecls.empty())
    getShared().notifyListenerOnRef(newFnDecls, baseName, expr, syntax);

  // If we found exactly one viable candidate then we succeed.
  if (newFnDecls.size() == 1) {
    ASTDecl *selectedDecl = newFnDecls[0];
    auto selectedFunc = cast<LIT::FuncOp>(selectedDecl);

    // If the target is static and there is a self operand, remove it from the
    // operand list so it doesn't get passed.
    if (operands.hasSelfOperand && selectedFunc.getIsStatic()) {
      operands.values.erase(operands.values.begin());
      operands.hasSelfOperand = false;
    }

    // It is possible this candidate needs some arguments emitted as MValues
    // (from PValue or SValues) to be passed as 'ref' arguments.  If this
    // happens, emit them now and then re-infer the correct lifetimes.
    const auto &argsNeedingLifetimes = bestFitness->getArgsNeedingLifetimes();
    std::optional<OverloadFitness> replaced;
    if (argsNeedingLifetimes.any()) {
      // Emit each of the arguments that needs a lifetime to an MValue.
      for (size_t i = 0, e = argsNeedingLifetimes.size(); i != e; ++i) {
        if (!argsNeedingLifetimes[i])
          continue;
        // If the operand is a positional argument it will be in the normal
        // operand list, otherwise it will be in the kwargs list.
        assert(i < operands.size() && "argument index incorrect");

        // We emit this as an MBValue instead of an MRValue specifically so we
        // do not infer mutability from the temporary.  We don't want ref's with
        // parametric lifetime to bind to these values.
        auto newVal =
            emitter.emitMBValue({operands[i]}, ExprContext::EC_CallRefArgValue);
        if (!newVal)
          return {}; // Could not emit the PValue/SValue to an MRValue.
        operands.values[i].ir = newVal;
      }

      // Now that we have the operands set, we re-evaluate the bindings, which
      // will reinfer parameters, getting the correct lifetimes from the MValues
      // that are required by this overload candidate.
      replaced.emplace(OverloadFitness::evaluate(
          selectedFunc.getFullSignature(), selectedDecl, *this, operands,
          allowImplicitConversions));
      bestFitness = &replaced.value();

      assert(bestFitness->isValid() &&
             "Re-emitting function to infer lifetimes didn't work");
      assert(bestFitness->getArgsNeedingLifetimes().none() &&
             "Re-emitting function infer lifetimes shouldn't need more MValues "
             "emitted");
    }

    // Finally, wrap things up into one callee.
    ParamBindings newBindings((const TypeCheckScopeInfo &)paramBindings);
    for (TypedAttr bind : bestFitness->getParamBindings())
      newBindings.addPrechecked(expr, bind);

    return getCallee(getShared(), selectedDecl, baseName, newBindings, expr);
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure && !isErroneous()) {
    size_t minConversions = bestFitness->getNumImplicitConversions() / 2;
    auto diag = getShared().emitError(expr->getLoc(), "ambiguous call to '")
                << baseName << "', each candidate requires " << minConversions
                << " implicit conversion" << plural(minConversions)
                << ", disambiguate with an explicit cast" << expr->getRange();
    for (ASTDecl *candidate : newFnDecls) {
      auto func = cast<LIT::FuncOp>(candidate);
      InflightDiag &note = diag.attachNote(candidate->getLoc());
      if (func.getIsSynthetic()) {
        note << "candidate generated with type "
             << ASTType(func.getFullSignature());
      } else {
        note << "candidate declared here";
      }
    }
  }
  return {};
}

PValue
OverloadSet::filterOverloadSetForValueType(ASTType functionType,
                                           bool emitDiagnosticOnFailure) const {
  if (!emitDiagnosticOnFailure)
    return filterOverloadSetForValueType(functionType, /*emitError=*/nullptr);

  std::optional<InflightDiag> diag;
  return filterOverloadSetForValueType(
      functionType, [&](SMLoc loc) -> InflightDiag & {
        return diag.emplace(getShared().emitError(loc));
      });
}

PValue OverloadSet::filterOverloadSetForValueType(
    ASTType functionType, function_ref<InflightDiag &(SMLoc)> emitError) const {
  // If the target type is something weird then don't filter.  Let the error be
  // reported another way.
  if (!isa<SignatureType>(functionType.mlirType)) {
    if (emitError) {
      auto &diag = emitError(expr->getLoc())
                   << "cannot convert function to non-function type "
                   << functionType;
      for (ASTDecl *candidate : fnDecls)
        diag.attachNote(candidate->getLoc())
            << "candidate declared here with type "
            << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());
    }
    return {};
  }

  // TODO(#22771): This is using an exact match which is perhaps too specific of
  // a check. We could do some amount of parameter inference to support cases
  // like:
  //
  //    fn foo[Type: mlirtype]() -> Type
  //    var f : ()-> Int = foo
  //
  // We could also support generating a lambda for fancy implicit conversions
  // and subtyping some day.
  auto getBindingsForSignature =
      [&](LITSignatureType candidateType) -> ParameterExprArrayAttr {
    // Fully apply any bound parameters to the candidate's type since they will
    // be applied when a reference is made.
    // TODO(#22771): Parameter inference.
    auto [newBindings, _] = paramBindings.verifyBindings(candidateType);
    return newBindings;
  };

  auto isValidCandidate = [&](LITSignatureType candidateType) -> bool {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.  We only do this if there are some
    // bindings present, because (unlike normal function calls) the result type
    // may have unbound parameters that we are trying to match, e.g. when in a
    // parameter expression context.
    if (!paramBindings.empty()) {
      auto newBindings = getBindingsForSignature(candidateType);
      if (!newBindings)
        return false; // If there is an error, return the problem.

      // If anything was bound, apply it to the signature so the expected
      // argument types are updated.
      if (!newBindings.empty())
        candidateType = candidateType.getSpecializedSignature(
            newBindings, getShared().translateLocation(expr->getLoc()));
    }

    return functionType.isEqualCanon(candidateType) ||
           canConvertWithRebind(candidateType, functionType, getShared());
  };

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<ASTDecl *> validCandidates;
  for (ASTDecl *candidate : fnDecls) {
    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*candidate).getFullSignature();
    if (isValidCandidate(candidateType))
      validCandidates.push_back(candidate);
  }

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!validCandidates.empty())
    getShared().notifyListenerOnRef(validCandidates, baseName, expr, syntax);

  // If we have exactly one viable candidate, then we succeed.
  if (validCandidates.size() == 1) {
    if (paramBindings.empty()) {
      return getCallee(getShared(), validCandidates[0], baseName, paramBindings,
                       expr);
    }

    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*fnDecls.front()).getFullSignature();

    ParamBindings newBindings((const TypeCheckScopeInfo &)paramBindings);
    for (TypedAttr bind : getBindingsForSignature(candidateType))
      newBindings.addPrechecked(expr, bind);
    return getCallee(getShared(), validCandidates[0], baseName, newBindings,
                     expr);
  }

  // If we aren't to emit a diagnostic, just return the failure.
  if (!emitError)
    return {};

  auto &diag = emitError(expr->getLoc());
  if (validCandidates.empty()) {
    diag << "no '" << baseName << "' candidates have type " << functionType
         << expr->getRange();
  } else {
    diag << "ambiguous use of '" << baseName << "' as type " << functionType
         << expr->getRange();
  }

  for (ASTDecl *candidate : fnDecls) {
    diag.attachNote(candidate->getLoc())
        << "candidate declared here with type "
        << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());
  }
  return {};
}

/// Perform substitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parameters specified.  They can be bound later.
TypedAttr OverloadSet::getBoundConstantAttr() const {
  if (fnDecls.size() == 1)
    return getCallee(getShared(), fnDecls[0], baseName, paramBindings, expr);

  // If we have multiple candidates, emit an ambiguity error.
  assert(!fnDecls.empty() && "DirectCallable malformed");
  auto diag = getShared().emitError(
                  expr->getLoc(),
                  "cannot form a reference to overloaded declaration of '")
              << baseName << "'" << expr->getRange();
  for (ASTDecl *candidate : fnDecls) {
    auto func = cast<LIT::FuncOp>(candidate);
    InflightDiag &note = diag.attachNote(candidate->getLoc());
    if (func.getIsSynthetic()) {
      note << "candidate generated with type "
           << ASTType(func.getFullSignature());
    } else {
      note << "candidate declared here";
    }
  }

  return {};
}

/// Get a OverloadSet for a lookup of a named method on the specified type.
/// If successful, this provides a non-null OverloadSet.
///
/// On failure, this returns a null OverloadSet and invokes errorHandler if
/// the problem hasn't already been diagnosed. This does not emit an error on
/// failure.
OverloadSet OverloadSet::lookup(const TypeCheckScopeInfo &scopeInfo,
                                ASTType type, StringRef methodName,
                                const ExprNode *expr, CallSyntax syntax,
                                function_ref<void()> errorHandler) {
  SharedState &shared = scopeInfo.shared;

  OverloadSet result(scopeInfo, expr, syntax, /*isErroneous=*/false);
  result.baseName = methodName;

  // If this is a previously-reported error, ignore and don't report an
  // additional error.
  if (type.isTypeCheckErrorType()) {
    result.erroneous = true;
    return result;
  }

  SMLoc callLoc = expr->getLoc();

  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type,
                                                  /*searchParentScopes=*/false);
  // If an error was already reported, propagate it.
  if (lookupResult.isErroneous()) {
    result.erroneous = true;
    return result;
  }

  // If we have candidates directly on the receiver, add them.
  if (lookupResult.isSuccess()) {
    ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
    assert(!resultDecls.empty() && "We know this succeeded");

    // If we find a vardecl or any other thing, then fail to find anything
    // because it cannot be called.
    if (!isa<LIT::FuncOp>(*resultDecls[0]))
      // FIXME: This seems wrong. why aren't we emitting an error??
      return result;

    assert(result.fnDecls.empty() && "Already have entries");
    result.fnDecls.assign(resultDecls.begin(), resultDecls.end());
  }

  // If the struct has a nonmaterializable target (e.g. "IntLiteral" will have
  // "Int" as a nonmaterializable target), then it is implicitly convertible to
  // that type.  Check to see if that type has the method: if so we can add them
  // into the overload set.
  if (ASTType nmTarget = type.getNonmaterializableTarget(shared)) {
    lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, nmTarget,
                                               /*searchParentScopes=*/false);
    if (lookupResult.isSuccess()) {
      ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
      assert(!resultDecls.empty() && "We know this succeeded");

      // If we find a vardecl or any other thing, then fail to find anything
      // because it cannot be called.
      if (!isa<LIT::FuncOp>(*resultDecls[0]))
        // FIXME: This seems wrong. why aren't we emitting an error??
        return result;
      result.fnDecls.append(resultDecls.begin(), resultDecls.end());
    }
  }

  // If we get this far and there are no candidates in the set, then we can't
  // find anything.  Emit the error.
  if (result.fnDecls.empty() && errorHandler)
    errorHandler();

  return result;
}

/// Lookup of a named named method on the specified type, filtered to match a
/// concrete operand set. If successful, this provides a non-null PValue for a
/// single callee.
///
/// NOTE: This can mutate the operand list, e.g. when calling a static method
/// that doesn't need a self value, and by emitting PValues when not in an
/// parameter context. The actual emission needs to use the updated argument
/// list.
PValue OverloadSet::lookupAndResolve(
    ASTType type, StringRef methodName, CallOperands &callOperands,
    const ExprNode *callExpr, CallSyntax syntax,
    function_ref<void()> lookupFailureErrorHandler,
    bool shouldPrintOverloadErrors, ExprEmitter &emitter) {
  auto ovSet = OverloadSet::lookup(emitter.getScopeInfo(), type, methodName,
                                   callExpr, syntax, lookupFailureErrorHandler);

  // If the core lookup failed, don't filter.
  if (ovSet.isNull())
    return {};

  // Filter the overload set with the actual operands list.  If this
  // fails, report an error (if we have an error handler) and reset to a
  // null state so the client can check this.
  return ovSet.filterOverloadSet(
      callOperands,
      /*allowImplicitConversions=*/true,
      /*emitDiagnosticOnFailure=*/shouldPrintOverloadErrors, emitter);
}

/// Try to resolve the overload set to a single function candidate, using the
/// expected type if provided or using current bindings if an emitter is
/// provided.  This emits errors if 'emitter' is non-null, but does not if it
/// is null.
PValue OverloadSet::getDirectSymbol(ASTType expectedType) const {
  // Handle the case of a single candidate.
  if (fnDecls.size() == 1) {
    // This is an unbound function. Just return a reference.
    if (paramBindings.empty())
      return cast<LIT::FuncOp>(*fnDecls.front()).getBoundReference();

    // Bind the parameters.
    return getBoundConstantAttr();
  }

  // With an emitter and an expected type, the overload set can definitely be
  // resolved to a single candidate or not.
  if (expectedType) {
    return filterOverloadSetForValueType(expectedType,
                                         /*emitDiagnosticOnFailure=*/true);
  }

  // If the overload set has parameter bindings, try to resolve the candidates
  // using them.
  if (!paramBindings.empty())
    return filterOverloadSetForParamBindings(/*allowImplicitConversions=*/true);

  // Otherwise, emit the "cannot form a reference to overloaded decl" error.
  return getBoundConstantAttr();
}

PValue OverloadSet::getIfPValue() const {
  // Overload sets with base values cannot be emitted as PValues since they
  // depend on a dynamic value.
  // TODO: A conversion can be emitted if the base value is a PValue.
  if (baseValue)
    return {};

  if (fnDecls.size() != 1)
    return {};

  return paramBindings.getBoundConstAttrFor(cast<LIT::FuncOp>(*fnDecls[0]),
                                            baseName, expr);
}

/// Emit this as a RValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue OverloadSet::emitAsCValue(ExprEmitter &emitter, ValueDest &dest) {
  // If we have an overload set with multiple possibilities, we'll fail to emit
  // this as a RValue.  Try to resolve it based on the destination's type.
  ASTType expectedType;
  if (fnDecls.size() > 1) {
    expectedType = dest.resolveImpliedType(expr->getLoc(),
                                           /*no implied type*/ Type(), emitter);
  }

  // We allow unbound symbols here which can be emitted as an PValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  PValue directSymbolAttr = getDirectSymbol(expectedType);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseValue)
    return emitter.emitCResult(directSymbolAttr, expr, dest);

  auto loc = baseValue.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  auto calleeSignature =
      cast<LITSignatureType>(directSymbolAttr.getType().mlirType);

  assert(!calleeSignature.isAnyVarArg(0) && "Error: self shouldn't be varargs");

  // TODO: Need to emit a closure instance that partially applies the 'self'
  // argument here.
  emitter.emitError(
      loc, "TODO: partial application of member methods is not yet supported");
  return {};
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
CValue OverloadSet::emitCall(CallOperands &&operands, ValueDest &dest,
                             ExprEmitter &emitter) {
  // The OverloadSet may have been formed in a parameter context (e.g. in an
  // alias) and used a a non-parameter context.
  // FIXME: isParamContext isn't like scope info.  We should eliminate this from
  // `TypeCheckScopeInfo`.
  llvm::SaveAndRestore x(paramBindings.isParamContext,
                         emitter.paramContext != EC_InvalidContext);

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;

  // If we have a bound self, add it to the operand list to simplify the logic
  // below.
  if (baseValue)
    operands.addSelf(baseValue);

  // Check the direct callees to see if they can be unambiguously resolved
  // with the bindings list and specified arguments.
  PValue callee = filterOverloadSet(operands,
                                    /*allowImplicitConversions=*/true,
                                    /*emitDiagnosticOnFailure=*/true, emitter);
  if (!callee) {
    dest.resetForError();
    return {};
  }
  return emitter.emitCallUnchecked(callee, operands, dest, expr);
}

CValue ExprEmitter::emitIndirectCall(CValue callee, CallOperands &&operands,
                                     ValueDest &dest,
                                     const ExprNode *callExpr) {
  auto calleeSig = dyn_cast<SignatureType>(callee.getRValueType());
  if (!calleeSig) {
    // If we are invoking something other than a SignatureType, try to invoke
    // its `__call__` method.
    operands.addSelf({callee, callExpr});
    return emitNamedMethodCall("__call__", std::move(operands), dest,
                               CallSyntax::kDirectCall, callExpr);
  }

  // If we have a function pointer, resolve it to an RValue.
  RValue calleeRV = emitRValue({callee, callExpr}, EC_CallCalleeValue);
  if (!calleeRV) {
    dest.resetForError();
    return {};
  }

  // Check to see if we can apply these operands to the callee signature.
  OverloadSet bindings{"callee", /*fnDecls=*/{}, ParamBindings(getScopeInfo()),
                       callExpr, CallSyntax::kIndirectCall};
  auto fitness = OverloadFitness::evaluate(calleeSig, /*indirect*/ nullptr,
                                           bindings, operands,
                                           /*allowImplicitConversions=*/true);
  if (!fitness.isValid()) {
    // If not, diagnose it with an error.
    emitError(callExpr->getLoc(), "invalid indirect call: ")
        << fitness.takeDiag();
    dest.resetForError();
    return {};
  }

  // If we have inferred parameters, bind them here. An indirect call with
  // inferred parameters must be a PValue.
  if (!fitness.getParamBindings().empty()) {
    SmallVector<TypedAttr> bindOperands;
    if (auto calleePVal = calleeRV.getIfPValue()) {
      bindOperands.push_back(calleePVal);
    } else {
      // The callee can be dynamic in cases where one of the parents had a
      // resolution error but we are inside the body of a closure. In this case
      // we want to silently error.
      for (ASTDecl *scope = &declScope; scope; scope = scope->getParentDecl()) {
        if (scope->isErroneous()) {
          dest.resetForError();
          return {};
        }
      }
      llvm_unreachable("binding a dynamic callee?");
    }
    llvm::append_range(bindOperands, fitness.getParamBindings());
    calleeRV = PValue(ParamOperatorAttr::get(POC::BindSignature, bindOperands));
  }

  return emitCallUnchecked(calleeRV, operands, dest, callExpr);
}

CValue ExprEmitter::emitNamedMethodCall(StringRef methodName,
                                        CallOperands &&operands,
                                        ValueDest &dest, CallSyntax syntax,
                                        const ExprNode *callNode) {
  assert(!operands.values.empty() &&
         "Cannot emit a method call without a receiver!");

  // Emit the first/self operand to a CValue so we can figure out which type to
  // lookup on.
  CValue selfVal = operands[0].ir.getIfCValue();
  if (!selfVal) {
    selfVal = emitCValue(operands[0], EC_CallArgValue);
    if (!selfVal) {
      dest.resetForError();
      return {};
    }
    operands[0].ir = selfVal;
  }

  ASTType type = selfVal.getRValueType();

  auto emitNoMethodError = [&]() {
    auto diag = emitError(callNode->getLoc(), "")
                << type << " does not implement the '" << methodName
                << "' method";
    switch (syntax) {
    case CallSyntax::kMethodCallSynthetic:
    case CallSyntax::kMethodCall:
      [[fallthrough]];
    case CallSyntax::kOperator:
      diag << operands[0].expr->getRange();
      break;
    case CallSyntax::kReversedOperator:
      diag << operands[1].expr->getRange();
      break;
    default:
      break;
    }
  };

  // If the type doesn't have the specified method, emit an error.
  PValue callee =
      OverloadSet::lookupAndResolve(type, methodName, operands, callNode,
                                    syntax, emitNoMethodError, true, *this);
  if (!callee) {
    dest.resetForError();
    return {};
  }

  return emitIndirectCall(callee, std::move(operands), dest, callNode);
}

/// Emit a call to __init__, returning an instance of the specified type.  If
/// `allowImplicitConversion` is true, the provided args are allowed to
/// implicitly convert to the expectations of the constructor signatures.
CValue ExprEmitter::emitConstructorCall(ASTType type,
                                        CallOperands &&callOperands,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        bool allowImplicitConversion) {
  // If the dest type is invalid, then an error has already been reported.
  if (type.isTypeCheckErrorType())
    return {};

  // Check to see if we can invoke an __init__ method to convert it.
  auto callee =
      OverloadSet::lookup(getScopeInfo(), type, "__init__", expr, syntax);
  shared.notifyListenerOnCall(callee.fnDecls, expr->getRangeEnd(), syntax,
                              callOperands);
  if (callee.isErroneous())
    return {};

  // If there are no candidates at all, diagnose specific errors.
  if (!callee) {
    if (!type.getDecl(shared) && syntax != CallSyntax::kImplicitConvert) {
      emitError(expr->getLoc())
          << "MLIR type " << type
          << " must be created with an MLIR operation, not constructor "
             "syntax";
      return {};
    }

    // Emit helpful error message when user tried to call a module.
    if (auto refType = dyn_cast<ParamRefType>(type)) {
      if (auto moduleAttr = dyn_cast<LIT::ModuleAttr>(refType.getParam())) {
        auto metaType = cast<AnyStructType>(moduleAttr.getType());
        auto diag = emitError(expr->getLoc());
        emitModuleCallSubscriptDiag(diag, metaType, "call", expr->getLoc(),
                                    shared);
        diag << expr->getRange();
        return {};
      }
    }

    // Diagnose implicit conversions with a custom message
    if (syntax == CallSyntax::kImplicitConvert) {
      ASTType singleOperandType;
      assert(callOperands.size() == 1 &&
             "implicit conversions have one operand");
      if (auto cValue = callOperands[0].ir.getIfCValue())
        singleOperandType = cValue.getRValueType();

      auto diag = emitError(expr->getLoc());
      if (isa<StructType>(type)) {
        diag << "invalid implicit conversion to " << type
             << ": no constructors found";
        return {};
      }

      // This is true if passing Int type to Int instead of Int() to Int.
      bool isConvertingTypeValue = type.getMetaType() == singleOperandType;
      bool isImplConvert = dest.getContext() != EC_CallParamValue &&
                           dest.getContext() != EC_CallArgValue;
      diag << "cannot " << (isImplConvert ? "implicitly convert " : "pass ");

      if (isConvertingTypeValue)
        diag << type << " type as a ";
      else if (singleOperandType)
        diag << singleOperandType << " ";
      diag << "value" << (isImplConvert ? " to " : ", expected ");
      diag << (isConvertingTypeValue ? "an instance of " : "") << type
           << getContextMessage(dest.getContext());

      if (isConvertingTypeValue)
        diag << "; did you mean to instantiate " << type << "?";
      diag << expr->getRange();
      return {};
    }
  }

  // Set the parameter bindings for the type we're creating - they can't be
  // inferred since from the result type.
  // FIXME: Should be able to remove this when kInitReg goes away.
  callee.paramBindings =
      ParamBindings::getForDeclaredType(getScopeInfo(), type, expr);

  // As a special extension, register-only types are allowed to return their
  // self directly as a register value instead of taking a memory value in.
  // Check to see if the init members in the overload set are the kInitReg form.
  // TODO: Eliminate special register form.
  bool hasInitSelfArg = true;
  if (type.isRegisterPassable(expr->getLoc(), shared)) {
    for (auto fnDecl : callee.fnDecls) {
      if (cast<LIT::FuncOp>(*fnDecl).getSpecialFunctionKind() ==
          SpecialFunctionKind::kInitReg) {
        hasInitSelfArg = false;
        break;
      }
    }

    // In the "-> Self" form of initializer, we may get ambiguity between
    // non-materializable "() -> IntLiteral" and "() -> Int" overloads which
    // cannot be resolved.  We're inferring based on result type, so manually
    // remove these.
    // TODO: Eliminate this special register form.
    if (!hasInitSelfArg) {
      auto *typeDecl = type.getDecl(shared);
      for (size_t i = 0; i != callee.fnDecls.size();) {
        if (callee.fnDecls[i]->getParentDecl() == typeDecl)
          ++i;
        else
          callee.fnDecls.erase(callee.fnDecls.begin() + i);
      }
    }
  }

  // Provide a self value so parameter inference can infer parameters from
  // typeof(self).
  if (hasInitSelfArg) {
    assert(!callee.baseValue && "Shouldn't have a self value yet");
    auto attr = UnknownAttr::get(RefType::getImmortal(type, true));
    callee.baseValue = {PValue(attr), expr};
  }

  return callee.emitCall(std::move(callOperands), dest, *this);
}

//===----------------------------------------------------------------------===//
// Type conversion helpers.

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkMLIRTypeConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveFully(traitDecl, loc)))
    return false; // an error was emitted
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto traitFn = dyn_cast<LIT::FuncOp>(*decl);
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

/// If the specified type can be constructed with the specified operands
/// return the initializer that would be invoked. If not, return null PValue.
/// If there were erroneous declarations when processing return failure so we
/// don't indicate downstream errors.
///
/// If there were erroneous declarations, an error has been raised about a
/// constructor that likely would have applied, which should be considered in
/// any error reporting. This does not generate any IR.
FailureOr<PValue> OverloadSet::canConstructType(
    ASTType requiredType, CallOperands &&operands, const ExprNode *expr,
    const TypeCheckScopeInfo &scopeInfo, bool allowImplicitConversions) {

  // Check to see if we can do an implicit conversion by invoking a `__init__`
  // method on the expected type.
  OverloadSet callee = OverloadSet::lookup(
      scopeInfo, requiredType, "__init__", expr, CallSyntax::kImplicitConvert,
      /*no error emission on failure */ {});

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee)
    return callee.isErroneous() ? FailureOr<PValue>(failure()) : PValue();

  // Initializers take 'inout self' as the first argument.
  // Register passable types have a funny exception that allow them to be
  // called without this.  Check to see if we're doing that.
  bool hasInitSelf = true;
  if (requiredType.isRegisterPassable(expr->getLoc(), scopeInfo.shared)) {
    if (!callee.fnDecls.empty() &&
        cast<FuncOp>(callee.fnDecls[0]).getSpecialFunctionKind() ==
            SpecialFunctionKind::kInitReg)
      hasInitSelf = false; // Using the register return convention.
  }

  // If this is InitSelf then we'll pass a self argument with the
  // destination when invoking the method.
  if (hasInitSelf) {
    // TODO: We should add a new magic InferSelfLValue() IRValue type.  This
    // would make the inference and overload resolution logic more consistent
    // because the selfexpr should really be an LValue.
    auto inferType =
        requiredType.getWithUnknownParametersReplaced(scopeInfo.shared);
    auto attr = UnknownAttr::get(RefType::getImmortal(inferType, true));
    operands.addSelf({PValue(attr), expr});
  }
  // Install the Self type parameters on the callee directly, since they cannot
  // always be inferred. This can happen if a constructor has more specific Self
  // type parameters or for the deprecated `-> Self` form of initializers.
  callee.paramBindings =
      ParamBindings::getForDeclaredType(scopeInfo, requiredType, expr);

  // Determine if we can emit this using an ExprEmitter in the parameter domain.
  // This ensures we don't emit any code converting parameters to MValues etc.
  ExprEmitter paramEmitter(scopeInfo.shared, scopeInfo.declScope,
                           ExprContext::EC_CallCalleeValue);

  // If we have at least one candidate, we check to see if any of them can
  // work. This needs to call filterOverloadSet manually because we might not
  // be able to allow implicit conversions.
  PValue result =
      callee.filterOverloadSet(operands, allowImplicitConversions,
                               /*emitDiagnosticOnFailure=*/false, paramEmitter);
  if (callee.isErroneous())
    return FailureOr<PValue>(failure());
  return result;
}

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
bool OverloadSet::canImplicitlyConvertToType(
    ASTExprAnd<CValue> value, ASTType requiredType,
    const TypeCheckScopeInfo &scopeInfo) {
  auto &shared = scopeInfo.shared;

  assert(value.ir && "Should only query valid values");
  // If it already matches, then we're done.
  ASTType rvType = value.ir.getRValueType();
  if (rvType.isEqualCanon(requiredType) ||
      canConvertWithRebind(rvType, requiredType, shared))
    return true;

  // Lifetimes and lifetime sets can convert between each other.
  // FIXME: This seems wrong, why isn't it checking for inclusion and
  // compatibility??
  if (isa<LifetimeType, LifetimeSetType>(rvType) &&
      isa<LifetimeType, LifetimeSetType>(requiredType))
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
  if (auto traitType = dyn_cast<TraitType>(requiredType)) {
    std::optional<InflightDiag> diag;
    // Struct types and Trait types can conform to traits.
    if (isa<AnyStructType, TraitType>(rvType) &&
        rvType.getDecl(shared)->doesNominalTypeConformsTo(traitType, diag,
                                                          shared))
      return cacheAndReturnVal(true);
    if (diag)
      diag->abandon();

    // MLIR types can conform to traits that have limited requirements.
    // AnyTraitType (the type of all traits) conforms to traits with only a
    // destructor (e.g. AnyType) since all traits have that.
    if (isa<TypeType>(rvType) &&
        checkMLIRTypeConformance(shared, value.expr->getLoc(), traitType))
      return cacheAndReturnVal(true);
    return cacheAndReturnVal(false);
  }

  // We can implicitly convert to the specified type if we can construct it with
  // the value.

  // Disable implicit conversions though, to prevent converting T -> S -> U in
  // one step.
  FailureOr<PValue> result = OverloadSet::canConstructType(
      requiredType, {{value}}, value.expr, scopeInfo,
      /*allowImplicitConversions=*/false);
  return cacheAndReturnVal(succeeded(result) && result.value());
}

void OverloadSet::dump() const {
  auto &os = llvm::errs();
  os << "OverloadSet{ ";
  os << baseName << " base name, ";
  os << " functions:\n";
  for (auto f : fnDecls) {
    os << "\t";
    f->dump();
    os << "\n";
  }
  if (paramBindings.empty()) {
    os << "no bound params, ";
  } else {
    os << "param bindings: ";
    paramBindings.dump();
  }
  os << syntax << " call syntax";
  if (erroneous)
    os << ", <ERRONEOUS>";
  os << "\n}\n";
}
