//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "IREvaluator.h"
#include "Elaborator.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// IR Interpreter
//===----------------------------------------------------------------------===//

ErrorOr<Region *> IREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  auto func = elaborator->getAnalysis().getTopLevelSymbolTable().lookup<FuncOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());

  // Make sure the function is inflated.
  if (auto err = elaborator->inflateFunc(func))
    return err.takeError();

  // Now we can return the function body.
  return &func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
  // Make sure the function is inflated.
  if (auto err = elaborator->inflateFunc(func))
    return ErrorTree(func.getLoc(), err.takeError());

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);
  ErrorTreeOr<SmallVector<Attribute>> result =
      startInterpreterAt(func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to evaluate 'apply'",
                     result.takeError());
  }

  // Apply operators only return one result.
  return cast<TypedAttr>(result.getValue().front());
}

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator,
                         DenseMap<StringAttr, Attribute> paramValues)
    : ParameterEvaluator(std::move(paramValues)),
      InterpreterState(elaborator.getTarget()),
      symtab(&elaborator.getAnalysis().getTopLevelSymbolTable()),
      elaborator(&elaborator) {}

FailureOr<TypedAttr> IREvaluator::evaluateExpression(ParamOperatorAttr op) {
  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  if (op.getOpcode() == POC::CurrentTarget) {
    // Retrieve the contextual compilation target info.
    return {TargetParamAttr::get(elaborator->getTarget(),
                                 TargetType::get(op.getContext()))};
  }

  if (op.getOpcode() == POC::GetAllImpls) {
    auto symbol = cast<SymbolConstantAttr>(op.getOperand(0));
    std::vector<FuncOp> funcs;
    if (auto err = elaborator->getAllConcreteFunctions(
            *errorLoc, symbol.getSymbol(), symbol.getParamValues(), funcs)) {
      emitError(std::move(*err));
      return failure();
    }

    std::vector<TypedAttr> refs;
    refs.reserve(funcs.size());
    for (FuncOp f : funcs)
      refs.emplace_back(SymbolConstantAttr::get(
          SymbolRefAttr::get(f.getSymNameAttr()), f.getFullSignature()));

    return {VariadicAttr::get(refs, cast<VariadicType>(op.getType()))};
  }

  if (op.getOpcode() == POC::Apply) {
    auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
    if (!symbol || !symbol.getType().getResultParamTypes().empty())
      return failure();
    ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
    auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
    if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant) || !ref)
      return failure();

    // Lookup the symbol reference and resolve it.
    ErrorTreeOr<FuncOp> func = elaborator->getConcreteFunction(
        *errorLoc, ref, symbol.getParamValues());
    if (func.isError()) {
      emitError(func.takeError());
      return failure();
    }

    ErrorTreeOr<TypedAttr> result = evaluateFunction(*func, operands);
    if (TypedAttr value = result.tryGetValue())
      return value;
    emitError(result.takeError());
    return failure();
  }

  if (op.getOpcode() == POC::Evaluate) {
    // Pull out the evaluator and ensure it's concretized.
    auto symbol =
        cast<SymbolConstantAttr>(op.getOperand(op.getNumOperands() - 1));
    ErrorTreeOr<FuncOp> evaluator = elaborator->getConcreteFunction(
        *errorLoc, symbol.getSymbol(), symbol.getParamValues());
    if (evaluator.isError()) {
      emitError(evaluator.takeError());
      return failure();
    }

    // Pull out the concrete functions from each option and evaluate them all.
    std::vector<FuncOp> options;
    auto optionsVariadic = cast<VariadicAttr>(op.getOperands().front());
    for (TypedAttr option : optionsVariadic.getValues()) {
      auto optionSym = cast<SymbolConstantAttr>(option);
      if (auto err = elaborator->getAllConcreteFunctions(
              *errorLoc, optionSym.getSymbol(), optionSym.getParamValues(),
              options)) {
        emitError(err->copy());
        return failure();
      }
    }

    auto bestOr = elaborator->getEvaluatorExecutorFn()(
        *evaluator, *symtab, elaborator->getTarget(), options);
    if (bestOr.isError()) {
      emitError(
          ErrorTree(UnknownLoc::get(op.getContext()), bestOr.takeError()));
      return failure();
    }

    // Have to create a new symbol constant because the best one could be an
    // implementation of one of the options.
    FuncOp best = options[*bestOr];
    return cast<TypedAttr>(SymbolConstantAttr::get(
        SymbolRefAttr::get(best.getSymNameAttr()), best.getFullSignature()));
  }

  return failure();
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute> IREvaluator::concretizeParameterExpr(Location loc,
                                                            Attribute expr,
                                                            bool allowUnknown) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Attribute result = getReboundAttribute(expr);
  if (error)
    return std::move(*error);

  // If we can fold this to a simple constant result, do.
  if (ParameterAttr::isSimpleConstant(result))
    return result;

  // If this was an unfoldable operator expression, error.  This can happen for
  // things like 'index' arithmetic that has target-specific results.
  if (auto oper = dyn_cast<ParamOperatorAttr>(result))
    return ErrorTree(loc,
                     "could not simplify operator " + getParamAsString(result));
  if (allowUnknown)
    return result;

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return ErrorTree(loc,
                   "unknown expression to fold: " + getParamAsString(result));
}

ErrorTreeOr<Type> IREvaluator::concretizeParameterExpr(Location loc,
                                                       Type expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);

  if (TypeConstantAttr::isConcreteType(result))
    return result;
  return ErrorTree(loc, Error("could not simplify type: " +
                              getParamAsString(TypeConstantAttr::get(result))));
}

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
std::optional<ErrorTree>
KGEN::evaluateConstraints(ArrayRef<ConstraintAttr> constraints,
                          IREvaluator &evaluator) {
  // Each constraint must be foldable, and must fold to true.
  for (ConstraintAttr constraint : constraints) {
    Location loc = constraint.getLoc();
    ErrorTreeOr<Attribute> result =
        evaluator.concretizeParameterExpr(loc, constraint.getExpr());
    if (!result)
      return ErrorTree(loc, "constraint evaluation failure",
                       result.takeError());

    auto resultInt = dyn_cast<IntegerAttr>(result.takeValue());
    if (!resultInt || resultInt.getValue().getBitWidth() != 1)
      return ErrorTree(loc,
                       "constraint evaluation didn't return true or false");

    // If this failed, indicate why.
    if (resultInt.getValue().isZero())
      return ErrorTree(loc, "constraint failed: " +
                                constraint.getMessage().getValue());
  }

  // If we made it this far, then everything folded to true.
  return {};
}
