//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SymbolicExpressions.h"
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

Region &IREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  auto func = getSymbolTable().lookup<FuncOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());

  // Make sure the function is inflated.
  elaborator.asyncMap.mapChained(func, [&](LLCL::AnyAsyncValueRef ch) {
    return Cache::inflateOp(func, elaborator.regionCache.copy(), std::move(ch));
  });
  elaborator.asyncMap.await(func);

  // Now we can return the function body.
  return func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
  // Make sure the function is inflated.
  elaborator.asyncMap.mapChained(func, [&](LLCL::AnyAsyncValueRef ch) {
    return Cache::inflateOp(func, elaborator.regionCache.copy(), std::move(ch));
  });
  elaborator.asyncMap.await(func);

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
  return result.getValue().front();
}

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator,
                         DenseMap<StringAttr, Attribute> paramValues)
    : ParameterEvaluator(std::move(paramValues)),
      InterpreterState(elaborator.analysis, elaborator.target),
      symtab(elaborator.analysis.getTopLevelSymbolTable()),
      elaborator(elaborator) {}

FailureOr<TypedAttr>
IREvaluator::evaluateSymbolicExpression(ParamOperatorAttr op) {
  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  if (op.getOpcode() == POC::Apply) {
    auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
    if (!symbol || !symbol.getType().getResultParamTypes().empty())
      return failure();
    ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
    auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
    if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant) || !ref)
      return failure();

    // Lookup the symbol reference.
    FuncInterface func = elaborator.lookupCallee(ref);
    if (!isa<FuncOp>(*func)) {
      // The symbol does not refer to a concrete function. Ask the elaborator to
      // instantiate the callee.
      SmallVector<Attribute> inputParams;
      for (ParamBindAttr bind : symbol.getParamValues())
        inputParams.push_back(bind.getValue());
      EvalContext &evalCtx = elaborator.getEvalContext(ref);
      auto paramValues = ArrayAttr::get(op.getContext(), inputParams);
      for (auto [decl, value] :
           llvm::zip(func.getInputParamDecls(), paramValues))
        evalCtx.evaluator.setOrOverwriteParameterValue(decl, value);
      ArrayRef<ErrorTreeOr<ElaboratedGenerator>> results =
          elaborator.getAllInstantiations(
              {cast<DeclInterface>(*func), paramValues},
              /*expansionDepth=*/0, evalCtx);

      // Since we are evaluating the callee at compile time, just pick the first
      // viable candidate.
      ErrorTree err(*errorLoc, "unable to evaluate generator or interface");
      for (const ErrorTreeOr<ElaboratedGenerator> &result : results) {
        if (result.isError()) {
          err.addCause(result.getError().copy());
          continue;
        }
        func = result.getValue().func;
        break;
      }
      if (!isa<FuncOp>(*func)) {
        emitError(std::move(err));
        return failure();
      }
    }

    ErrorTreeOr<TypedAttr> result =
        evaluateFunction(cast<FuncOp>(*func), operands);
    if (TypedAttr value = result.tryGetValue())
      return value;
    emitError(result.takeError());
    return failure();
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
  Optional<ErrorTree> error;
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
  Optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);

  if (isa<ConcreteTypeConstantAttr>(TypeConstantAttr::get(result)))
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
Optional<ErrorTree>
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
