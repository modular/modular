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
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  FuncOp func =
      elaborator->getSymbolTable().read([name](const SymbolTable &symtab) {
        return symtab.lookup<FuncOp>(name);
      });

  // Now we can return the function body.
  return &func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
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
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> IREvaluator::evaluateExpression(ParamOperatorAttr op) {
  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  if (op.getOpcode() == POC::CurrentTarget) {
    // Retrieve the contextual compilation target info.
    return {TargetParamAttr::get(elaborator->getTarget())};
  }

  if (op.getOpcode() == POC::GetAllImpls)
    return evaluateGetAllImpls(op);

  if (op.getOpcode() == POC::Apply)
    return evaluateApply(op);

  if (op.getOpcode() == POC::GetEnv)
    return evaluateGetEnv(op);

  return failure();
}

FailureOr<TypedAttr> IREvaluator::evaluateGetAllImpls(ParamOperatorAttr op) {
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
  if (!symbol)
    return failure();
  std::vector<FuncOp> funcs;
  std::optional<ErrorTreeOrSuccess> err = elaborator->getAllConcreteFunctions(
      parent, *errorLoc, cast<FlatSymbolRefAttr>(symbol.getSymbol()),
      symbol.getParamValues(), funcs);
  if (!err) {
    return TypedAttr();
  }
  if (err->isError()) {
    emitError(err->takeError());
    return TypedAttr();
  }

  std::vector<TypedAttr> refs;
  refs.reserve(funcs.size());
  for (FuncOp f : funcs)
    refs.emplace_back(SymbolConstantAttr::get(
        SymbolRefAttr::get(f.getSymNameAttr()), f.getFullSignature()));

  return {VariadicAttr::get(refs, cast<VariadicType>(op.getType()))};
}

FailureOr<TypedAttr> IREvaluator::evaluateApply(ParamOperatorAttr op) {
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
  if (!symbol || !symbol.getType().getResultParamTypes().empty())
    return failure();
  ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
  auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
  if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant) || !ref)
    return failure();

  // Lookup the symbol reference and resolve it.
  std::optional<ErrorTreeOr<FuncOp>> func = elaborator->getConcreteFunction(
      parent, *errorLoc, ref, symbol.getParamValues());
  if (!func) {
    return TypedAttr();
  }
  if (func->isError()) {
    emitError(func->takeError());
    return TypedAttr();
  }

  ErrorTreeOr<TypedAttr> result = evaluateFunction(**func, operands);
  if (TypedAttr value = result.tryGetValue())
    return value;
  emitError(result.takeError());
  return TypedAttr();
}

FailureOr<TypedAttr> IREvaluator::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  auto module = cast<ModuleOp>(elaborator->getSymbolTable().get().getOp());
  auto env = module->getAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName());
  assert(env && "expected an environment attribute on the module");

  auto name = dyn_cast<StringAttr>(op.getOperands().front());
  if (!name) {
    emitError({*errorLoc, "'get_env' name did not narrow to a constant"});
    return failure();
  }

  // Get the `StringRef` out of the `StringAttr` because the latter comes with
  // a `StringType` type that makes pointer comparisons fails.
  Attribute value = env.getValues().get(name.getValue());
  if (isa<IndexType, StringType>(op.getType()) && !value) {
    emitError({*errorLoc, "environment variable '" + name.getValue() +
                              "' does not exist"});
    return failure();
  }

  if (isa<IndexType>(op.getType())) {
    if (auto intVal = dyn_cast<IntegerAttr>(value))
      return {intVal};
    emitError({*errorLoc, "environment variable '" + name.getValue() +
                              "' is not an integer, got " +
                              mlir::debugString(value)});
    return failure();
  }

  if (isa<StringType>(op.getType())) {
    if (auto strVal = dyn_cast<StringAttr>(value))
      return {strVal};
    emitError({*errorLoc, "environment variable '" + name.getValue() +
                              "' is not a string, got " +
                              mlir::debugString(value)});
    return failure();
  }

  // This must be an `i1` type. Return true or false based on whether the
  // environment variable is present.
  assert(cast<IntegerType>(op.getType()).isSignlessInteger(1));
  return {BoolAttr::get(op.getContext(), static_cast<bool>(value))};
}

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator,
                         DenseMap<StringAttr, Attribute> paramValues)
    : ParameterEvaluator(std::move(paramValues)),
      InterpreterState(elaborator.getTarget()), elaborator(&elaborator) {}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                            Location loc,
                                                            Attribute expr,
                                                            bool allowUnknown) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Attribute result = getReboundAttribute(expr);
  if (error)
    return std::move(*error);

  if (!result)
    return Attribute();

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

ErrorTreeOr<Type> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                       Location loc, Type expr,
                                                       bool allowUnknown) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);

  if (!result)
    return Type();

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
std::optional<ErrorTreeOrSuccess>
KGEN::evaluateConstraints(ImplNode *parent,
                          ArrayRef<ConstraintAttr> constraints,
                          IREvaluator &evaluator) {
  // Each constraint must be foldable, and must fold to true.
  for (ConstraintAttr constraint : constraints) {
    Location loc = constraint.getLoc();
    ErrorTreeOr<Attribute> result = evaluator.concretizeParameterExpr(
        parent, loc, constraint.getExpr(), /*allowUnknown=*/false);
    if (result.isError())
      return ErrorTree(loc, "constraint evaluation failure",
                       result.takeError());
    if (!*result)
      return std::nullopt;

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
  return success();
}
