//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SymbolicExpressions.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
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
// Struct Layout
//===----------------------------------------------------------------------===//

/// Lookup the struct declaration and rebind it.
static std::pair<StructDeclOp, ParameterEvaluator>
lookupStructDecl(SymbolTable &symtab, DeclRefType type) {
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : type.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());
  auto decl = symtab.lookup<StructDeclOp>(type.getName());
  return {decl, std::move(evaluator)};
}

/// Get the alignemnt of a type.
static ErrorOr<int64_t> computeAlignof(SymbolTable &symtab,
                                       TargetInfoAttr target, Type type);

/// Build the expression to compute the alignment of a struct type. Returns none
/// if it could not be computed.
static ErrorOr<int64_t> computeStructAlignof(SymbolTable &symtab,
                                             TargetInfoAttr target,
                                             DeclRefType type) {
  auto [decl, evaluator] = lookupStructDecl(symtab, type);

  // The alignment of a struct is the strictest alignment requirement of its
  // fields. The smallest alignment is 1.
  int64_t align = 1;
  for (StructFieldOp field : decl.getFieldDecls()) {
    ErrorOr<int64_t> fieldAlign = computeAlignof(
        symtab, target, evaluator.getReboundType(field.getType()));
    if (fieldAlign.isError())
      return fieldAlign.takeError();
    align = std::max(align, *fieldAlign);
  }
  return align;
}

/// Get the alignemnt of a type.
static ErrorOr<int64_t> computeAlignof(SymbolTable &symtab,
                                       TargetInfoAttr target, Type type) {
  if (auto ref = dyn_cast<DeclRefType>(type))
    return computeStructAlignof(symtab, target, ref);
  Optional<int64_t> align =
      DataLayoutInterface::getTypeAlignInBytes(target, type);
  if (!align)
    return Error("could not compute alignment of type " +
                 mlir::debugString(type));
  return *align;
}

/// Get the size of a type.
static ErrorOr<int64_t> computeSizeof(SymbolTable &symtab,
                                      TargetInfoAttr target, Type type);

/// Build the expression to compute the size of a struct type. Returns none if
/// it could not be computed.
static ErrorOr<int64_t> computeStructSizeof(SymbolTable &symtab,
                                            TargetInfoAttr target,
                                            DeclRefType type) {
  auto [decl, evaluator] = lookupStructDecl(symtab, type);

  // The smallest size is 0.
  int64_t size = 0, align = 1;
  for (StructFieldOp field : decl.getFieldDecls()) {
    // Add padding to the current size of the struct to align it to the
    // alignment of the field type before adding its size.
    Type type = evaluator.getReboundType(field.getType());
    ErrorOr<int64_t> fieldAlign = computeAlignof(symtab, target, type);
    if (fieldAlign.isError())
      return fieldAlign.takeError();
    ErrorOr<int64_t> fieldSize = computeSizeof(symtab, target, type);
    if (fieldSize.isError())
      return fieldSize.takeError();
    size = llvm::alignTo(size, *fieldAlign) + *fieldSize;
    align = std::max(align, *fieldAlign);
  }
  // Pad the struct to satisfy its own alignment.
  return llvm::alignTo(size, align);
}

/// Get the size of a type.
static ErrorOr<int64_t> computeSizeof(SymbolTable &symtab,
                                      TargetInfoAttr target, Type type) {
  if (auto ref = dyn_cast<DeclRefType>(type))
    return computeStructSizeof(symtab, target, ref);
  Optional<int64_t> size =
      DataLayoutInterface::getTypeSizeInBytes(target, type);
  if (!size)
    return Error("could not compute size of type " + mlir::debugString(type));
  return *size;
}

//===----------------------------------------------------------------------===//
// IR Interpreter
//===----------------------------------------------------------------------===//

/// Report an error with folding an operation.
static ErrorTree reportFoldError(FuncOp func, Operation &op,
                                 ArrayRef<Attribute> operands,
                                 const Twine &prefix,
                                 const Twine &suffix = "") {
  ErrorTree error(func.getLoc(),
                  "failed to interpret function @" + func.getName());
  std::string note;
  llvm::raw_string_ostream os(note);
  os << prefix << op.getName() << '(';
  llvm::interleaveComma(operands, os);
  os << ')' << suffix;
  return std::move(error.addCause(op.getLoc(), Error(os.str())));
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
  // Make sure the function is inflated.
  asyncMap.mapChained(func, [&](LLCL::AnyAsyncValueRef ch) {
    return Cache::inflateOp(func, regionCache.copy(), std::move(ch));
  });
  asyncMap.await(func);

  DenseMap<Value, Attribute> values;
  // Map the function argument values.
  for (auto [arg, input] : llvm::zip(func.getArguments(), inputs))
    values.try_emplace(arg, input);

  // This is the top-level error that will be returned if one occurs.
  ErrorTree error(*errorLoc, "failed to evaluate 'apply'");

  // Interpret the IR.
  SmallVector<Attribute> operands;
  SmallVector<OpFoldResult> results;
  for (Operation &op : func.getBody()->without_terminator()) {
    operands.clear();
    results.clear();
    for (Value operand : op.getOperands())
      operands.push_back(values.lookup(operand));

    // Check for an interpreter interface implementation.
    if (auto interpItf = dyn_cast<InterpreterOpInterface>(op)) {
      ErrorOrSuccess err = interpItf.interpret(operands, *this, results);
      if (err.isError()) {
        return std::move(error.addCause(
            std::move(reportFoldError(func, op, operands,
                                      "failed to interpret operation ")
                          .addCause(op.getLoc(), err.takeError()))));
      }
    } else {
      // Otherwise, try to use the operation folder.
      if (failed(op.fold(operands, results)))
        return std::move(error.addCause(
            reportFoldError(func, op, operands, "failed to fold operation ")));
    }
    for (auto [i, result, output] :
         llvm::zip(llvm::seq<unsigned>(0, op.getNumResults()), results,
                   op.getResults())) {
      auto value = result.dyn_cast<Attribute>();
      if (!value) {
        return std::move(error.addCause(reportFoldError(
            func, op, operands, "operation evaluation ",
            " did not return a value for result #" + Twine(i))));
      }
      values.try_emplace(output, value);
    }
  }

  // Extract the result.
  assert(func.getNumResults() == 1);
  Attribute result = values.lookup(func.getReturnOp().getOperand(0));
  if (auto expr = dyn_cast<TypedAttr>(result))
    return expr;

  return std::move(error.addCause(
      func.getLoc(), Error("function @" + func.getName() +
                           " result is not a parameter expression")));
}

//=----------------------------------------------------------------------===//
// IR Evaluator
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
IREvaluator::evaluateSymbolicExpression(ParamOperatorAttr op) {
  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  if (op.getOpcode() == POC::GetSizeOf || op.getOpcode() == POC::GetAlignOf) {
    auto typeCst = dyn_cast<TypeConstantAttr>(op.getOperand(0));
    auto target = dyn_cast<TargetParamAttr>(op.getOperand(1));
    if (!typeCst || !target)
      return failure();
    auto ref = dyn_cast<DeclRefType>(typeCst.getValue());
    if (!ref)
      return failure();

    ErrorOr<int64_t> indexResult = 0;
    if (op.getOpcode() == POC::GetSizeOf)
      indexResult = computeStructSizeof(symtab, target.getTarget(), ref);
    else
      indexResult = computeStructAlignof(symtab, target.getTarget(), ref);

    if (indexResult.isError()) {
      emitError({*errorLoc, indexResult.takeError()});
      return failure();
    }
    return {Builder(op.getContext()).getIndexAttr(*indexResult)};
  }

  if (op.getOpcode() == POC::Apply) {
    auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
    if (!symbol || !symbol.getType().isConcrete())
      return failure();
    ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
    auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
    if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant) || !ref)
      return failure();
    auto func = symtab.lookup<FuncOp>(ref.getAttr());
    if (!func)
      return failure();

    ErrorTreeOr<TypedAttr> result = evaluateFunction(func, operands);
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
