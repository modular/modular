//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
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
  ParameterEvaluator evaluator(&symtab);
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

static ErrorOr<TypedAttr> evaluateFunction(FuncOp func,
                                           ArrayRef<TypedAttr> inputs) {
  DenseMap<Value, Attribute> values;
  // Map the function argument values.
  for (auto [arg, input] : llvm::zip(func.getArguments(), inputs))
    values.try_emplace(arg, input);

  std::string errMsg("IR interpreter failed: ");
  llvm::raw_string_ostream err(errMsg);

  // "Interpret" the IR by calling operation folders.
  SmallVector<Attribute> operands;
  SmallVector<OpFoldResult> results;
  for (Operation &op : func.getBody()->without_terminator()) {
    operands.clear();
    for (Value operand : op.getOperands())
      operands.push_back(values.lookup(operand));
    if (failed(op.fold(operands, results))) {
      // FIXME: We need to report the location error, but the elaborator isn't
      // set up to attach "notes" the elaboration diagnostics.
      err << "could not fold operation " << op.getName() << '(';
      llvm::interleaveComma(operands, err);
      err << ") in function @" << func.getName();
      return Error(err.str());
    }
    for (auto [result, output] : llvm::zip(results, op.getResults())) {
      auto value = result.dyn_cast<Attribute>();
      if (!value) {
        err << "folder for operation " << op.getName()
            << " did not return a constant value, in function @"
            << func.getName();
        return Error(err.str());
      }
      values.try_emplace(output, value);
    }
  }

  // Extract the result.
  assert(func.getNumResults() == 1);
  Attribute result = values.lookup(func.getReturnOp().getOperand(0));
  if (auto expr = dyn_cast<TypedAttr>(result))
    return expr;
  err << "function @" << func.getName()
      << " result is not a parameter expression";
  return Error(err.str());
}

//===----------------------------------------------------------------------===//
// Parameter Evaluator
//===----------------------------------------------------------------------===//

ErrorOr<TypedAttr>
ParameterEvaluator::evaluateSymbolicExpression(ParamOperatorAttr op) {
  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  if (op.getOpcode() == POC::GetSizeOf || op.getOpcode() == POC::GetAlignOf) {
    auto typeCst = dyn_cast<TypeConstantAttr>(op.getOperand(0));
    auto target = dyn_cast<TargetInfoAttr>(op.getOperand(1));
    if (!typeCst || !target)
      return {op};
    auto ref = dyn_cast<DeclRefType>(typeCst.getValue());
    if (!ref)
      return {op};

    ErrorOr<int64_t> indexResult = 0;
    if (op.getOpcode() == POC::GetSizeOf)
      indexResult = computeStructSizeof(*symtab, target, ref);
    else
      indexResult = computeStructAlignof(*symtab, target, ref);
    if (indexResult.isError())
      return indexResult.takeError();
    return {Builder(op.getContext()).getIndexAttr(*indexResult)};
  }

  if (op.getOpcode() == POC::Apply) {
    auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
    if (!symbol || !symbol.getType().isConcrete())
      return {op};
    ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
    if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant))
      return {op};
    if (auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol()))
      if (auto func = symtab->lookup<FuncOp>(ref.getAttr()))
        return evaluateFunction(func, operands);
    return {op};
  }

  return {op};
}
