//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// ParserParamEvaluator
//===----------------------------------------------------------------------===//

ParserParamEvaluator::ParserParamEvaluator(DeclResolver &resolver,
                                           ArrayRef<ParamDeclAttr> paramDecls,
                                           ArrayRef<TypedAttr> paramValues)
    : ParameterEvaluator(paramDecls, paramValues),
      InterpreterState(resolver.getContext()), resolver(resolver) {}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateFunctionCall(SymbolRefAttr symbol,
                                           ArrayRef<Attribute> arguments) {
  ErrorOr<Region *> body = lookupFunctionBody(symbol);
  if (body.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator", llvm::errs()
                                                   << "[ParserParamEvaluator] "
                                                   << body.getError() << "\n");
    return failure();
  }

  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(*body.takeValue(), arguments);
  if (result.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator",
                    result.takeError().emit(
                        (InFlightDiagnostic(*)(Location))mlir::emitError));
    return failure();
  }

  return cast<TypedAttr>(result->front());
}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateExpression(ParamOperatorAttr op) {
  if (op.getOpcode() != POC::Apply)
    return failure();

  // We can only fold direct calls.
  auto ref = dyn_cast<SymbolConstantAttr>(op.getOperands().front());
  if (!ref)
    return failure();

  // All inputs must be simple constants.
  ArrayRef<TypedAttr> inputs = op.getOperands().drop_front();
  if (!llvm::all_of(inputs, ParameterAttr::isSimpleConstant))
    return failure();

  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  return evaluateFunctionCall(ref.getSymbol(), arguments);
}

ErrorOr<Region *>
ParserParamEvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  ASTDecl *decl = resolver.getDeclForFuncSymbol(symbol);
  if (!decl)
    return Error("function not found: " + mlir::debugString(symbol));

  // Fail if the function is parameterized.
  if (failed(resolver.resolveSignature(*decl, decl->getLoc())))
    return Error("failed to resolve function signature");

  auto func = cast<LIT::FuncOp>(*decl);
  if (func.getInlineLevel() == InlineLevel::Automatic ||
      func.getInlineLevel() == InlineLevel::Never)
    return Error("function is not always_inline");
  LITSignatureType fullSig = func.getFullSignature();
  if (!fullSig.getParamTypes().empty() ||
      !fullSig.getResultParamTypes().empty())
    return Error("function is parametric");

  // Use of the interpreter's memory model requires a target specification,
  // which the parser does not have.
  if (fullSig.hasMemoryOnlyResult() || fullSig.hasInitSelfArg())
    return Error("function has memory-only result");

  // Make sure to fully resolve the body and everything within it.
  if (failed(resolver.resolveFully(*decl, decl->getLoc())))
    return Error("failed to fully resolve function");
  return &func.getBodyRegion();
}

Type ParserParamEvaluator::refineType(Type type) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamOperatorAttr op) -> TypedAttr {
    FailureOr<TypedAttr> result = evaluateExpression(op);
    if (failed(result))
      return op;
    return *result;
  });
  return replacer.replace(type);
}
