//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/Interpreter/InterpreterState.h"
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

namespace {
/// This class is an implementation of IRInterpreter that knows about parser
/// state for function lookup.
class ParserInterpreter : public IRInterpreter {
public:
  ParserInterpreter(DeclResolver &resolver)
      : IRInterpreter(resolver.getContext()), resolver(resolver) {}

  /// Lookup the body of the referenced function using the DeclResolver.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;
  /// Lookup the body of a reference type using the DeclResolver.
  Operation *lookupTypeDefinition(SymbolRefAttr symbol) override;

  DeclResolver &resolver;
};
} // namespace

ErrorOr<Region *> ParserInterpreter::lookupFunctionBody(SymbolRefAttr symbol) {
  ASTDecl *decl = resolver.getDeclForFuncSymbol(symbol);
  if (!decl)
    return Error("function not found: " + mlir::debugString(symbol));

  // Fail if the function is parameterized.
  if (failed(resolver.resolveSignature(*decl, decl->getLoc())))
    return Error("failed to resolve function signature");

  auto func = cast<FnOp>(*decl);
  if (func.getInlineLevel() == InlineLevel::Automatic ||
      func.getInlineLevel() == InlineLevel::Never ||
      func.getInlineLevel() == InlineLevel::Always)
    return Error("function is not always_inline");
  FnTypeGeneratorType fullSig = func.getFullSignature();
  if (!fullSig.getInputParamTypes().empty())
    return Error("function is parametric");

  // Make sure to fully resolve the body and everything within it.
  if (failed(resolver.resolveFully(*decl, decl->getLoc())))
    return Error("failed to fully resolve function");
  return &func.getBodyRegion();
}

Operation *ParserInterpreter::lookupTypeDefinition(SymbolRefAttr symbol) {
  ASTDecl &decl = resolver.getDeclForTypeSymbol(symbol);
  if (failed(resolver.resolveFully(decl, decl.getLoc())))
    return {};
  return decl.getIfOperation();
}

//===----------------------------------------------------------------------===//
// ParserParamEvaluator
//===----------------------------------------------------------------------===//

ParserParamEvaluator::ParserParamEvaluator(DeclResolver &resolver,
                                           ArrayRef<ParamDeclAttr> paramDecls,
                                           ArrayRef<TypedAttr> paramValues)
    : ParameterEvaluator(paramDecls, paramValues), resolver(resolver) {}

ParserParamEvaluator::ParserParamEvaluator(DeclResolver &resolver,
                                           ArrayRef<TypedAttr> paramValues)
    : ParameterEvaluator(paramValues), resolver(resolver) {}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateFunctionCall(SymbolRefAttr symbol,
                                           ArrayRef<Attribute> arguments) {
  auto &map = resolver.shared.getInterpreterCache().interpCache;
  ParserInterpreterCache::Key key{
      symbol, ArrayAttr::get(symbol.getContext(), arguments)};

  if (auto it = map.find(key); it != map.end())
    return it->second;

  // `evaluateFunctionCallImpl` can invalidate the iterator.
  FailureOr<TypedAttr> result = evaluateFunctionCallImpl(symbol, arguments);
  map.try_emplace(key, result);
  return result;
}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateFunctionCallImpl(SymbolRefAttr symbol,
                                               ArrayRef<Attribute> arguments) {
  // Use the interpreter to execute the function call.
  ParserInterpreter interpreter(resolver);
  ErrorOr<Region *> bodyOr = interpreter.lookupFunctionBody(symbol);
  if (bodyOr.isError()) {
    // Swallow the error.
    DEBUG_WITH_TYPE("lit-parameter-evaluator",
                    llvm::errs() << "[ParserParamEvaluator] "
                                 << bodyOr.getError() << "\n");
    return failure();
  }
  Region &body = **bodyOr;
  FnTypeGeneratorType sig =
      cast<FnOp>(body.getParentOp()).getFuncTypeGenerator();

  TypedAttr value;
  if (sig.hasMemoryOnlyResult()) {
    Value resultArg = body.getArguments().back();
    Type resultType = cast<RefType>(resultArg.getType()).getElementType();
    ErrorOr<TypedAttr> init =
        createUninitializedValueOf(resultType, interpreter);
    if (init.isError())
      return failure();

    ErrorTreeOr<TypedAttr> result =
        interpreter.executeRegionWithResultSlot(body, arguments, *init);
    if (result.isError()) {
      // Swallow the error.
      DEBUG_WITH_TYPE("lit-parameter-evaluator",
                      llvm::errs() << "[ParserParamEvaluator] "
                                   << bodyOr.getError() << "\n");
      return failure();
    }
    value = result.takeValue();
  } else {
    ErrorTreeOr<SmallVector<Attribute>> result =
        interpreter.executeRegion(body, arguments);
    if (result.isError()) {
      // Swallow the error.
      DEBUG_WITH_TYPE(
          "lit-parameter-evaluator",
          result.takeError().emit(
              (InFlightDiagnostic(*)(Location))mlir::emitError, "called from"));
      return failure();
    }
    value = cast<TypedAttr>(result->front());
  }
  return value;
}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateExpression(ParamOperatorAttr op) {
  if (op.getOpcode() != POC::Apply && op.getOpcode() != POC::ApplyResultSlot)
    return failure();

  // We can only fold direct calls.
  SymbolConstantAttr ref = dyn_cast<SymbolConstantAttr>(
      ParamOperatorAttr::stripRebind(op.getOperands().front()));
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

Type ParserParamEvaluator::refine(Type type) { return refineImpl(type); }

Attribute ParserParamEvaluator::refine(Attribute attr) {
  return refineImpl(attr);
}

template <typename T>
T ParserParamEvaluator::refineImpl(T arg) {
  if (auto it = refineCache.find(arg.getAsOpaquePointer());
      it != refineCache.end())
    return T::getFromOpaquePointer(it->second);

  // Refine starting from the leaves, so we visit apply expressions sooner and
  // fold them along the way. `AttrTypeReplacer` visits in the opposite
  // direction.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  bool changed = false;
  arg.walkImmediateSubElements(
      [&](Attribute attr) {
        newAttrs.push_back(refine(attr));
        changed |= newAttrs.back() != attr;
      },
      [&](Type type) {
        newTypes.push_back(refine(type));
        changed |= newTypes.back() != type;
      });
  T value = arg;
  if (changed)
    value = arg.replaceImmediateSubElements(newAttrs, newTypes);
  if constexpr (std::is_same_v<Attribute, T>)
    if (auto op = dyn_cast<ParamOperatorAttr>(value))
      if (FailureOr<TypedAttr> result = evaluateExpression(op);
          succeeded(result))
        value = *result;

  refineCache.try_emplace(arg.getAsOpaquePointer(), value.getAsOpaquePointer());
  return value;
}
