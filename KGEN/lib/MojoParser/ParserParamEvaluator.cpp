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
  return Error("not interpreting functions anymore");
}

Operation *ParserInterpreter::lookupTypeDefinition(SymbolRefAttr symbol) {
  return {};
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
  return failure();
}

FailureOr<TypedAttr>
ParserParamEvaluator::evaluateExpression(ParamOperatorAttr op) {
  return failure();
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
