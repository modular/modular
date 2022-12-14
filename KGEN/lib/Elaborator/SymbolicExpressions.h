//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"

namespace M::KGEN {
class FuncOp;
struct EvalError;

using EvalDiagnostic = std::pair<Location, EvalError>;

/// An expression evaluation error is an error with extra notes.
struct EvalError {
  EvalError(Error error) : error(std::move(error)) {}

  /// Create a new error with this error as the cause of it.
  EvalError(Error error, Location loc, EvalError causes);

  /// The main error.
  Error error;
  /// The causes of the main error.
  std::vector<EvalDiagnostic> notes;
};

/// This IR evaluator is a parameter evaluator that can work during elaboration
/// to concretize parameter expressions and compute symbolic parameter
/// expressions, such as `apply` on a symbol constant or `get_sizeof` and
/// `get_alignof` a decl type.
class IREvaluator : public ParameterEvaluator {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(
      SymbolTable &symtab, LLCL::AsyncSideEffectMap &asyncMap,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      DenseMap<StringAttr, Attribute> paramValues =
          DenseMap<StringAttr, Attribute>())
      : ParameterEvaluator(std::move(paramValues)), symtab(symtab),
        asyncMap(asyncMap), regionCache(std::move(regionCache)),
        transformCache(std::move(transformCache)) {}

  IREvaluator(const IREvaluator &eval)
      : ParameterEvaluator(eval), symtab(eval.symtab), asyncMap(eval.asyncMap),
        regionCache(eval.regionCache.copy()),
        transformCache(eval.transformCache.copy()) {}

  IREvaluator &operator=(const IREvaluator &eval) {
    return *(new (this) IREvaluator(eval));
  }

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr>
  evaluateSymbolicExpression(ParamOperatorAttr op) override;

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available). If `allowUnknown`
  /// is set, only unevaluated parameter operators are rejected.
  std::variant<EvalError, Attribute>
  concretizeParameterExpr(Attribute expr, bool allowUnknown = false);
  std::variant<EvalError, Type> concretizeParameterExpr(Type expr);

private:
  Attribute getReboundAttribute(Attribute attr) {
    return ParameterEvaluator::getReboundAttribute(attr);
  }
  Type getReboundType(Type type) {
    return ParameterEvaluator::getReboundType(type);
  }

  /// Evaluate the function with the provided constant inputs.
  std::variant<EvalError, TypedAttr>
  evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs);

  /// The symbol table to lookup symbol references.
  SymbolTable &symtab;

  /// The async map to use for inflating ops.
  LLCL::AsyncSideEffectMap &asyncMap;

  /// The region cache to use for inflating ops.
  LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache;

  /// The transform cache to use for caching interpretation.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The function to use to emit an error.
  std::function<void(EvalError)> emitError;
};

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
Optional<EvalDiagnostic>
evaluateConstraints(ArrayRef<ConstraintAttr> constraints,
                    IREvaluator &evaluator);

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
Optional<EvalDiagnostic>
evaluateConstraints(DeclInterface decl, ArrayRef<Attribute> inputParamValues,
                    IREvaluator &evaluator);

} // namespace M::KGEN
