//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATOR_H
#define KGEN_ELABORATOR_ELABORATOR_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/Elaborator.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "Support/Compiler/ErrorTree.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

namespace M::KGEN {
class IREvaluator;

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

class Elaborator {
public:
  /// Initialize the elaborator and its symbol table.
  Elaborator(
      mlir::SymbolTableAnalysis &analysis,
      ParameterCollector::Analysis &paramCache, TargetInfoAttr target,
      LLCL::Runtime &runtime, LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      EvaluatorExecutorFnRef evaluatorExecutorFn)
      : analysis(analysis), paramCache(paramCache), target(target),
        runtime(runtime), asyncMap(map),
        transformCache(std::move(transformCache)),
        regionCache(std::move(regionCache)),
        evaluatorExecutorFn(evaluatorExecutorFn) {}

  virtual ~Elaborator() = default;

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation.
  virtual ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) = 0;

  /// Get all the concrete functions for the given symbol. If the symbol is a
  /// function already, append it to the list and move on, otherwise,
  /// elaborate it and append all the concrete implementations.
  virtual std::optional<ErrorTree>
  getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                          ArrayRef<TypedAttr> paramValues,
                          std::vector<FuncOp> &funcs) = 0;

  /// Get the SymbolTableAnalysis object associated with this instance of the
  /// elaborator.
  mlir::SymbolTableAnalysis &getAnalysis() { return analysis; }
  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() { return target; }

  /// Inflate the provided function.
  ErrorOrSuccess inflateFunc(FuncOp func) {
    asyncMap.mapChained(func, [&](LLCL::AnyAsyncValueRef ch) {
      return Cache::inflateOp(func, regionCache.copy(), std::move(ch));
    });
    return asyncMap.await(func);
  }

  /// Return the LLCL runtime.
  LLCL::Runtime &getRuntime() { return runtime; }

  /// Return the evaluator to use when specializing generators.
  EvaluatorExecutorFnRef getEvaluatorExecutorFn() const {
    return evaluatorExecutorFn;
  }

protected:
  /// This symbol table analysis allows efficient lookups across the module.
  mlir::SymbolTableAnalysis &analysis;

  /// This is the cached parameter collector analysis.
  ParameterCollector::Analysis &paramCache;

  /// The target we are compiling code for.
  TargetInfoAttr target;

  /// This provides a runtime reference for the Elaborator and all its
  /// functionality.
  LLCL::Runtime &runtime;

  /// This gives us a map of operation -> in-flight side effect. This is
  /// important because we do async mutations on the IR and we may need await
  /// those mutations to materialize.
  LLCL::AsyncSideEffectMap &asyncMap;

  /// These are the caches the Elaborator will use to run its operations.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;
  LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache;

  /// The functor used for evaluating generator specializations.
  EvaluatorExecutorFnRef evaluatorExecutorFn;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
