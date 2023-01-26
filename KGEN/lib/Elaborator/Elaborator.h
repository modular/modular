//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATOR_H
#define KGEN_ELABORATOR_ELABORATOR_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "Support/Compiler/ErrorTree.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

namespace M::KGEN {
struct EvalContext;
class IREvaluator;

//===----------------------------------------------------------------------===//
// New elaborator entry point
//===----------------------------------------------------------------------===//

LogicalResult elaborateGeneratorsV2(mlir::SymbolTableAnalysis &analysis,
                                    LLCL::Runtime &runtime,
                                    ArrayRef<GeneratorOp> primaryGenerators,
                                    bool enableSearch);

//===----------------------------------------------------------------------===//
// ElaboratedGenerator
//===----------------------------------------------------------------------===//

/// This typedef represents a generator declaration + a set of input
/// parameters that provide a complete binding for something that can be
/// resolved.
using DeclAndInputParamsPair = std::pair<DeclInterface, ArrayAttr>;

/// This class keeps track of one result from binding a generator to a set of
/// input parameters.  It holds both the func that gets produced as well as
/// the (transitive) set of generator bindings used to create it.  This is used
/// to ensure that further-derived generators are only elaborated with
/// consistent bindings.
class ElaboratedGenerator {
public:
  explicit ElaboratedGenerator(FuncOp func) : func(func) {}

  /// This is the func that is produced.
  FuncOp func;

  /// These are the bindings used to produce the func.  The results are
  /// transitively flattened, so we don't need to maintain a tree of bindings.
  SmallDenseMap<DeclAndInputParamsPair, FuncOp> bindings;

  /// If we have a binding for the specified generator+InputParamSet, return it,
  /// otherwise return null.
  FuncOp getBinding(DeclAndInputParamsPair key) const;

  /// Return true if the set of bindings in this elaborated func are
  /// consistent with the specified set of bindings.
  bool isConsistentWith(const ElaboratedGenerator &other) const;

  /// Declare that we're resolving the specified `declAndInputParams` to a
  /// specified callee.  The callee is known to have bindings that are
  /// consistent with ours, but may have additional entries to merge in.
  void addBinding(DeclAndInputParamsPair declAndInputParams,
                  const ElaboratedGenerator &newCallee);

  LLVM_DUMP_METHOD void dump() const;

private:
  void addOneBinding(DeclAndInputParamsPair declAndInputParams, FuncOp result);
};

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

class Elaborator {
public:
  /// Initialize the elaborator and its symbol table.
  Elaborator(
      mlir::SymbolTableAnalysis &analysis, TargetInfoAttr target,
      LLCL::Runtime &runtime, LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      bool enableSearch = false)
      : analysis(analysis), target(target), runtime(runtime), asyncMap(map),
        transformCache(std::move(transformCache)),
        regionCache(std::move(regionCache)), enableSearch(enableSearch) {}

  virtual ~Elaborator() = default;

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation.
  virtual ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                      ArrayRef<ParamBindAttr> paramValues) = 0;

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

protected:
  /// This symbol table analysis allows efficient lookups across the module.
  mlir::SymbolTableAnalysis &analysis;

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

  /// Enable search during interface elaboration. This defaults to `false`
  /// because we want search to be opt-in.
  bool enableSearch = false;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
