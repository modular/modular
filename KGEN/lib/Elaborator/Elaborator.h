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

namespace M::KGEN {
struct EvalContext;
class IREvaluator;

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
      SymbolTableAnalysis &analysis, TargetInfoAttr target,
      LLCL::Runtime &runtime, LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      bool enableSearch = false)
      : analysis(analysis), target(target), runtime(runtime), asyncMap(map),
        transformCache(std::move(transformCache)),
        regionCache(std::move(regionCache)), enableSearch(enableSearch) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Return the operation that defines the specified symbol.
  FuncInterface lookupCallee(SymbolRefAttr symbolRef);

  /// Return all instantiations of the specified declaration (a func,
  /// generator, or interface) with the specified input parameter values.
  ArrayRef<ErrorTreeOr<ElaboratedGenerator>>
  getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                       size_t expansionDepth, EvalContext &evalCtx);

  /// Insert a variant of an existing func into the primary file.
  void insertFuncVariant(FuncOp existing, FuncOp newFunc);

  /// Indicate that a function should be removed from the module at the end of
  /// elaboration. These functions are either invalid instantiations or
  /// inlined.
  void markFuncForRemoval(FuncOp func) { funcsToRemove.insert(func); }

  /// Return true if the function was invalid or inlined and should be removed
  /// from the module.
  bool shouldRemoveFunc(FuncOp func) { return funcsToRemove.contains(func); }

  /// Returns true if search is enabled.
  bool isSearchEnabled() { return enableSearch; }

  ArrayRef<GeneratorOp> getGeneratorsImplementing(GeneratorInterfaceOp itf) {
    auto it = interfaceImpls.find(itf.getNameAttr());
    return it == interfaceImpls.end() ? ArrayRef<GeneratorOp>() : it->second;
  }

  const DenseMap<GeneratorOp, FuncOp> &
  getFirstConcreteFuncForGenerator() const {
    return firstConcreteFuncForGenerator;
  }

  /// Bind the result parameters of a fully-specialized function and clear them
  /// from the function.
  void bindResultParameters(FuncOp func);

  /// Lookup bound result parameters of a function.
  ParameterExprArrayAttr lookupResultParameters(FuncOp func) {
    auto it = resultParams.find(func);
    assert(it != resultParams.end() && "results parameters not bound");
    return it->second;
  }

  /// Set the evaluation context of a region body.
  void setEvalContext(SymbolRefAttr ref, EvalContext evalCtx);

  /// Get the evaluation context of the base symbol reference, or set it to the
  /// default context.
  EvalContext &getEvalContext(SymbolRefAttr ref);

  /// Instantiate a new evaluator with the given parameters.
  IREvaluator createEvaluator(DenseMap<StringAttr, Attribute> values =
                                  DenseMap<StringAttr, Attribute>());

  /// Get the symbol table analysis.
  SymbolTableAnalysis &getAnalysis() const { return analysis; }

private:
  /// Specialize a func body, generating one variant or each viable
  /// instantiation of that body.  Funcs do not have input parameters, but
  /// they can invoke interfaces etc which can cause them to produce multiple
  /// variants.
  ///
  /// SourceModule indicates which module in the included library this
  /// originally came from (likely not the primary module).
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeFunc(FuncOp func, ModuleOp sourceModule, size_t expansionDepth,
                 EvalContext &evalCtx);

  /// Specialize a generator with the specified input parameters and return
  /// the generated func.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                      size_t expansionDepth, EvalContext &evalCtx);

  /// Specialize a generator interface with the specified input parameters and
  /// return the generated func.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeInterface(DeclAndInputParamsPair declAndInputParams,
                      size_t expansionDepth, EvalContext &evalCtx);

  /// Report an error given an interface and an error string - just reduces
  /// boilerplate around CalleeExpansionError creation.
  ErrorTreeOr<ElaboratedGenerator>
  reportCalleeExpansionError(GeneratorInterfaceOp itf, Twine err) {
    return ErrorTree(itf.getLoc(), err);
  };

  /// This symbol table analysis allows efficient lookups across the module.
  SymbolTableAnalysis &analysis;

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

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;

  /// This map contains bindings for result parameteres from specialized
  /// functions.
  DenseMap<FuncOp, ParameterExprArrayAttr> resultParams;

  /// This is a cache of already-instantiated declarations.  The key is the
  /// generator/interface and input parameters, the result are all-possible
  /// funcs that could be generated from this.
  DenseMap<DeclAndInputParamsPair,
           SmallVector<ErrorTreeOr<ElaboratedGenerator>>>
      generatedFuncs;

  /// This keeps tracks the evaluation context of region bodies. It keeps a flag
  /// of whether the region is isolated from above (and thus all nodes along the
  /// callgraph down to the callsite need to be inlined) and the parameter
  /// context.
  DenseMap<SymbolRefAttr, EvalContext> evaluationContext;

  /// This map keeps track of the first func that a generator with no
  /// parameters expanded into.  We rename it to have the same symbol as the
  /// original generator in a post-pass.
  DenseMap<GeneratorOp, FuncOp> firstConcreteFuncForGenerator;

  /// This tracks generated functions that should be removed after
  /// elaboration. These functions were either inlined by a parameter rewriter
  /// or are malformed. These functions need to be cleaned up at the end of
  /// the pass.
  DenseSet<FuncOp> funcsToRemove;

  /// Enable search during interface elaboration. This defaults to `false`
  /// because we want search to be opt-in.
  bool enableSearch = false;

  /// Allow the evaluator to access elaborator internals.
  friend class IREvaluator;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
