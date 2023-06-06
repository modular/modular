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
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

namespace M::KGEN {
class IREvaluator;

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

class Elaborator {
public:
  /// Initialize the elaborator and its symbol table.
  Elaborator(SymbolTable &symtab, ParameterCollector::Analysis &paramCache,
             TargetInfoAttr target, EvaluatorExecutorFnRef evaluatorExecutorFn)
      : symtab(symtab), paramCache(paramCache, /*maxNumThreads=*/1),
        target(target), evaluatorExecutorFn(evaluatorExecutorFn) {}

  virtual ~Elaborator() = default;

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation.
  virtual ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, FlatSymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) = 0;

  /// Get all the concrete functions for the given symbol. If the symbol is a
  /// function already, append it to the list and move on, otherwise,
  /// elaborate it and append all the concrete implementations.
  virtual ErrorTreeOrSuccess
  getAllConcreteFunctions(Location loc, FlatSymbolRefAttr symbolRef,
                          ArrayRef<TypedAttr> paramValues,
                          std::vector<FuncOp> &funcs) = 0;

  /// Get the symbol table associated with this instance of the elaborator.
  Shared<SymbolTable &> &getSymbolTable() { return symtab; }
  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() { return target; }

  /// Return the evaluator to use when specializing generators.
  EvaluatorExecutorFnRef getEvaluatorExecutorFn() const {
    return evaluatorExecutorFn;
  }

protected:
  /// This symbol table allows efficient lookups across the module.
  Shared<SymbolTable &> symtab;

  /// This is the cached parameter collector analysis.
  ThreadLocalCache<ParameterCollector::Analysis> paramCache;

  /// The target we are compiling code for.
  TargetInfoAttr target;

  /// The functor used for evaluating generator specializations.
  EvaluatorExecutorFnRef evaluatorExecutorFn;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
