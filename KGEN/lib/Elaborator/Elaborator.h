//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATOR_H
#define KGEN_ELABORATOR_ELABORATOR_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "IREvaluator.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
std::string mangleParameterValues(GeneratorOp generator,
                                  ArrayRef<TypedAttr> inputParamValues);

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

using EvaluatorExecutorFnRef = function_ref<ErrorOr<ElaboratorSearchFn>(
    FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<FuncOp>)>;
using ElaboratorCompileAsmFnRef = function_ref<ErrorOr<CrossDeviceFunction>(
    GeneratorOp, SymbolConstantAttr, StringAttr, const SymbolTable &,
    TargetInfoAttr, EmissionKind)>;

class Elaborator {
public:
  /// Enumeration of the compile assembly format.
  enum ASMFormat : uint8_t { ASM, LLVM };

  /// Initialize the elaborator and its symbol table.
  Elaborator(SymbolTable &symtab, TargetInfoAttr target,
             const ElaborateGeneratorsOptions &config)
      : symtab(symtab), target(target), config(config) {}

  virtual ~Elaborator() = default;

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation. Return none if the specialization is not ready yet.
  virtual std::optional<ErrorTreeOr<FuncOp>>
  getConcreteFunction(ImplNode *parent, Location loc,
                      FlatSymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) = 0;

  /// Get all the concrete functions for the given symbol. If the symbol is a
  /// function already, append it to the list and move on, otherwise,
  /// elaborate it and append all the concrete implementations.
  virtual std::optional<ErrorTreeOrSuccess> getAllConcreteFunctions(
      ImplNode *parent, Location loc, FlatSymbolRefAttr symbolRef,
      ArrayRef<TypedAttr> paramValues, std::vector<FuncOp> &funcs) = 0;

  /// Get the functor for compiling a generator to assembly.
  virtual ElaboratorCompileAsmFnRef
  getCompileAsmFn(ASMFormat format = ASMFormat::ASM) const = 0;

  /// Add an owned function operation that should be appended to the module at
  /// the end of elaboration. This is where generated functions during
  /// elaboration should go.
  virtual void addDeferredFunction(OwningOpRef<FuncOp> func) = 0;

  /// Get the symbol table associated with this instance of the elaborator.
  Shared<SymbolTable &> &getSymbolTable() { return symtab; }
  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() const { return target; }
  /// Get the elaborator config.
  const ElaborateGeneratorsOptions &getOptions() const { return config; }

protected:
  /// This symbol table allows efficient lookups across the module.
  Shared<SymbolTable &> symtab;

  /// The target we are compiling code for.
  TargetInfoAttr target;

  /// The elaborator config.
  ElaborateGeneratorsOptions config;
};

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

/// This struct represents the expansion of a callgraph during elaboration.
struct ExpansionGraph {
  ExpansionGraph(LLCL::Runtime &runtime)
      : worklistCh(LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime)) {}

  virtual ~ExpansionGraph();

  /// Increment the task count.
  void didAddTask();

  /// Decrement the task count.
  void didCompleteTask();

  /// Wait on all outstanding tasks.
  AsyncValueRef<Chain> quiesce();

  /// Map from generator instantiation to expansion tree node.
  Shared<DenseMap<std::pair<ParameterExprArrayAttr, GeneratorOp>,
                  std::unique_ptr<ParamNode>>>
      nodes;

  /// Map from concrete function to implementation node.
  Shared<DenseMap<FuncOp, ImplNode *>> concreteNodes;

  /// The current number of tasks scheduled anywhere in the elaborator on the
  /// worklist.
  std::atomic<size_t> numWorkItems = 1;
  /// This chain is signalled when all active work items have completed. This is
  /// used to starve the workqueue before running evaluators, because evaluation
  /// cannot be reliably performed while the compiler is doing work on other
  /// threads.
  LLCL::AsyncValueRef<LLCL::Chain> worklistCh;

  /// Concrete functions added directly to the expansion graph.
  std::vector<std::unique_ptr<ImplNode>> elaboratedNodes;

  /// Protect access to quiesceChain.
  std::mutex quiesceMu;

  /// Protect access to worklistChain.
  std::mutex worklistMu;

  /// Number of outstanding resources created from this runtime.
  size_t numOutstandingResources = 0;

  /// If quiesce() has been called, the chain it returned. Otherwise null.
  AsyncValueRef<Chain> quiesceChain;

  /// Get or create the node for a generator instantiation.
  ParamNode *getOrCreate(LLCL::Runtime &runtime, ParameterExprArrayAttr values,
                         GeneratorOp gen, size_t depth);
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
