//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_H
#define KGEN_ELABORATOR_H

#include "KGEN/KGENDialect/KGENParameters.h"
#include <filesystem>

namespace mlir {
class SymbolTableAnalysis;
} // namespace mlir

namespace M {
class TargetInfoAttr;
namespace LLCL {
class Runtime;
} // namespace LLCL
namespace KGEN {
class GeneratorOp;
class FuncOp;
} // namespace KGEN

/// This function kind represents a callback to invoke a compiled evaluator
/// function with the compiled candidate functions. This function performs the
/// actual benchmarking of search and must be invoked in isolation. The
/// elaborator ensures that the compiler process is quiet before invoking this
/// function, which is required for stable and accurate results.
using ElaboratorSearchFn = llvm::unique_function<ErrorOr<ssize_t>()>;

/// This function kind represents a callback given the IR for an evaluator
/// function and a list of candidate functions and should perform all necessary
/// JIT compilation on those functions, in preparation for search. The function
/// should return a search execute function, which the elaborator then
/// guarantees executes in isolation.
using EvaluatorExecutorFn = std::function<ErrorOr<ElaboratorSearchFn>(
    KGEN::FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<KGEN::FuncOp>)>;
using EvaluatorExecutorFnRef = function_ref<ErrorOr<ElaboratorSearchFn>(
    KGEN::FuncOp, const SymbolTable &, TargetInfoAttr, ArrayRef<KGEN::FuncOp>)>;

/// Elaborator config.
struct ElaboratorConfig {
  /// Enable search during interface elaboration. This defaults to `false`
  /// because we want search to be opt-in.
  bool enableSearch;
  /// If this is true, emit diagnostics for certain conditions that are
  /// interesting to test for.
  bool testDiagnostics;
  /// The maximum instantiation depth before the elaborator gives up.
  unsigned maxDepth;
};

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.  On error, diagnostics are emitted and the
/// primary file isn't completely lowered.
LogicalResult elaborateGenerators(
    mlir::SymbolTableAnalysis &symtab,
    KGEN::ParameterCollector::Analysis &paramCache, LLCL::Runtime &runtime,
    TargetInfoAttr target, ArrayRef<KGEN::GeneratorOp> generators,
    EvaluatorExecutorFnRef evaluatorExecutorFn, const ElaboratorConfig &config);

} // namespace M

#endif // KGEN_ELABORATOR_H
