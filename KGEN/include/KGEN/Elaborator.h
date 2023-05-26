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

/// This function provides support for executing a given specialization
/// evaluator, and returning either the index of the best specialization, or
/// error.
using EvaluatorExecutorFn = std::function<ErrorOr<size_t>(
    KGEN::FuncOp evaluator, const SymbolTable &symtab, TargetInfoAttr target,
    ArrayRef<KGEN::FuncOp> specializations)>;
using EvaluatorExecutorFnRef = function_ref<ErrorOr<size_t>(
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
