//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_INLININGUTILS_H
#define KGEN_TRANSFORMUTILS_INLININGUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ThreadPool.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::KGEN {
class CallOp;
class FuncOp;
class KGENCallOpInterface;
class GeneratorOp;
class SourceLocOp;

/// Replace the call operation with the given region using values from args for
/// the region inputs.
///
/// The region is inserted into its own scope - either a loop or async execute
/// op (depending on the type of the call). This scope is returned from the
/// function.
std::pair<Operation *, bool> inlineRegion(mlir::RewriterBase &b, IRMapping &map,
                                          Operation *call, Region &region,
                                          bool takeBody = false);

/// Decrement the counter of a SourceLocOp and lower it to file, line, and
/// column constants (extracted from the given call location) if needed.
void processSourceLocOp(SourceLocOp sourceLocOp, Location callLoc,
                        mlir::RewriterBase &b);

/// Inlining might create trivial loops with a single break at the end. This
/// function cleans it up.
void foldTrivialLoop(Operation *op);

/// Starting from an inlining scope, update debug information as appropriate and
/// fold the scope if requested. Recurse on nested scopes.
void updateScopeDebugInfoFrom(Operation *scope, IntegerAttr tag,
                              StringAttr updateAttrName);

/// Given a function, find the top-level scopes and start processing debug info
/// from there.
void updateScopeDebugInfo(FuncOp func, StringAttr updateAttrName);

/// After inlining a region, update its debuginfo if required.
void maybeUpdateDebugInfo(Operation *scope,
                          std::optional<StringAttr> updateAttrName,
                          bool singleExit);

/// This class manages a pass manager instance for each thread.
class PerThreadPassManagers {
public:
  explicit PerThreadPassManagers(
      MLIRContext *ctx,
      function_ref<void(mlir::OpPassManager &)> buildFuncPasses);

  /// Get the pass manager for the current thread, initializing it if one does
  /// not exist.
  mlir::PassManager &getPassManager();

private:
  /// The MLIR context.
  MLIRContext *ctx;
  /// The functor to populate the passes.
  function_ref<void(mlir::OpPassManager &)> buildFuncPasses;
  /// The pass managers for each thread.
  DenseMap<uint64_t, std::unique_ptr<mlir::PassManager>> pms;
  /// The mutex guarding the per-thread pass managers map.
  llvm::sys::SmartRWMutex<true> mutex;
};

/// Get number of operations in this function.
uint64_t getNumOperations(Operation *op);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_INLININGUTILS_H
