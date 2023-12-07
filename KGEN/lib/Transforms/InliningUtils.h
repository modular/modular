//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_TRANSFORMS_INLININGUTILS_H
#define KGEN_LIB_TRANSFORMS_INLININGUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/ThreadPool.h"

namespace M::KGEN {
class CallOp;
class FuncOp;
class KGENCallOpInterface;
class GeneratorOp;

/// Replace the call operation with the given region using values from args for
/// the region inputs.
///
/// The region is inserted into its own scope - either a loop or async execute
/// op (depending on the type of the call). This scope is returned from the
/// function.
std::pair<Operation *, bool> inlineRegion(IRMapping &map,
                                          KGENCallOpInterface call,
                                          Region &region,
                                          bool takeBody = false);

/// Inlining might create trivial loops with a single break at the end. This
/// function cleans it up.
void foldTrivialLoop(Operation *op);

/// Starting from an inlining scope, update debug information as appropriate.
void updateScopeDebugInfoFrom(Operation *scope, bool noDebug);

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

#endif // KGEN_LIB_TRANSFORMS_INLININGUTILS_H
