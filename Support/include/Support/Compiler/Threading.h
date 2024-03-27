//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_THREADING_H
#define SUPPORT_COMPILER_THREADING_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/ThreadPool.h"

namespace M {
/// Invoke the given function on the elements in the provided range
/// asynchronously, if threading is enabled in the MLIR context, otherwise the
/// elements are processed sequentially. This function takes a reference to a
/// cache type that should be sharded to each worker.
template <typename RangeT, typename FuncT, typename CacheT,
          typename ConsolidateCacheFnT>
LogicalResult failableParallelForEach(MLIRContext *ctx, RangeT &&range,
                                      FuncT &&func, CacheT &cache,
                                      ConsolidateCacheFnT &&consolidate) {
  unsigned numElements = llvm::size(range);
  if (!numElements)
    return success();

  // If threading is not enabled, there is a single element, or the threadpool
  // is single-threaded, process the elements sequentially. Pass in the cache
  // directly.
  auto begin = std::begin(range);
  if (!ctx->isMultithreadingEnabled() || numElements == 1 ||
      ctx->getThreadPool().getMaxConcurrency() == 1) {
    for (auto e = std::end(range); begin != e; ++begin)
      if (failed(func(cache, *begin)))
        return failure();
    return success();
  }

  // Otherwise, process the elements in parallel.
  llvm::ThreadPoolInterface &threadPool = ctx->getThreadPool();
  llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
  size_t numActions = std::min(numElements, threadPool.getMaxConcurrency());

  // Each worker gets a copy of the cache.
  std::vector<CacheT> workerCaches(numActions - 1, cache);

  // Build a wrapper processing function that properly initializes a parallel
  // diagnostic handler.
  mlir::ParallelDiagnosticHandler handler(ctx);
  std::atomic<unsigned> curIndex = 0;
  std::atomic<bool> processingFailed = false;
  auto workFn = [&](CacheT &cache) {
    unsigned index;
    while (!processingFailed && (index = curIndex++) < numElements) {
      handler.setOrderIDForThread(index);
      if (failed(func(cache, *std::next(begin, index))))
        processingFailed = true;
      handler.eraseOrderIDForThread();
    }
  };

  // Save 1 copy of the cache.
  tasksGroup.async([&] { workFn(cache); });
  for (CacheT &cache : workerCaches)
    tasksGroup.async([&] { workFn(cache); });
  // If the current thread is a worker thread from the pool, then waiting for
  // the task group allows the current thread to also participate in processing
  // tasks from the group, which avoid any deadlock/starvation.
  tasksGroup.wait();

  // Consolidate the caches.
  consolidate(cache, workerCaches);
  return failure(processingFailed);
}
} // namespace M

#endif // SUPPORT_COMPILER_THREADING_H
