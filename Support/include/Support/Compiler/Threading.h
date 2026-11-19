//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_THREADING_H
#define SUPPORT_COMPILER_THREADING_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ThreadPool.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

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
    return llvm::success();

  // If threading is not enabled, there is a single element, or the threadpool
  // is single-threaded, process the elements sequentially. Pass in the cache
  // directly.
  auto begin = std::begin(range);
  if (!ctx->isMultithreadingEnabled() || numElements == 1 ||
      ctx->getThreadPool().getMaxConcurrency() == 1) {
    for (auto e = std::end(range); begin != e; ++begin)
      if (failed(func(cache, *begin)))
        return llvm::failure();
    return llvm::success();
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
  return llvm::failure(processingFailed);
}

/// This version of the function has a no-op consolidate function.
template <typename RangeT, typename FuncT, typename CacheT>
LogicalResult failableParallelForEach(MLIRContext *ctx, RangeT &&range,
                                      FuncT &&func, CacheT &cache) {
  return failableParallelForEach(ctx, std::forward<RangeT>(range),
                                 std::forward<FuncT>(func), cache,
                                 [](auto &&...) {});
}

/// This version of the function is not failable.
template <typename RangeT, typename FuncT, typename CacheT>
void parallelForEach(MLIRContext *ctx, RangeT &&range, FuncT &&func,
                     CacheT &cache) {
  (void)failableParallelForEach(
      ctx, std::forward<RangeT>(range),
      [&](CacheT &cache, auto &&arg) {
        return func(cache, std::forward<decltype(arg)>(arg)), llvm::success();
      },
      cache, [](auto &&...) {});
}

/// The most chunks that the chunk helpers below split a range into. More chunks
/// than threads keep the load even when a few items cost far more than the
/// rest.
inline constexpr size_t kMaxParallelChunks = 512;

namespace Detail {
template <typename RangeT>
using RangeItemT = std::remove_cv_t<
    std::remove_reference_t<decltype(*std::begin(std::declval<RangeT &>()))>>;

inline size_t getNumParallelChunks(size_t numItems) {
  return std::min(numItems, kMaxParallelChunks);
}
} // namespace Detail

/// Splits `items` into chunks and calls `fn(size_t chunkIndex, ArrayRef<T>
/// chunk)` on each chunk in parallel.
///
/// The split depends only on the number of items, so a result that depends on
/// the chunks is the same at every thread count. With `n` chunks, chunk `c`
/// holds items `c`, `c + n`, `c + 2n` and so on. This interleaving spreads a
/// run of expensive neighbors over many chunks.
///
/// `chunkIndex` is less than `std::min(items.size(), kMaxParallelChunks)`, and
/// no two calls of `fn` that run at once get the same one. A caller can thus
/// keep one state per chunk index across several calls of this function, such
/// as a walker cache.
///
/// `items` must be a contiguous container of values that are cheap to copy.
/// `fn` runs on several threads at once. The diagnostics it emits are ordered
/// by chunk.
template <typename RangeT, typename FnT>
void parallelForEachIndexedChunk(MLIRContext *ctx, RangeT &&items, FnT &&fn) {
  using ItemT = Detail::RangeItemT<RangeT>;
  ArrayRef<ItemT> itemsRef(items);
  size_t numChunks = Detail::getNumParallelChunks(itemsRef.size());
  mlir::parallelFor(ctx, 0, numChunks, [&](size_t chunk) {
    SmallVector<ItemT> chunkItems;
    for (size_t i = chunk; i < itemsRef.size(); i += numChunks)
      chunkItems.push_back(itemsRef[i]);
    fn(chunk, ArrayRef<ItemT>(chunkItems));
  });
}

/// Calls `fn(ArrayRef<T> chunk)` on each chunk as `parallelForEachIndexedChunk`
/// does. State that `fn` keeps for its chunk, such as a replacer cache, is
/// reused across the items of that chunk.
template <typename RangeT, typename FnT>
void parallelForEachChunk(MLIRContext *ctx, RangeT &&items, FnT &&fn) {
  parallelForEachIndexedChunk(ctx, items,
                              [&](size_t, auto chunk) { fn(chunk); });
}

/// Calls `fn` on each chunk as `parallelForEachChunk` does, and returns the
/// value that `fn` returned for each chunk, in chunk order. A caller that
/// merges these values in order gets the same result at every thread count.
///
/// The result type must be default-constructible.
template <typename RangeT, typename FnT>
auto parallelMapChunks(MLIRContext *ctx, RangeT &&items, FnT &&fn) {
  ArrayRef<Detail::RangeItemT<RangeT>> itemsRef(items);
  using ResultT = std::decay_t<std::invoke_result_t<FnT &, decltype(itemsRef)>>;
  SmallVector<ResultT, 0> results(
      Detail::getNumParallelChunks(itemsRef.size()));
  parallelForEachIndexedChunk(
      ctx, itemsRef,
      [&](size_t chunk, auto chunkItems) { results[chunk] = fn(chunkItems); });
  return results;
}

/// Calls `fn(item)` on each item in parallel and returns the results in item
/// order. The result type must be default-constructible.
template <typename RangeT, typename FnT>
auto parallelMap(MLIRContext *ctx, RangeT &&items, FnT &&fn) {
  ArrayRef<Detail::RangeItemT<RangeT>> itemsRef(items);
  using ResultT = std::decay_t<
      std::invoke_result_t<FnT &, const Detail::RangeItemT<RangeT> &>>;
  // `std::vector<bool>` packs its elements, so writes to different elements
  // from different threads would race.
  SmallVector<ResultT, 0> results(itemsRef.size());
  mlir::parallelFor(ctx, 0, itemsRef.size(),
                    [&](size_t i) { results[i] = fn(itemsRef[i]); });
  return results;
}

/// Returns true if `pred(item)` is true for an item. `pred` runs on several
/// threads at once. After a call returns true no new call starts, but calls
/// that are already running finish.
template <typename RangeT, typename PredT>
bool parallelAnyOf(MLIRContext *ctx, RangeT &&items, PredT &&pred) {
  ArrayRef<Detail::RangeItemT<RangeT>> itemsRef(items);
  std::atomic<bool> found = false;
  mlir::parallelFor(ctx, 0, itemsRef.size(), [&](size_t i) {
    if (!found && pred(itemsRef[i]))
      found = true;
  });
  return found;
}

} // namespace M

#endif // SUPPORT_COMPILER_THREADING_H
