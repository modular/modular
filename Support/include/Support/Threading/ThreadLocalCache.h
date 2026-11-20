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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"
#include <concepts>
#include <cstdint>
#include <memory>

#ifndef SUPPORT_THREADING_THREADLOCALCACHE_H
#define SUPPORT_THREADING_THREADLOCALCACHE_H

namespace M {

/// A cache type can define `static CacheT makeThreadCache(const CacheT &)` to
/// create the cache each thread of a `ThreadLocalCache` starts with, for
/// example as a layer that reads through to the original. Otherwise each
/// thread starts with a copy of the original.
template <typename CacheT>
concept HasThreadCacheFactory = requires(const CacheT &original) {
  { CacheT::makeThreadCache(original) } -> std::same_as<CacheT>;
};

//===----------------------------------------------------------------------===//
// ThreadLocalCache
//===----------------------------------------------------------------------===//

/// This class manages a set of thread-local caches. This class is useful when
/// an in-memory analysis cache needs to be shared across multiple threads,
/// where each thread needs to update the cache (deterministically), but it is
/// more efficient to let threads mutate their own local caches, which can be
/// merged back into the main cache. See `HasThreadCacheFactory` for how each
/// thread's cache is created.
template <typename CacheT>
class ThreadLocalCache {
public:
  /// Construct a thread-local cache manager given the expected number of
  /// threads that access the cache.
  explicit ThreadLocalCache(CacheT &original, unsigned expectedNumThreads)
      : original(original) {
    threadCaches.reserve(expectedNumThreads);
  }

  /// Get the cache instance for the current thread.
  CacheT &getThreadLocalCache() {
    int64_t threadId = llvm::get_threadid();
    {
      llvm::sys::SmartScopedReader<true> lock(mutex);
      if (auto it = threadCaches.find(threadId); it != threadCaches.end())
        return *it->second;
    }
    llvm::sys::SmartScopedWriter<true> lock(mutex);
    return *threadCaches.try_emplace(threadId, createThreadCache())
                .first->second;
  }

  /// Consolidate the thread-local caches back into the main cache instance, if
  /// desired.
  void consolidate(llvm::function_ref<void(CacheT &, const CacheT &)> unionFn) {
    for (const std::unique_ptr<CacheT> &cache :
         llvm::make_second_range(threadCaches))
      unionFn(original, *cache);
  }

private:
  /// Returns the cache a thread starts with: `CacheT::makeThreadCache` if the
  /// cache type defines it, otherwise a copy of `original`.
  std::unique_ptr<CacheT> createThreadCache() const {
    if constexpr (HasThreadCacheFactory<CacheT>)
      return std::make_unique<CacheT>(CacheT::makeThreadCache(original));
    else
      return std::make_unique<CacheT>(original);
  }

  /// The original cache that each thread's cache starts from.
  CacheT &original;
  /// Thread-local instances of the cache.
  llvm::DenseMap<uint64_t, std::unique_ptr<CacheT>> threadCaches;
  /// The mutex guarding the thread-local cache instances.
  llvm::sys::SmartRWMutex<true> mutex;
};

} // namespace M

#endif // SUPPORT_THREADING_THREADLOCALCACHE_H
