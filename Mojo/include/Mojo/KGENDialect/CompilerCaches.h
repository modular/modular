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

#ifndef KGEN_COMPILERCACHES_H
#define KGEN_COMPILERCACHES_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <tuple>

namespace M::KGEN {

/// Thread-safe cache from `KeyT` to the opaque pointer of an attribute or type.
/// Entries are pure functions of uniqued values, so they never go stale.
template <typename KeyT>
class ShardedCache {
public:
  /// Returns null on a miss.
  const void *lookup(const KeyT &key) {
    Shard &shard = getShard(key);
    std::shared_lock<std::shared_mutex> lock(shard.mutex);
    auto it = shard.map.find(key);
    return it == shard.map.end() ? nullptr : it->second;
  }

  /// Records `value` for `key`; the first entry wins when threads race.
  void insert(const KeyT &key, const void *value) {
    Shard &shard = getShard(key);
    std::unique_lock<std::shared_mutex> lock(shard.mutex);
    shard.map.try_emplace(key, value);
  }

private:
  // Sharded to avoid one global lock.
  static constexpr size_t numShards = 64;
  struct Shard {
    std::shared_mutex mutex;
    llvm::DenseMap<KeyT, const void *> map;
  };

  Shard &getShard(const KeyT &key) {
    return shards[llvm::DenseMapInfo<KeyT>::getHashValue(key) % numShards];
  }

  std::array<Shard, numShards> shards;
};

/// `IndexDepthAdjuster` results, keyed by the value, `adjustDepth` and
/// `onlyAdjustIndexRef`.
using DepthAdjustCache =
    ShardedCache<std::tuple<const void *, int64_t, unsigned>>;

/// Caches for pure helpers of the parameter machinery. Every parser state on an
/// `MLIRContext` shares one instance, owned through the `shared_ptr` that
/// `getOrCreate` returns; it lives until the last owner releases it. Helpers
/// that only have the context find it with `get`, and compute without caching
/// when no owner is alive.
class CompilerCaches {
public:
  /// Returns the caches of `context`, or null if no owner is alive.
  static CompilerCaches *get(mlir::MLIRContext *context);

  /// Returns the caches shared by all owners on `context`, creating them for
  /// the first one.
  static std::shared_ptr<CompilerCaches>
  getOrCreate(mlir::MLIRContext *context);

  /// Canonical forms, keyed by the value's opaque pointer.
  ShardedCache<const void *> canonical;
  DepthAdjustCache depthAdjust;

private:
  CompilerCaches() = default;
};

} // namespace M::KGEN

#endif // KGEN_COMPILERCACHES_H
