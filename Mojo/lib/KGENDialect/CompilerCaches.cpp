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

#include "Mojo/KGENDialect/CompilerCaches.h"

#include <atomic>

using namespace M;
using namespace KGEN;

namespace {
struct Registry {
  std::shared_mutex mutex;
  /// Weak, so the caches go away with their last owner.
  llvm::DenseMap<mlir::MLIRContext *, std::weak_ptr<CompilerCaches>> caches;
  /// Bumped on every change so threads can reuse their last lookup.
  std::atomic<uint64_t> generation{1};
};

/// Intentionally never destroyed, so lookups during shutdown stay valid.
Registry &getRegistry() {
  static Registry *registry = new Registry;
  return *registry;
}

/// Drops the entry of a context whose caches are gone. A newer instance may
/// already have replaced it, and that one stays.
void forget(mlir::MLIRContext *context) {
  Registry &registry = getRegistry();
  std::unique_lock<std::shared_mutex> lock(registry.mutex);
  auto it = registry.caches.find(context);
  if (it != registry.caches.end() && it->second.expired())
    registry.caches.erase(it);
  registry.generation.fetch_add(1, std::memory_order_release);
}
} // namespace

CompilerCaches *CompilerCaches::get(mlir::MLIRContext *context) {
  // One-entry per-thread cache in front of the locked registry lookup.
  // `lastContext` is the key: a process can have several contexts (the LSP
  // server, tests), and a result applies only to the context it was found for.
  // `lastGeneration` is the registry version it was found at: every creation
  // and release bumps it, so a changed registry invalidates the saved result.
  thread_local mlir::MLIRContext *lastContext = nullptr;
  thread_local CompilerCaches *lastCaches = nullptr;
  thread_local uint64_t lastGeneration = 0;
  Registry &registry = getRegistry();
  uint64_t generation = registry.generation.load(std::memory_order_acquire);
  if (lastContext == context && lastGeneration == generation)
    return lastCaches;
  std::weak_ptr<CompilerCaches> weak;
  {
    std::shared_lock<std::shared_mutex> lock(registry.mutex);
    weak = registry.caches.lookup(context);
    lastGeneration = registry.generation.load(std::memory_order_acquire);
  }
  // Locked outside the registry: if this were the last owner, its release
  // takes the registry lock.
  lastCaches = weak.lock().get();
  lastContext = context;
  return lastCaches;
}

std::shared_ptr<CompilerCaches>
CompilerCaches::getOrCreate(mlir::MLIRContext *context) {
  Registry &registry = getRegistry();
  std::unique_lock<std::shared_mutex> lock(registry.mutex);
  std::weak_ptr<CompilerCaches> &slot = registry.caches[context];
  if (std::shared_ptr<CompilerCaches> existing = slot.lock())
    return existing;
  std::shared_ptr<CompilerCaches> created(new CompilerCaches,
                                          [context](CompilerCaches *caches) {
                                            forget(context);
                                            delete caches;
                                          });
  slot = created;
  registry.generation.fetch_add(1, std::memory_order_release);
  return created;
}
