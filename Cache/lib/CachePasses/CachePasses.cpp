//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachePasses/CachePasses.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "Cache/CacheDialect/CacheOps.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// DeflateSymbolsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_DEFLATESYMBOLS
#include "Cache/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class DeflateSymbolsPass : public impl::DeflateSymbolsBase<DeflateSymbolsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  DeflateSymbolsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    if (!runtime) {
      new (runtime) Runtime(createLeakCheckAllocator(createMallocAllocator()),
                            createSingleThreadWorkQueue());
    }

    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(getFilesystemBackend(
        *runtime, std::filesystem::path(cacheDir.getValue())));
    // Deflate each symbol.
    SmallVector<AnyAsyncValueRef> results;
    for (auto &op : getOperation()) {
      if (!op.hasAttr(SymbolTable::getSymbolAttrName()))
        continue;

      results.push_back(deflateOp(&op, cache));
    }

    await(results);
    for (auto &r : results)
      if (failed(r->get<LogicalResult>()))
        signalPassFailure();
  }

private:
  Runtime *runtime;
};
} // namespace

std::unique_ptr<mlir::Pass> M::Cache::createDeflateSymbolsPass(Runtime &rt) {
  return std::make_unique<DeflateSymbolsPass>(rt);
}

//===----------------------------------------------------------------------===//
// InflateSymbolsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_INFLATESYMBOLS
#include "Cache/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class InflateSymbolsPass : public impl::InflateSymbolsBase<InflateSymbolsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  InflateSymbolsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    if (!runtime) {
      new (runtime) Runtime(createLeakCheckAllocator(createMallocAllocator()),
                            createSingleThreadWorkQueue());
    }

    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(getFilesystemBackend(
        *runtime, std::filesystem::path(cacheDir.getValue())));
    // Inflate each deflated op.
    SmallVector<AnyAsyncValueRef> results;
    for (auto &sym : getOperation()) {
      if (!sym.hasAttr(getRegionHashAttrName()))
        continue;

      results.push_back(inflateOp(&sym, cache));
    }

    await(results);
    for (auto &r : results)
      if (failed(r->get<LogicalResult>()))
        signalPassFailure();
  }

private:
  Runtime *runtime;
};
} // namespace

std::unique_ptr<mlir::Pass> M::Cache::createInflateSymbolsPass(Runtime &rt) {
  return std::make_unique<InflateSymbolsPass>(rt);
}

void M::Cache::registerCachePasses(Runtime &rt) {
  // Register the passes with the correct constructor - one that takes the
  // runtime as an argument.
  mlir::registerPass([&]() { return Cache::createDeflateSymbolsPass(rt); });
  mlir::registerPass([&]() { return Cache::createInflateSymbolsPass(rt); });
}
