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

// TODO: delete this in favor of passing in an LLCL runtime to the pass.
static Runtime getDefaultRuntime() {
  return {createLeakCheckAllocator(createMallocAllocator()),
          createSingleThreadWorkQueue(), llvm::StringLiteral(__FILE__)};
}

namespace M::Cache {
#define GEN_PASS_DEF_DEFLATESYMBOLS
#include "Cache/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class DeflateSymbolsPass : public impl::DeflateSymbolsBase<DeflateSymbolsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Runtime rt = getDefaultRuntime();
    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(
        getFilesystemBackend(rt, std::filesystem::path(cacheDir.getValue())));
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
};
} // namespace

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
  using Base::Base;

  void runOnOperation() override {
    Runtime rt = getDefaultRuntime();
    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(
        getFilesystemBackend(rt, std::filesystem::path(cacheDir.getValue())));
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
};
} // namespace
