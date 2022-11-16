//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CachePasses/CachePasses.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "Support/CacheDialect/CacheOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>

using namespace M;
using namespace Cache;

//===----------------------------------------------------------------------===//
// DeflateSymbolsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_DEFLATESYMBOLS
#include "Support/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class DeflateSymbolsPass : public impl::DeflateSymbolsBase<DeflateSymbolsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(
        getFilesystemBackend(std::filesystem::path(cacheDir.getValue())));
    // Deflate each symbol.
    for (auto &op : getOperation()) {
      if (!op.hasAttr(SymbolTable::getSymbolAttrName()))
        continue;

      if (failed(deflateOp(&op, cache)))
        return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// InflateSymbolsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_INFLATESYMBOLS
#include "Support/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class InflateSymbolsPass : public impl::InflateSymbolsBase<InflateSymbolsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    // Bring up the cache.
    BlobCache<RegionCacheKey> cache(
        getFilesystemBackend(std::filesystem::path(cacheDir.getValue())));
    // Inflate each deflated op.
    for (auto &sym : getOperation()) {
      if (!sym.hasAttr(getRegionHashAttrName()))
        continue;

      if (failed(inflateOp(&sym, cache)))
        signalPassFailure();
    }
  }
};
} // namespace
