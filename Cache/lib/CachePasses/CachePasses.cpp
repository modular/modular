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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
template <typename T>
static ErrorOr<RCRef<BlobCache<T>>> createCache(StringRef cacheDir) {
  auto uriOr = URI::parse(cacheDir);
  if (uriOr.isError())
    return uriOr.takeError();
  auto errOr = getDefaultBackendChain(*uriOr);
  if (errOr.isError())
    return errOr.takeError();
  return RCRef<BlobCache<T>>::create(std::move(*errOr));
}

class DeflateSymbolsPass : public impl::DeflateSymbolsBase<DeflateSymbolsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  DeflateSymbolsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    auto cacheOr = createCache<RegionCacheKey>(cacheDir.getValue());
    if (cacheOr.isError()) {
      getOperation()->emitError() << cacheOr.getError();
      signalPassFailure();
      return;
    }

    // Deflate each symbol.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (!op->hasAttr(SymbolTable::getSymbolAttrName()))
        return;

      results.push_back(deflateOp(op, cacheOr->copy(),
                                  AsyncValueRef<Chain>::createReady(*runtime)));
      // Gotta wait cause it could be nested.
      await(results.back());
    });

    for (auto &r : results)
      if (r.isError()) {
        getOperation()->emitError() << r.getDiagnostic().getMessage();
        signalPassFailure();
      }
  }

private:
  Runtime *runtime = nullptr;
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
    auto cacheOr = createCache<RegionCacheKey>(cacheDir.getValue());
    if (cacheOr.isError()) {
      getOperation()->emitError() << cacheOr.getError();
      signalPassFailure();
      return;
    }

    // Inflate each deflated op.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (!op->hasAttr(getRegionHashAttrName()))
        return;

      results.push_back(inflateOp(op, cacheOr->copy(),
                                  AsyncValueRef<Chain>::createReady(*runtime)));
      // Gotta wait cause it could be nested.
      await(results.back());
    });

    for (auto &r : results)
      if (r.isError()) {
        getOperation()->emitError() << r.getDiagnostic().getMessage();
        signalPassFailure();
      }
  }

private:
  Runtime *runtime = nullptr;
};
} // namespace

std::unique_ptr<mlir::Pass> M::Cache::createInflateSymbolsPass(Runtime &rt) {
  return std::make_unique<InflateSymbolsPass>(rt);
}

//===----------------------------------------------------------------------===//
// InflateSymbolsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_INFLATECONSTANTS
#include "Cache/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class InflateConstantsPass
    : public impl::InflateConstantsBase<InflateConstantsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  InflateConstantsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    auto cacheOr = createCache<DataCacheKey>(cacheDir.getValue());
    if (cacheOr.isError()) {
      getOperation()->emitError() << cacheOr.getError();
      signalPassFailure();
      return;
    }

    // Inflate each constant.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        results.push_back(inflateConstant(
            op, cacheOr->copy(), AsyncValueRef<Chain>::createReady(*runtime)));
    });

    await(results);
    for (auto &r : results)
      if (r.isError()) {
        getOperation()->emitError() << r.getDiagnostic().getMessage();
        signalPassFailure();
      }
  }

private:
  Runtime *runtime = nullptr;
};
} // namespace

std::unique_ptr<mlir::Pass> M::Cache::createInflateConstantsPass(Runtime &rt) {
  return std::make_unique<InflateConstantsPass>(rt);
}

void M::Cache::registerCachePasses(Runtime &rt) {
  // Register the passes with the correct constructor - one that takes the
  // runtime as an argument.
  mlir::registerPass([&]() { return Cache::createDeflateSymbolsPass(rt); });
  mlir::registerPass([&]() { return Cache::createInflateSymbolsPass(rt); });
  mlir::registerPass([&]() { return Cache::createInflateConstantsPass(rt); });
}
