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
#include "Support/STLExtras.h"
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
class DeflateSymbolsPass : public impl::DeflateSymbolsBase<DeflateSymbolsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  DeflateSymbolsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    AsyncValue::registerType<LogicalResult>();

    auto rt = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
        runtime, createLeakCheckAllocator(createMallocAllocator()),
        createSingleThreadWorkQueue());

    // Bring up the cache.
    auto cache = RCRef<BlobCache<RegionCacheKey>>::create(
        getFilesystemBackend(*rt, std::filesystem::path(cacheDir.getValue())));
    // Deflate each symbol.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (!op->hasAttr(SymbolTable::getSymbolAttrName()))
        return;

      results.push_back(
          deflateOp(op, cache.copy(), AsyncValueRef<Chain>::createReady(*rt)));
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
    AsyncValue::registerType<LogicalResult>();

    auto rt = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
        runtime, createLeakCheckAllocator(createMallocAllocator()),
        createSingleThreadWorkQueue());

    // Bring up the cache.
    auto cache = RCRef<BlobCache<RegionCacheKey>>::create(
        getFilesystemBackend(*rt, std::filesystem::path(cacheDir.getValue())));
    // Inflate each deflated op.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (!op->hasAttr(getRegionHashAttrName()))
        return;

      results.push_back(
          inflateOp(op, cache.copy(), AsyncValueRef<Chain>::createReady(*rt)));
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
  Runtime *runtime;
};
} // namespace

std::unique_ptr<mlir::Pass> M::Cache::createInflateSymbolsPass(Runtime &rt) {
  return std::make_unique<InflateSymbolsPass>(rt);
}

//===----------------------------------------------------------------------===//
// DeflateConstantsPass
//===----------------------------------------------------------------------===//

namespace M::Cache {
#define GEN_PASS_DEF_DEFLATECONSTANTS
#include "Cache/CachePasses/CachePasses.h.inc"
} // namespace M::Cache

namespace {
class DeflateConstantsPass
    : public impl::DeflateConstantsBase<DeflateConstantsPass> {
public:
  /// Construct an instance of this pass with the provided runtime.
  DeflateConstantsPass(Runtime &rt) : Base::Base(), runtime(&rt) {}

  /// Construct an instance of this pass without a runtime - the pass will
  /// construct its own.
  using Base::Base;

  void runOnOperation() override {
    AsyncValue::registerType<LogicalResult>();

    auto rt = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
        runtime, createLeakCheckAllocator(createMallocAllocator()),
        createSingleThreadWorkQueue());

    // Bring up the cache.
    auto cache = RCRef<BlobCache<DataCacheKey>>::create(
        getFilesystemBackend(*rt, std::filesystem::path(cacheDir.getValue())));
    // Deflate each constant.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        results.push_back(deflateConstant(
            op, cache.copy(), AsyncValueRef<Chain>::createReady(*rt)));
    });

    await(results);
    for (auto &r : results)
      if (r.isError()) {
        getOperation()->emitError() << r.getDiagnostic().getMessage();
        signalPassFailure();
      }
  }

private:
  Runtime *runtime;
};
} // namespace

std::unique_ptr<mlir::Pass>
M::Cache::createDeflateConstantsPass(LLCL::Runtime &rt) {
  return std::make_unique<DeflateConstantsPass>(rt);
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
    AsyncValue::registerType<LogicalResult>();

    auto rt = ConditionallyOwnedPointer<Runtime>::allocateIfNeeded(
        runtime, createLeakCheckAllocator(createMallocAllocator()),
        createSingleThreadWorkQueue());

    // Bring up the cache.
    auto cache = RCRef<BlobCache<DataCacheKey>>::create(
        getFilesystemBackend(*rt, std::filesystem::path(cacheDir.getValue())));
    // Inflate each constant.
    SmallVector<AnyAsyncValueRef> results;
    getOperation().walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        results.push_back(inflateConstant(
            op, cache.copy(), AsyncValueRef<Chain>::createReady(*rt)));
    });

    await(results);
    for (auto &r : results)
      if (r.isError()) {
        getOperation()->emitError() << r.getDiagnostic().getMessage();
        signalPassFailure();
      }
  }

private:
  Runtime *runtime;
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
  mlir::registerPass([&]() { return Cache::createDeflateConstantsPass(rt); });
  mlir::registerPass([&]() { return Cache::createInflateConstantsPass(rt); });
}
