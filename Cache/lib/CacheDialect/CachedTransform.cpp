//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "LLCL/Runtime/Algorithms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SHA256.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Caching transforms
//===----------------------------------------------------------------------===//

std::string TransformCacheKey::hashKey(TransformCacheKey::KeyTy key) {
  // This is just a (usually relatively small) string - the hash is just the
  // SHA256 hash of the input.
  std::array<uint8_t, 32> hash = llvm::SHA256::hash(
      llvm::makeArrayRef((const uint8_t *)key.begin(), key.size()));
  return {hash.begin(), hash.end()};
}

/// Do a transform that can be cached. The transform must be named, see the
/// PassManager overload for an example.
LLCL::AsyncValueRef<LogicalResult> Cache::cachedTransform(
    Operation *target, BlobCache<RegionCacheKey> &regionCache,
    BlobCache<TransformCacheKey> &transformCache,
    LLCL::AsyncValueRef<LogicalResult> chain, WriteableBufferRef transformKey,
    TransformFn transformFn, CacheAccessFn cacheAccessFn) {
  AsyncValue::registerType<CacheFindResult>();

  // Allocate an output variable for the overall success/failure of this
  // function.
  auto out = AsyncValueRef<LogicalResult>::allocate(regionCache.getRuntime());
  // Create the cache key by caching the target op's inputs. `deflate` is
  // sequenced on `chain` here.
  auto deflate = deflateOp(target, regionCache, std::move(chain));
  deflate.andThen([target, &regionCache, &transformCache,
                   writeableKeyBuffer = std::move(transformKey), transformFn,
                   cacheAccessFn, out = out.copy(),
                   deflate = deflate.copy()]() mutable {
    // If we failed to deflate, then there's not much we can do.
    if (failed(*deflate)) {
      out.emplace(failure());
      return;
    }

    // Basically we just want to key off the deflated op, so write the target op
    // into the key buffer and get a read-only ref to it.
    target->print(*writeableKeyBuffer);
    BufferRef keyBuffer = std::move(writeableKeyBuffer);

    // Try to find the key in the cache. The cache hit function should chain off
    // that and do the right for the cache state.
    auto foundOr = transformCache.find(keyBuffer->getBuffer());
    auto cacheHit = cacheAccessFn(target, std::move(foundOr));
    // Sequence actually doing the transform off the cache hit result.
    cacheHit.andThen([target, &regionCache, &transformCache, transformFn,
                      /*passthrough*/ keyBuffer = std::move(keyBuffer),
                      out = out.copy(), cacheHit = cacheHit.copy()]() mutable {
      // If there was an error, nothing we can do.
      if (cacheHit->isError()) {
        out.emplace(mlir::emitError(target->getLoc()) << cacheHit->getError());
        return;
      }

      // Cache hit, we're done!
      if (cacheHit->hasValue()) {
        out.emplace(success());
        return;
      }

      // No error but no cache hit.

      // Now we do the transform on the inflated operation. This is so
      // that we don't have to change the MLIR pass manager or anything
      // - the CacheHitFn provides a way to avoid inflating/deflating
      // ops when there's a cache hit.

      // First re-inflate the target op.
      auto inflate = inflateOp(target, regionCache,
                               AsyncValueRef<LogicalResult>::createReady(
                                   cacheHit.getRuntime(), success()));

      // Run the transform.
      WriteableBufferRef writeableTransformResult = WriteableBuffer::get();
      auto xform = transformFn(target, writeableTransformResult.copy(),
                               std::move(inflate));

      // Re-deflate the target op.
      auto deflate = deflateOp(target, regionCache, std::move(xform));

      // Insert the transform result into the cache.
      deflate.andThen(
          [target, &transformCache, out = out.copy(),
           keyBuffer = std::move(keyBuffer), deflate = deflate.copy(),
           transformResult = std::move(writeableTransformResult)]() mutable {
            // Only at this point (so the transform has finished, and deflate
            // has too) should we change the transform result ref to be
            // read-only.
            auto hashOr = transformCache.insert(keyBuffer->getBuffer(),
                                                std::move(transformResult));
            hashOr.andThen([&, hashOr = hashOr.copy(), out = out.copy()] {
              if (failed(*hashOr))
                return out.emplace(mlir::emitError(target->getLoc())
                                   << hashOr->getError());

              // Finally done, return success.
              out.emplace(success());
            });
          });
    });
  });

  return out;
}

/// Run a pass manager's passes as a cached transform.
LLCL::AsyncValueRef<LogicalResult> Cache::cachedTransform(
    Operation *target, BlobCache<RegionCacheKey> &regionCache,
    BlobCache<TransformCacheKey> &transformCache,
    LLCL::AsyncValueRef<LogicalResult> chain, mlir::PassManager &pm) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform =
      [&pm, &regionCache](
          Operation *op, WriteableBufferRef buf,
          AsyncValueRef<LogicalResult> chain) -> AsyncValueRef<LogicalResult> {
    // Allocate a space to put the result of the pass manager. We'll chain
    // off that for the deflation.
    auto pmResult = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
    chain.andThen(
        [op, &pm, chain = chain.copy(), pmResult = pmResult.copy()]() mutable {
          if (failed(*chain) || failed(pm.run(op)))
            return pmResult.emplace(failure());

          pmResult.emplace(success());
        });

    // Hang the deflation off the pass manager result chain.
    auto deflate = deflateOp(op, regionCache, std::move(pmResult));
    // Once deflation has gone through, we can get the new region hash and
    // store it in the cache.
    auto out = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
    deflate.andThen([op, buf = std::move(buf), out = out.copy(),
                     deflate = deflate.copy()]() mutable {
      // Get the new region hashes and stuff them in the
      // cache.
      auto resultRegionHashes =
          op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
      if (!resultRegionHashes) {
        out.emplace(mlir::emitError(op->getLoc())
                    << "could not find region hashes");
        return;
      }
      *buf << resultRegionHashes;
      // TODO: This currently requires a null terminator (MLIR bug #58964)
      buf->write((char)0);
      out.emplace(success());
    });
    return out;
  };

  // Callback that on a cache hit reads the region hashes out of the cache and
  // places them on the operation.
  auto onCacheHit = [](Operation *op, AsyncValueRef<CacheFindResult> foundOr)
      -> AsyncValueRef<CacheFindResult> {
    auto out = AsyncValueRef<CacheFindResult>::allocate(foundOr.getRuntime());
    foundOr.andThen([op, out = out.copy(), foundOr = foundOr.copy()] {
      // If it's an error, return the error.
      if (foundOr->isError()) {
        out.emplace(CacheFindResult::error(foundOr->takeError()));
        return;
      }

      // Nothing in the cache, say that.
      if (!foundOr->hasValue()) {
        out.emplace(CacheFindResult::notInCache());
        return;
      }

      // We found the value in the cache, handle that.
      BufferRef buf = foundOr->takeValue();
      // TODO: We have to drop the null terminator, the underlying memory needs
      //       it but the StringRef shouldn't have it (MLIR bug #58964).
      StringRef attrStr = buf->getBuffer().drop_back();
      Attribute newHashes = mlir::parseAttribute(attrStr, op->getContext());
      if (!newHashes || !isa<RegionHashArrayAttr>(newHashes))
        return out.emplace(
            CacheFindResult::error("failed to parse region hashes"));

      // Otherwise, replace the region hash array attr on the target, and
      // we're done.
      op->setAttr(getRegionHashAttrName(),
                  cast<RegionHashArrayAttr>(newHashes));
      // Forward the BufferRef because we found something in the cache.
      out.emplace(CacheFindResult::value(std::move(buf)));
    });
    return out;
  };

  return cachedTransform(target, regionCache, transformCache, std::move(chain),
                         std::move(keyBuf), runTransform, onCacheHit);
}
