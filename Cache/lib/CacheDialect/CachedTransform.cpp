//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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
LLCL::AnyAsyncValueRef
Cache::cachedTransform(Operation *target,
                       BlobCache<TransformCacheKey> &transformCache,
                       AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                       TransformFn transformFn, CacheAccessFn cacheAccessFn) {
  AsyncValue::registerType<CacheFindResult>();

  mlir::writeBytecodeToFile(target, *transformKey);
  BufferRef keyBuffer = std::move(transformKey);

  // Try to find the key in the cache. The cache hit function should chain off
  // that and do the right for the cache state.

  auto foundOr = LLCL::AsyncValueRef<CacheFindResult>::allocate(
      transformCache.getRuntime());
  chain->andThen([foundOr = foundOr.copy(), keyBuffer = keyBuffer.copy(),
                  &transformCache] {
    auto f = transformCache.find(keyBuffer->getBuffer());
    f.andThen([f = f.copy(), foundOr = foundOr.copy()] {
      foundOr.emplace(std::move(*f));
    });
  });

  auto cacheHit = cacheAccessFn(target, std::move(foundOr));

  // Allocate an output variable for the overall success/failure of this
  // function.
  AnyAsyncValueRef out = AsyncValue::createIndirect(cacheHit.getRuntime());

  // Sequence actually doing the transform off the cache hit result.
  cacheHit.andThen([target, &transformCache, transformFn,
                    /*passthrough*/ keyBuffer = std::move(keyBuffer),
                    out = out.copy(), cacheHit = cacheHit.copy()]() mutable {
    // If there was an error, nothing we can do.
    if (cacheHit->isError()) {
      return out->setToError(
          getMLIRDiagnostic(cacheHit->takeError(), target->getLoc()));
    }

    // Cache hit, we're done!
    if (cacheHit->hasValue())
      return out->emplaceIndirect<Chain>();

    // No error but no cache hit.

    // Run the transform.
    WriteableBufferRef writeableTransformResult = WriteableBuffer::get();
    auto xform = transformFn(target, writeableTransformResult.copy(),
                             std::move(cacheHit));

    // Insert the transform result into the cache.
    xform->andThen(
        [target, &transformCache, out = out.copy(), xform = xform.copy(),
         keyBuffer = std::move(keyBuffer),
         transformResult = std::move(writeableTransformResult)]() mutable {
          // Only at this point (so the transform has finished) should we change
          // the transform result ref to be read-only.
          auto hashOr = transformCache.insert(keyBuffer->getBuffer(),
                                              std::move(transformResult));
          hashOr.andThen([&, hashOr = hashOr.copy(), out = out.copy()] {
            if (failed(*hashOr))
              return out->setToError(
                  getMLIRDiagnostic(hashOr->takeError(), target->getLoc()));

            return out->resolveIndirect(std::move(xform));
          });
        });
  });

  return out;
}

/// Run a pass manager's passes as a cached transform.
AnyAsyncValueRef
Cache::cachedTransform(Operation *target,
                       BlobCache<RegionCacheKey> &regionCache,
                       BlobCache<TransformCacheKey> &transformCache,
                       AnyAsyncValueRef chain, mlir::PassManager &pm) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform =
      [&pm, &regionCache](Operation *op, WriteableBufferRef buf,
                          AnyAsyncValueRef chain) -> AsyncValueRef<Chain> {
    // Allocate a space to put the result of the pass manager. We'll chain
    // off that for the deflation.
    auto pmResult = AsyncValueRef<Chain>::allocate(chain->getRuntime());
    chain->andThen(
        [op, &pm, chain = chain.copy(), pmResult = pmResult.copy()]() mutable {
          if (chain->isError())
            pmResult.setToError(chain->takeDiagnostic());

          if (failed(pm.run(op))) {
            return pmResult.setToError(getMLIRDiagnostic(
                Error("failed to run the pass manager"), op->getLoc()));
          }

          pmResult.emplace();
        });

    // Hang the deflation off the pass manager result chain.
    auto deflate = deflateOp(op, regionCache, std::move(pmResult));
    // Once deflation has gone through, we can get the new region hash and
    // store it in the cache.
    auto out = AsyncValueRef<Chain>::allocate(chain->getRuntime());
    deflate.andThen([op, buf = std::move(buf), out = out.copy(),
                     deflate = deflate.copy()]() mutable {
      // Get the new region hashes and stuff them in the
      // cache.
      auto resultRegionHashes =
          op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
      if (!resultRegionHashes) {
        return out.setToError(getMLIRDiagnostic(
            Error("could not find region hashes"), op->getLoc()));
      }
      *buf << resultRegionHashes;
      // TODO: This currently requires a null terminator (MLIR bug #58964)
      buf->write((char)0);
      out.emplace();
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
      if (foundOr->isError())
        return out.emplace(CacheFindResult::error(foundOr->takeError()));

      // Nothing in the cache, say that.
      if (!foundOr->hasValue())
        return out.emplace(CacheFindResult::notInCache());

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

  return cachedTransform(target, transformCache, std::move(chain),
                         std::move(keyBuf), runTransform, onCacheHit);
}
