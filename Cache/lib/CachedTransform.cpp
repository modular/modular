//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachedTransform.h"
#include "LLCL/Runtime/Algorithms.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

std::string TransformCacheKey::hashKey(TransformCacheKey::KeyTy key) {
  // This is just a (usually relatively small) string - the hash is just the
  // SHA256 hash of the input.
  std::array<uint8_t, 32> hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)key->getBufferStart(), key->getBufferSize()));
  return {hash.begin(), hash.end()};
}

/// Do a transform that can be cached. The transform must be named, see the
/// PassManager overload for an example.
AnyAsyncValueRef Cache::cachedTransform(EncodedLocation loc,
                                        RCRef<TransformCache> transformCache,
                                        AnyAsyncValueRef chain,
                                        WriteableBufferRef transformKey,
                                        TransformFn transformFn,
                                        CacheHitFn cacheHitFn) {
  TimeTraceScope traceScope(
      RuntimeCacheProfilerEntry::create("Cache::cachedTransform"));
  BufferRef keyBuffer = std::move(transformKey);

  // Try to find the key in the cache. The cache hit function should chain off
  // that and do the right for the cache state.
  auto foundOr = AsyncValueRef<std::optional<BufferRef>>::allocate(
      transformCache->getRuntime());
  chain.andThenSync([foundOr = foundOr.copy(), keyBuffer = keyBuffer.copy(),
                     transformCache = transformCache.copy(),
                     loc = std::move(loc)]() mutable {
    // Find the thing in the cache with the target op's location. This copy of
    // `keyBuffer` is local, so it's safe to move.
    auto f = transformCache->find(std::move(keyBuffer), std::move(loc));
    std::move(f).andThenSync(
        [foundOr = foundOr.copy()](
            AsyncValueRef<std::optional<BufferRef>> &&f) mutable {
          if (f.isError())
            return std::move(foundOr).setToError(f.takeDiagnostic());

          std::move(foundOr).emplace(std::move(*f));
        });
  });

  // Allocate space for the output.
  AnyAsyncValueRef out =
      AnyAsyncValueRef::createIndirect(transformCache->getRuntime());
  std::move(foundOr).andThenSync(
      [out = out.copy(), transformCache = transformCache.copy(),
       transformFn = std::move(transformFn), keyBuffer = std::move(keyBuffer),
       cacheHitFn = std::move(cacheHitFn)](
          AsyncValueRef<std::optional<BufferRef>> &&foundOr) mutable {
        if (foundOr.isError())
          return std::move(out).setToError(
              foundOr.getPointer()->takeDiagnostic());

        if (foundOr->has_value())
          return std::move(out).resolveIndirect(
              cacheHitFn(std::move(**foundOr)));

        // No error but no cache hit.

        // Run the transform. Use a 1 MB in-memory buffer.
        WriteableBufferRef writeableTransformResult = WriteableBuffer::get(
            /*size=*/0, /*aligment=*/{}, /*capacity=*/1024 * 1024);
        auto xform =
            transformFn(writeableTransformResult.copy(), std::move(foundOr));

        // Insert the transform result into the cache.
        std::move(xform).andThenSync(
            [transformCache = transformCache.copy(), out = out.copy(),
             keyBuffer = std::move(keyBuffer),
             transformResult = std::move(writeableTransformResult)](
                AnyAsyncValueRef &&xform) mutable {
              if (xform.isError())
                return std::move(out).setToError(xform.takeDiagnostic());

              // Only at this point (so the transform has finished successfully)
              // should we change the transform result ref to be read-only.
              // Again, this keyBuffer is local, so it's safe to move.
              AsyncValueRef<std::string> hashOr = transformCache->insert(
                  std::move(keyBuffer), std::move(transformResult));
              std::move(hashOr).andThenSync(
                  [out = out.copy(), xform = xform.copy()](
                      AsyncValueRef<std::string> &&hashOr) mutable {
                    if (hashOr.isError())
                      return std::move(out).setToError(hashOr.takeDiagnostic());

                    return std::move(out).resolveIndirect(xform.copy());
                  });
            });
      });

  return out;
}
