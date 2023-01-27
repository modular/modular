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
  std::array<uint8_t, 32> hash =
      llvm::SHA256::hash(ArrayRef((const uint8_t *)key.begin(), key.size()));
  return {hash.begin(), hash.end()};
}

/// Do a transform that can be cached. The transform must be named, see the
/// PassManager overload for an example.
AnyAsyncValueRef
Cache::cachedTransform(Operation *target, RCRef<TransformCache> transformCache,
                       AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                       TransformFn transformFn, CacheHitFn cacheHitFn) {
  AsyncValue::registerType<CacheFindResult>();

  mlir::writeBytecodeToFile(target, *transformKey);
  BufferRef keyBuffer = std::move(transformKey);

  // Try to find the key in the cache. The cache hit function should chain off
  // that and do the right for the cache state.

  auto foundOr =
      AsyncValueRef<CacheFindResult>::allocate(transformCache->getRuntime());
  chain.andThenSync([foundOr = foundOr.copy(), keyBuffer = keyBuffer.copy(),
                     transformCache = transformCache.copy()] {
    auto f = transformCache->find(keyBuffer->getBuffer());
    std::move(f).andThenSync(
        [foundOr = foundOr.copy()](AsyncValueRef<CacheFindResult> &&f) mutable {
          std::move(foundOr).emplace(std::move(*f));
        });
  });

  // Allocate space for the output.
  AnyAsyncValueRef out =
      AnyAsyncValueRef::createIndirect(transformCache->getRuntime());
  std::move(foundOr).andThenSync(
      [out = out.copy(), target, transformCache = transformCache.copy(),
       transformFn = std::move(transformFn), keyBuffer = std::move(keyBuffer),
       cacheHitFn = std::move(cacheHitFn)](
          AsyncValueRef<CacheFindResult> &&foundOr) mutable {
        if (foundOr.isError())
          return out.setToError(foundOr.getPointer()->takeDiagnostic());

        if (foundOr->hasValue())
          return out.resolveIndirect(cacheHitFn(target, foundOr->takeValue()));

        // No error but no cache hit.

        // Run the transform.
        WriteableBufferRef writeableTransformResult = WriteableBuffer::get();
        auto xform = transformFn(target, writeableTransformResult.copy(),
                                 std::move(foundOr));

        // Insert the transform result into the cache.
        std::move(xform).andThenSync(
            [target, transformCache = transformCache.copy(), out = out.copy(),
             keyBuffer = std::move(keyBuffer),
             transformResult = std::move(writeableTransformResult)](
                AnyAsyncValueRef &&xform) mutable {
              if (xform.isError())
                return out.setToError(xform.takeDiagnostic());

              // Only at this point (so the transform has finished successfully)
              // should we change the transform result ref to be read-only.
              auto hashOr = transformCache->insert(keyBuffer->getBuffer(),
                                                   std::move(transformResult));
              std::move(hashOr).andThenSync(
                  [target, out = out.copy(), xform = xform.copy()](
                      AsyncValueRef<ErrorOr<std::string>> &&hashOr) mutable {
                    if (failed(*hashOr))
                      return out.setToError(getMLIRDiagnostic(
                          hashOr->takeError(), target->getLoc()));

                    return out.resolveIndirect(xform.copy());
                  });
            });
      });

  return out;
}

/// Run a pass manager's passes as a cached transform.
AnyAsyncValueRef Cache::cachedTransform(Operation *target,
                                        RCRef<RegionCache> regionCache,
                                        RCRef<TransformCache> transformCache,
                                        AnyAsyncValueRef chain,
                                        mlir::PassManager &pm) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform = [&pm, regionCache = regionCache.copy()](
                          Operation *op, WriteableBufferRef buf,
                          AnyAsyncValueRef chain) -> AsyncValueRef<Chain> {
    // Allocate a space to put the result of the pass manager. We'll chain
    // off that for the deflation.
    auto pmResult = AsyncValueRef<Chain>::allocate(chain.getRuntime());
    std::move(chain).andThenSync([op, &pm, pmResult = pmResult.copy()](
                                     AnyAsyncValueRef &&chain) mutable {
      if (chain.isError())
        pmResult.setToError(chain.takeDiagnostic());

      if (failed(pm.run(op))) {
        return pmResult.setToError(getMLIRDiagnostic(
            Error("failed to run the pass manager"), op->getLoc()));
      }

      std::move(pmResult).emplace();
    });

    // Hang the deflation off the pass manager result chain.
    auto deflate = deflateOp(op, regionCache.copy(), std::move(pmResult));
    // Once deflation has gone through, we can get the new region hash and
    // store it in the cache.
    auto out = AsyncValueRef<Chain>::allocate(deflate.getRuntime());
    std::move(deflate).andThenSync([op, buf = std::move(buf), out = out.copy()](
                                       AsyncValueRef<Chain> &&deflate) mutable {
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
      std::move(out).emplace();
    });
    return out;
  };

  // Callback that on a cache hit reads the region hashes out of the cache and
  // places them on the operation.
  auto onCacheHit = [](Operation *op,
                       BufferRef regionHashes) -> ErrorOrSuccess {
    // TODO: This currently requires a null terminator (MLIR bug #58964)
    StringRef attrStr = regionHashes->getBuffer().drop_back();
    Attribute newHashes = mlir::parseAttribute(attrStr, op->getContext());
    if (!newHashes || !isa<RegionHashArrayAttr>(newHashes))
      return Error("failed to parse the region hashes");

    // Otherwise, replace the region hash array attr on the target, and
    // we're done.
    op->setAttr(getRegionHashAttrName(), cast<RegionHashArrayAttr>(newHashes));
    return success();
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(keyBuf), std::move(runTransform),
                         std::move(onCacheHit));
}
