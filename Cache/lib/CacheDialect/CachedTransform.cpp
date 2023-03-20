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

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

void Cache::writeOperationToCacheKey(Operation *op, WriteableBufferRef key) {
  // Use bytecode when writing cache keys to ensure determinism across different
  // builds.
  mlir::writeBytecodeToFile(op, *key);
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
        return std::move(pmResult).setToError(chain.takeDiagnostic());

      if (failed(pm.run(op))) {
        return std::move(pmResult).setToError(getMLIRDiagnostic(
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
        return std::move(out).setToError(getMLIRDiagnostic(
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
