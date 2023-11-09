//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

LogicalResult Cache::writeOperationToCacheKey(Operation *op,
                                              WriteableBufferRef key) {
  // Use bytecode when writing cache keys to ensure determinism across different
  // builds.
  return mlir::writeBytecodeToFile(op, *key);
}

/// Run a pass manager's passes as a cached transform.
AnyAsyncValueRef Cache::cachedTransform(Operation *target,
                                        RCRef<RegionCache> regionCache,
                                        RCRef<TransformCache> transformCache,
                                        AnyAsyncValueRef chain,
                                        mlir::PassManager &pm,
                                        bool deflateTarget) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);
  *keyBuf << "deflate=" << deflateTarget;

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform = [&pm, regionCache = regionCache.copy(), deflateTarget](
                          Operation *op, WriteableBufferRef buf,
                          AnyAsyncValueRef chain) -> AsyncValueRef<Chain> {
    TimeTraceScope<> traceScope(
        "Cache::cachedTransform(Operation *)::runTransform",
        deflateTarget ? "(deflated)" : "");
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

    auto out = AsyncValueRef<Chain>::allocate(pmResult.getRuntime());
    // If we aren't deflating the target, then we need to just write the
    // bytecode and return.
    if (!deflateTarget) {
      std::move(pmResult).andThenSync(
          [op, buf = std::move(buf),
           out = out.copy()](AsyncValueRef<Chain> &&pmResult) mutable {
            if (pmResult.isError())
              return std::move(out).setToError(pmResult.takeDiagnostic());

            TimeTraceScope<> traceScope("writeBytecodeToFile");
            if (failed(mlir::writeBytecodeToFile(op, *buf))) {
              return std::move(out).setToError(getMLIRDiagnostic(
                  "failed to write bytecode file", op->getLoc()));
            }
            std::move(out).emplace();
          });
      return out;
    }

    // Hang the deflation off the pass manager result chain.
    auto deflate = deflateOp(op, regionCache.copy(), std::move(pmResult));
    // Once deflation has gone through, we can get the new region hash and
    // store it in the cache.
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
  auto onCacheHit = [deflateTarget](Operation *op,
                                    BufferRef buf) -> ErrorOrSuccess {
    TimeTraceScope<> traceScope(
        "Cache::cachedTransform(Operation *)::onCacheHit",
        deflateTarget ? "(deflated)" : "");
    if (deflateTarget) {
      // TODO: This currently requires a null terminator (MLIR bug #58964)
      StringRef attrStr = buf->getBuffer().drop_back();
      Attribute newHashes = mlir::parseAttribute(attrStr, op->getContext());
      if (!newHashes || !isa<RegionHashArrayAttr>(newHashes))
        return Error("failed to parse the region hashes");

      // Otherwise, replace the region hash array attr on the target, and
      // we're done.
      op->setAttr(getRegionHashAttrName(),
                  cast<RegionHashArrayAttr>(newHashes));
      return success();
    }

    std::unique_ptr<llvm::MemoryBuffer> bytecode =
        llvm::MemoryBuffer::getMemBuffer(buf->getBuffer(),
                                         /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);

    // Create a dummy block that we can use to inflate container ops.
    Block b;
    if (failed(mlir::readBytecodeFile(
            *bytecode, &b,
            mlir::ParserConfig(op->getContext(),
                               /*verifyAfterParse=*/false)))) {
      return Error("reading bytecode file failed");
    }
    // Get the body from the parsed op and onto the op we're using.
    for (auto [cached, opRegion] :
         llvm::zip(b.front().getRegions(), op->getRegions()))
      opRegion.takeBody(cached);

    return success();
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(keyBuf), std::move(runTransform),
                         std::move(onCacheHit));
}
