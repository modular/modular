//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "LLCL/Runtime/Algorithms.h"
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

/// The symbol start code is a single 0xff byte - in unicode this prefix
/// indicates a special code point so it should almost always result in an
/// unknown code point (and therefore be distinguishable).
static constexpr char kSymbolStartCode = (char)0xff;

/// Because MLIR hides the implementation details of how things are
/// parsed/printed, if we want to print it in a way we can recover it we have to
/// do it ourselves. This writes a RegionHashArrayAttr in a simple binary format
/// that is very similar to BEF.
static void printParseableRegionHashes(RegionHashArrayAttr hashes,
                                       WriteableBufferRef buf) {
  uint16_t numHashes = hashes.size();
  buf->write((char *)&numHashes, sizeof(uint16_t));

  for (auto hash : hashes) {
    buf->write(hash.getHash().begin(), hash.getHash().size());
    uint16_t numSymbols = hash.getSymbols().size();
    buf->write((char *)&numSymbols, sizeof(uint16_t));
    for (auto sym : hash.getSymbols()) {
      buf->write(kSymbolStartCode);
      *buf << sym;
    }
  }
}

/// Perform the inverse of the operation above - parse the region hashes out of
/// the buffer and return a new RegionHashArrayAttr.
static FailureOr<RegionHashArrayAttr>
parseRegionHashes(BufferRef buf, MLIRContext *ctx, Location loc) {
  StringRef buffer = buf->getBuffer();
  uint16_t numHashes = *((const uint16_t *)buffer.begin());
  buffer = buffer.drop_front(sizeof(uint16_t));
  SmallVector<RegionHashAttr> hashes;
  for (uint16_t i = 0; i < numHashes; ++i) {
    // Consume the hash.
    StringRef hashBytes = buffer.take_front(32);
    buffer = buffer.drop_front(32);
    // Consume the number of symbols.
    uint16_t numSymbols = *((const uint16_t *)buffer.begin());
    buffer = buffer.drop_front(sizeof(uint16_t));
    SmallVector<SymbolRefAttr> syms;
    for (uint16_t s = 0; s < numSymbols; ++s) {
      if (buffer.front() != kSymbolStartCode)
        return mlir::emitError(loc)
               << "corrupted symbol, did not start with the start code";

      // Consume the symbol.
      auto [symbol, b] = buffer.split(kSymbolStartCode);
      if (b.empty())
        return mlir::emitError(loc)
               << "could not split binary field on the symbol start code, "
                  "corrupted input detected";

      buffer = b;
      syms.push_back(SymbolRefAttr::get(ctx, symbol));
    }
    // Now we can build the hash.
    hashes.push_back(RegionHashAttr::get(ctx, hashBytes, syms));
  }

  // Return the full array.
  return RegionHashArrayAttr::get(ctx, hashes);
}

/// Do a transform that can be cached. The transform must be named, see the
/// PassManager overload for an example.
LLCL::AsyncValueRef<LogicalResult> Cache::cachedTransform(
    Operation *target, BlobCache<RegionCacheKey> &regionCache,
    BlobCache<TransformCacheKey> &transformCache,
    LLCL::AsyncValueRef<LogicalResult> chain, StringRef transformName,
    TransformFn transformFn, CacheAccessFn cacheAccessFn) {
  // Write the transform name to the key buffer immediately - we can't worry
  // about things getting deallocated.
  WriteableBufferRef writeableKeyBuffer = WriteableBuffer::get();
  *writeableKeyBuffer << transformName;

  // Allocate an output variable for the overall success/failure of this
  // function.
  auto out = AsyncValueRef<LogicalResult>::allocate(regionCache.getRuntime());
  // Create the cache key by caching the target op's inputs. `deflate` is
  // sequenced on `chain` here.
  auto deflate = deflateOp(target, regionCache, std::move(chain));
  deflate.andThen([target, &regionCache, &transformCache,
                   writeableKeyBuffer = std::move(writeableKeyBuffer),
                   transformFn, cacheAccessFn, out = out.copy(),
                   deflate = deflate.copy()]() mutable {
    // If we failed to deflate, then there's not much we can do.
    if (failed(*deflate)) {
      out.emplace(failure());
      return;
    }

    // Construct the key buffer for communicating with the transform cache.
    auto targetRegionHashes =
        target->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());

    // Write the target region hashes into the key buffer and get a read-only
    // ref to it.
    *writeableKeyBuffer << targetRegionHashes;
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
      BufferRef transformResult = std::move(writeableTransformResult);

      // Re-deflate the target op.
      auto deflate = deflateOp(target, regionCache, std::move(xform));

      // Insert the transform result into the cache.
      deflate.andThen([target, &transformCache, out = out.copy(),
                       keyBuffer = std::move(keyBuffer),
                       deflate = deflate.copy(),
                       transformResult = std::move(transformResult)]() mutable {
        auto hashOr = transformCache.insert(keyBuffer->getBuffer(),
                                            std::move(transformResult));
        hashOr.andThen([&, hashOr = hashOr.copy(), out = out.copy()] {
          if (failed(*hashOr)) {
            out.emplace(mlir::emitError(target->getLoc())
                        << hashOr->getError());
            return;
          }

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
  std::string pipeline;
  llvm::raw_string_ostream stream(pipeline);
  pm.printAsTextualPipeline(stream);

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
    deflate.andThen([op, buf = buf.copy(), out = out.copy(),
                     deflate = deflate.copy()]() mutable {
      // Get the new region hashes and stuff them in the
      // cache. This rewrites the mapping hash(pipeline,
      // inputRegionHashes) -> resultRegionHashes.
      auto resultRegionHashes =
          op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
      if (!resultRegionHashes) {
        out.emplace(mlir::emitError(op->getLoc())
                    << "could not find region hashes");
        return;
      }
      // Print them in a way we can parse them.
      printParseableRegionHashes(resultRegionHashes, std::move(buf));
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
      auto newHashesOr =
          parseRegionHashes(buf.copy(), op->getContext(), op->getLoc());
      if (failed(newHashesOr)) {
        out.emplace(CacheFindResult::error("failed to parse region hashes"));
        return;
      }

      // Otherwise, replace the region hash array attr on the target, and
      // we're done.
      op->setAttr(getRegionHashAttrName(), *newHashesOr);
      // Forward the BufferRef because we found something in the cache.
      out.emplace(CacheFindResult::value(std::move(buf)));
    });
    return out;
  };

  return cachedTransform(target, regionCache, transformCache, std::move(chain),
                         stream.str(), runTransform, onCacheHit);
}
