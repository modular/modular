//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEDTRANSFORM_H
#define CACHE_CACHEDTRANSFORM_H

#include "Cache/BlobCache.h"
#include "Cache/Buffer.h"
#include "Cache/CacheDialect/CacheOps.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class PassManager;
}

namespace M::Cache {
//===----------------------------------------------------------------------===//
// Caching a transformation
//===----------------------------------------------------------------------===//

/// The Cache dialect provides a method to cache a transform as defined by a set
/// of MLIR Passes. This struct defines the cache key - it's always a StringRef
/// because we're hashing small strings (in this case, generally a name and a
/// precomputed hash over regions) so the extra complexity of allowing us to
/// pass in a hash directly isn't worth it.
struct TransformCacheKey {
  using KeyTy = StringRef;
  static std::string hashKey(KeyTy key);
};

/// The most basic function that takes a target operation and transforms it -
/// returning success or failure. The function can write the result of the
/// transform to the provided WriteableBufferRef - there is a cachedTransform
/// overload for IR to IR transformations. The Operation provided to the
/// TransformFn is NOT pre-inflated by the caller - that is the responsibility
/// of the TransformFn if so desired. The transform should chain itself on the
/// provided AsyncValueRef.
///
/// For example:
///   auto runTransform = [](Operation *op, WriteableBufferRef buf,
///                          AsyncValueRef<LogicalResult> chain)
///     -> AsyncValueRef<LogicalResult> {
///    auto xform = doAsyncTransform(op, buf, std::move(chain));
///    // Allocate a space to put the result of the pass manager. We'll chain
///    // off that for the deflation.
///    auto result = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
///    xform.andThen([&]() mutable {
///      result.emplace(doSyncTransform(op, buf));
///    });
///
///    return result;
///  };
using TransformFn = llvm::function_ref<LLCL::AsyncValueRef<LogicalResult>(
    Operation *, WriteableBufferRef, LLCL::AsyncValueRef<LogicalResult>)>;

/// This is the function that's called on a cache access. It provides the user
/// with the Operation pointer (still deflated) and CacheFindResult for the
/// requested lookup. The function is called irrespective of cache hit or not,
/// so the function must properly propagate the input CacheFindResult. The
/// function should chain itself on the provided AsyncValueRef.
///
/// For example:
///   auto onCacheHit = [](Operation *op,
///                        AsyncValueRef<CacheFindResult> foundOr)
///      -> AsyncValueRef<CacheFindResult> {
///    auto out =
///    AsyncValueRef<CacheFindResult>::allocate(foundOr.getRuntime());
///    foundOr.andThen([] {
///      // If it's an error, return the error.
///      if (foundOr->isError()) {
///        out.emplace(CacheFindResult::error(foundOr->takeError()));
///        return;
///      }
///
///      // Nothing in the cache, say that.
///      if (!foundOr->hasValue()) {
///        out.emplace(CacheFindResult::notInCache());
///        return;
///      }
///
///      // We found the value in the cache, handle that.
///      handleCacheHit(foundOr->takeValue());
///
///      // Forward the BufferRef because we found something in the cache.
///      out.emplace(CacheFindResult::value(std::move(buf)));
///    });
///    return out;
///  };
using CacheAccessFn = llvm::function_ref<LLCL::AsyncValueRef<CacheFindResult>(
    Operation *, LLCL::AsyncValueRef<CacheFindResult>)>;

/// Run the specified transform on the target operation. The transform must have
/// a key of some kind that can be associated with the operation. The semantics
/// of `cachedTransform` are that it will combine the input IR with the name of
/// the transform to map to a cached result. If deflation/inflation is desired,
/// the user should either deflate before calling this funciton, or
/// deflate/inflate as part of the provided transform. See the PassManager
/// overload below for an example.
LLCL::AsyncValueRef<LogicalResult>
cachedTransform(Operation *target, BlobCache<TransformCacheKey> &transformCache,
                LLCL::AsyncValueRef<LogicalResult> chain,
                WriteableBufferRef transformKey, TransformFn transformFn,
                CacheAccessFn cacheAccessFn);

/// Run the specified passes over the target operation (i.e. ModulePasses over a
/// ModuleOp). If the target operation and pass pipeline result in a cache hit,
/// that cache hit will simply replace the operation's region hash attribute
/// with the updated region hash attribute. The granularity of the result is a
/// region on the operation `target`. This function manifests its result as an
/// update to the RegionHashArrayAttr on `target` - it will update the region
/// hashes from the old versions (pre-transform) to the new versions (transform
/// applied).
LLCL::AsyncValueRef<LogicalResult>
cachedTransform(Operation *target, BlobCache<RegionCacheKey> &regionCache,
                BlobCache<TransformCacheKey> &transformCache,
                LLCL::AsyncValueRef<LogicalResult> chain,
                mlir::PassManager &pm);
} // namespace M::Cache

#endif // CACHE_CACHEDTRANSFORM_H
