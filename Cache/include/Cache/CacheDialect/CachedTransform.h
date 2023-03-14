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
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class PassManager;
}

namespace M::Cache {
//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

/// The Cache dialect provides a method to cache generic transformations. This
/// struct defines the cache key - it's always a StringRef because we're hashing
/// small strings so the extra complexity of allowing us to pass in a hash
/// directly isn't worth it.
struct TransformCacheKey {
  using KeyTy = StringRef;
  static std::string hashKey(KeyTy key);
};

/// Convenience typedef to reduce typing.
using TransformCache = BlobCache<TransformCacheKey>;

/// The most basic function that performs a transformation, writing the
/// cacheable results to the provided buffer. The transform should chain itself
/// on the provided AsyncValueRef.
///
/// For example:
///   auto runTransform = [](WriteableBufferRef buf, AnyAsyncValueRef chain)
///     -> AsyncValueRef<Chain> {
///    auto xform = doAsyncTransform(op, buf, std::move(chain));
///
///    // Allocate a space to put the result of the transformation. We'll chain
///    // off that.
///    auto result = AsyncValueRef<Chain>::allocate(chain.getRuntime());
///    xform.andThenSync([&]() mutable {
///      result.emplace(doSyncTransform(op, buf));
///    });
///    return result;
///  };
using TransformFn = llvm::unique_function<LLCL::AnyAsyncValueRef(
    WriteableBufferRef, LLCL::AnyAsyncValueRef)>;

/// This is the function that's called on a cache hit. It provides the buffer
/// that was in the cache for the requested lookup.
using CacheHitFn = llvm::unique_function<LLCL::AnyAsyncValueRef(BufferRef)>;

/// Run the specified transform, using the associated key for caching. When the
/// transform is run, the result AnyAsyncValueRef is resolved to the result of
/// the transform. If the transform is *not* run, then the result
/// AnyAsyncValueRef simply contains a Chain.
LLCL::AnyAsyncValueRef
cachedTransform(EncodedLocation loc, LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformFn transformFn, CacheHitFn cacheHitFn);

namespace Detail {
/// These three detectors check for the ErrorOr-style APIs we care about for the
/// templated version of `cachedTransform` below.
template <typename T>
using HasIsError = decltype(std::declval<T>().isError());
template <typename T>
using HasTakeError = decltype(std::declval<T>().takeError());
template <typename T>
using HasTakeValue = decltype(std::declval<T>().takeValue());

/// Given a CacheHitFn-like callable, get the result type.
template <typename CacheHitFnT>
using ResultT = std::invoke_result_t<CacheHitFnT, BufferRef>;

template <typename CacheHitFnT>
using AsyncValueRefResultT = LLCL::AsyncValueRef<Detail::ResultT<CacheHitFnT>>;

/// Package up detection of member functions of ErrorOr.
template <typename CacheHitFnT>
constexpr bool is_result_error_or_v =
    llvm::is_detected<Detail::HasIsError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeValue,
                      Detail::ResultT<CacheHitFnT>>::value;

} // namespace Detail

/// This provides a templated version of `cachedTransform` that provides a sync
/// API for the cache hit function.
template <typename CacheHitFnT>
LLCL::AnyAsyncValueRef
cachedTransform(EncodedLocation loc, LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformFn transformFn, CacheHitFnT cacheHitFn) {
  // Get the runtime pointer to hand to the closure.
  LLCL::CompactRuntimePtr rt = transformCache->getRuntime();

  CacheHitFn onCacheHit;

  // If the cache hit function return something like an ErrorOr<T> propagate
  // failures properly.
  if constexpr (Detail::is_result_error_or_v<CacheHitFnT>) {
    onCacheHit = [loc = loc.copy(), cacheHitFn = std::move(cacheHitFn),
                  rt](BufferRef buf) mutable {
      auto resultOr = cacheHitFn(std::move(buf));
      if (resultOr.isError())
        return Detail::AsyncValueRefResultT<CacheHitFnT>::createError(
            rt, EncodedDiagnostic(resultOr.takeError(), std::move(loc)));

      return Detail::AsyncValueRefResultT<CacheHitFnT>::createReady(
          rt, resultOr.takeValue());
    };
  } else {
    onCacheHit = [cacheHitFn = std::move(cacheHitFn), rt](BufferRef buf) {
      auto result = Detail::AsyncValueRefResultT<CacheHitFnT>::allocate(rt);
      result.copy().emplace(cacheHitFn(std::move(buf)));
      return result;
    };
  }
  return cachedTransform(std::move(loc), std::move(transformCache),
                         std::move(chain), std::move(transformKey),
                         std::move(transformFn), std::move(onCacheHit));
}

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

/// Transformation and cache functions that operate on a given operation. The
/// Operation provided to the TransformFn is NOT pre-inflated by the caller -
/// that is the responsibility of the TransformFn if so desired.
using OpTransformFn = llvm::unique_function<LLCL::AnyAsyncValueRef(
    Operation *, WriteableBufferRef, LLCL::AnyAsyncValueRef)>;
using OpCacheHitFn =
    llvm::unique_function<LLCL::AnyAsyncValueRef(Operation *, BufferRef)>;

/// Helper method to write the given operation to the provided cache key.
void writeOperationToCacheKey(Operation *op, WriteableBufferRef key);

/// Run the specified transform on the target operation. The transform must have
/// a key of some kind that can be associated with the operation. The semantics
/// of `cachedTransform` are that it will combine the input IR with the name of
/// the transform to map to a cached result. If deflation/inflation is desired,
/// the user should either deflate before calling this function, or
/// deflate/inflate as part of the provided transform. See the PassManager
/// overload below for an example.
///
/// When the transform is run, the result AnyAsyncValueRef is resolved to the
/// result of the transform. If the transform is *not* run, then the result
/// AnyAsyncValueRef simply contains a Chain.
template <typename TransformationFnT, typename CacheHitFnT>
LLCL::AnyAsyncValueRef
cachedTransform(Operation *target, LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformationFnT &&transformFn, CacheHitFnT &&cacheHitFn) {
  writeOperationToCacheKey(target, transformKey.copy());

  return cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(target->getLoc()),
      std::move(transformCache), std::move(chain), std::move(transformKey),
      [target, transformFn = std::forward<TransformationFnT>(transformFn)](
          WriteableBufferRef buf, LLCL::AnyAsyncValueRef chain) {
        return transformFn(target, std::move(buf), std::move(chain));
      },
      [target, cacheHitFn = std::forward<CacheHitFnT>(cacheHitFn)](
          BufferRef buf) { return cacheHitFn(target, std::move(buf)); });
}

/// Run the specified passes over the target operation (i.e. ModulePasses over a
/// ModuleOp). If the target operation and pass pipeline result in a cache hit,
/// that cache hit will simply replace the operation's region hash attribute
/// with the updated region hash attribute. The granularity of the result is a
/// region on the operation `target`. This function manifests its result as an
/// update to the RegionHashArrayAttr on `target` - it will update the region
/// hashes from the old versions (pre-transform) to the new versions (transform
/// applied).
LLCL::AnyAsyncValueRef
cachedTransform(Operation *target, LLCL::RCRef<RegionCache> regionCache,
                LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, mlir::PassManager &pm);
} // namespace M::Cache

#endif // CACHE_CACHEDTRANSFORM_H
