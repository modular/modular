//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHED_TRANSFORM_H
#define CACHE_CACHED_TRANSFORM_H

#include "Cache/BlobCache.h"
#include "Support/Buffer.h"
#include "Support/LLVMForwardDecls.h"

namespace M::Cache {

/// Profiler entry for run-time cache transforms.
using RuntimeCacheProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1)>;

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

/// The Cache dialect provides a method to cache generic transformations. This
/// struct defines the cache key as a BufferRef; the contents of which are
/// hashed; this is to disambiguate it from StringRef keys, which generally
/// have no hashing applied.
struct TransformCacheKey {
  using KeyTy = BufferRef;
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
cachedTransform(EncodedLocation loc, RCRef<TransformCache> transformCache,
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
cachedTransform(EncodedLocation loc, RCRef<TransformCache> transformCache,
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
} // namespace M::Cache

#endif // CACHE_CACHED_TRANSFORM_H
