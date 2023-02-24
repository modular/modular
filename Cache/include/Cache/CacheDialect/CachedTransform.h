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

/// Convenience typedef to reduce typing.
using TransformCache = BlobCache<TransformCacheKey>;

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
///                          AnyAsyncValueRef chain)
///     -> AsyncValueRef<Chain> {
///    auto xform = doAsyncTransform(op, buf, std::move(chain));
///    // Allocate a space to put the result of the pass manager. We'll chain
///    // off that for the deflation.
///    auto result = AsyncValueRef<Chain>::allocate(chain.getRuntime());
///    xform.andThenSync([&]() mutable {
///      result.emplace(doSyncTransform(op, buf));
///    });
///
///    return result;
///  };
using TransformFn = llvm::unique_function<LLCL::AnyAsyncValueRef(
    Operation *, WriteableBufferRef, LLCL::AnyAsyncValueRef)>;

/// This is the function that's called on a cache hit. It provides the user
/// with the Operation pointer and the buffer that was in the cache for the
/// requested lookup.
using CacheHitFn =
    llvm::unique_function<LLCL::AnyAsyncValueRef(Operation *, BufferRef)>;

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
LLCL::AnyAsyncValueRef
cachedTransform(Operation *target, LLCL::RCRef<TransformCache> transformCache,
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
using ResultT = std::invoke_result_t<CacheHitFnT, Operation *, BufferRef>;

/// Package up detection of member functions of ErrorOr.
template <typename CacheHitFnT>
constexpr bool ReturnsErrorOrLike =
    llvm::is_detected<Detail::HasIsError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeError,
                      Detail::ResultT<CacheHitFnT>>::value &&
    llvm::is_detected<Detail::HasTakeValue,
                      Detail::ResultT<CacheHitFnT>>::value;
} // namespace Detail

/// This provides a templated version of `cachedTransform` that provides a sync
/// API for the cache hit function. The only restriction is that the cache hit
/// function must return something like an ErrorOr<T> so we can propagate
/// failures properly.
template <typename CacheHitFnT>
std::enable_if_t<!std::is_convertible_v<Detail::ResultT<CacheHitFnT>,
                                        LLCL::AnyAsyncValueRef> &&
                     Detail::ReturnsErrorOrLike<CacheHitFnT>,
                 LLCL::AnyAsyncValueRef>
cachedTransform(Operation *target, LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformFn transformFn, CacheHitFnT cacheHitFn) {
  // Get the runtime pointer to hand to the closure.
  LLCL::CompactRuntimePtr rt = transformCache->getRuntime();
  // Register the result type before we try and allocate it.
  LLCL::TypeID::registerType<Detail::ResultT<CacheHitFnT>>();
  auto onCacheHit = [target, cacheHitFn = std::move(cacheHitFn),
                     rt](Operation *op, BufferRef buf) {
    // Call the provided function and act accordingly.
    auto resultOr = cacheHitFn(op, std::move(buf));
    if (resultOr.isError())
      return LLCL::AsyncValueRef<Detail::ResultT<CacheHitFnT>>::createError(
          rt, LLCL::getMLIRDiagnostic(resultOr.takeError(), target->getLoc()));

    return LLCL::AsyncValueRef<Detail::ResultT<CacheHitFnT>>::createReady(
        rt, resultOr.takeValue());
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(transformKey), std::move(transformFn),
                         onCacheHit);
}

/// This provides a templated version of `cachedTransform` that provides a sync
/// API for the cache hit function. This propagates the result as an
/// AsyncValueRef<T> directly, without unwrapping anything that may be inside
/// the result type.
template <typename CacheHitFnT>
std::enable_if_t<!std::is_convertible_v<Detail::ResultT<CacheHitFnT>,
                                        LLCL::AnyAsyncValueRef> &&
                     !Detail::ReturnsErrorOrLike<CacheHitFnT>,
                 LLCL::AnyAsyncValueRef>
cachedTransform(Operation *target, LLCL::RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformFn transformFn, CacheHitFnT cacheHitFn) {
  // Get the runtime pointer to hand to the closure.
  LLCL::CompactRuntimePtr rt = transformCache->getRuntime();
  // Register the result type before we try and allocate it.
  LLCL::TypeID::registerType<Detail::ResultT<CacheHitFnT>>();
  auto onCacheHit = [cacheHitFn = std::move(cacheHitFn), rt](Operation *op,
                                                             BufferRef buf) {
    auto result =
        LLCL::AsyncValueRef<Detail::ResultT<CacheHitFnT>>::allocate(rt);
    result.copy().emplace(cacheHitFn(op, std::move(buf)));
    return result;
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(transformKey), std::move(transformFn),
                         onCacheHit);
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
