//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEDTRANSFORM_H
#define CACHE_CACHEDTRANSFORM_H

#include "Cache/CacheDialect/CacheOps.h"
#include "Cache/CachedTransform.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class PassManager;
}

namespace M::Cache {
//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

/// Transformation and cache functions that operate on a given operation.
using OpTransformFn = llvm::unique_function<LLCL::AnyAsyncValueRef(
    Operation *, WriteableBufferRef, LLCL::AnyAsyncValueRef)>;
using OpCacheHitFn =
    llvm::unique_function<LLCL::AnyAsyncValueRef(Operation *, BufferRef)>;

/// Helper method to write the given operation to the provided cache key.
LogicalResult writeOperationToCacheKey(Operation *op, WriteableBufferRef key);

/// Run the specified transform on the target operation. The transform must have
/// a key of some kind that can be associated with the operation. The semantics
/// of `cachedTransform` are that it will combine the input IR with the name of
/// the transform to map to a cached result.
///
/// When the transform is run, the result AnyAsyncValueRef is resolved to the
/// result of the transform. If the transform is *not* run, then the result
/// AnyAsyncValueRef simply contains a Chain.
template <typename TransformationFnT, typename CacheHitFnT>
LLCL::AnyAsyncValueRef
cachedTransform(Operation *target, RCRef<TransformCache> transformCache,
                LLCL::AnyAsyncValueRef chain, WriteableBufferRef transformKey,
                TransformationFnT &&transformFn, CacheHitFnT &&cacheHitFn) {
  if (failed(writeOperationToCacheKey(target, transformKey.copy()))) {
    chain.copy().setToError(LLCL::getMLIRDiagnostic(
        "failed to write bytecode file", target->getLoc()));
    return chain;
  }

  return cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(target->getLoc()),
      std::move(transformCache), std::move(chain), std::move(transformKey),
      [target, transformFn = std::forward<TransformationFnT>(transformFn)](
          WriteableBufferRef buf, LLCL::AnyAsyncValueRef chain) mutable {
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
LLCL::AnyAsyncValueRef cachedTransform(Operation *target,
                                       RCRef<RegionCache> regionCache,
                                       RCRef<TransformCache> transformCache,
                                       LLCL::AnyAsyncValueRef chain,
                                       mlir::PassManager &pm);
} // namespace M::Cache

#endif // CACHE_CACHEDTRANSFORM_H
