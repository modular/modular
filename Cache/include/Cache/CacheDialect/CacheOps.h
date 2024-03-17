//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEDIALECT_CACHEOPS_H
#define CACHE_CACHEDIALECT_CACHEOPS_H

#include "Cache/BlobCache.h"
#include "Cache/CacheDialect/CacheAttrs.h"
#include "LLCL/Support/Chain.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace M::Cache {
/// Profiler entry for compile-time cache transforms.
using CacheProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 1), Trace::kCompiler>;

/// The Cache dialect can store a large constant - this struct defines the cache
/// key. It can be an Attribute (indicating that we should hash the data itself)
/// or a string (indicating we already know the hash).
struct DataCacheKey {
  using KeyTy = std::variant<Attribute, StringRef>;
  static std::string hashKey(KeyTy key);
};

/// Convenience typedef to reduce typing.
using DataCache = BlobCache<DataCacheKey>;

/// Store large constant attrs on `constant` in the cache. Currently, this is
/// done by outlining the DenseResourceElementsAttrs on `constant` and replacing
/// them with a ConstantHashAttr with the same type.
LLCL::AsyncValueRef<LLCL::Chain> deflateConstant(Operation *constant,
                                                 RCRef<DataCache> cache,
                                                 LLCL::AnyAsyncValueRef chain);

/// Pull cached constants represented by `ConstantHashAttr` from the cache and
/// replace them on `constant`.
LLCL::AsyncValueRef<LLCL::Chain> inflateConstant(Operation *constant,
                                                 RCRef<DataCache> cache,
                                                 LLCL::AnyAsyncValueRef chain);

/// The Cache dialect can store the region of an op - this struct defines the
/// cache key. It can be a region (indicating that we need to hash the region)
/// or a string (indicating that we already know the hash).
struct RegionCacheKey {
  using KeyTy = std::variant<Region *, StringRef>;
  static std::string hashKey(KeyTy key);
};

/// Convenience typedef to reduce typing.
using RegionCache = BlobCache<RegionCacheKey>;

/// Return the name used to denote that an op's regions have been cached.
inline llvm::StringLiteral getRegionHashAttrName() { return "region_hashes"; }

/// This function allows the user to deflate an operation by eliding the body
/// and storing it in the cache. If the operation is already deflated this is a
/// no-op. The deflation is implemented as an `andThenSync` on `chain` - this is
/// to simplify calling code which is also likely async.
LLCL::AsyncValueRef<LLCL::Chain> deflateOp(Operation *op,
                                           RCRef<RegionCache> cache,
                                           LLCL::AnyAsyncValueRef chain);

/// This function allows the user to inflate a cached op into its original
/// form by pulling the regions attached to it from the cache and re-attaching
/// them to the op. If the op is not deflated, this is a no-op. The inflation is
/// implemented as an `andThenSync` on `chain` - this is to simplify calling
/// code which is also likely async.
LLCL::AsyncValueRef<LLCL::Chain> inflateOp(Operation *cached,
                                           RCRef<RegionCache> cache,
                                           LLCL::AnyAsyncValueRef chain);
} // namespace M::Cache

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Cache/CacheDialect/Cache.h.inc"

#endif // CACHE_CACHEDIALECT_CACHEOPS_H
