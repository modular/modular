//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEDIALECT_CACHEOPS_H
#define CACHE_CACHEDIALECT_CACHEOPS_H

#include "Cache/BlobCache.h"
#include "Cache/CacheDialect/CacheAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace M::Cache {
/// The Cache dialect stores the region of an op - this defines the cache key.
/// It can be a region (indicating that we need to hash the region) or a string
/// (indicating that we already know the hash).
struct RegionCacheKey {
  using KeyTy = std::variant<Region *, StringRef>;
  static std::string hashKey(KeyTy key);
};

/// Return the name used to denote that an op's regions have been cached.
inline llvm::StringLiteral getRegionHashAttrName() { return "region_hashes"; }

/// This function allows the user to deflate an operation by eliding the body
/// and storing it in the cache.
LogicalResult deflateOp(Operation *symbol, BlobCache<RegionCacheKey> &cache);

/// This function allows the user to inflate a cached op into its original
/// form by pulling the regions attached to it from the cache and re-attaching
/// them to the op.
LogicalResult inflateOp(Operation *cached, BlobCache<RegionCacheKey> &cache);
} // namespace M::Cache

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Cache/CacheDialect/Cache.h.inc"

#endif // CACHE_CACHEDIALECT_CACHEOPS_H
