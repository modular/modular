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

} // namespace M::Cache

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Cache/CacheDialect/Cache.h.inc"

#endif // CACHE_CACHEDIALECT_CACHEOPS_H
