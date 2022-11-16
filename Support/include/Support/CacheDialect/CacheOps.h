//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CACHEDIALECT_CACHEOPS_H
#define SUPPORT_CACHEDIALECT_CACHEOPS_H

#include "Support/BlobCache.h"
#include "Support/CacheDialect/CacheAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace M::Cache {
// Forward declare SymbolOp so we can use it here.
class SymbolOp;

/// The Cache dialect stores the region of an op - this defines the cache key.
/// It can be a region (indicating that we need to hash the region) or a string
/// (indicating that we already know the hash).
struct RegionCacheKey {
  using KeyTy = std::variant<Region *, StringRef>;
  static std::string hashKey(KeyTy key);
};

/// This function allows the user to deflate a symbol op, store its body
/// in the cache, and replace it with a `cache.symbol`.
FailureOr<SymbolOp> deflateSymbol(Operation *symbol, SymbolTable &symtab,
                                  BlobCache<RegionCacheKey> &cache);

/// This function allows the user to inflate a cached symbol into its original
/// operation.
FailureOr<Operation *> inflateSymbol(SymbolOp cached, SymbolTable &symtab,
                                     BlobCache<RegionCacheKey> &cache);
} // namespace M::Cache

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/CacheDialect/Cache.h.inc"

#endif // SUPPORT_CACHEDIALECT_CACHEOPS_H
