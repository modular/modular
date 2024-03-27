//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHEDIALECT_TD
#define CACHEDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// CacheDialect
//===----------------------------------------------------------------------===//

def CacheDialect : Dialect {
  let name = "cache";

  let summary = "Cache dialect";
  let description = [{
    The `Cache` dialect contains transforms required for generic caching.
  }];

  let dependentDialects = ["::M::MDialect"];

  let cppNamespace = "::M::Cache";
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // CACHEDIALECT_TD
