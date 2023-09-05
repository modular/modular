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
    The `Cache` dialect contains types, attributes, and operations required
    for generic caching.
  }];

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect ops.
    void registerOps();
    /// Inject attribute interfaces.
    void injectAttrInterfaces();
  }];

  let dependentDialects = ["::M::MDialect"];

  let cppNamespace = "::M::Cache";
  let useDefaultAttributePrinterParser = 1;
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // CACHEDIALECT_TD
