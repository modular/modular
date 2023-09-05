//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MDIALECT_TD
#define MDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

def MDialect : Dialect {
  let name = "M";

  let summary = "M dialect";
  let description = [{
    The `M` dialect contains core types and attributes that depend only on core
    MLIR and which can be shared across all Modular code. Conceptually, this
    dialect is like the `builtin` dialect.
  }];

  let cppNamespace = "::M";

  let extraClassDeclaration = [{
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect types.
    void registerTypes();
    /// Inject any external type / attribute models.
    void injectTypeInterfaces();
    void injectAttrInterfaces();
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // MDIALECT_TD
