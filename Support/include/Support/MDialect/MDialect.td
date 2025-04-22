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

    This dialect also holds a general purpose context that has effective global
    singletons within the Modular stack, such as the runtime.
  }];

  let cppNamespace = "::M";

  let extraClassDeclaration = [{
  public:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect types.
    void registerTypes();
    /// Inject any external type / attribute models.
    void injectTypeInterfaces();
    void injectAttrInterfaces();

    /// Get the registered context.
    ContextRef getInternal() { return context.copy(); }
    /// Set the registered context.
    void setInternal(ContextRef &&ref) { context = std::move(ref); }
  private:
    ContextRef context;
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // MDIALECT_TD
