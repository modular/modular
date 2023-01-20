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
    /// Inject any external attribute models.
    void injectAttrInterfaces();
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  // https://discourse.llvm.org/t/psa-new-improved-fold-method-signature-has-landed-please-update-your-downstream-projects/67618
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // MDIALECT_TD
