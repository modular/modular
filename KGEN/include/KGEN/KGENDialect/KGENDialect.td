//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the KGENDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDIALECT_TD
#define KGEN_KGENDIALECT_KGENDIALECT_TD

include "mlir/IR/OpBase.td"

def KGENDialect : Dialect {
  let name = "kgen";

  let summary = "The 'kgen' dialect";
  let description = [{
    This dialect defines the `kgen` dialect, which defines the top level
    and other support operations for kernel generation framework.
  }];

  let cppNamespace = "::M::KGEN";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = ["MDialect"];

  let extraClassDeclaration = [{
    /// Register all KGEN attributes.
    void registerAttributes();
    /// Register all KGEN types.
    void registerTypes();
    /// Inject any external attribute models.
    void injectAttrInterfaces();
  }];

  // https://discourse.llvm.org/t/psa-new-improved-fold-method-signature-has-landed-please-update-your-downstream-projects/67618
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // KGEN_KGENDIALECT_KGENDIALECT_TD
