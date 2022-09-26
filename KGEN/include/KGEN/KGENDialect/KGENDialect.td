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

  let extraClassDeclaration = [{
    /// Register all KGEN attributes.
    void registerAttributes();
  }];
}

#endif // KGEN_KGENDIALECT_KGENDIALECT_TD
