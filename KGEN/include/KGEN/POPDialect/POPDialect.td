//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPDIALECT_TD
#define KGEN_POPDIALECT_POPDIALECT_TD

include "mlir/IR/OpBase.td"

def POPDialect : Dialect {
  let name = "pop";

  let summary = "The 'pop' dialect";
  let description = [{
    This dialect defines the `pop` dialect, which defines parametric operations
    on top of the LLVM dialect.
  }];

  let cppNamespace = "::M::KGEN::POP";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    /// Register all POP attributes.
    void registerAttributes();
    /// Register all POP types.
    void registerTypes();
  }];
}

#endif // KGEN_POPDIALECT_POPDIALECT_TD
