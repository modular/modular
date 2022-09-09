//===- POPDialect.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPDIALECT_TD
#define KGEN_POPDIALECT_POPDIALECT_TD

include "mlir/IR/DialectBase.td"

def POPDialect : Dialect {
  let name = "pop";

  let summary = "The 'pop' dialect";
  let description = [{
    This dialect defines the `pop` dialect, which defines parametric operations
    on top of the LLVM dialect.
  }];

  let cppNamespace = "::M::KGEN::POP";

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all POP attributes.
    void registerAttributes();
  }];
}

#endif // KGEN_POPDIALECT_POPDIALECT_TD
