//===----------------------------------------------------------------------===//
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

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    /// Register all POP attributes.
    void registerAttributes();
    /// Register all POP types.
    void registerTypes();
    /// Custom parser for POP types. The reason we customize it is to be able
    /// "scalar<t>" as "simd<1, t>".
    Type parseType(DialectAsmParser &parser) const override;
    /// Custom printer for POP types.
    void printType(Type type, DialectAsmPrinter &p) const override;
  }];
}

#endif // KGEN_POPDIALECT_POPDIALECT_TD
