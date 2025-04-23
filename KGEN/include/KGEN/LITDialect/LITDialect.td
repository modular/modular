//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the LITDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITDIALECT_TD
#define KGEN_LITDIALECT_LITDIALECT_TD

include "mlir/IR/DialectBase.td"

def LITDialect : Dialect {
  let name = "lit";

  let summary = "The 'lit' dialect for Lightning";
  let description = [{
    This dialect defines the `lit` dialect, a set of language-level
    abstractions that lower to the `kgen` dialect.
  }];

  let cppNamespace = "::M::KGEN::LIT";
  let dependentDialects = [
    "KGENDialect",
    "POP::POPDialect",
    "CO::CODialect",
    "HLCF::HLCFDialect",
    "DebugInfo::DebugInfoDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &p) const override;
    void printType(Type type, DialectAsmPrinter &p) const override;
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // KGEN_LITDIALECT_LITDIALECT_TD
