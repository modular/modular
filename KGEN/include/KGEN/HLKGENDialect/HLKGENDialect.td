//===- HLKGENDialect.td ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the HLKGENDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLKGENDIALECT_HLKGENDIALECT_TD
#define KGEN_HLKGENDIALECT_HLKGENDIALECT_TD

def HLKGENDialect : Dialect {
  let name = "hlkgen";

  let summary = "The 'hlkgen' dialect";
  let description = [{
    This dialect defines the `hlkgen` dialect, a set of higher level
    abstractions that lower to the `kgen` dialect.
  }];

  let cppNamespace = "::M::KGEN";
  let dependentDialects = ["::M::KGEN::KGENDialect"];

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // KGEN_HLKGENDIALECT_HLKGENDIALECT_TD
