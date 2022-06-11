//===- KGENDialect.td --------------------------------------*- tablegen -*-===//
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

def KGENDialect : Dialect {
  let name = "kgen";

  let summary = "The 'kgen' dialect";
  let description = [{
    This dialect defines the `kgen` dialect, which defines the top level
    and other support operations for kernel generation framework.
  }];

  let cppNamespace = "::M::KGEN";

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // KGEN_KGENDIALECT_KGENDIALECT_TD
