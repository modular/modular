//===- MetaDialect.td -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the MetaDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_METADIALECT_METADIALECT_TD
#define KGEN_METADIALECT_METADIALECT_TD

def MetaDialect : Dialect {
  let name = "meta";

  let summary = "The 'meta' dialect";
  let description = [{
    This dialect defines the `meta` dialect, which includes types and operations
    used with parameterized generators.  This is separate from the kgen dialect
    because it is not core to the working of the compiler, more part of the
    "standard library".
  }];

  // We don't need an extra C++ namespace, keep this in the KGEN namespace.
  let cppNamespace = "::M::KGEN";
  let dependentDialects = ["::M::KGEN::KGENDialect"];

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif // KGEN_METADIALECT_METADIALECT_TD
