//===- POPDialect.td ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPDIALECT_TD
#define KGEN_POPDIALECT_POPDIALECT_TD

def POPDialect : Dialect {
  let name = "pop";

  let summary = "The 'pop' dialect";
  let description = [{
    This dialect defines the `pop` dialect, which defines parametric
    operations on top of both the LLVM and arith dialects.
  }];

  let cppNamespace = "::M::KGEN";

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let hasConstantMaterializer = 1;
}

#endif // KGEN_POPDIALECT_POPDIALECT_TD
