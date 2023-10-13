//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines enums in the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITENUMS_TD
#define KGEN_LITDIALECT_LITENUMS_TD

include "mlir/IR/EnumAttr.td"

/// Defines how an argument/parameter is allowed (by declaratopm) to be passed.
def LIT_PassingKind : I32EnumAttr<"PassingKind",
    "How an argument or parameter is allowed to be passed", [
  I32EnumAttrCase<"PosOrKw", 0, "pos_or_kw">,
  I32EnumAttrCase<"PosOnly", 1, "pos">,
  I32EnumAttrCase<"KwOnly", 2, "kw">,
]> {
  let cppNamespace = "M::KGEN::LIT";
}

#endif // KGEN_LITDIALECT_LITENUMS_TD
