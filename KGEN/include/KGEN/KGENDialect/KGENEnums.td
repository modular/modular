//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGENENUMS_TD
#define KGENENUMS_TD

include "KGEN/KGENDialect/KGENDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// POCAttr
//===----------------------------------------------------------------------===//

/// Parameter Expression Opcodes.
let cppNamespace = "M::KGEN" in {

/// This defines the Parameter Operator Code enum attribute.
def POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
  /// Fully associative variadic expressions.
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Mul", 1, "mul">,
  I32EnumAttrCase<"And", 2, "and">,
  I32EnumAttrCase<"Or",  3, "or">,
  I32EnumAttrCase<"Xor", 4, "xor">,
  I32EnumAttrCase<"Max", 5, "max">,
  I32EnumAttrCase<"Min", 6, "min">,

  // Binary expressions.
  I32EnumAttrCase<"Shl", 7, "shl">,
  I32EnumAttrCase<"Shr", 8, "shr">,
  I32EnumAttrCase<"Div", 9, "div">,
  I32EnumAttrCase<"Mod", 10, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 11, "eq">,
  I32EnumAttrCase<"LT", 12, "lt">,
  I32EnumAttrCase<"LE", 13, "le">,

  // Complex variadic operations.
  I32EnumAttrCase<"In", 14, "in">,

  // Unary type expressions.
  I32EnumAttrCase<"GetDType",   15, "get_dtype">,
  I32EnumAttrCase<"GetSizeOf",  16, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 17, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"BindSignature", 18, "bind_signature">,
  I32EnumAttrCase<"Apply", 19, "apply">

  // TODO: Can add concat_string, etc as needed.
]>;
}

//===----------------------------------------------------------------------===//
// InputGenKindAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def InputGenKindAttr : I32EnumAttr<"InputGenKind", "Input Generation Kind", [
  I32EnumAttrCase<"Zeros", 0, "zeros">,
  I32EnumAttrCase<"Ones", 1, "ones">,
  I32EnumAttrCase<"Random", 2, "random">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

#endif // KGENENUMS_TD
