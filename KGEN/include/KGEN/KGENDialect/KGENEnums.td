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
def KGEN_POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
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

  // Target comparisons & utilities.
  I32EnumAttrCase<"TargetEq", 14, "target_eq">,
  I32EnumAttrCase<"TargetHasFeature", 15, "target_has_feature">,
  I32EnumAttrCase<"TargetIsArch", 16, "target_is_arch">,
  I32EnumAttrCase<"TargetGetField", 17, "target_get_field">,

  // Complex variadic operations.
  I32EnumAttrCase<"In", 18, "in">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 19, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 20, "get_alignof">,
  I32EnumAttrCase<"GetListElement", 21, "get_list_element">,

  // Signature processing.
  I32EnumAttrCase<"BindSignature", 22, "bind_signature">,
  I32EnumAttrCase<"Apply", 23, "apply">
]>;
}

//===----------------------------------------------------------------------===//
// InputGenKindAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def KGEN_InputGenKindAttr : I32EnumAttr<"InputGenKind", "Input Generation Kind", [
  I32EnumAttrCase<"Zeros", 0, "zeros">,
  I32EnumAttrCase<"Ones", 1, "ones">,
  I32EnumAttrCase<"Random", 2, "random">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Conventions
//===----------------------------------------------------------------------===//

/// This describes the encoding of the value parameter conventions in a
/// SignatureType.
def KGEN_ValueInputConvention : I8BitEnumAttr<"ValueInputConvention",
    "Function value input convention", [
  I8BitEnumAttrCaseNone<"ByVal", "byval">,
  I8BitEnumAttrCaseBit<"ByRef", 0, "byref">,
  I8BitEnumAttrCaseBit<"VarArg", 1, "vararg">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in a SignatureType.
def KGEN_FnEffects : I8BitEnumAttr<"FnEffects", "Function effects", [
  I8BitEnumAttrCaseNone<"None", "none">,
  I8BitEnumAttrCaseBit<"Throws", 0, "throws">,
  I8BitEnumAttrCaseBit<"ForceInline", 1, "force_inline">,
  I8BitEnumAttrCaseBit<"Async", 2, "async">,
]> {
  let cppNamespace = "M::KGEN";
}

#endif // KGENENUMS_TD
