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

/// Parameter Expression Opcodes: This defines the Parameter Operator Code enum
/// attribute.
let cppNamespace = "M::KGEN" in
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
  I32EnumAttrCase<"CurrentTarget", 14, "current_target">,
  I32EnumAttrCase<"TargetHasFeature", 15, "target_has_feature">,
  I32EnumAttrCase<"TargetGetField", 16, "target_get_field">,

  // BuildInfo comparisons & utilities.
  I32EnumAttrCase<"BuildInfoGetField", 17, "build_info_get_field">,

  // Complex variadic operations.
  I32EnumAttrCase<"In", 18, "in">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 19, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 20, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"BindSignature", 21, "bind_signature">,
  I32EnumAttrCase<"Apply", 22, "apply">,
  I32EnumAttrCase<"Rebind", 23, "rebind">,

  // Evaluating a set of candidates.
  I32EnumAttrCase<"Evaluate", 24, "evaluate">,

  // Enumerating all concrete implementations for a given signature.
  I32EnumAttrCase<"GetAllImpls", 25, "get_all_impls">
]>;

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
// AlwaysInlineLevelAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def KGEN_AlwaysInlineLevelAttr : I32EnumAttr<"AlwaysInlineLevel",
          "Whether and how to always_inline this function", [
  I32EnumAttrCase<"Disabled", 0, "disabled">,
  I32EnumAttrCase<"Enabled", 1, "enabled">,
  I32EnumAttrCase<"EnabledNoDebug", 2, "enabled_nodebug">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 1;
}

//===----------------------------------------------------------------------===//
// Conventions
//===----------------------------------------------------------------------===//

/// This describes the encoding of input argument calling conventions, stored
/// in the metadata of a SignatureType.
def KGEN_ValueInputConvention : I8BitEnumAttr<"ValueInputConvention",
    "Function value input convention", [
  // TODO: Modeling this as a bitfield is a hack, this is really two different
  // non-bitfield enums put together: a varargs marker and an ownership
  // qualifier.
  I8BitEnumAttrCaseNone<"ByVal",       "byval">,
  I8BitEnumAttrCaseBit <"ByRef",    0, "byref">,
  I8BitEnumAttrCaseBit <"VarArg",   1, "vararg">,
  I8BitEnumAttrCaseBit <"KWVarArg", 2, "kwvararg">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in the metadata of a
/// SignatureType.
def KGEN_FnEffects : I8BitEnumAttr<"FnEffects", "Function effects", [
  I8BitEnumAttrCaseNone<"None", "none">,
  I8BitEnumAttrCaseBit<"Throws", 0, "throws">,
  I8BitEnumAttrCaseBit<"Async", 1, "async">,
]> {
  let cppNamespace = "M::KGEN";
}

#endif // KGENENUMS_TD
