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
  I32EnumAttrCase<"MulNuw", 2, "mul_nuw">,
  I32EnumAttrCase<"And", 3, "and">,
  I32EnumAttrCase<"Or",  4, "or">,
  I32EnumAttrCase<"Xor", 5, "xor">,
  I32EnumAttrCase<"Max", 6, "max">,
  I32EnumAttrCase<"Min", 7, "min">,

  // Binary expressions.
  I32EnumAttrCase<"Shl", 8, "shl">,
  I32EnumAttrCase<"Shr", 9, "shr">,
  I32EnumAttrCase<"Div", 10, "div">,
  I32EnumAttrCase<"Mod", 11, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 12, "eq">,
  I32EnumAttrCase<"LT", 13, "lt">,
  I32EnumAttrCase<"LE", 14, "le">,

  // Target comparisons & utilities.
  I32EnumAttrCase<"CurrentTarget", 15, "current_target">,
  I32EnumAttrCase<"TargetHasFeature", 16, "target_has_feature">,
  I32EnumAttrCase<"TargetGetField", 17, "target_get_field">,

  // BuildInfo comparisons & utilities.
  I32EnumAttrCase<"BuildInfoGetField", 18, "build_info_get_field">,

  // Complex variadic operations.
  I32EnumAttrCase<"In", 19, "in">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 20, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 21, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"BindSignature", 22, "bind_signature">,
  I32EnumAttrCase<"Apply", 23, "apply">,
  I32EnumAttrCase<"Rebind", 24, "rebind">,

  // Evaluator.
  I32EnumAttrCase<"GetAllImpls", 25, "get_all_impls">,
  I32EnumAttrCase<"VariadicGet", 26, "variadic_get">,

  // Conditional expression.
  I32EnumAttrCase<"Cond", 27, "cond">,

  // Environment variables.
  I32EnumAttrCase<"GetEnv", 28, "get_env">
]>;

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
}

//===----------------------------------------------------------------------===//
// Conventions
//===----------------------------------------------------------------------===//

/// This describes the encoding of input argument calling conventions, stored
/// in the metadata of a SignatureType.
def KGEN_ValueInputConvention : I32EnumAttr<"ValueInputConvention",
    "Function value input convention", [
  /// This is an owned register-passable value an SSA register.  The MLIR type
  /// may be anything that can be lowered to LLVM (notably not a memory-only
  /// type).
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedInReg", 0, "owned">,

  /// This is an owned memory-only value, which is passed indirectly.
  ///
  /// The MLIR type is always a POP::PointerType, and the underlying type is a
  /// memory-only type.  The caller initializes the memory slot and the callee
  /// destroys it, returning with it uninitialized.
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedInMem", 1, "owned_in_mem">,

  /// This is an immutable borrow of a register-passable value in an SSA
  /// register.  The MLIR type may be anything that can be lowered to LLVM
  /// (but notably not a memory-only type).
  ///
  /// The caller guarantees the value is live for the duration of the call.
  I32EnumAttrCase<"BorrowedInReg", 2, "borrow">,

  /// This is an immutably borrowed memory-only value, which is passed
  /// indirectly.
  ///
  /// The MLIR type is always a POP::PointerType, and the underlying type is a
  /// memory-only type.  The caller passes a reference to an uninitialized
  /// memory slot that may be read but not mutated by the callee.
  I32EnumAttrCase<"BorrowedInMem", 3, "borrow_in_mem">,

  /// This is used for by-ref arguments like `fn x(a&: Int)`.
  ///
  /// The MLIR type is always a POP::PointerType, and the underlying type can be
  /// anything.  The value is initialized on entry to the function and must be
  /// initialized when the function returns.
  I32EnumAttrCase<"ByRef", 4, "byref">,

  /// This is used on the first argument of a function that returns a
  /// memory-only value, requiring it to be returned through memory.
  ///
  /// The MLIR type is always a POP::PointerType, and the underlying type can be
  /// anything.  The value is uninitialized on entry to the function and defined
  /// on exit from the function if it returns.
  ///
  /// The caller is responsible for destroying the value.
  I32EnumAttrCase<"ByRefResult", 5, "byref_result">,

  /// This is used on the 'self' argument to `__init__`.  It has identical
  /// semantics to ByRefResult on the callee side, but isn't a result on the
  /// caller side.
  I32EnumAttrCase<"InitSelf", 6, "init_self">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in the metadata of a
/// SignatureType.
def KGEN_FnEffects : I8BitEnumAttr<"FnEffects", "Function effects", [
  /// This function has no effects.
  I8BitEnumAttrCaseNone<"None", "none">,
  /// This function can throw.
  I8BitEnumAttrCaseBit<"Throws", 0, "throws">,
  /// This is an async function.
  I8BitEnumAttrCaseBit<"Async", 1, "async">,
  /// This function has vararg arguments.
  I8BitEnumAttrCaseBit<"Vararg", 2, "vararg">,
  /// This function has pack arguments.
  I8BitEnumAttrCaseBit<"PackVararg", 3, "packvararg">,
  /// This function has keyword vararg arguments.
  I8BitEnumAttrCaseBit<"KWVararg", 4, "kwvararg">,
  /// This function has vararg input parameters.
  I8BitEnumAttrCaseBit<"ParamVararg", 5, "param_vararg">,
  /// This function captures state.
  I8BitEnumAttrCaseBit<"Capturing", 6, "capturing">,
  /// The return value of this function is a non-trivial owned register value.
  I8BitEnumAttrCaseBit<"OwnedResult", 7, "ownedresult">,
]> {
  let cppNamespace = "M::KGEN";
}

#endif // KGENENUMS_TD
