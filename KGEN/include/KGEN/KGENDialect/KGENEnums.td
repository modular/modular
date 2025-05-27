//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENENUMS_TD
#define KGEN_KGENDIALECT_KGENENUMS_TD

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
  I32EnumAttrCase<"Div", 10, "div">, // NOTE division by 0 is undefined behavior
  I32EnumAttrCase<"Mod", 11, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 12, "eq">,
  I32EnumAttrCase<"LT", 13, "lt">,
  I32EnumAttrCase<"LE", 14, "le">,

  // Complex variadic operations.
  I32EnumAttrCase<"In", 15, "in">,

  // Conditional expression.
  I32EnumAttrCase<"Cond", 16, "cond">,

  // Target and build comparisons & utilities.
  I32EnumAttrCase<"CurrentTarget", 17, "current_target">,
  I32EnumAttrCase<"TargetHasFeature", 18, "target_has_feature">,
  I32EnumAttrCase<"TargetGetField", 19, "target_get_field">,
  I32EnumAttrCase<"AcceleratorArch", 20, "accelerator_arch">,
  I32EnumAttrCase<"CrossCompilation", 21, "cross_compilation">,

  // Environment variables.
  I32EnumAttrCase<"GetEnv", 22, "get_env">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 23, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 24, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"Apply", 25, "apply">,
  I32EnumAttrCase<"ApplyResultSlot", 26, "apply_result_slot">,
  I32EnumAttrCase<"Rebind", 27, "rebind">,

  // Evaluator.
  I32EnumAttrCase<"VariadicGet", 28, "variadic_get">,
  I32EnumAttrCase<"CompileAssembly", 30, "compile_assembly">,
  I32EnumAttrCase<"CompileOffloadClosure", 31, "compile_offload_closure">,

  // Generics.
  I32EnumAttrCase<"GetVTableEntry", 33, "get_vtable_entry">,
  I32EnumAttrCase<"PtrBitcast", 34, "ptr_bitcast">,
  I32EnumAttrCase<"LoadFromMem", 35, "load_from_mem">,

  // Given a list of !variadic<sometype> type, return the same list with each
  // element wrapped in a kgen pointer type: [i32, f32] -> [ptr<i32>, ptr<f32>].
  // The first operand is the list, the second operand is an integer addrspace.
  I32EnumAttrCase<"VariadicPtrMap", 36, "variadic_ptr_map">,
  // Given a list of !variadic<pointer<sometype>> type, return the same list
  // with each element stripped of kgen pointer type:
  //     [ptr<i32>, ptr<f32>] -> [i32, f32].
  // This takes one operand, the list of pointer types.
  I32EnumAttrCase<"VariadicPtrRemoveMap", 37, "variadic_ptrremove_map">,

  /// Convert an arbitrary MLIR attribute into a !kgen.string using
  /// mlir::debugString on the attribute.
  I32EnumAttrCase<"AttrToStr", 39, "attr_to_str">,
  /// Convert a length+pointer representation in interpreter memory into a
  /// !kgen.string, used to convert comp-time String (and related types) to
  /// StringLiteral.
  I32EnumAttrCase<"DataToStr", 40, "data_to_str">,
  /// Get the address of a string.
  I32EnumAttrCase<"StringAddress", 41, "string_address">,
  /// Concatenate two values of !kgen.string type.
  I32EnumAttrCase<"StrConcat", 42, "str_concat">,
  /// Get the arguments from a function signature as a kgen.variadic.
  I32EnumAttrCase<"FunctionGetArgTypes", 43, "function_get_arg_types">,

  /// Division and remainder operations from `mlir.index`.
  I32EnumAttrCase<"DivS", 44, "div_s">,
  I32EnumAttrCase<"DivU", 45, "div_u">,
  I32EnumAttrCase<"CeilDivS", 46, "ceil_div_s">,
  I32EnumAttrCase<"CeilDivU", 47, "ceil_div_u">,
  I32EnumAttrCase<"FloorDivS", 48, "floor_div_s">,
  I32EnumAttrCase<"RemS", 49, "rem_s">,
  I32EnumAttrCase<"RemU", 50, "rem_u">,

  /// HELLO INTREPID COMPILER DEVELOPER!  Are you thinking about adding another
  /// POC opcode for something that isn't a trivial expression like add or sub?
  /// The list above ^^^ is massively out of control, these things are not
  /// "operators" like ParamOperatorAttr was designed for.  MLIR allows us to
  /// define new attributes, and it's really good at it!  Let's stop this
  /// pattern and progressively refactor these away over time so POC can get
  /// back to being simple arithmetic stuff some day.  Thank you, -ChrisL
]>;

//===----------------------------------------------------------------------===//
// InlineLevelAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def KGEN_InlineLevelAttr : I32EnumAttr<"InlineLevel",
          "Whether and how to inline this function", [
  I32EnumAttrCase<"Automatic", 0, "automatic">,
  I32EnumAttrCase<"Always", 1, "always">,
  I32EnumAttrCase<"AlwaysNoDebug", 2, "always_nodebug">,
  I32EnumAttrCase<"AlwaysBuiltin", 3, "always_builtin">,
  I32EnumAttrCase<"Never", 4, "never">,
]> {
  let cppNamespace = "M::KGEN";
}

//===----------------------------------------------------------------------===//
// ExportKind
//===----------------------------------------------------------------------===//

def KGEN_ExportKind : I32EnumAttr<"ExportKind", "export kind", [
  I32EnumAttrCase<"NotExported", 0, "not_exported">,
  I32EnumAttrCase<"Exported", 1, "exported">,
  I32EnumAttrCase<"CExported", 2, "c_exported">,
  I32EnumAttrCase<"PackageExported", 3, "package_exported">
]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TailKind
//===----------------------------------------------------------------------===//

def KGEN_TailKind : I32EnumAttr<"TailKind", "tail kind", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"MustTail", 1, "musttail">,
  I32EnumAttrCase<"NoTail", 2, "notail">
]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Conventions
//===----------------------------------------------------------------------===//

/// This describes the encoding of input argument calling conventions, stored
/// in the metadata of a FuncType.
///
/// That these enums have helper functions defined in KGENEnums.h for working
/// with various classes.
def KGEN_ArgConvention : I32EnumAttr<"ArgConvention",
    "Function argument convention", [
  /// This is an immutable borrow of a register-passable value in an SSA
  /// register.  The MLIR type may be anything that can be lowered to LLVM
  /// (but notably not a memory-only type).
  ///
  /// The caller guarantees the value is live for the duration of the call.
  I32EnumAttrCase<"ReadReg", 0, "read">,

  /// This is an immutably borrowed memory-only value, which is passed
  /// indirectly.
  ///
  /// The MLIR type is always a PointerType, and the underlying type is a
  /// memory-only type.  The caller passes a reference to an initialized memory
  /// slot that may be read but not mutated by the callee.
  I32EnumAttrCase<"ReadMem", 1, "read_mem">,

  /// This is an owned register-passable value an SSA register.  The MLIR type
  /// may be anything that can be lowered to LLVM (notably not a memory-only
  /// type).  The Mojo parser does not use this argument convention but
  /// LowerArgConventions produces it.
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedReg", 2, "owned">,

  /// This is an owned memory-only value, which is passed indirectly.
  ///
  /// The MLIR type is always a PointerType, and the underlying type is a
  /// memory-only type.  The caller initializes the memory slot and the callee
  /// destroys it, returning with it uninitialized.
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedMem", 3, "owned_in_mem">,

  /// This is used for mut arguments like `fn x(mut a: Int)`.
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function and must be initialized when the function returns.
  I32EnumAttrCase<"Mut", 4, "mut">,

  /// This is used for ref arguments like:
  ///   `fn x[origin: X](ref [origin, addrspace] T)`
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function.
  I32EnumAttrCase<"Ref", 5, "ref">,

  /// This is used for `ref []` arguments that are guaranteed to always be
  /// mutable, and thus are known to be no-alias.
  I32EnumAttrCase<"MutRef", 6, "mutref">,

  /// This is used on the last argument of a function that returns a
  /// memory-only value, requiring it to be returned through memory.
  ///
  /// The MLIR type is always a PointerType, and the underlying type can be
  /// anything.  The value is uninitialized on entry to the function and defined
  /// on exit from the function if it returns.
  ///
  /// The caller is responsible for destroying the value.
  I32EnumAttrCase<"ByRefResult", 7, "byref_result">,

  /// This is used on the second-last argument of a function that may raise,
  /// returning a potentially initialized error value if the function throws.
  I32EnumAttrCase<"ByRefError", 9, "byref_error">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in the metadata of a
/// FuncType.
def KGEN_FnEffectsImpl : I16BitEnumAttr<"FnEffects", "Function effects", [
  /// This function has no effects.
  I16BitEnumAttrCaseNone<"None", "none">,
  /// This function can throw.
  I16BitEnumAttrCaseBit<"Throws", 0, "throws">,
  /// This is an async function.
  I16BitEnumAttrCaseBit<"Async", 1, "async">,
  /// This function captures state.
  I16BitEnumAttrCaseBit<"Capturing", 2, "capturing">,
  /// This local function has state that may live beyond its defined scope.
  I16BitEnumAttrCaseBit<"Escaping", 4, "escaping">,
  /// This function has a auto-derefed result of !lit.ref or !kgen.pointer type.
  I16BitEnumAttrCaseBit<"RefResult", 5, "refresult">,
]> {
  let cppNamespace = "M::KGEN::impl";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ComputeKindAttr
//===----------------------------------------------------------------------===//

def ComputeKind : I32EnumAttr<"ComputeKind", "compute kind", [
  I32EnumAttrCase<"Addition",       0, "addition">,
  I32EnumAttrCase<"Comparison",     1, "comparison">,
  I32EnumAttrCase<"Division",       2, "division">,
  I32EnumAttrCase<"Multiplication", 3, "multiplication">,
  I32EnumAttrCase<"MultiplyAdd",    4, "multiply_add">,
  I32EnumAttrCase<"Other",          5, "other">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// EmitAs
//===----------------------------------------------------------------------===//

def EmitAs : I32EnumAttr<"EmitAs", "compile_assembly emission kind", [
  I32EnumAttrCase<"ASM",  0, "asm">,
  I32EnumAttrCase<"LLVM", 1, "llvm">,
  I32EnumAttrCase<"LLVM_OPT", 2, "llvm-opt">,
  I32EnumAttrCase<"OBJECT", 3, "object">
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ClosureMethod
//===----------------------------------------------------------------------===//

def KGEN_ClosureMethod : I32EnumAttr<"ClosureMethod", "closure method", [
  I32EnumAttrCase<"CALL",  0, "call">,
  I32EnumAttrCase<"DEL", 1, "del">,
  I32EnumAttrCase<"MOVE", 2, "move">,
  I32EnumAttrCase<"COPY", 3, "copy">
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ClosureMemoryKind
//===----------------------------------------------------------------------===//

def KGEN_ClosureMemoryKind : I32EnumAttr<"ClosureMemoryKind", "closure memory kind", [
  I32EnumAttrCase<"ESCAPING",  0, "escaping">,
  I32EnumAttrCase<"NONESCAPING", 1, "nonescaping">,
  I32EnumAttrCase<"REGISTER_PASSABLE", 2, "registerpassable">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

#endif // KGEN_KGENDIALECT_KGENENUMS_TD
