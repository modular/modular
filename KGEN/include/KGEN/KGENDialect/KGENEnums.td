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

  // Environment variables.
  I32EnumAttrCase<"GetEnv", 20, "get_env">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 21, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 22, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"BindSignature", 23, "bind_signature">,
  I32EnumAttrCase<"Apply", 24, "apply">,
  I32EnumAttrCase<"ApplyResultSlot", 25, "apply_result_slot">,
  I32EnumAttrCase<"Rebind", 26, "rebind">,

  // Evaluator.
  I32EnumAttrCase<"VariadicGet", 28, "variadic_get">,
  I32EnumAttrCase<"CompileAssembly", 30, "compile_assembly">,
  I32EnumAttrCase<"GetLinkageName", 31, "get_linkage_name">,

  // Generics.
  I32EnumAttrCase<"GetTypeMethod", 32, "get_type_method">,
  I32EnumAttrCase<"PtrBitcast", 33, "ptr_bitcast">,
  I32EnumAttrCase<"LoadFromMem", 34, "load_from_mem">,

  // Given a list of !variadic<sometype> type, return the same list with each
  // element wrapped in a kgen pointer type: [i32, f32] -> [ptr<i32>, ptr<f32>].
  // The first operand is the list, the second operand is an integer addrspace.
  I32EnumAttrCase<"VariadicPtrMap", 35, "variadic_ptr_map">,
  // Given a list of !variadic<pointer<sometype>> type, return the same list
  // with each element stripped of kgen pointer type:
  //     [ptr<i32>, ptr<f32>] -> [i32, f32].
  // This takes one operand, the list of pointer types.
  I32EnumAttrCase<"VariadicPtrRemoveMap", 36, "variadic_ptrremove_map">,
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
  I32EnumAttrCase<"Never", 3, "never">,
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
/// in the metadata of a SignatureType.
def KGEN_ArgConvention : I32EnumAttr<"ArgConvention",
    "Function argument convention", [
  /// This is an immutable borrow of a register-passable value in an SSA
  /// register.  The MLIR type may be anything that can be lowered to LLVM
  /// (but notably not a memory-only type).
  ///
  /// The caller guarantees the value is live for the duration of the call.
  I32EnumAttrCase<"BorrowedInReg", 0, "borrow">,

  /// This is an immutably borrowed memory-only value, which is passed
  /// indirectly.
  ///
  /// The MLIR type is always a PointerType, and the underlying type is a
  /// memory-only type.  The caller passes a reference to an initialized memory
  /// slot that may be read but not mutated by the callee.
  I32EnumAttrCase<"BorrowedInMem", 1, "borrow_in_mem">,

  /// This is an owned register-passable value an SSA register.  The MLIR type
  /// may be anything that can be lowered to LLVM (notably not a memory-only
  /// type).
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedInReg", 2, "owned">,

  /// This is an owned memory-only value, which is passed indirectly.
  ///
  /// The MLIR type is always a PointerType, and the underlying type is a
  /// memory-only type.  The caller initializes the memory slot and the callee
  /// destroys it, returning with it uninitialized.
  ///
  /// The callee is responsible for destroying the value before returning.
  I32EnumAttrCase<"OwnedInMem", 3, "owned_in_mem">,

  /// This is used for inout arguments like `fn x(inout a: Int)`.
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function and must be initialized when the function returns.
  I32EnumAttrCase<"InOut", 4, "inout">,

  /// This is used for ref arguments like:
  ///   `fn x[,,](ref [lifetime, addrspace] T)`
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function.
  I32EnumAttrCase<"Ref", 5, "ref">,

  /// This is used on the last argument of a function that returns a
  /// memory-only value, requiring it to be returned through memory.
  ///
  /// The MLIR type is always a PointerType, and the underlying type can be
  /// anything.  The value is uninitialized on entry to the function and defined
  /// on exit from the function if it returns.
  ///
  /// The caller is responsible for destroying the value.
  I32EnumAttrCase<"ByRefResult", 6, "byref_result">,

  /// This is used on the 'self' argument to `__init__`.  It has identical
  /// semantics to ByRefResult on the callee side, but isn't a result on the
  /// caller side.
  I32EnumAttrCase<"InitSelf", 7, "init_self">,

  /// This is used on the second-last argument of a function that may raise,
  /// returning a potentially initialized error value if the function throws.
  I32EnumAttrCase<"ByRefError", 8, "byref_error">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in the metadata of a
/// SignatureType.
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
// IntLiteral support
//===----------------------------------------------------------------------===//

/// This encodes which specific comparison operation is to be performed
/// for `kgen.int_literal.cmp` ops.
def KGEN_IntLiteralCmpPred : I32EnumAttr<"IntLiteralCmpPred",
  "operation specification for IntLiteral comparison", [
  I32EnumAttrCase<"Eq", 0, "eq">,
  I32EnumAttrCase<"Ne", 1, "ne">,
  I32EnumAttrCase<"Lt", 2, "lt">,
  I32EnumAttrCase<"Le", 3, "le">,
  I32EnumAttrCase<"Gt", 4, "gt">,
  I32EnumAttrCase<"Ge", 5, "ge">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_IntLiteralCmpPredAttr : EnumAttr<KGENDialect, KGEN_IntLiteralCmpPred,
                                          "int_literal.cmp_pred">;

/// This encodes which specific operation is to be performed
/// for `kgen.int_literal.binop` ops.
def KGEN_IntLiteralBinopKind : I32EnumAttr<"IntLiteralBinopKind",
  "operation specification for IntLiteral binary operations", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"FloorDiv", 3, "floordiv">,
  I32EnumAttrCase<"Mod", 4, "mod">,
  I32EnumAttrCase<"Lshift", 5, "lshift">,
  I32EnumAttrCase<"Rshift", 6, "rshift">,
  I32EnumAttrCase<"And", 7, "and">,
  I32EnumAttrCase<"Or", 8, "or">,
  I32EnumAttrCase<"Xor", 9, "xor">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_IntLiteralBinopKindAttr : EnumAttr<KGENDialect,
                                            KGEN_IntLiteralBinopKind,
                                            "int_literal.binop_kind">;

//===----------------------------------------------------------------------===//
// FloatLiteral support
//===----------------------------------------------------------------------===//

/// This encodes special values for `#kgen.float_literal` attributes.
def KGEN_FloatLiteralSpecialValues : I32EnumAttr<"FloatLiteralSpecialValues",
  "Special value kinds for float literals", [
  I32EnumAttrCase<"Normal", 0, "normal">,
  I32EnumAttrCase<"NegZero", 1, "neg_zero">,
  I32EnumAttrCase<"Inf", 2, "inf">,
  I32EnumAttrCase<"NegInf", 3, "neg_inf">,
  I32EnumAttrCase<"Nan", 4, "nan">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_FloatLiteralSpecialValuesAttr :
  EnumAttr<KGENDialect, KGEN_FloatLiteralSpecialValues,
           "float_literal.special_values">;

/// This encodes which specific comparison operation is to be performed
/// for `kgen.float_literal.cmp` ops.
def KGEN_FloatLiteralCmpPred : I32EnumAttr<"FloatLiteralCmpPred",
  "operation specification for FloatLiteral comparison", [
  I32EnumAttrCase<"Eq", 0, "eq">,
  I32EnumAttrCase<"Ne", 1, "ne">,
  I32EnumAttrCase<"Lt", 2, "lt">,
  I32EnumAttrCase<"Le", 3, "le">,
  I32EnumAttrCase<"Gt", 4, "gt">,
  I32EnumAttrCase<"Ge", 5, "ge">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_FloatLiteralCmpPredAttr : EnumAttr<KGENDialect,
                                            KGEN_FloatLiteralCmpPred,
                                            "float_literal.cmp_pred">;

/// This encodes which specific operation is to be performed
/// for `kgen.float_literal.binop` ops.
def KGEN_FloatLiteralBinopKind : I32EnumAttr<"FloatLiteralBinopKind",
  "operation specification for FloatLiteral binary operations", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"TrueDiv", 3, "truediv">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_FloatLiteralBinopKindAttr : EnumAttr<KGENDialect,
                                              KGEN_FloatLiteralBinopKind,
                                              "float_literal.binop_kind">;

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

#endif // KGEN_KGENDIALECT_KGENENUMS_TD
