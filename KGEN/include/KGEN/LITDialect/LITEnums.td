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

//===----------------------------------------------------------------------===//
// VariadicKind
//===----------------------------------------------------------------------===//

def LIT_VariadicKind : I32EnumAttr<"VariadicKind",
    "Whether an argument is a homogenous variadic or pack", [
  I32EnumAttrCase<"None", 0, "not_vararg">,
  I32EnumAttrCase<"PosVarArg", 1, "pos_vararg">,
  I32EnumAttrCase<"PackVarArg", 2, "pack_vararg">,
  I32EnumAttrCase<"KwVarArg", 3, "kw_vararg">
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TypeConvention
//===----------------------------------------------------------------------===//

def LIT_TypeConvention : I32EnumAttr<"TypeConvention",
    "Whether a type is register passable or memory-only", [
  I32EnumAttrCase<"MemoryOnly", 0, "memory_only">,
  I32EnumAttrCase<"RegisterPassable", 1, "register_passable">,
  // TODO(#14086): Trivial should be a separate thing.
  I32EnumAttrCase<"RegisterPassableTrivial", 2, "register_passable_trivial">,
  I32EnumAttrCase<"Unspecified", 3, "unspecified">
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// PassingKind
//===----------------------------------------------------------------------===//

/// Defines how an argument/parameter is allowed (by declaratopm) to be passed.
def LIT_PassingKind : I32EnumAttr<"PassingKind",
    "How an argument or parameter is allowed to be passed", [
  I32EnumAttrCase<"PosOrKw", 0, "pos_or_kw">,
  I32EnumAttrCase<"PosOnly", 1, "pos">,
  I32EnumAttrCase<"KwOnly", 2, "kw">,
  // Things like byref results/errors and _Self in Traits.
  I32EnumAttrCase<"Implicit", 3, "implicit">,
  // User-declared inferred-only params like: fn x[a: Bool, //, b: Int](...
  I32EnumAttrCase<"Inferred", 4, "inferred"> 
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// VarDeclKind
//===----------------------------------------------------------------------===//

def LIT_VarDeclKind : I32EnumAttr<"VarDeclKind",
    "origin of a VarDeclOp", [
  /// Value is declared explicitly by the user to be mutable.
  I32EnumAttrCase<"Var", 0, "var">,
  /// Value is declared implicitly by the user via non-var/let syntax.
  /// For example, implicit declarations in `def` functions, target declarations
  /// in `for` statements.
  I32EnumAttrCase<"Implicit", 1, "imp">,
  /// Value is not declared by the user at all.
  I32EnumAttrCase<"Synthesized", 2, "synth">,
  /// Value is a shadow of a function argument, for arguments that need a
  /// variable location.
  I32EnumAttrCase<"Arg", 3, "arg">,
  /// Value is a shadow of an 'out' argument for a register passable
  /// initializer.
  I32EnumAttrCase<"InitOutArg", 4, "initoutarg">,
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

#endif // KGEN_LITDIALECT_LITENUMS_TD
