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
// PassingKind
//===----------------------------------------------------------------------===//

/// Defines how an argument/parameter is allowed (by declaratopm) to be passed.
def LIT_PassingKind : I32EnumAttr<"PassingKind",
    "How an argument or parameter is allowed to be passed", [
  I32EnumAttrCase<"PosOrKw", 0, "pos_or_kw">,
  I32EnumAttrCase<"PosOnly", 1, "pos">,
  I32EnumAttrCase<"KwOnly", 2, "kw">,
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// VarLetDeclKind
//===----------------------------------------------------------------------===//

def LIT_VarLetDeclKind : I32EnumAttr<"VarLetDeclKind",
    "origin of a VarLetDeclOp", [
  /// Value is declared explicitly by the user to be immutable.
  I32EnumAttrCase<"Let", 0, "let">,
  /// Value is declared explicitly by the user to be mutable.
  I32EnumAttrCase<"Var", 1, "var">,
  /// Value is declared implicitly by the user via non-var/let syntax.
  /// For example, implicit declarations in `def` functions, target declarations
  /// in `for` statements.
  I32EnumAttrCase<"Implicit", 2, "imp">,
  /// Value is not declared by the user at all.
  I32EnumAttrCase<"Synthesized", 3, "synth">,
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

#endif // KGEN_LITDIALECT_LITENUMS_TD
