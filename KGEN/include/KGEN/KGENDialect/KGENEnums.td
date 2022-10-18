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

  // Binary expressions.
  I32EnumAttrCase<"Shl", 5, "shl">,
  I32EnumAttrCase<"Shr", 6, "shr">,
  I32EnumAttrCase<"Div", 7, "div">,
  I32EnumAttrCase<"Mod", 8, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 9, "eq">,
  I32EnumAttrCase<"LT", 10, "lt">,
  I32EnumAttrCase<"LE", 11, "le">,

  // Complex variadic operations.
  I32EnumAttrCase<"IN", 12, "in">,

  // Unary type expressions.
  I32EnumAttrCase<"GET_DTYPE", 13, "get_dtype">,
  I32EnumAttrCase<"GET_SIZEOF", 14, "get_sizeof">,
  I32EnumAttrCase<"GET_ALIGNOF", 15, "get_alignof">,

  // TODO: Can add string concat, etc as needed.
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

//===----------------------------------------------------------------------===//
// LinkageAttr
//===----------------------------------------------------------------------===//

/// This defines the LinkageAttrInfo backing for the LinkageAttr enum.
def LinkageAttrInfo : I32EnumAttr<"Linkage", "Linkage", [
  /// Public means this symbol must be kept alive and the signature
  /// unmodified because external consumers might use it.
  I32EnumAttrCase<"Public", 0, "public">,
  /// Module Private means that nobody outside this module will use this
  /// symbol. Module in this context means mlir::ModuleOp - this will need
  /// renaming once lightning has the concept of modules.
  I32EnumAttrCase<"ModulePrivate", 1, "module_private">,
  /// LibraryPrivate means that we might have a user in our current library.
  /// It won't be visible outside the current library, but it's visible and
  /// link-able in the current library.
  I32EnumAttrCase<"LibraryPrivate", 2, "library_private">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

def LinkageAttr : EnumAttr<KGENDialect, LinkageAttrInfo, "linkage">;

#endif // KGENENUMS_TD
