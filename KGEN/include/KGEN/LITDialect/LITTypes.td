//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITTYPES_TD
#define KGEN_LITDIALECT_LITTYPES_TD

include "KGEN/LITDialect/LITDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class LITType<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<LITDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TypeCheckErrorType
//===----------------------------------------------------------------------===//

def TypeCheckErrorType : LITType<"TypeCheckError", "typecheckerror"> {
  let summary = "Type used for values and decls that can't be type checked";
  let description = [{
    This type is assigned to values and declarations when their signature cannot
    be type checked or there is some other problem that might make references to
    it produce spurious downstream errors.  Because the presence of this type
    silences downstream errors, it should only be used when a primary error has
    already been emitted (about whatever is wrong with the decl).
  }];
}

//===----------------------------------------------------------------------===//
// ObjectType
//===----------------------------------------------------------------------===//

def ObjectType : LITType<"Object", "object"> {
  let summary = "Type used for implicit object values";
  let description = [{
    This type is used for implicitly dynamically typed values, e.g. parameters
    to functions without explicit types.
  }];
}

//===----------------------------------------------------------------------===//
// UnresolvedType
//===----------------------------------------------------------------------===//

def UnresolvedType : LITType<"Unresolved", "unresolved"> {
  let summary = "Type placeholder used during parsing";
  let description = [{
    Declarations are parsed with a multiphase approach where the declaration
    is parsed, then name binding is performed after that.  This type is used
    to represent the type of declarations before name binding is complete.

    This only occurs as the root type for a declaration, it never occurs in a
    nested position.
  }];
}

#endif // KGEN_LITDIALECT_LITTYPES_TD
