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

def LIT_TypeCheckErrorType : LITType<"TypeCheckError", "typecheckerror"> {
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
// UnresolvedType
//===----------------------------------------------------------------------===//

def LIT_UnresolvedType : LITType<"Unresolved", "unresolved"> {
  let summary = "Type placeholder used during parsing";
  let description = [{
    Declarations are parsed with a multiphase approach where the declaration
    is parsed, then name binding is performed after that.  This type is used
    to represent the type of declarations before name binding is complete.

    This only occurs as the root type for a declaration, it never occurs in a
    nested position.
  }];
}

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

def LIT_NoneType : LITType<"None", "none"> {
  let summary = "Type used for 'void' function results";
  let description = [{
    This type is nominal static type of the 'None' keyword, which is the result
    type of functions with no declared result type.
    TODO: This should be replaced with an empty tuple when we have variadics.
  }];
}

//===----------------------------------------------------------------------===//
// LifetimeType
//===----------------------------------------------------------------------===//

def LIT_LifetimeType : LITType<"Lifetime", "lifetime"> {
  let summary = "Lifetime type";
  let description = [{
    This type is used for lifetime parameters when tracking ownership and
    provenance in a signature.
  }];
}


//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

def LIT_RefType : LITType<"Ref", "ref"> {
  let summary = "lifetime-bound reference type";
  let description = [{
    This type represents a typed reference to a memory object with a lifetime.
    This is lowered to a !pop.pointer after CheckLifetimes verifies that the
    uses of it are all correct.  Examples:

    ```mlir
    // A reference to a scalar.
    !lit.ref<f32, some_lifetime>
    ```
  }];

  let parameters = (ins "BoolAttr":$isMutable,
                        "TypedAttr":$elementType, "TypedAttr":$lifetime);
  let assemblyFormat = [{
    `<` custom<MutFlag>($isMutable) `` custom<PrettyType>($elementType) `,`
        custom<LifetimeParamValue>($lifetime) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "bool":$isMutable,
                                        "TypedAttr":$elementType,
                                        "TypedAttr":$lifetime)>,
    TypeBuilderWithInferredContext<(ins "bool":$isMutable,
                                        "Type":$elementType,
                                        "TypedAttr":$lifetime)>,
  ];

  let extraClassDeclaration = [{
    /// Return the elementType attribute as a type.
    Type getElementAsType();
  }];
}

//===----------------------------------------------------------------------===//
// Types for magic declarations.
//===----------------------------------------------------------------------===//

def LIT_MagicMLIRAttrType : LITType<"MagicMLIRAttr", "magic.__mlir_attr"> {
  let summary = "Type of the __mlir_attr declaration";
}

def LIT_MagicMLIROpType : LITType<"MagicMLIROp", "magic.__mlir_op"> {
  let summary = "Type of the __mlir_op declaration";
}

def MagicMLIRTypeType : LITType<"MagicMLIRType", "magic.__mlir_type"> {
  let summary = "Type of the __mlir_type declaration";
}

#endif // KGEN_LITDIALECT_LITTYPES_TD
