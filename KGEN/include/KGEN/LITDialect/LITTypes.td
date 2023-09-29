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
// SignatureType
//===----------------------------------------------------------------------===//

def LIT_SignatureType : DialectType<LITDialect,
    CPred<"llvm::isa<LITSignatureType>($_self)">, "LIT signature type",
    "LITSignatureType">;

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
// MetaTypeType
//===----------------------------------------------------------------------===//

def LIT_MetaTypeType : LITType<"MetaType", "metatype"> {
  let summary = "Type of a type value.";
  let description = [{
    The `!lit.metatype` type represents the type of a type value (tongue
    twister). The meta-type stores the prototype of a type value in its type so
    that member inspection can be performed through the type system during
    member lookup or implicit conversions to other meta-types.

    Example:

    ```mlir
    !lit.metatype<@SomeStruct>
    ```

    The meta-type is propagated through values by attaching itself to type
    values.

    Example:

    ```mlir
    // Concrete type value.
    #kgen.concretetype.constant<!kgen.declref<@SomeStruct>>
      : !lit.metatype<@SomeStruct>

    // Parametric type value.
    #kgen.param.decl.ref<"some_type"> : !lit.metatype<@SomeStruct>
    ```
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        "ParamBindArrayAttr":$paramValues);
  let assemblyFormat = [{
    `<` $symbol `` custom<OptionalParamBindSpec>($paramValues) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol), [{
      return get(symbol.getContext(), symbol,
                 ParamBindArrayAttr::get(symbol.getContext(), {}));
    }]>
  ];
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
    This is lowered to a !kgen.pointer after CheckLifetimes verifies that the
    uses of it are all correct.  Examples:

    ```mlir
    // A reference to a scalar.
    !lit.ref<f32, some_lifetime>
    ```
  }];

  let parameters = (ins "bool":$isMutable,
                        "TypedAttr":$elementType, "TypedAttr":$lifetime);
  let assemblyFormat = [{
    `<` custom<MutFlag>($isMutable) `` custom<TypeParamValue>($elementType) `,`
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

    /// Return the pointer type that corresponds to this reference type,
    /// ignoring the lifetime and the mutability.
    PointerType getAsPointerType();
  }];
}

class ReferenceElementType<string lhs, string rhs> : TypesMatchWith<
    "reference element type", lhs, rhs,
    "$_self.cast<RefType>().getElementAsType()">;

class RefOf<Type type> : ConfinedType<LIT_RefType,
    [CPred<"$_self.cast<::M::KGEN::LIT::RefType>().getElementType()">,
     SubstLeaves<"$_self",
      "$_self.cast<::M::KGEN::LIT::RefType>().getElementAsType()",
      type.predicate>],
    "reference to " # type.summary> {
  let cppClassName = "::M::KGEN::LIT::RefType";
}

//===----------------------------------------------------------------------===//
// REPLResultRefType
//===----------------------------------------------------------------------===//

def LIT_REPLResultRefType : LITType<"REPLResultRef", "replresultref"> {
  let summary = "reference type for results from the REPL";
  let description = [{
    This type represents a typed reference to a memory object contained in the
    REPL, usually used for persistent variables in the REPL. Note that the
    REPLResultRefType is hidden from the user, it is not printed after the REPL
    executes expressions.

    ```mlir
    // A reference to a scalar.
    !lit.replresultref<scalar<f32>>
    ```
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
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
