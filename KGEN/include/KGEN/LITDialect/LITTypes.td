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

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
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
// DiscardType
//===----------------------------------------------------------------------===//

def LIT_DiscardType : LITType<"Discard", "discard"> {
  let summary = "Type placeholder used for explicitly unbound parameters";
  let description = [{
    When expressing explicitly unbound parameters using e.g. the `_` syntax, a
    this type is uses to serve as placeholder in cases when the actualy type of
    the parameter is not known yet.
  }];
}

//===----------------------------------------------------------------------===//
// TypeSignatureType
//===----------------------------------------------------------------------===//

def LIT_TypeSignatureType : LITType<"TypeSignature", "type_signature",
    [ParameterScopeTypeInterface]> {
  let summary = "type parameter signature";
  let description = [{
    The `!lit.type_signature` type describes the parameter signature of a type:
    its input parameter types, names, default values, and parameter passing
    kinds.

    Example:

    ```mlir
    !lit.type_signature
    !lit.type_signature<index, |>
    !lit.type_signature<"dt": dtype = f32>
    !lit<type_signature<"i": variadic<index>> param_vararg>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$paramTypes,
    ArrayRefParameter<"StringAttr">:$paramNames,
    ArrayRefParameter<"PassingKind">:$paramPassingKinds,
    ArrayRefParameter<"TypedAttr">:$defaultParameters,
    "bool":$paramVarArg
  );

  let assemblyFormat = [{
    `` custom<TypeSignature>($paramTypes, $paramNames, $paramPassingKinds,
                             $defaultParameters, $paramVarArg)
  }];

  let builders = [
    TypeBuilder<(ins), [{
      using std::nullopt;
      return $_get($_ctxt, nullopt, nullopt, nullopt, nullopt, false);
    }]>,
    TypeBuilder<(ins "ArrayRef<Type>":$inputParams,
                     "ArrayRef<StringAttr>":$paramNames,
                     "ArrayRef<PassingKind>":$passingKinds), [{
      return $_get($_ctxt, inputParams, paramNames, passingKinds, std::nullopt,
                   false);
    }]>
  ];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    ArrayRef<Type> getInputParamTypes() const { return getParamTypes(); }
    ArrayRef<Type> getResultParamTypes() const { return {}; }

    /// Get the number of input parameters.
    size_t getNumInputParams() { return getInputParamTypes().size(); }
    /// Return true if the parameter at the given index is a vararg.
    bool isVarArg(size_t index) {
      return index + 1 == getNumInputParams() && getParamVarArg();
    }

    /// Construct a type signature from named parameter declarations. This
    /// helper is used to convert between a named signature structure to a
    /// nameless representation.
    static TypeSignatureType remapToSignature(
        function_ref<InFlightDiagnostic()> emitError,
        ParamDeclArrayAttr paramDecls, ArrayRef<StringAttr> paramNames,
        ArrayRef<PassingKind> passingKinds, ArrayRef<TypedAttr> defaults,
        bool paramVarArg);
  }];
}

//===----------------------------------------------------------------------===//
// UnpackedType
//===----------------------------------------------------------------------===//

def LIT_UnpackedType : LITType<"Unpacked", "unpacked"> {
  let summary = "Parameterized unpacked expression type.";
  let description = [{
    The `!lit.unpacked` type represent the type of an unpacked expression before
    the unpacking is actually performed. It stores the type of the expression to
    be unpacked.
  }];
  let parameters = (ins "Type":$exprType);
  let assemblyFormat = "`<` custom<ParamType>($exprType) `>`";
}

//===----------------------------------------------------------------------===//
// MetaTypeType
//===----------------------------------------------------------------------===//

def LIT_MetaTypeType : LITType<"MetaType", "metatype",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
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

  let parameters = (ins
    "SymbolRefAttr":$symbol,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    DefaultValuedParameter<"TypeSignatureType",
                           "TypeSignatureType::get($_ctxt)">:$signature
  );
  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) (`,` $signature^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ArrayRef<TypedAttr>":$values,
                                        "TypeSignatureType":$signature), [{
      return get(symbol.getContext(), symbol, values, signature);
    }]>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "TypeSignatureType":$signature), [{
      return get(symbol, {}, signature);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Bind parameter values to the metatype.
    MetaTypeType bind(ArrayRef<TypedAttr> values) const;
  }];
}

//===----------------------------------------------------------------------===//
// TraitType
//===----------------------------------------------------------------------===//

def LIT_TraitType: LITType<"Trait", "trait",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Type of a trait value.";
  let description = [{
    The `!lit.trait` type represents the type of a lit.trait.decl value.

    Example:

    ```mlir
    !lit.trait<@SomeTrait>
    ```
  }];

  let parameters = (ins "SymbolRefAttr":$symbol);
  let assemblyFormat = "`<` $symbol `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol), [{
      return get(symbol.getContext(), symbol);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// LifetimeType
//===----------------------------------------------------------------------===//

def LIT_LifetimeType : LITType<"Lifetime", "lifetime",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Lifetime type";
  let description = [{
    This type is used for lifetime parameters when tracking ownership and
    provenance in a signature.
  }];
  let assemblyFormat = "";
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
                        "Type":$elementType, "TypedAttr":$lifetime);
  let assemblyFormat = [{
    `<` custom<MutFlag>($isMutable) `` custom<ParamType>($elementType) `,`
        custom<LifetimeParamValue>($lifetime) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "bool":$isMutable,
                                        "Type":$elementType,
                                        "TypedAttr":$lifetime)>,
    TypeBuilderWithInferredContext<(ins "bool":$isMutable,
                                        "TypedAttr":$elementType,
                                        "TypedAttr":$lifetime)>,
  ];

  let extraClassDeclaration = [{
    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return getElementType().cast<ElTy>(); }

    /// Return the pointer type that corresponds to this reference type,
    /// ignoring the lifetime and the mutability.
    PointerType getAsPointerType();

    /// Return this RefType but with a different element type.
    RefType getWithElement(Type newElement);

    /// Return this RefType but with a different lifetime.
    RefType getWithLifetime(TypedAttr newLifetime);

    /// Return this RefType but with a different mutability.
    RefType getWithMutability(bool isMut);

    /// Given the specified pointer type, return a reference type of the same
    /// element but with a hacked lifetime.
    /// TODO(references): Remove. This is just for migration.
    static RefType getRefForPointerHACK(PointerType type, bool isMut);
  }];
}

class ReferenceElementType<string lhs, string rhs> : TypesMatchWith<
    "reference element type", lhs, rhs,
    "$_self.cast<RefType>().getElementType()">;

class RefOf<Type type> : ConfinedType<LIT_RefType,
    [CPred<"$_self.cast<::M::KGEN::LIT::RefType>().getElementType()">,
     SubstLeaves<"$_self",
      "$_self.cast<::M::KGEN::LIT::RefType>().getElementType()",
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
