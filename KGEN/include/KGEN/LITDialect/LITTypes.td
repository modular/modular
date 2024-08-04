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
    !lit<type_signature<"i": variadic<index>>>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$paramTypes,
    "PogListAttr":$paramListAttrs
  );

  let assemblyFormat = [{
    `` custom<TypeSignature>($paramTypes, $paramListAttrs)
  }];

  let builders = [
    TypeBuilder<(ins)>,
  ];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    ArrayRef<Type> getInputParamTypes() const { return getParamTypes(); }
    ArrayRef<Type> getResultParamTypes() const { return {}; }
    StringAttr getParamName(size_t idx) const;
    ArrayRef<TypedAttr> getDefaultPosParams() const;
    ArrayRef<TypedAttr> getDefaultKwOnlyParams() const;

    /// Get the number of input parameters.
    size_t getNumParams() { return getParamTypes().size(); }
    /// Return true if the parameter at the given index is variadic.
    bool isVarParam(size_t idx) const;

    /// Construct a type signature from parameter declarations and corresponding
    /// metadata. This helper is used to convert between a named signature
    /// structure to a nameless representation.
    static TypeSignatureType remapToSignature(
      function_ref<InFlightDiagnostic()> emitError,
      ParamDeclArrayAttr paramDecls, PogListAttr paramListAttrs);

    /// Return true if the signature has a variadic parameter.
    /// TODO: remove this when migration to variadic indices is complete.
    bool hasVariadicParam() const;
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
  let parameters = (ins);
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def LIT_StructType : LITType<"Struct", "struct", [
    DeclareTypeInterfaceMethods<StructTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Reference to a type declaration.";
  let description = [{
    A `!lit.struct` type references a type declaration and binds parameter
    values to the type.

    Example:

    ```mlir
    !lit.struct<@MyStruct<size = 1, T:dtype = fooType>>
    ```
  }];

  let parameters = (ins
    "SymbolAttr":$value,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    "Type":$metaType
  );
  let assemblyFormat = [{
    `<` custom<ParameterizedSymbol>($value, $paramValues) ``
        custom<OptionalMetaType>($metaType, ref($value), ref($paramValues)) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "Type":$metatype)>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "Type":$metatype)>
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getSymbol() const;

    /// Get the name of the referenced type, ignoring packages.
    StringAttr getName();

    /// Return an alias name if the StructType should be printed with an alias.
    std::optional<StringRef> getAliasName();
    static std::optional<StringRef> getAliasName(SymbolRefAttr symbol);
  }];
}

//===----------------------------------------------------------------------===//
// AnyStructType
//===----------------------------------------------------------------------===//

def LIT_AnyStructType : LITType<"AnyStruct", "anystruct",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Type of a struct value.";
  let description = [{
    The `!lit.anystruct` type represents the metatype of a struct, e.g.
    `typeof(Int)`.  The meta-type stores the prototype of a type value in its
    type so that member inspection can be performed through the type system
    during member lookup or implicit conversions to other meta-types.

    Example:

    ```mlir
    !lit.anystruct<@SomeStruct>
    ```

    The meta-type is propagated through values by attaching itself to type
    values.

    Example:

    ```mlir
    // Concrete type value.
    #kgen.type<!lit.struct<@SomeStruct>> : !lit.anystruct<@SomeStruct>

    // Parametric type value.
    #kgen.param.decl.ref<"some_type"> : !lit.anystruct<@SomeStruct>
    ```
  }];

  let parameters = (ins
    "SymbolAttr":$value,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    DefaultValuedParameter<"TypeSignatureType",
                           "TypeSignatureType::get($_ctxt)">:$signature
  );
  let assemblyFormat = [{
    `<` custom<ParameterizedSymbol>($value, $paramValues) ``
        (`,` $signature^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ArrayRef<TypedAttr>":$values,
                                        "TypeSignatureType":$signature)>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "TypeSignatureType":$signature), [{
      return get(symbol, {}, signature);
    }]>
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getSymbol() const;

    /// Return the struct type this metatype corresponds to.
    StructType getStructType();

    /// Bind parameter values to the metatype.
    AnyStructType bind(ArrayRef<TypedAttr> values) const;
    /// Return whether the type is a metatype.
    bool isMetaType() { return true; }
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

  let extraClassDeclaration = [{
    /// Return whether the type is a metatype.
    bool isMetaType() { return true; }

    /// Return the metatype for this this trait as a value.
    AnyTraitType getMetaType();

    /// Return a TypeConstantAttr for a reference to this trait as a value, e.g.
    /// uttering 'Stringable' in code.
    TypedAttr getPValue();
  }];
}


//===----------------------------------------------------------------------===//
// AnyTraitType
//===----------------------------------------------------------------------===//

def LIT_AnyTraitType : LITType<"AnyTrait", "anytrait",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Type of a trait value.";
  let description = [{
    The `!lit.anytrait` type represents the type of a trait value, e.g. the
    result of `typeof(Stringable)`. The meta-type stores the prototype of a type
    value in its type so that member inspection can be performed through the type system during
    member lookup or implicit conversions to other meta-types.

    Example:

    ```mlir
    !lit.anytrait<@Stringable>
    ```

    The meta-type is propagated through values by attaching itself to type
    values.

    Example:

    ```mlir
    // Concrete type value for 'Int' converted for use as Stringable value.
    #kgen.type<!lit.struct<@Int>> : !lit.anytrait<@Stringable>

    // Parametric type value.
    #kgen.param.decl.ref<"some_type"> : !lit.anytrait<@Stringable>
    ```
  }];

  let parameters = (ins "TraitType":$traitType);
  let assemblyFormat = "`<` $traitType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "TraitType":$traitType), [{
      return get(traitType.getContext(), traitType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return whether the type is a metatype.
    bool isMetaType() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// LifetimeType
//===----------------------------------------------------------------------===//

def LIT_LifetimeType : LITType<"Lifetime", "lifetime",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Lifetime type";
  let description = [{
    This type is used for lifetime parameters when tracking accesses in a
    signature.  The `isMutable` parameter indicates whether the access is
    mutating or just reading.
  }];

  let parameters = (ins "TypedAttr":$isMutable);
  let assemblyFormat = "`<` custom<I1ParamValue>($isMutable) `>`";

  let builders = [
    TypeBuilder<(ins "bool":$isMutable)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$isMutable)>,
  ];

  let extraClassDeclaration = [{
    /// Return true if the mutable attribute is known to be the specific
    /// constant.  This returns false if parametric or if the other value.
    bool isMutableKnown(bool value);

    /// Return a (possibly parameteric) specification for whether this reference
    /// is a mutation or a read.
    TypedAttr isMutable() { return getIsMutable(); }

    enum MutabilityClass {
      Mutable, Immutable, Parametric
    };

    /// Classify the mutability into Mutable/Immutable/Parametric.
    MutabilityClass getMutabilityClass();
  }];
}

//===----------------------------------------------------------------------===//
// LifetimeSetType
//===----------------------------------------------------------------------===//

def LIT_LifetimeSetType : LITType<"LifetimeSet", "lifetime.set",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a singleton type representing a lifetime set";
  let description = [{
    The `!lit.lifetime.set` type is a singleton type that represents a lifetime
    set. A lifetime set is a collection of different lifetimes.

    Example:

    ```mlir
    !lit.lifetime.set
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

def LIT_RefType : LITType<"Ref", "ref",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "lifetime-bound reference type";
  let description = [{
    This type represents a typed reference to a memory object with a lifetime.
    This is lowered to a !kgen.pointer after CheckLifetimes verifies that the
    uses of it are all correct.  Examples:

    ```mlir
    // A reference to a scalar.
    !lit.ref<f32, some_lifetime>

    // A reference with parametric mutability.
    !lit.ref<f32, mut=sometimes_mutable, some_lifetime, 0>
    ```
  }];

  let parameters = (ins "Type":$elementType, "TypedAttr":$lifetime,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<ParamType>($elementType) `,`
        custom<LifetimeParamValue>($lifetime) ``
        (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$lifetime,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$lifetime,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return llvm::cast<ElTy>(getElementType()); }

    /// Return the pointer type that corresponds to this reference type,
    /// ignoring the lifetime and the mutability.
    PointerType getAsPointerType();

    /// Return this RefType but with a different element type.
    RefType getWithElement(Type newElement);

    /// Return this RefType but with a different lifetime.
    RefType getWithLifetime(TypedAttr newLifetime);

    /// Return this RefType but with a different mutability.
    RefType getWithMutability(bool isMut);

    /// Return true if the mutable attribute is known to be the specific
    /// constant.  This returns false if parametric or if the other value.
    bool isMutableKnown(bool value);

    /// Return a (possibly parameteric) specification for whether this reference
    /// is a mutation or a read.
    TypedAttr isMutable();

    /// Classify the mutability into Mutable/Immutable/Parametric.
    LifetimeType::MutabilityClass getMutabilityClass();

    /// Return the type of the lifetime reference, which is always a
    /// `!lit.lifetime<mutability>` type.
    LifetimeType getLifetimeType();

    /// Return true if this is in address space 0.
    bool isDefaultAddrSpace();

    /// Return a reference to the specified element type and mutability with an
    /// immortal (#lit.lifetime) lifetime.
    static RefType getImmortal(Type elementType, bool isMut,
                               unsigned addrSpace = 0);
    static RefType getImmortal(Type elementType, bool isMut,
                               TypedAttr addrSpace);
  }];
}

class ReferenceElementType<string lhs, string rhs> : TypesMatchWith<
    "reference element type", lhs, rhs,
    "llvm::cast<RefType>($_self).getElementType()">;

class RefOf<Type type> : ConfinedType<LIT_RefType,
    [CPred<"llvm::cast<::M::KGEN::LIT::RefType>($_self).getElementType()">,
     SubstLeaves<"$_self",
      "llvm::cast<::M::KGEN::LIT::RefType>($_self).getElementType()",
      type.predicate>],
    "reference to " # type.summary> {
  let cppClassName = "::M::KGEN::LIT::RefType";
}

//===----------------------------------------------------------------------===//
// RefPackType
//===----------------------------------------------------------------------===//

def LIT_RefPackType : LITType<"RefPack", "ref.pack",
                        [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a parameterized pack of by-ref values";
  let description = [{
    A `!lit.ref.pack` type represents a sequence of heterogeneously typed
    references to elements with a common lifetime.  As generated by the Mojo
    parser, the "variadic" will be a parametric list of types (e.g.
    `[f32, String]`) that is only expandable after elaboration, and this pack's
    runtime representation is semantically
    `!kgen.pack<[!lit.ref<f32, lt>, !lit.ref<String, lt>]>`.

    In addition to the variadic, this also takes the lifetime of the references,
    which indicates mutability of the references.

    Example:

    ```mlir
    // Pack with 'Ts' elements, with a parametric lifetime and addrspace=42.
    !lit.ref.pack<Ts, life, 42>

    lit.func @pack<Ts: variadic<!kgen.type>>(
      %0: !lit.ref.pack<Ts, life, 42>
    )
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic,  "TypedAttr":$lifetime,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($variadic) `,`
        custom<LifetimeParamValue>($lifetime) ``
        (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$variadic,
                                        "TypedAttr":$lifetime,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// If the pack type is concrete, returns the attribute representing the
    /// variadic sequence that the pack type is based upon. Otherwise, returns
    /// a null attribute.
    VariadicAttr getVariadicIfResolved() const;

    /// This returns the element type of the variadic list parameter, typically
    /// something like !kgen.type or a trait type.
    Type getVariadicElementType();

    /// Return the effective type (always a reference) of each element given
    /// the type according to the type list.
    RefType getElementRefTypeFor(Type elementType);
  }];
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
