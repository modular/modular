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
// LITGeneratorType
//===----------------------------------------------------------------------===//

def LIT_GeneratorType : DialectType<LITDialect,
    CPred<"llvm::isa<LITGeneratorType>($_self)">, "LIT generator type",
    "LITGeneratorType">;

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def LIT_FnType : DialectType<LITDialect,
    CPred<"llvm::isa<FnType>($_self)">, "LIT function type", "FnType">;

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

def LIT_FnTypeGeneratorType : DialectType<LITDialect,
    CPred<"llvm::isa<FnTypeGeneratorType>($_self)">,
    "LIT function type generator type", "FnTypeGeneratorType">;

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

    /// Bind parameter values to the signature, returning a new one.
    TypeSignatureType bind(ArrayRef<TypedAttr> values) const;
  }];
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
    DefaultValuedParameter<"TypeSignatureType",
                           "TypeSignatureType::get($_ctxt)">:$signature
  );
  let assemblyFormat = [{
    `<` custom<ParameterizedSymbol>($value, $paramValues) ``
        (`,` $signature^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "TypeSignatureType":$signature)>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "TypeSignatureType":$signature)>
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getSymbol() const;

    /// Get the name of the referenced type, ignoring packages.
    StringAttr getName();

    /// Return an alias name if the StructType should be printed with an alias.
    std::optional<StringRef> getAliasName();
    static std::optional<StringRef> getAliasName(SymbolRefAttr symbol);

    /// Bind parameter values to the struct type, returning a new struct type.
    /// Expects the number of values to match the number of param values. Only
    /// positions that are currently unbound can be updated.
    StructType bindAll(ArrayRef<TypedAttr> values) const;

    /// Bind parameter values to the struct type, returning a new struct type.
    /// Expects the number of values to match the number of unbound parameters
    /// in the current param values list.
    StructType bindUnbound(ArrayRef<TypedAttr> values) const;
  }];
}

//===----------------------------------------------------------------------===//
// TraitType
//===----------------------------------------------------------------------===//

def LIT_TraitType: LITType<"Trait", "trait",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Type of a trait value.";
  let description = [{
    The `!lit.trait` type represents the type of some lit.trait.decl values.
    The list of decls must be non-empty and uniqued.

    Example:

    ```mlir
    !lit.trait<@SomeTrait>
    !lit.trait<@SomeTrait, @SomeOtherTrait>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"SymbolRefAttr">:$symbols);
  let assemblyFormat = "`<` $symbols `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol), [{
      return get(symbol.getContext(), {symbol});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return whether the type is a metatype.
    bool isMetaType() { return true; }

    /// Return the metatype for this this trait as a value.
    AnyTraitType getMetaType();

    /// Return a TypeParamAttr for a reference to this trait as a value, e.g.
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
// MetaType
//===----------------------------------------------------------------------===//

def LIT_MetaType : LITType<"Meta", "meta",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "The singleton metatype of a type.";
  let description = [{
    The `!lit.meta` type represents a singleton metatype, which is a metatype
    that has only one constituent type. It is the “tightest” metatype a type can
    have.

    Example:

    ```mlir
    !lit.meta<@Int>
    ```
  }];

  let parameters = (ins "Type":$type);
  let assemblyFormat = "`<` $type `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// OriginType
//===----------------------------------------------------------------------===//

def LIT_OriginType : LITType<"Origin", "origin",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Origin type";
  let description = [{
    This type is used for origin parameters when tracking accesses in a
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

    /// Remove any OriginMutCast and ._mlir_origin if present.
    static TypedAttr stripMutCastAndFieldExtract(TypedAttr origin);
  }];
}

//===----------------------------------------------------------------------===//
// OriginSetType
//===----------------------------------------------------------------------===//

def LIT_OriginSetType : LITType<"OriginSet", "origin.set",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a singleton type representing a origin set";
  let description = [{
    The `!lit.origin.set` type is a singleton type that represents a origin
    set. An origin set is a collection of different origins.

    Example:

    ```mlir
    !lit.origin.set
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

def LIT_RefType : LITType<"Ref", "ref",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "pointer type that tracks an origin";
  let description = [{
    This type represents a typed reference to a memory object with an origin.
    This is lowered to a !kgen.pointer after CheckLifetimes verifies that the
    uses of it are all correct.  Examples:

    ```mlir
    // A reference to a scalar.
    !lit.ref<f32, some_origin>

    // A reference with parametric mutability.
    !lit.ref<f32, mut=sometimes_mutable, some_origin, 0>
    ```
  }];

  let parameters = (ins "Type":$elementType, "TypedAttr":$origin,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<ParamType>($elementType) `,`
        custom<OriginParamValue>($origin) ``
        (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$origin,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$origin,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return llvm::cast<ElTy>(getElementType()); }

    /// Return the pointer type that corresponds to this reference type,
    /// ignoring the origin and the mutability.
    PointerType getAsPointerType();

    /// Return this RefType but with a different element type.
    RefType getWithElement(Type newElement);

    /// Return this RefType but with a different origin.
    RefType getWithOrigin(TypedAttr newOrigin);

    /// Return this RefType but with a different mutability.
    RefType getWithMutability(bool isMut);

    /// Return true if the mutable attribute is known to be the specific
    /// constant.  This returns false if parametric or if the other value.
    bool isMutableKnown(bool value);

    /// Return a (possibly parameteric) specification for whether this reference
    /// is a mutation or a read.
    TypedAttr isMutable();

    /// Classify the mutability into Mutable/Immutable/Parametric.
    OriginType::MutabilityClass getMutabilityClass();

    /// Return the type of the origin reference, which is always a
    /// `!lit.origin<mutability>` type.
    OriginType getOriginType();

    /// Return true if this is in address space 0.
    bool isDefaultAddrSpace();

    /// Return a reference to the specified element type and mutability with a
    /// #lit.any.origin.
    static RefType getAnyOrigin(Type elementType, bool isMut,
                                  unsigned addrSpace = 0);
    static RefType getAnyOrigin(Type elementType, bool isMut,
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
  let cppType = "::M::KGEN::LIT::RefType";
}

//===----------------------------------------------------------------------===//
// RefPackType
//===----------------------------------------------------------------------===//

def LIT_RefPackType : LITType<"RefPack", "ref.pack",
                        [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "a parameterized pack of by-ref values";
  let description = [{
    A `!lit.ref.pack` type represents a sequence of heterogeneously typed
    references to elements with a common origin.  As generated by the Mojo
    parser, the "variadic" will be a parametric list of types (e.g.
    `[f32, String]`) that is only expandable after elaboration, and this pack's
    runtime representation is semantically
    `!kgen.pack<[!lit.ref<f32, lt>, !lit.ref<String, lt>]>`.

    In addition to the variadic, this also takes the origin of the references,
    which indicates mutability of the references.

    Example:

    ```mlir
    // Pack with 'Ts' elements, with a parametric origin and addrspace=42.
    !lit.ref.pack<Ts, origin, 42>

    lit.fn @pack<Ts: variadic<!kgen.type>>(
      %0: !lit.ref.pack<Ts, origin, 42>
    )
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic,  "TypedAttr":$origin,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($variadic) `,`
        custom<OriginParamValue>($origin) ``
        (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$variadic,
                                        "TypedAttr":$origin,
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
