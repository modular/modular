//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_TD
#define KGEN_LITDIALECT_LITATTRS_TD

include "KGEN/KGENDialect/KGENAttrInterfaces.td"
include "KGEN/LITDialect/LITEnums.td"
include "KGEN/LITDialect/LITDialect.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

/// An attribute in the LIT dialect.
class LITAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LITDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// A typed attribute in the LIT dialect.
class LITTypedAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LITAttr<attrName, attrMnemonic, [TypedAttrInterface] # traits>;

//===----------------------------------------------------------------------===//
// SymbolAttr
//===----------------------------------------------------------------------===//

def LIT_SymbolAttr : LITAttr<"Symbol", "symbol"> {
  let summary = "a type symbol reference";
  let description = [{
    This attribute wraps a `SymbolRefAttr` in another attribute type so that
    aliases can be generated for it selectively.
  }];

  let parameters = (ins "SymbolRefAttr":$value);
  let assemblyFormat = "`` $value";

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// PogMetadataAttr
//===----------------------------------------------------------------------===//

def LIT_PogMetadataAttr : LITAttr<"PogMetadata", "pog_metadata"> {
  let summary = "Metadata for a single Argument or parameter.";
  let description = [{
    The `#lit.pog_metadata` attribute contains metadata about an argument or
    parameter of a function, including the name, passing kind, and variadicness

    Example:

    ```mlir
    #lit.pog_metadata<"some_keyword_param", pos_or_kw, false>
    #lit.pog_metadata<"some_variadic_param", pos_or_kw, true>
    ```
  }];

  let parameters = (ins
    "StringAttr":$name, "PassingKind":$passingKind, "bool":$isVariadic
  );

  let assemblyFormat = "`<` $name `,` $passingKind `,` $isVariadic `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "StringAttr":$name,
      "PassingKind":$passingKind,
      CArg<"bool", "false">:$isVariadic
    ), [{
      return $_get(name.getContext(), name, passingKind, isVariadic);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isVariadic() const { return getIsVariadic(); }
  }];
}

//===----------------------------------------------------------------------===//
// PogListAttr
//===----------------------------------------------------------------------===//

def LIT_PackIndexParameter : OptionalParameter<"ssize_t"> {
  let defaultValue = "ssize_t(-1)";
}

def LIT_PogListAttr : LITAttr<"PogList", "pog_list"> {
  let summary = "Argument or parameter list metadata.";
  let description = [{
    The `#lit.pog_list` attribute contains metadata about an argument or
    parameter list of a function, including names, passing kinds, default
    values, and information about variadic arguments. The positional default
    values correspond to the trailing positional (i.e. pos-only or pos-or-kw)
    args/params, and the keyword-only default values similarly correspond to the
    trailing keyword-only args/params.

    Example:

    ```mlir
    // Metadata specifying arg/param names "a", "b", "c", "d" in order; "a" is
    // positional-only, "b" is positional-or-keyword, and "c" and "d" are
    // keyword-only. The last positional arg/param ("b") has a default value of
    // `4.2`, and the last keyword-only arg/param ("d") has a default value of
    // 1. The second argument is a (positional) variadic pack, and the last
    // argument is a variadic keyword argument
    #lit.pog_list<
      ["a", "b", "c", "d"],
      [pos, pos_or_kw, kw, kw],
      [4.2 : f32],
      [1: i64],
      [1],
      [3]
    >
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"PogMetadataAttr">:$pogs,
    OptionalArrayRefParameter<"TypedAttr">:$defaultPos,
    OptionalArrayRefParameter<"TypedAttr">:$defaultKwOnly,
    LIT_PackIndexParameter:$packIndex,
    OptionalParameter<"std::optional<ArgConvention>">:$origPackConvention
  );

  let assemblyFormat = [{
    `<` `[` (`]`) : ($pogs^            `]`)? `,`
        `[` (`]`) : ($defaultPos^      `]`)? `,`
        `[` (`]`) : ($defaultKwOnly^   `]`)?
        (`,` custom<PackInfo>($packIndex, $origPackConvention)^)?
    `>`
  }];

  let builders = [
    AttrBuilder<(ins)>,
    AttrBuilder<(ins "ArrayRef<PogMetadataAttr>":$pogs)>,
    AttrBuilder<(ins "ArrayRef<PogMetadataAttr>":$pogs,
                     "ArrayRef<TypedAttr>":$defaultPos,
                     "ArrayRef<TypedAttr>":$defaultKwOnly)>,
    AttrBuilder<(ins "ArrayRef<StringAttr>":$names,
                     "ArrayRef<PassingKind>":$passingKinds)>,
    AttrBuilder<(ins "ArrayRef<StringAttr>":$names,
                     "ArrayRef<PassingKind>":$passingKinds,
                     "ArrayRef<TypedAttr>":$defaultPos,
                     "ArrayRef<TypedAttr>":$defaultKwOnly,
                     "ArrayRef<size_t>":$variadicIndices,
                     "ssize_t":$packIndex,
                     "std::optional<ArgConvention>":$origPackConvention)>
  ];

  let extraClassDeclaration = [{
    /// Create a new argument/parametere list attr with the given array of
    /// individual argument/parameter metadata attributes.
    PogListAttr cloneWith(ArrayRef<PogMetadataAttr> pogs) const;

    /// Return true if the argument/param at `idx` is a variadic.
    bool isVariadic(size_t idx) const;
    /// Return true if the argument/param at `idx` is a pack.
    bool isPack(size_t idx) const;
    /// Return true if the argument/param at `idx` is a positional variadic.
    bool isPosVariadic(size_t idx) const;
    /// Return true if the argument/param at `idx` is a keyword variadic.
    bool isKwVariadic(size_t idx) const;
    /// Return true if there is a variadic (pos or keyword) argument/param.
    bool hasVariadic() const;
    /// Return true if there is a variadic pack argument/param.
    bool hasPack() const;
    /// Return true if there is a variadic keyword argument/param.
    bool hasKwVariadics() const;

    /// Return the name for the argument/parameter at the given index.
    StringAttr getName(size_t idx) const;
    /// Return the passing kind for the argument/parameter at the given index.
    PassingKind getPassingKind(size_t idx) const;

    /// Create an array of PogMetadataAttr from the given names, passing kinds,
    /// and variadic indices.
    static SmallVector<PogMetadataAttr> toPogs(
      ArrayRef<StringAttr> names, ArrayRef<PassingKind> passingKinds,
      ArrayRef<size_t> indices);
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

def LIT_FnMetadataAttr : LITAttr<"FnMetadata", "fn_metadata",
    [DeclareAttrInterfaceMethods<FnMetadataAttrInterface>]> {
  let summary = "Function metadata.";
  let description = [{
    The `#lit.fn_metadata` attribute contains additional information about a
    function, such as the argument and parameter list metadata and the number of
    implicit lifetime declarations.

    Example:

    ```mlir
    // Metadata specifying that: argument #0 is called "someRef" is
    // positional-only and a default value of `13`; argument #1 and #2 are keyword-only
    // and are called "v" and "kwargs", the former has a default value of `17`,
    // and the latter is variadic (i.e. `**kwargs`). Moveover, parameter #0 is
    // called "someParam", parameter #1 is called "paramWithDefault", they have
    // positional-only and positional-or-keyword passing kinds respectively, and
    // the latter has the default value `3.14`. The list of keyword-only default
    // parameter values is empty (since there are no keyword-only parameters).
    // The signature has two implicit lifetime declarations.
    #lit.fn_metadata<
      <["someRef", "v", "kwargs"], [pos, kw], [13 : index], [17 : i64]>,
      <["someParam", "paramWithDefault"], [pos, pos_or_kw], [3.14: f32], []>,
      2
    >
    ```
  }];

  let parameters = (ins
    LIT_PogListAttr:$argListAttrs,
    LIT_PogListAttr:$paramListAttrs,
    "size_t":$numImplicitLifetimeDecls
  );
  let assemblyFormat = [{
    `<` $argListAttrs `,` $paramListAttrs `,` $numImplicitLifetimeDecls `>`
  }];

  let builders = [
    AttrBuilder<(ins)>,
    AttrBuilder<(ins "size_t":$numParams, "size_t":$numArgs,
                     CArg<"size_t", "0">:$numImplicitLifetimeDecls)>,
    AttrBuilderWithInferredContext<(ins
      "PogListAttr":$argListAttrs,
      "PogListAttr":$paramListAttrs,
      CArg<"size_t", "0">:$numImplicitLifetimeDecls)>,
    AttrBuilderWithInferredContext<(ins
      "PogListAttr":$argListAttrs,
      CArg<"size_t", "0">:$numImplicitLifetimeDecls)>,
  ];

  let extraClassDeclaration = [{
    size_t getNumArgs() const;
    ArrayRef<TypedAttr> getDefaultPosArgs() const {
      return getArgListAttrs().getDefaultPos();
    }
    ArrayRef<TypedAttr> getDefaultKwOnlyArgs() const {
      return getArgListAttrs().getDefaultKwOnly();
    }
    ArrayRef<TypedAttr> getDefaultPosParams() const {
      return getParamListAttrs().getDefaultPos();
    }
    ArrayRef<TypedAttr> getDefaultKwOnlyParams() const {
      return getParamListAttrs().getDefaultKwOnly();
    }

    /// Return true if the signature has a variadic argument.
    bool hasVarArgs() const;
    /// Return true if the signature has a variadic pack argument.
    bool hasPackVarArgs() const;
    /// Return true if the signature has a variadic parameter.
    bool hasParamVarArgs() const;
    /// Return true if the signature has a keyword variadic argument.
    bool hasKwVarArgs() const;
    /// Return true if the argument at `idx` is a any variadic or pack argument.
    bool isAnyVarArg(size_t idx) const;
    /// Return true if the argument at `idx` is a positional variadic argument.
    bool isPosVarArg(size_t idx) const;
    /// Return true if the argument at `idx` is a keyword variadic argument.
    bool isKwVarArg(size_t idx) const;
    /// Return true if the argument at `idx` is a variadic pack argument.
    bool isPackVarArg(size_t idx) const;

    /// Get a new metadata attribute for a signature with the given number of
    /// positional input parameters prepended to the signature. An additional
    /// array of bool corresponding to the variadic mask of the prepended
    /// parameters is also required.
    FnMetadataAttr prependPosParams(size_t numNewParams,
                                    ArrayRef<bool> variadicMask) const;
  }];
}

//===----------------------------------------------------------------------===//
// DocStringAttr
//===----------------------------------------------------------------------===//

def LIT_DocStringAttr : LITAttr<"DocString", "doc.string"> {
  let summary = "A source documentation string";
  let description = [{
    The `#lit.doc.string` attribute represents a documentation string within
    Mojo source. It contains the raw string, and optionally the source location
    of the start of the string.

    Example:

    ```mlir
    #lit.doc.string<"This is a doc string">
    #lit.doc.string<"This is a doc string", loc("file.mojo":10:8)>
    ```
  }];

  let parameters = (ins
    "StringAttr":$string,
    OptionalParameter<"FileLineColLoc">:$location
  );
  let assemblyFormat = "`<` $string (`,` $location^)? `>`";

  let builders = [AttrBuilderWithInferredContext<(ins
    "StringAttr":$string, CArg<"FileLineColLoc", "{}">:$location),
  [{
    return get(string.getContext(), string, location);
  }]>];
}

//===----------------------------------------------------------------------===//
// UnboundMLIROperationAttr
//===----------------------------------------------------------------------===//

def UnboundMLIROperationAttr
  : LITTypedAttr<"UnboundMLIROperation", "unbound_mlir_operation"> {
  let summary = "An unbound mlir operation";
  let description = [{
    This attribute is used by the Lit parser as a reference to an MLIR
    operation by name: it may have bound attributes as well.  When applied to
    value operands, it materializes into a specific MLIR operation instance
  }];

  let parameters = (ins "StringAttr":$name, "DictionaryAttr":$attrs);
  let assemblyFormat = "$name $attrs";
  let extraClassDeclaration = [{ Type getType() const; }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "DictionaryAttr":$attrs), [{
      return $_get(name.getContext(), name, attrs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// UnpackedAttr
//===----------------------------------------------------------------------===//

def LIT_UnpackedAttr : LITTypedAttr<"Unpacked", "unpacked"> {
  let summary = "An attribute representing a `*_` value";
  let description = [{
    Represents *_ expressions which are magic expressions that can appear in
    parameter lists that autoparameterize the containing declaration.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "UnpackedType">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilder<(ins), [{
      return get($_ctxt, UnpackedType::get($_ctxt));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ModuleAttr
//===----------------------------------------------------------------------===//

def ModuleAttr : LITTypedAttr<"Module", "module"> {
  let summary = "A module attribute.";
  let description = [{
    The `#lit.module` attribute represents an unresolved module value. It always
    has a meta-type type.

    Example:

    ```mlir
    #lit.module : !lit.anystruct<@some_module>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "AnyStructType">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "AnyStructType":$type),
                                   [{ return get(type.getContext(), type); }]>
  ];
}

//===----------------------------------------------------------------------===//
// BindTypeAttr
//===----------------------------------------------------------------------===//

def BindTypeAttr : LITTypedAttr<"BindType", "bind_type"> {
  let summary = "bind parameter values to a type";
  let description = [{
    The `#lit.bind_type` attribute is a parameter operator attribute that
    binds parameter values to a type instance.

    Example:

    ```mlir
    // Fully unbound to fully bound.
    #lit.bind_type<
      :anystruct<@SIMD<:dtype #kgen.unbound, #kgen.unbound>, <dtype, index>>
                 @SIMD<:dtype #kgen.unbound, #kgen.unbound>,
      [f32, 16]>
      : !lit.anystruct<@SIMD<:dtype f32, 16>>

    // Fully unbound to partially bound.
    #lit.bind_type<
      :anystruct<@SIMD<:dtype #kgen.unbound, #kgen.unbound>, <dtype, index>>
                 @SIMD<:dtype #kgen.unbound, #kgen.unbound>,
      [#kgen.unbound, 16]>
      : !lit.anystruct<@SIMD<:dtype #kgen.unbound, 16>, <dtype>>

    // Partially unbound to fully bound with a default.
    #lit.bind_type<
      :anystruct<@SIMD<:dtype #kgen.unbound, 16>, <dtype = f32>>
                 @SIMD<:dtype #kgen.unbound, 16>,
      [si32, 16]>
      : !lit.anystruct<@SIMD<:dtype si32, 16>>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    ArrayRefParameter<"TypedAttr">:$values,
    AttributeSelfTypeParameter<"", "AnyStructType">:$type
  );

  let assemblyFormat = [{
    `<` custom<ColonTypeParamValue>($typeValue) `,` `[`
    custom<BindTypeParams>($values, ref($typeValue)) `]` `>`
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "ArrayRef<TypedAttr>":$values,
                                        "AnyStructType":$type), [{
      return get(type.getContext(), typeValue, values, type);
    }], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "ArrayRef<TypedAttr>":$values), [{
      return get(typeValue.getContext(), typeValue, values,
                 getBindTypeResultType(typeValue, values));
    }], "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$typeValue, "ArrayRef<TypedAttr>":$values,
                     "AnyStructType":$type), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static BindTypeAttr getFromBytecode(
        TypedAttr typeValue, ArrayRef<TypedAttr> values, AnyStructType type);
  }];
}

//===----------------------------------------------------------------------===//
// InvalidRefLifetimeAttr
//===----------------------------------------------------------------------===//

def InvalidRefLifetimeAttr
  : LITTypedAttr<"InvalidRefLifetime", "invalid.ref.lifetime"> {
  let summary = "An attribute value of !lit.lifetime<mut> type";
  let description = [{
    This attribute is a singleton lifetime value used to indicate when a
    !lit.ref type should not be bindable to a Reference library type.  This
    happens in generic borrowed arguments that may be bound to a
    @register_passable type that doesn't have a lifetime.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "LifetimeType">:$type
  );

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      assert(::isa<LifetimeType>(type) && "LifetimeAttr must have LifetimeType");
      return get(type.getContext(), ::cast<LifetimeType>(type));
    }]>,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$isMutable), [{
      return get(isMutable.getContext(), LifetimeType::get(isMutable));
    }]>,
    AttrBuilder<(ins "bool":$isMutable), [{
      return get($_ctxt, LifetimeType::get($_ctxt, isMutable));
    }]>
 ];
}

//===----------------------------------------------------------------------===//
// LifetimeAttr
//===----------------------------------------------------------------------===//

def LifetimeAttr : LITTypedAttr<"Lifetime", "lifetime"> {
  let summary = "A !lit.lifetime<mut> value";
  let description = [{
    This attribute is a singleton (for each mutability) used when lowering
    lifetimes, and represents an immortal value.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "LifetimeType">:$type
  );

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      assert(::isa<LifetimeType>(type) && "LifetimeAttr must have LifetimeType");
      return get(type.getContext(), ::cast<LifetimeType>(type));
    }]>,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$isMutable), [{
      return get(isMutable.getContext(), LifetimeType::get(isMutable));
    }]>,
    AttrBuilder<(ins "bool":$isMutable), [{
      return get($_ctxt, LifetimeType::get($_ctxt, isMutable));
    }]>
 ];
}

//===----------------------------------------------------------------------===//
// LifetimeUnionAttr
//===----------------------------------------------------------------------===//

def LifetimeUnionAttr : LITTypedAttr<"LifetimeUnion", "lifetime.union"> {
  let summary = "A #lit.lifetime.union set";
  let description = [{
    This attribute represents a set of symbolic lifetime attributes.  It is
    canonicalized on construction, and always has !lit.lifetime type.
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "LifetimeType">:$type);
  let assemblyFormat = "`<` $operands `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>":$operands,
      "LifetimeType":$type
    ), "", "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "LifetimeType":$type), [{
      return get(operands, type);
    }], "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands), [{}], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static LifetimeUnionAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, LifetimeType type);
  }];
}

//===----------------------------------------------------------------------===//
// LifetimeMutCastAttr
//===----------------------------------------------------------------------===//

def LifetimeMutCastAttr : LITTypedAttr<"LifetimeMutCast", "lifetime.mutcast"> {
  let summary = "Cast a !lit.lifetime value to a different mutability";
  let description = [{
    This attribute is a cast of a lifetime value from one mutability to another.
    It is often folded away by the builder.
  }];

  let parameters = (ins "TypedAttr":$operand,
                        AttributeSelfTypeParameter<"", "LifetimeType">:$type);

  let assemblyFormat = "`<` $operand `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "TypedAttr":$isMutable), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "mlir::Type":$type), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "bool":$isMutable), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$operand, "LifetimeType":$type), [{
      return get(operand, type);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified value has a LifetimeMutCastAttr on it, return the
    /// operand, otherwise return the value.
    static TypedAttr strip(TypedAttr value) {
      if (auto castAttr = llvm::dyn_cast<LifetimeMutCastAttr>(value))
        return castAttr.getOperand();
      return value;
    }

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static LifetimeMutCastAttr getFromBytecode(
        TypedAttr operand, LifetimeType type);
  }];
}

//===----------------------------------------------------------------------===//
// ImplicitLifetimeRefAttr
//===----------------------------------------------------------------------===//

def ImplicitLifetimeRefAttr
  : LITTypedAttr<"ImplicitLifetimeRef", "lifetime_ref", [
    DeclareAttrInterfaceMethods<IndexRefAttrInterface>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Reference to an unbound implicit lifetime parameter.";
  let description = [{
    The `#lit.lifetime_ref` attribute is a reference to an unbound implicit
    lifetime parameter.

    Example:

    ```mlir
    #lit.lifetime_ref<0, 1>
    ```
  }];

  let parameters = (ins "size_t":$depth, "size_t":$index,
                        AttributeSelfTypeParameter<"", "LifetimeType">:$type);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "size_t":$depth, "size_t":$index, "Type":$type), [{
      return get(type.getContext(), depth, index, ::cast<LifetimeType>(type));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// LifetimeSetAttr
//===----------------------------------------------------------------------===//

def LifetimeSetAttr : LITTypedAttr<"LifetimeSet", "lifetime.set"> {
  let summary = "a set of lifetimes with different mutabilities";
  let description = [{
    The `#lit.lifetime.set` attribute contains a set of distinct lifetimes with
    distict mutabilities. This tracks the individual mutabilities of each
    element and propagates them. It can be converted to a single lifetime value
    via `#lit.lifetime.union`.

    Example:

    ```mlir
    #lit.lifetime.set<{mut a, imm b, (is_mut) c}>
    ```
  }];


  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "LifetimeSetType">:$type);
  let assemblyFormat = "`<` custom<LifetimeSet>($operands) `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "LifetimeSetType":$type)>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$operands,
                                        "LifetimeSetType":$type)>,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands)>
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static LifetimeSetAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, LifetimeSetType type);
  }];
}

//===----------------------------------------------------------------------===//
// LifetimeSetUnionAttr
//===----------------------------------------------------------------------===//

def LifetimeSetUnionAttr : LITTypedAttr<"LifetimeSetUnion",
                                        "lifetime.set.union"> {
  let summary = "converts a lifetime set to a single coarse lifetime";
  let description = [{
    The `#lit.lifetime.set.union` attribute converts a `!lit.lifetime.set` value
    to a single coarse-grained lifetime value by converting passing the elements
    into a `#lit.lifetime.union`.
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"", "LifetimeType">:$type);
  let assemblyFormat = "`<` $value `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "TypedAttr":$value, "LifetimeType":$type), [{
      return get(value, type);
    }], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value,
                                        "LifetimeType":$type),
    [{}], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static LifetimeSetUnionAttr getFromBytecode(
        TypedAttr value, LifetimeType type);
  }];
}

//===----------------------------------------------------------------------===//
// UnresolvedAliasValueAttr
//===----------------------------------------------------------------------===//

def UnresolvedAliasValueAttr
  : LITTypedAttr<"UnresolvedAliasValue", "unresolved.alias"> {
  let summary = "The value of an 'alias' before it is bound to a value";

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), type);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def LIT_StructAttr : LITAttr<"LITStruct", "struct", [TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>,
    DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "A constant struct value";
  let description = [{
    The `#lit.struct` attribute represents a constant struct value. It stores
    the names of the fields and its values.

    Example:

    ```mlir
    #lit.struct<{foo = 5, bar: dtype = dt}> : !lit.struct<@Struct>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"std::tuple<StringAttr, TypedAttr>">:$values,
    AttributeSelfTypeParameter<"", "StructType">:$type);
  let assemblyFormat = "`<` custom<StructElements>($values) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "ArrayRef<std::tuple<StringAttr, TypedAttr>>":$values,
        "StructType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def LIT_StructExtractAttr : LITAttr<"StructExtract", "struct.extract", [
    TypedAttrInterface
  ]> {
  let summary = "Extract an element from a struct value";
  let description = [{
    The `#lit.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #lit.struct.extract<p, "indexfield"> : index
    ```
  }];

  let parameters = (ins "TypedAttr":$structValue, "StringAttr":$field,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat =
    "`<` custom<ColonTypeParamValue>($structValue) `,` $field `>`";

  let skipDefaultBuilders = 1;

  let builders = [
    // Override the default builder used by the generated parser.  This computes
    // the result type correctly and performs inline folding.
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "StructFieldOp":$fieldOp),
                "", "TypedAttr">,
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "StringAttr":$field,
                     "Type":$resultType), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$structValue, "StringAttr":$field,
                     "Type":$resultType), "", "TypedAttr">

  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static StructExtractAttr getFromBytecode(
        TypedAttr structValue, StringAttr field, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// StructGERAttr
//===----------------------------------------------------------------------===//

def LIT_StructGERAttr : LITAttr<"StructGER", "struct.ger",
    [TypedAttrInterface]> {
  let summary = "an element reference to a struct in symbolic memory";
  let description = [{
    The `#lit.struct.ger` attribute is leveraged to implement the interpreter on
    `lit.ref.struct.ger` when using symbolic memory. Since raw offsets cannot be
    computed on symbolic memory, this attribute keeps a stack of references to
    the base pointer, which are processed by any load or store operations.

    The base value must be a `SymbolicPointerAttr` or another `StructGERAttr`.
  }];

  let parameters = (ins "TypedAttr":$value, "StringAttr":$field,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $value `,` $field `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "StringAttr":$field,
                                        "Type":$type), [{
      return $_get(value.getContext(), value, field, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TypeLineageAttr
//===----------------------------------------------------------------------===//

def LIT_TypeLineageAttr : LITAttr<"TypeLineage", "type_lineage"> {
  let summary = "a type and where it was inherited from";
  let description = [{
    The `#lit.type_lineage` attribute contains a type and a lineage of types it
    was inherited from. This attribute is used to provide nicer diagnostics in
    the parser when diagnosing errors with inherited types.

    The attribute simply encodes a straight line of ancestor types that the type
    was inherited from, instead of the full graph representation of the
    ancestry.

    Example:

    ```mlir
    #lit.type_lineage<trait<@Foo>, [trait<@Bar>]>
    ```
  }];

  let parameters = (ins "Type":$type,
                        OptionalArrayRefParameter<"Type">:$inheritedFrom);
  let assemblyFormat = "`<` $type (`,` `[` $inheritedFrom^ `]`)? `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "ArrayRef<Type>":$inheritedFrom), [{
      return get(type.getContext(), type, inheritedFrom);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type, {});
    }]>
  ];
}

def TypeLineageArrayAttr : ArrayOfAttr<LITDialect, "TypeLineageArray",
                                       "type_lineage.array", "TypeLineageAttr">;

//===----------------------------------------------------------------------===//
// RefPackAttr
//===----------------------------------------------------------------------===//

def LIT_RefPackAttr : LITAttr<"RefPack", "ref.pack", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "heterogeneous list of constant elements";
  let description = [{
    The `#kgen.pack` attribute contains a heterogenously typed list of constant
    references.

    Example:

    ```mlir
    // A pack of 3 elements.
    %0 = kgen.param.constant: !lit.ref.pack<[i8, ui4, i32]> =
         <store_to_mem(3), store_to_mem(1), store_to_mem(4)>>
    // An empty pack.
    %1 = kgen.param.constant: !lit.ref.pack<[]> = <<>>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "RefPackType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<PackElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "RefPackType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TypeConventionAttr
//===----------------------------------------------------------------------===//

def LIT_TypeConventionAttr : EnumAttr<LITDialect, LIT_TypeConvention,
    "type_convention">;

//===----------------------------------------------------------------------===//
// VarDeclKindAttr
//===----------------------------------------------------------------------===//

def LIT_VarDeclKindAttr : EnumAttr<LITDialect, LIT_VarDeclKind,
    "var.decl.kind">;

#endif // KGEN_LITDIALECT_LITATTRS_TD
