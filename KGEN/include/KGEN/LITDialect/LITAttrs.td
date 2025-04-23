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
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
      "PassingKind":$passingKind, CArg<"bool", "false">:$isVariadic
    ), [{
      return $_get(name.getContext(), name, passingKind, isVariadic);
    }]>,
    AttrBuilder<(ins), [{
      return $_get($_ctxt, StringAttr::get($_ctxt), PassingKind::PosOrKw,
                   /*isVariadic=*/false);
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

def LIT_PogListAttr : LITAttr<"PogList", "pog_list",
    [DeclareAttrInterfaceMethods<GeneratorMetadataAttrInterface>]> {
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
    AttrBuilder<(ins "size_t":$numPogs)>,
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

    /// Return the number of parameter entries.
    size_t size() const { return getPogs().size(); }

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
    /// Return true if there are inferred parameters.
    bool hasInferredParams() const;

    /// Return the name for the argument/parameter at the given index.
    StringAttr getName(size_t idx) const;
    /// Return the passing kind for the argument/parameter at the given index.
    PassingKind getPassingKind(size_t idx) const;

    /// Return the number of leading "implicit" parameters by PassingKinds.
    size_t getNumImplicit() const;

    /// Create an array of PogMetadataAttr from the given names, passing kinds,
    /// and variadic indices.
    static SmallVector<PogMetadataAttr> toPogs(
      ArrayRef<StringAttr> names, ArrayRef<PassingKind> passingKinds,
      ArrayRef<size_t> indices);

    /// Get a new metadata attribute for a generator with the given number of
    /// positional input parameters prepended to the generator. An additional
    /// array of bool corresponding to the variadic mask of the prepended
    /// parameters is also required.
    PogListAttr prependPosParams(size_t numNewParams,
                                 ArrayRef<bool> variadicMask) const;
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
    function, such as the argument list metadata and the number of implicit
    origin declarations.

    Example:

    ```mlir
    // Metadata specifying that: argument #0 is called "someRef" is
    // positional-only and a default value of `13`; argument #1 and #2 are keyword-only
    // and are called "v" and "kwargs", the former has a default value of `17`,
    // and the latter is variadic (i.e. `**kwargs`).
    // The signature has two implicit origin declarations.
    #lit.fn_metadata<
      <["someRef", "v", "kwargs"], [pos, kw], [13 : index], [17 : i64]>,
      2
    >
    ```
  }];

  let parameters = (ins
    LIT_PogListAttr:$argListAttrs,
    "size_t":$numImplicitOriginDecls,
    "TypedAttr":$captureOrigins,
    // TODO: Generalize this to a set of origin parameters that are guaranteed
    // to not be dereferenced.
    OptionalParameter<"bool">:$isNestedOriginExclusivityCheckingDisabled
  );
  let assemblyFormat = [{
    `<` $argListAttrs `,` $numImplicitOriginDecls
    `` custom<FnMetadataOrigins>($captureOrigins)
    (`,` $isNestedOriginExclusivityCheckingDisabled^)? `>`
  }];

  let builders = [
    AttrBuilder<(ins)>,
    AttrBuilder<(ins "size_t":$numArgs,
                     CArg<"size_t", "0">:$numImplicitOriginDecls)>,
    AttrBuilderWithInferredContext<(ins
      "PogListAttr":$argListAttrs,
      CArg<"size_t", "0">:$numImplicitOriginDecls)>,
    AttrBuilderWithInferredContext<(ins
      "PogListAttr":$argListAttrs, "size_t":$numImplicitOriginDecls,
      "TypedAttr":$captureOrigins, "bool":$nestedOriginFlag)>,
  ];

  let extraClassDeclaration = [{
    size_t getNumArgs() const;
    ArrayRef<TypedAttr> getDefaultPosArgs() const {
      return getArgListAttrs().getDefaultPos();
    }
    ArrayRef<TypedAttr> getDefaultKwOnlyArgs() const {
      return getArgListAttrs().getDefaultKwOnly();
    }

    /// Reconstruct the metadata with the union of the current capture origins
    /// with the new capture origins.
    FnMetadataAttr addCaptureOrigins(TypedAttr origins);

    /// Return true if the signature has a variadic argument.
    bool hasVarArgs() const;
    /// Return true if the signature has a variadic pack argument.
    bool hasPackVarArgs() const;
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

  let parameters = (ins "TypedAttr":$value, "bool":$kwOnly,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{
    `<` custom<ColonTypeParamValue>($value) `,` custom<UnpackKind>($kwOnly) `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "bool":$kwOnly,
                                        "Type":$type), [{
      return $_get(type.getContext(), value, kwOnly, type);
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
    #lit.module : meta<!lit.struct<@some_module>>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "StructMetaType">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StructMetaType":$type),
                                   [{ return get(type.getContext(), type); }]>
  ];
}

//===----------------------------------------------------------------------===//
// AnyOriginAttr
//===----------------------------------------------------------------------===//

def AnyOriginAttr : LITTypedAttr<"AnyOrigin", "any.origin"> {
  let summary = "A !lit.origin<mut> value that could refer to anything";
  let description = [{
    This attribute is a singleton (for each mutability) which represents
    origin that might alias anything.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "OriginType">:$type
  );

  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get(type.getContext(), ::cast<OriginType>(type));
    }]>,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$isMutable), [{
      return get(isMutable.getContext(), OriginType::get(isMutable));
    }]>,
    AttrBuilder<(ins "bool":$isMutable), [{
      return get($_ctxt, OriginType::get($_ctxt, isMutable));
    }]>
 ];
}

//===----------------------------------------------------------------------===//
// StaticOriginAttr
//===----------------------------------------------------------------------===//

def StaticOriginAttr : LITTypedAttr<"StaticOrigin", "static.origin", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "A !lit.origin<mut> value that refers to static memory";
  let description = [{
    This attribute is a singleton (for each mutability) which represents the
    origin of global static memory, used for static strings and global
    variables.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "OriginType">:$type
  );

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// OriginUnionAttr
//===----------------------------------------------------------------------===//

def OriginUnionAttr : LITTypedAttr<"OriginUnion", "origin.union", [
      DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "A #lit.origin.union set";
  let description = [{
    This attribute represents a union of symbolic origin attributes.  It is
    canonicalized on construction, and always has !lit.origin type.
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = "`<` $operands `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>":$operands,
      "OriginType":$type
    ), "", "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "OriginType":$type), [{
      return get(operands, type);
    }], "TypedAttr">,
    AttrBuilder<(ins CArg<"ArrayRef<TypedAttr>", "{}">:$operands),
                [{}], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return get({}, ::cast<OriginType>(type));
    }], "TypedAttr">,
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
    static OriginUnionAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginMutCastAttr
//===----------------------------------------------------------------------===//

def OriginMutCastAttr : LITTypedAttr<"OriginMutCast", "origin.mutcast", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Cast a !lit.origin value to a different mutability";
  let description = [{
    This attribute is a cast of a origin value from one mutability to another.
    It is often folded away by the builder.
  }];

  let parameters = (ins "TypedAttr":$operand,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $operand `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "TypedAttr":$isMutable), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "Type":$type), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$operand, "bool":$isMutable), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$operand, "OriginType":$type), [{
      return get(operand, type);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified value has a OriginMutCastAttr on it, return the
    /// operand, otherwise return the value.
    static TypedAttr strip(TypedAttr value) {
      if (auto castAttr = llvm::dyn_cast<OriginMutCastAttr>(value))
        return castAttr.getOperand();
      return value;
    }

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginMutCastAttr getFromBytecode(
        TypedAttr operand, OriginType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginFieldAttr
//===----------------------------------------------------------------------===//

def OriginFieldAttr : LITTypedAttr<"OriginField", "origin.field", [
          DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
  ]> {
  let summary = "Get a field-sensitive subfield of another origin";
  let description = [{
    This attribute refines a larger origin to that of a subfield.  This
    subfield is known to alias the larger origin, but not other sibling origins.
  }];

  let parameters = (ins "TypedAttr":$base, "StringAttr":$field,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $base `,` $field `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$base, "StringAttr":$field), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$base, "StringAttr":$field,
                "OriginType":$type), [{
      return get(base, field);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginFieldAttr getFromBytecode(TypedAttr base,
        StringAttr field, OriginType type);
  }];
}


//===----------------------------------------------------------------------===//
// IndirectOriginAttr
//===----------------------------------------------------------------------===//

def IndirectOriginAttr : LITTypedAttr<"IndirectOrigin", "indirect.origin",  [
          DeclareAttrInterfaceMethods<ParameterAttrInterface>
  ]> {
  let summary = "Access an indirectly defined origin";
  let description = [{
    This attribute specifies an indirectly accessed memory location that is
    usually (but not necessarily) a pointer indirection away from the base.  The
    base origin controls invalidation of the indirect origin, and may either be
    a "shared" base (which is used to model Mojo concepts like ArcPointer or
    a "unique" base (used to model things like OwnedPointer, Variant, List,
    etc).
  }];

  let parameters = (ins "TypedAttr":$base,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);

  let assemblyFormat = "`<` $base `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // These builders fold the request automatically.
    AttrBuilderWithInferredContext<(ins
      "TypedAttr":$base), "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$base, "OriginType":$type), [{
      return get(base);
    }], "TypedAttr">,
  ];
}

//===----------------------------------------------------------------------===//
// ImplicitOriginRefAttr
//===----------------------------------------------------------------------===//

def ImplicitOriginRefAttr
  : LITTypedAttr<"ImplicitOriginRef", "origin.ref", [
    DeclareAttrInterfaceMethods<IndexRefAttrInterface>,
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Reference to an unbound implicit origin parameter.";
  let description = [{
    The `#lit.origin.ref` attribute is a reference to an unbound implicit
    origin parameter.

    Example:

    ```mlir
    #lit.origin.ref<0, 1>
    ```
  }];

  let parameters = (ins "size_t":$depth, "size_t":$index,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "size_t":$depth, "size_t":$index, "Type":$type), [{
      return get(type.getContext(), depth, index, ::cast<OriginType>(type));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// OriginSetAttr
//===----------------------------------------------------------------------===//

def OriginSetAttr : LITTypedAttr<"OriginSet", "origin.set"> {
  let summary = "a set of origins with different mutabilities";
  let description = [{
    The `#lit.origin.set` attribute contains a set of distinct origins with
    distict mutabilities. This tracks the individual mutabilities of each
    element and propagates them. It can be converted to a single origin value
    via `#lit.origin.union`.

    Example:

    ```mlir
    #lit.origin.set<{mut a, imm b, (is_mut) c}>
    ```
  }];


  let parameters = (ins ArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"", "OriginSetType">:$type);
  let assemblyFormat = "`<` custom<OriginSet>($operands) `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands, "OriginSetType":$type),
                "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$operands,
                                        "OriginSetType":$type), "", "TypedAttr">,
    AttrBuilder<(ins "ArrayRef<TypedAttr>":$operands), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginSetAttr getFromBytecode(
        ArrayRef<TypedAttr> operands, OriginSetType type);
  }];
}

//===----------------------------------------------------------------------===//
// OriginSetUnionAttr
//===----------------------------------------------------------------------===//

def OriginSetUnionAttr : LITTypedAttr<"OriginSetUnion",
                                      "origin.set.union"> {
  let summary = "converts a origin set to a single coarse origin";
  let description = [{
    The `#lit.origin.set.union` attribute converts a `!lit.origin.set` value
    to a single coarse-grained origin value by converting passing the elements
    into a `#lit.origin.union`.
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"", "OriginType">:$type);
  let assemblyFormat = [{
    `<` custom<ParamValue>($value, "OriginSetType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "TypedAttr":$value, "OriginType":$type), [{
      return get(value, type);
    }], "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value,
                                        "OriginType":$type),
    [{}], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static OriginSetUnionAttr getFromBytecode(
        TypedAttr value, OriginType type);
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

  // Disable default builders so we can canonicalize on construction.
  let skipDefaultBuilders = 1;

  let builders = [
    // Returns TypedAttr instead of StructAttr because it canonicalizes on
    // construction.
    AttrBuilder<(ins "ArrayRef<std::tuple<StringAttr, TypedAttr>>":$values,
                     "StructType":$type), "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins
        "ArrayRef<std::tuple<StringAttr, TypedAttr>>":$values,
        "StructType":$type), [{
      return get(type.getContext(), values, type);
    }], "TypedAttr">
  ];
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def LIT_StructExtractAttr : LITAttr<"StructExtract", "struct.extract", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>
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
