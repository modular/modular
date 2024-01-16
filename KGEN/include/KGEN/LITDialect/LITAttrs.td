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
// FnMetadataAttr
//===----------------------------------------------------------------------===//

def LIT_FnMetadataAttr : LITAttr<"FnMetadata", "fn_metadata",
    [DeclareAttrInterfaceMethods<FnMetadataAttrInterface>]> {
  let summary = "Function metadata.";
  let description = [{
    The `#lit.fn_metadata` attribute contains additional information about a
    function, such as the argument/parameter names and kinds (e.g. positional or
    keyword-only) and the default values for positional and keyword-only
    arguments/parameters.

    Example:

    ```mlir
    // Metadata specifying argument #0 is called "a", argument #1 is called "b",
    // argument #1 has a default value of `4.2`, and the default value for the
    // parameter with name "p" is equal to 1.
    #lit.fn_metadata<["a", "b"], ["p"], [4.2 : f32], [1: i64]>

    // Metadata specifying argument names only for arguments #1 and #2, no
    // default arguments, and no default parameters.
    #lit.fn_metadata<["", "y", "z"], [], []>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"StringAttr">:$argNames,
    OptionalArrayRefParameter<"PassingKind">:$argPassingKinds,
    OptionalArrayRefParameter<"StringAttr">:$paramNames,
    OptionalArrayRefParameter<"PassingKind">:$paramPassingKinds,
    OptionalArrayRefParameter<"TypedAttr">:$defaultPosArgs,
    OptionalArrayRefParameter<"TypedAttr">:$defaultPosParams,
    OptionalArrayRefParameter<"TypedAttr">:$defaultKwOnlyArgs,
    OptionalArrayRefParameter<"TypedAttr">:$defaultKwOnlyParams,
    "size_t":$numImplicitLifetimeDecls
  );
  let assemblyFormat = [{
    `<` `[` (`]`) : ($argNames^            `]`)? `,`
        `[` (`]`) : ($argPassingKinds^     `]`)? `,`
        `[` (`]`) : ($paramNames^          `]`)? `,`
        `[` (`]`) : ($paramPassingKinds^   `]`)? `,`
        `[` (`]`) : ($defaultPosArgs^      `]`)? `,`
        `[` (`]`) : ($defaultPosParams^    `]`)? `,`
        `[` (`]`) : ($defaultKwOnlyArgs^   `]`)? `,`
        `[` (`]`) : ($defaultKwOnlyParams^ `]`)? `,`
        $numImplicitLifetimeDecls
    `>`
  }];

  let builders = [
    AttrBuilder<(ins "ArrayRef<StringAttr>":$argNames,
                     "ArrayRef<PassingKind>":$argPassingKinds,
                     "size_t":$numImplicitLifetimeDecls)>
  ];

  let extraClassDeclaration = [{
    /// Create a new metadata attribute with the given argument names. Only use
    /// this helper if certain that the default arguments do not need to be
    /// modified (e.g. because prepending a positional-only argument).
    FnMetadataAttr cloneWith(ArrayRef<StringAttr> argNames,
                             ArrayRef<PassingKind> argPassingKinds) const;
  }];

  let genVerifyDecl = 1;
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
  let summary = "An attribute representing an unpacked value.";
  let description = [{
    Represents unpacked expressions (a.k.a. starred expressions in Python).
    The contained expression must be a single iterable attribute e.g. a
    `#kgen.variadic` (in which case this represents unpacking it), or an
    `UnboundAttr` (whose unpacking means repeating it).
  }];

  let parameters = (ins "TypedAttr":$value,
                        AttributeSelfTypeParameter<"", "UnpackedType">:$type);
  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `>`";
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
    #lit.module : !lit.metatype<@some_module>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "MetaTypeType">:$type);
  let assemblyFormat = "";

  let builders = [
    AttrBuilderWithInferredContext<(ins "MetaTypeType":$type),
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
      :metatype<@SIMD<:dtype #kgen.unbound, #kgen.unbound>, <dtype, index>>
                @SIMD<:dtype #kgen.unbound, #kgen.unbound>,
      [f32, 16]>
      : !lit.metatype<@SIMD<:dtype f32, 16>>

    // Fully unbound to partially bound.
    #lit.bind_type<
      :metatype<@SIMD<:dtype #kgen.unbound, #kgen.unbound>, <dtype, index>>
                @SIMD<:dtype #kgen.unbound, #kgen.unbound>,
      [#kgen.unbound, 16]>
      : !lit.metatype<@SIMD<:dtype #kgen.unbound, 16>, <dtype>>

    // Partially unbound to fully bound with a default.
    #lit.bind_type<
      :metatype<@SIMD<:dtype #kgen.unbound, 16>, <dtype = f32>>
                @SIMD<:dtype #kgen.unbound, 16>,
      [si32, 16]>
      : !lit.metatype<@SIMD<:dtype si32, 16>>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$typeValue,
    ArrayRefParameter<"TypedAttr">:$values,
    AttributeSelfTypeParameter<"", "MetaTypeType">:$type
  );

  let assemblyFormat = [{
    `<` custom<ColonTypeParamValue>($typeValue) `,` `[`
    custom<BindTypeParams>($values, ref($typeValue)) `]` `>`
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "ArrayRef<TypedAttr>":$values,
                                        "MetaTypeType":$type), [{
      return get(type.getContext(), typeValue, values, type);
    }], "TypedAttr">,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$typeValue,
                                        "ArrayRef<TypedAttr>":$values), [{
      return get(typeValue.getContext(), typeValue, values,
                 getBindTypeResultType(typeValue, values));
    }], "TypedAttr">,
    TypeBuilder<(ins "TypedAttr":$typeValue, "ArrayRef<TypedAttr>":$values,
                     "MetaTypeType":$type), "", "TypedAttr">
  ];
}

//===----------------------------------------------------------------------===//
// LifetimeAttr
//===----------------------------------------------------------------------===//

def LifetimeAttr : LITTypedAttr<"Lifetime", "lifetime"> {
  let summary = "A !lit.lifetime value";
  let description = [{
    This attribute is a singleton used when lowering lifetimes.
  }];

  let assemblyFormat = "";
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// LifetimeUnionAttr
//===----------------------------------------------------------------------===//

def LifetimeUnionAttr : LITTypedAttr<"LifetimeUnion", "lifetime.union"> {
  let summary = "A !lit.lifetime.union set";
  let description = [{
    This attribute represents a set of symbolic lifetime attributes.  It is
    canonicalized on construction, and always has !lit.lifetime type.
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$operands);
  let assemblyFormat = "`<` $operands `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins
      "ArrayRef<TypedAttr>":$operands,
      "mlir::Type":$type
    ), "", "TypedAttr">,

    // The standard builder that infers lifetime type.
    AttrBuilder<(ins
      "ArrayRef<TypedAttr>":$operands
    ), "", "TypedAttr">
  ];

   let extraClassDeclaration = [{
    Type getType() const;
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }
  }];
}

//===----------------------------------------------------------------------===//
// ImplicitLifetimeRefAttr
//===----------------------------------------------------------------------===//

def ImplicitLifetimeRefAttr
    : LITTypedAttr<"ImplicitLifetimeRef", "lifetime_ref"> {
  let summary = "Reference to an unbound implicit lifetime parameter.";
  let description = [{
    The `#lit.lifetime_ref` attribute is a reference to an unbound implicit
    lifetime parameter.

    Example:

    ```mlir
    #lit.lifetime_ref<0, 1>
    ```
  }];

  let parameters = (ins "size_t":$depth, "size_t":$index);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let extraClassDeclaration = [{ Type getType() const; }];
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
    #lit.struct<{foo = 5, bar: dtype = dt}> : !kgen.declref<@Struct>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"std::tuple<StringAttr, TypedAttr>">:$values,
    AttributeSelfTypeParameter<"", "DeclRefType">:$type);
  let assemblyFormat = "`<` custom<StructElements>($values) `>`";
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
// TypeConventionAttr
//===----------------------------------------------------------------------===//

def LIT_TypeConventionAttr : EnumAttr<LITDialect, LIT_TypeConvention,
    "type_convention">;

//===----------------------------------------------------------------------===//
// PassingKindArrayAttr
//===----------------------------------------------------------------------===//

def LIT_PassingKindArrayAttr : LITAttr<"PassingKindArray", "passing.kinds"> {
  let parameters = (ins OptionalArrayRefParameter<"PassingKind">:$value);
  let assemblyFormat = "`<` `[` (`]`) : ($value^ `]`)? `>`";
}

//===----------------------------------------------------------------------===//
// VarLetDeclKindAttr
//===----------------------------------------------------------------------===//

def LIT_VarLetDeclKindAttr : EnumAttr<LITDialect, LIT_VarLetDeclKind,
    "varlet.decl.kind">;

#endif // KGEN_LITDIALECT_LITATTRS_TD
