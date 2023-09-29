//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the LITDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_TD
#define KGEN_LITDIALECT_LITATTRS_TD

include "KGEN/LITDialect/LITDialect.td"
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
// PackageArchiveAttr
//===----------------------------------------------------------------------===//

def LIT_PackageArchiveAttr : LITAttr<"PackageArchive", "package.archive"> {
  let summary = "A pre-compiled, standalone archive for a package";
  let description = [{
    The `#lit.package.archive` attribute stores binary artifacts for an
    elaborated Mojo package module that has been compiled into a standalone
    archive.

    Example:

    ```mlir
    #archive = #lit.package.archive<
      elaboratedModule = dense_resource<foo> : tensor<42xui8>,
      archive = dense_resource<bar> : tensor<13xui8>
    >
    ```
  }];

  let parameters = (ins
    Builtin_DenseResourceElementsAttr:$elaboratedModule,
    Builtin_DenseResourceElementsAttr:$archive
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DenseResourceElementsAttr":$elaboratedModule,
      "DenseResourceElementsAttr":$archive), [{
        return get(elaboratedModule.getContext(), elaboratedModule, archive);
      }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

def KGEN_FnMetadataAttr : LITAttr<"FnMetadata", "fn_metadata",
    [DeclareAttrInterfaceMethods<FnMetadataAttrInterface>]> {
  let summary = "Function metadata.";
  let description = [{
    The `#lit.fn_metadata` attribute contains additional information about a
    function, such as its argument calling conventions (pass-by-value or
    pass-by-reference), any default argument values, the argument names and
    kinds (e.g. positional or keyword-only), parameter names, and the default
    parameter values.

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
    OptionalArrayRefParameter<"StringAttr">:$paramNames,
    OptionalArrayRefParameter<"TypedAttr">:$defaultArguments,
    OptionalArrayRefParameter<"TypedAttr">:$defaultParameters
  );
  let assemblyFormat = [{
    `<` `[` (`]`) : ($argNames^          `]`)? `,`
        `[` (`]`) : ($paramNames^        `]`)? `,`
        `[` (`]`) : ($defaultArguments^  `]`)? `,`
        `[` (`]`) : ($defaultParameters^ `]`)?
    `>`
  }];

  let builders = [
    AttrBuilder<(ins "unsigned":$numArgs, "unsigned":$numParams)>,
    AttrBuilder<(ins "ArrayRef<StringAttr>":$argNames,
                     "ArrayRef<StringAttr>":$paramNames)>,
  ];

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
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "DictionaryAttr":$attrs), [{
      return $_get(name.getContext(), name, attrs);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

def NoneAttr : LITTypedAttr<"None", "none"> {
  let summary = "A None value";
  let description = [{
    This attribute represents None values.
  }];

  let assemblyFormat = "";
  let extraClassDeclaration = [{ Type getType() const; }];
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

def LIT_StructAttr : LITAttr<"Struct", "struct", [TypedAttrInterface,
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

#endif // KGEN_LITDIALECT_LITATTRS_TD
