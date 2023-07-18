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

/// An attribute in the LIT dialect.
class LITAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LITDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// A typed attribute in the LIT dialect.
class LITTypedAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : LITAttr<attrName, attrMnemonic, [TypedAttrInterface] # traits>;

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
    "StringAttr":$string, "FileLineColLoc":$location),
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

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "StringAttr":$name, "DictionaryAttr":$attrs);
  let assemblyFormat = "$name $attrs $type";
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

def NoneAttr : LITTypedAttr<"None", "none"> {
  let summary = "A None value";
  let description = [{
    This attribute represents None values.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "LIT::NoneType">:$type);
  let assemblyFormat = "";

  let builders = [AttrBuilder<(ins), [{
    return get(context, LIT::NoneType::get(context));
  }]>];
}

//===----------------------------------------------------------------------===//
// LifetimeAttr
//===----------------------------------------------------------------------===//

def LifetimeAttr : LITTypedAttr<"Lifetime", "lifetime"> {
  let summary = "A !lit.lifetime value";
  let description = [{
    This attribute is a singleton used when lowering lifetimes.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "LIT::LifetimeType">:$type);
  let assemblyFormat = "";

  let builders = [AttrBuilder<(ins), [{
    return get(context, LIT::LifetimeType::get(context));
  }]>];
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
