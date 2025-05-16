//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_TD
#define KGEN_POPDIALECT_POPATTRS_TD

include "KGEN/KGENDialect/KGENAttrInterfaces.td"
include "KGEN/POPDialect/POPDialect.td"
include "KGEN/POPDialect/POPEnums.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class POP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<POPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// The base class of all parameter expression attributes. Expression attributes
/// are all typed attributes.
class POP_ExprAttr<string attrName, string attrMnemonic,
                    list<Trait> traits = []>
    : POP_Attr<attrName, attrMnemonic,
               [TypedAttrInterface,
                DeclareAttrInterfaceMethods<ParameterAttrInterface>] # traits>;

//===----------------------------------------------------------------------===//
// UnionAttr
//===----------------------------------------------------------------------===//

def POP_UnionAttr : POP_Attr<"Union", "union", [TypedAttrInterface]> {
  let summary = "a union with a specific typed value";
  let description = [{
    The `#pop.union` attribute represents a union with a value of one of the
    union's element types. The attribute knows which union element the value is
    and contains an attribute of said value type.

    Example:

    ```mlir
    #pop.union<0> : !pop.union<index, struct<()>>
    #pop.union<:i64 42> : !pop.union<i64, none>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                         AttributeSelfTypeParameter<"", "UnionType">:$type);
  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "UnionType":$type), [{
      return $_get(type.getContext(), value, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// SIMDAttr
//===----------------------------------------------------------------------===//

def POP_SIMDAttr : POP_ExprAttr<"SIMD", "simd"> {
  let summary = "A SIMD vector attribute.";
  let description = [{
    The `#pop.simd` attribute represents a constant SIMD vector value. It
    contains `N` values of a particular dtype. Only integer, floating point, and
    bool dtypes are supported.

    Example:

    ```mlir
    #pop.simd<1, 2> : !pop.simd<2, si32>
    #pop.simd<1.5, 2.5> : !pop.simd<2, f64>
    #pop.simd<true, false> : !pop.simd<2, bool>
    ```

    When all values of the SIMD vector are equal, the attribute has a special
    splat syntax:

    ```mlir
    #pop<simd 0> : !pop.simd<4, si32>
    #pop<simd "1.5"> : !pop.simd<4, f32>
    #pop<simd false> : !pop.simd<4, bool>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"DTypeValue">:$values,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);
  let assemblyFormat = "`` custom<SIMDValues>($values, ref($type))";

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<DTypeValue>":$values,
                                        "SIMDType":$type), [{
      return get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "DTypeValue":$value,
                                        "SIMDType":$type), [{
      return get(ArrayRef(value), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "uint64_t":$intVal, "SIMDType":$type)>
  ];


  let extraClassDeclaration = [{
    /// Create a "zero" value initialized to 0, 0.0, false, etc.  NOTE: This can
    /// return null if the SIMD type is parametric or we don't know how to make
    /// a zero value of the specified dtype.
    static SIMDAttr getZeroValue(SIMDType type);
  }];

  let genVerifyDecl = 1;

  /// Note: DTypeValue is not POD since APInt will allocate if its data exceeds
  /// 64 bits. Use a custom storage class to own the memory.
  let genStorageClass = 0;
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def POP_ArrayAttr : POP_ExprAttr<"Array", "array"> {
  let summary = "An array attribute.";
  let description = [{
    The `#pop.array` attribute contains a homogeneous list of elements of an
    array type. It is used to represent array constant values.

    Example:

    ```mlir
    #pop.array<1, 2> : !pop.array<2, index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "ArrayType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "ArrayType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralAttr : POP_ExprAttr<"IntLiteral", "int_literal"> {
  let summary = "Attribute for holding infinite precision integer literals.";
  let description = [{
    The `#pop.int_literal` represents an integer literal, storing its
    data in an IPInt.
  }];

  let parameters = (ins "IPInt":$value);

  let assemblyFormat = "`<` custom<IntLiteral>($value) `>`";

  let extraClassDeclaration = [{
    Type getType() const;
  }];

  let builders = [
    AttrBuilder<(ins "int64_t":$value), [{
      return $_get($_ctxt, IPInt(value));
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralConvertAttr : POP_ExprAttr<"IntLiteralConvert",
                                         "int_literal_convert", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["validateForElaborator"]>]> {
  let summary = "Convert an integer literal to a different type.";
  let description = [{
    The `#pop.int_literal_convert` attribute is used to convert an integer
    literal to a different type.  It is the parameter domain form of
    the `pop.int_literal.convert` operation.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$input,
    "bool":$treatIndexAsUnsigned
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "IntLiteralType::get($_ctxt)")
    `,` $treatIndexAsUnsigned `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$input,
                     "bool":$treatIndexAsUnsigned), "", "TypedAttr">,
  ];
}

//===----------------------------------------------------------------------===//
// IntLiteralBinAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralBinAttr : POP_ExprAttr<"IntLiteralBin",
                                         "int_literal_bin"> {
  let summary = "Convert an integer literal to a different type.";
  let description = [{
    The `#pop.int_literal_binop` attr does standard arithmetic binary
    operations on `!pop.int_literal` values that return a `!pop.int_literal`.
  }];

  let parameters = (ins
    "TypedAttr":$lhs, "TypedAttr":$rhs,
    "IntLiteralBinKindAttr":$oper
  );

  let assemblyFormat = "`<` $oper $lhs `,` $rhs `>`";
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs,
                     "IntLiteralBinKindAttr":$oper), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// IntLiteralCmpAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralCmpAttr : POP_ExprAttr<"IntLiteralCmp", "int_literal_cmp"> {
  let summary = "Comparison operator for int_literals.";
  let description = [{
    The `#pop.int_literal_cmp` attr does comparison testing
    operations on `!pop.int_literal` values that return an `i1`.
  }];

  let parameters = (ins
    "IntLiteralCmpPredAttr":$pred,
    "TypedAttr":$lhs, "TypedAttr":$rhs
  );

  let assemblyFormat = [{
    `<` $pred custom<ParamValue>($lhs, "IntLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "IntLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "IntLiteralCmpPredAttr":$pred, "TypedAttr":$lhs,
                     "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralAttr : POP_ExprAttr<"FloatLiteral", "float_literal"> {
  let summary = "Attribute for holding infinite precision float literals.";
  let description = [{
    The `#pop<float_literal` represents an float literal, though it
    is actually stored as a rational number IPRational.

    The `special` field tells whether it is a normal number or one of
    the special values (infinities, negative zero, or NAN).  When it
    is a non-normal number, it does not include a rational number
    value.

    TODO(#23387): MLIR's AsmParser doesn't have `parseSlash` or a more
    generic way to parse literal strings/characters, so we use
    the pipe "|" character instead for rationals.

    Example:

    ```mlir
    #pop.float_literal<normal(5|3)>
    #pop.float_literal<neg_zero>
    #pop.float_literal<inf>
    #pop.float_literal<neg_inf>
    #pop.float_literal<nan>
    ```
  }];

  let parameters = (ins
    POP_FloatLiteralSpecialValuesAttr:$special,
    "std::optional<IPRational>":$rational
  );

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can canonicalize.
    AttrBuilder<(ins "FloatLiteralSpecialValuesAttr":$input,
                     "std::optional<IPRational>":$rational)>,
    // Always a normal value, set the SpecialValue to Normal.
    AttrBuilder<(ins "IPRational":$value)>,
  ];

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsaAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralIsaAttr : POP_ExprAttr<"FloatLiteralIsa",
                                           "float_literal_isa"> {
  let summary = "Check if the float_literal is of the given special value.";
  let description = [{
    The `#pop<float_literal_isa` attribute is used to check if the float
    literal is of the given special value.  It is the parameter domain form of
    the `pop.float_literal.isa` operation.
  }];

  let parameters = (ins
    "FloatLiteralSpecialValuesAttr":$kind,
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` $kind custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "FloatLiteralSpecialValuesAttr":$kind,
                     "TypedAttr":$input), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralConvertAttr : POP_ExprAttr<"FloatLiteralConvert",
                                               "float_literal_convert", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["validateForElaborator"]>]> {
  let summary = "Convert an float literal to a different type.";
  let description = [{
    The `#pop<float_literal_convert` attribute is used to convert an float
    literal to a different type.  It is the parameter domain form of
    the `pop.float_literal.convert` operation.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$input), "", "TypedAttr">,
  ];
}

//===----------------------------------------------------------------------===//
// IntToFloatLiteralAttr
//===----------------------------------------------------------------------===//

def POP_IntToFloatLiteralAttr : POP_ExprAttr<"IntToFloatLiteral",
                                             "int_to_float_literal"> {
  let summary = "Convert an int literal to an float literal";
  let description = [{
    The `#pop.int_to_float_literal` attr converts an int literal to an
    float literal.
  }];

  let parameters = (ins "TypedAttr":$input);

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "IntLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "TypedAttr":$input), "", "TypedAttr">,
  ];
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatToIntLiteralAttr
//===----------------------------------------------------------------------===//

def POP_FloatToIntLiteralAttr : POP_ExprAttr<"FloatToIntLiteral",
                                             "float_to_int_literal"> {
  let summary = "Convert an float literal to an int literal.";
  let description = [{
    The `#pop<float_to_int_literal` attr converts an float literal to an
    int literal.
  }];

  let parameters = (ins
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an FloatToIntLiteralAttr.
    AttrBuilder<(ins "TypedAttr":$input), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralBinAttr : POP_ExprAttr<"FloatLiteralBin",
                                           "float_literal_bin"> {
  let summary = "Convert an float literal to a different type.";
  let description = [{
    The `#pop<float_literal_bin` attr does standard arithmetic binary
    operations on `!pop.float_literal` values that return a `!pop.float_literal`.
  }];

  let parameters = (ins
    "TypedAttr":$lhs, "TypedAttr":$rhs,
    "FloatLiteralBinKindAttr":$oper
  );

  let assemblyFormat = [{
    `<` $oper custom<ParamValue>($lhs, "FloatLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs,
                     "FloatLiteralBinKindAttr":$oper), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmpAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralCmpAttr : POP_ExprAttr<"FloatLiteralCmp",
                                           "float_literal_cmp"> {
  let summary = "Comparison operator for float literals.";
  let description = [{
    The `#pop<float_literal_cmp` attr does comparison testing
    operations on `!pop.float_literal` values that return an `i1`.
  }];

  let parameters = (ins
    "FloatLiteralCmpPredAttr":$pred,
    "TypedAttr":$lhs, "TypedAttr":$rhs
  );

  let assemblyFormat = [{
    `<` $pred custom<ParamValue>($lhs, "FloatLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "FloatLiteralType::get($_ctxt)") `>`
  }];
  
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "FloatLiteralCmpPredAttr":$pred, "TypedAttr":$lhs,
                     "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StringSizeAttr
//===----------------------------------------------------------------------===//

def POP_StringSizeAttr : POP_ExprAttr<"StringSize", "string_size"> {
  let summary = "Get the size of a string.";
  let description = [{
    The `#pop.string_size` attribute gets the size of a !kgen.string value.
  }];

  let parameters = (ins "TypedAttr":$str);

  let assemblyFormat = [{
    `<` custom<ParamValue>($str, "StringType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return a StringSizeAttr.
    AttrBuilder<(ins "TypedAttr":$str), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StringConcatAttr
//===----------------------------------------------------------------------===//

def POP_StringConcatAttr : POP_ExprAttr<"StringConcat", "string_concat"> {
  let summary = "Concatenate two strings.";
  let description = [{
    The `#pop.string_concat` attribute concatenates two !kgen.string values.
  }];

  let parameters = (ins "TypedAttr":$lhs, "TypedAttr":$rhs);

  let assemblyFormat = [{
    `<` custom<ParamValue>($lhs, "StringType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "StringType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an StringConcatAttr.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

#endif // KGEN_POPDIALECT_POPATTRS_TD
