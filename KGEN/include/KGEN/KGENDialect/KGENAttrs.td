//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines KGEN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_TD
#define KGEN_KGENDIALECT_KGENATTRS_TD

include "KGEN/KGENDialect/KGENAttrInterfaces.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "Support/Compiler/AttrBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

/// An attribute in the KGEN dialect.
class KGENAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KGENDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// The base class of all parameter expression attributes. Expression attributes
/// are all typed attributes.
class KGEN_ExprAttr<string attrName, string attrMnemonic,
                    list<Trait> traits = []>
    : KGENAttr<attrName, attrMnemonic,
               [TypedAttrInterface,
                DeclareAttrInterfaceMethods<ParameterAttrInterface>] # traits>;

// IndexAttr whose return value is an `int64_t`.
def UnsignedIndexAttr :
    TypedAttrBase<
      Index, "IntegerAttr",
      And<[CPred<"isa<IntegerAttr>($_self)">,
           CPred<"isa<IndexType>(cast<IntegerAttr>($_self).getType())">]>,
      "index attribute"> {
  let returnType = "unsigned";
  let convertFromStorage = "$_self.getInt()";
}

//===----------------------------------------------------------------------===//
// ParameterExprArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_ParameterExprArrayAttr
    : ArrayOfAttr<KGENDialect, "ParameterExprArray", "exprs", "TypedAttr">;

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

/// Declaration of a named parameter.
def KGEN_ParamDeclAttr : KGENAttr<"ParamDecl", "param.decl"> {
  let summary = "Parameter declaration";
  let description = [{
    This is a declaration of a parameter in the meta-programming domain for
    generators and related infrastructure.  These are typically owned by
    kgen.generator instances or other things that produce new attributes.
  }];

  /// ParamDeclAttr's always have a `name` and `type`: an omitted type in the
  /// asmsyntax defaults to `index`.
  let parameters = (ins "StringAttr":$name, "Type":$type);

  let assemblyFormat = "custom<ParamName>($name) `:` $type";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type),
      "return $_get(type.getContext(), name, type);">,
    AttrBuilderWithInferredContext<(ins "ParamDeclRefAttr":$ref),
      "return $_get(ref.getContext(), ref.getName(), ref.getType());">,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">
  ];
}

//===----------------------------------------------------------------------===//
// ParamDeclArrayAttr
//===----------------------------------------------------------------------===//

/// An array of ParamDeclAttr's typically used on a kgen.generator and other
/// ops that define parameters.
def KGEN_ParamDeclArrayAttr : ArrayOfAttr<KGENDialect, "ParamDeclArray",
    "param.decls", "ParamDeclAttr"> {
  let builders = [
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$paramDecl), [{
      return get(paramDecl.getContext(), paramDecl);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// VariadicAttr
//===----------------------------------------------------------------------===//

def KGEN_VariadicAttr : KGEN_ExprAttr<"Variadic", "variadic", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A parameterized variadic attribute.";
  let description = [{
    The `#kgen.variadic` attribute contains a homogeneous list of elements of an
    variadic type. It is used to represent constant variadic sequence values.

    Example:

    ```mlir
    #kgen.variadic<1, 2> : !kgen.variadic<index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "VariadicType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "VariadicType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

def KGEN_UnknownAttr : KGEN_ExprAttr<"Unknown", "unknown"> {
  let summary = "An unknown parameter value.";
  let description = [{
    The `#kgen.unknown` attribute represents an unknown parameter value. It is
    a special placeholder that represents a parameter that may only be known
    dynamically.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

def KGEN_UnboundAttr : KGEN_ExprAttr<"Unbound", "unbound"> {
  let summary = "An unbound parameter value.";
  let description = [{
    The `#kgen.unbound` attribute represents an unbound parameter value. It is
    a special placeholder that appears in collections of parameters that have
    been partially bound.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";

  let builders = [TypeBuilderWithInferredContext<(ins "Type":$type), [{
    return get(type.getContext(), type);
  }]>];
}

//===----------------------------------------------------------------------===//
// NoneAttr
//===----------------------------------------------------------------------===//

def KGEN_NoneAttr : KGEN_ExprAttr<"None", "none"> {
  let summary = "A none value";
  let description = [{
    The `#kgen.none` attribute is a value of `!kgen.none` type.
  }];

  let assemblyFormat = "";
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// This attribute models a reference to a named parameter.  The type of the
/// ParamDeclRefAttr must always be the same as the type of the parameter being
/// referenced.
def KGEN_ParamDeclRefAttr : KGEN_ExprAttr<"ParamDeclRef", "param.decl.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a parameter value.";
  let description = [{
    The `#kgen.param.decl.ref` attribute is a typed attribute that represents
    a reference to a declared parameter. It contains the type of the referenced
    parameter and its name.

    Example:

    ```mlir
    // A reference to parameter "p" with type "i1".
    #kgen.param.decl.ref<"p"> : i1
    ```
  }];

  let parameters = (ins "StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type), [{
      return get(name.getContext(), name, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringRef":$name, "Type":$type), [{
      return get(StringAttr::get(type.getContext(), name), type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ParamDeclAttr":$decl), [{
      return get(decl.getName(), decl.getType());
    }]>
  ];

  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr
//===----------------------------------------------------------------------===//

def KGEN_ParamIndexRefAttr : KGEN_ExprAttr<"ParamIndexRef", "param.index.ref",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "A reference to a signature parameter.";
  let description = [{
    The `#kgen.param.index.ref` attribute is a reference to an input
    parameter of an enclosing signature. This attribute can only be used inside
    a `GeneratorType`. This attribute contains two fields: the depth of the
    referenced signature relative to the nearest signature (non-negative
    integer), and the index of the parameter (non-negative integer).

    Example:

    ```mlir
    // Second input parameter of the nearest signature.
    #kgen.param.index.ref<0, 1> : index
    // First input parameter of next enclosing signature.
    #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
    ```
  }];

  let parameters = (ins "size_t":$depth,"size_t":$index,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $depth `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "size_t":$depth, "size_t":$index,
                                        "Type":$type), [{
      return $_get(type.getContext(), depth, index, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "size_t":$index, "Type":$type), [{
      return $_get(type.getContext(), /*depth=*/0, index, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// VTableAttr
//===----------------------------------------------------------------------===//

def KGEN_VTableAttr : KGENAttr<"VTable", "vtable"> {
  let summary = "Static vtable for structs";
  let description = [{
    Static vtable for structs.  Used to implement traits.

    Example:

    ```mlir
    #kgen<vtable "entry1" : <() -> index> = @entry1,
                 "entry2" : <() -> index> = @entry2>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"VTableEntryAttr">:$entries);
  let assemblyFormat = "`` $entries";
}

def KGEN_VTableEntryAttr : KGENAttr<"VTableEntry", "vtable.entry"> {
  let summary = "Entry for VTable attr";
  let description = [{
    Entry for a VTableAttr.  The name and signature are keys in the
    table, while the method is the lookup value.

    Example:

    ```mlir
    #kgen<vtable.entry "methodName" : <() -> index> = @methodImpl>
    ```
  }];

  let parameters = (ins "StringAttr":$name, "TypedAttr":$method);
  let assemblyFormat = "`` custom<VTableEntry>($name, $method)";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "TypedAttr":$method), [{
      return $_get(name.getContext(), name, method);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// TypeParamAttr
//===----------------------------------------------------------------------===//

def VTableParam
  : DefaultValuedParameter<"VTableAttr", "VTableAttr::get($_ctxt, {})">;

def KGEN_TypeParamAttr : KGEN_ExprAttr<"TypeParam", "type"> {
  let summary = "Convert an MLIR type to a parameter expression";
  let description = [{
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute that refers to a type.  The difference
    is that it is a `TypedAttr` so it can be a parameter expression, and has a
    metatype.

    The `typeValue` field encodes the value-representation of a type, while the
    `mlirType` field encodes the type-representation of the type.

    Example:

    ```mlir
    // Default asm format.
    #kgen.type<!myTypeValue, !myMlirType, {
      "method1" : <() -> index> = @method1,
      "method2" : <() -> index> = @method2
    }> : !kgen.type

    // MlirType is omitted if same as typeValue.
    #kgen.type<!myTypeValue, {
      "method1" : <() -> index> = @method1
    }> : !kgen.type

    // VTable is omitted if empty.
    #kgen.type<!myTypeValue> : !kgen.type
    ```
  }];

  let parameters = (ins
    "Type":$typeValue,
    "Type":$mlirType,
    AttributeSelfTypeParameter<"", "Type">:$type,
    VTableParam:$vTable
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "MLIRContext *":$ctx, "Type":$typeValue,
                                        "Type":$mlirType, "Type":$type,
                                        "VTableAttr":$vtable),
                                   "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$typeValue, "Type":$mlirType,
                                        "Type":$type),
                                   "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$typeValue, "Type":$mlirType,
                                        "Type":$type, "VTableAttr":$vtable),
                                   "", "TypedAttr">,
    // Builders for identical typeValue & mlirType cases.
    // Only available temporarily until proper the type/value-domain split.
    AttrBuilderWithInferredContext<(ins "Type":$mlirType, "Type":$type),
                                   "", "TypedAttr">,
    AttrBuilderWithInferredContext<(ins "Type":$mlirType, "Type":$type,
                                        "VTableAttr":$vtable),
                                   "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static TypeParamAttr getFromBytecode(
        Type typeValue, Type mlirType, Type type, VTableAttr vtable);

    /// Whether the type constant has identical value & type representation.
    bool hasIdenticalRepresentation();
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UpcastAttr
//===----------------------------------------------------------------------===//

def KGEN_UpcastAttr : KGEN_ExprAttr<"Upcast", "upcast"> {
  let summary = "Rebind a type with metatype information to base trait.";
  let description = [{
    The `#kgen.upcast` attribute is used to convert from a typeValue (which
    carries a vtable and other metadate) to a typeValue of a less-derived trait.
    For example, this can represent a cast from Movable to AnyType, handling the
    rebind of the `__del__` member.

    This aggressively canonicalizes, e.g. when the operand is a simple type
    value like a struct, it will return a TypeParamAttr.

    Example:

    ```mlir
    #kgen.upcast<#kgen.param.decl.ref<"T"> : !lit.trait<Movable>,
        "__del__" : !lit.generator<("self": !lit.ref<:trait<Movable> T> owned_in_mem, |) -> !kgen.none>
            = get_vtable_entry(:trait<Movable> T, "__del__")> : !lit.trait<AnyType>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$inputTypeValue,
    VTableParam:$vTable
  );
  let assemblyFormat = "`<` $inputTypeValue `,` $vTable `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an UpcastAttr.
    AttrBuilderWithInferredContext<(ins
       "Type":$type, "TypedAttr":$inputTypeValue,
       "VTableAttr":$vTable), "", "TypedAttr">,

    // Verbose form with MLIRContext, needed for ODS constructors.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$inputTypeValue,
                     "VTableAttr":$vTable), [{
      return get(type, inputTypeValue, vTable);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    /// If the specified attribute is an upcast, remove it.
    static TypedAttr strip(TypedAttr value) {
      if (auto upcast = llvm::dyn_cast<UpcastAttr>(value))
        return upcast.getInputTypeValue();
      return value;
    }
  }];
}

//===----------------------------------------------------------------------===//
// BindParams
//===----------------------------------------------------------------------===//

def KGEN_BindParamsAttr : KGEN_ExprAttr<"BindParams", "bind_params"> {
  let summary = "Bind params onto an expression with generator type.";
  let description = [{
    The `#kgen.bind_params` attribute is used to bind parameters onto a
    generator parameter (a parameter whose type is some GeneratorType).

    Partial & out-of-order parameter binding is allowed by specifying the
    unbound positions with UnboundAttr.

    Example:

    Given
    ```mlir
      @SIMD : !kgen.generator<<dtype, index> AnyStruct<@SIMD<*(0,0), *(0,1)>>>
    ```

    This attr binds the index parameter but not the dtype
    ```mlir
    #kgen.bind_param<@SIMD<?, 2>>
      : !kgen.generator<<dtype> AnyStruct<@SIMD<*(0,0), 2>>>
    ```
  }];

  let parameters = (ins
    "TypedAttr":$generator,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "Type">:$type
  );
  let assemblyFormat = "`<` custom<BindParams>($generator, $paramValues) `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins
       "TypedAttr":$generator, "ArrayRef<TypedAttr>":$paramValues,
       "Type":$type), "", "TypedAttr">,
    // Builder that derives the result type from the input paramValues.
    AttrBuilderWithInferredContext<(ins
       "TypedAttr":$generator, "ArrayRef<TypedAttr>":$paramValues), "",
       "TypedAttr">
  ];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// TypeGeneratorRefAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeGeneratorRefAttr : KGEN_ExprAttr<"TypeGeneratorRef", "genref", [
    EvaluatableAttrInterface,
    DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "type generator reference";
  let description = [{
    This is a symbolic reference to a type-value generator. Its type is the
    metatype of the type-value. If the type-value is parametric, additional
    parameter values may be bound.

    TODO: Merge SymbolConstantAttr into this.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        OptionalArrayRefParameter<"TypedAttr">:$paramValues,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, {}, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "ArrayRef<TypedAttr>":$paramValues,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) `` `>`
  }];
}

//===----------------------------------------------------------------------===//
// TypeInstanceRefAttr
//===----------------------------------------------------------------------===//

def KGEN_TypeInstanceRefAttr : KGEN_ExprAttr<"TypeInstanceRef", "instref"> {
  let summary = "type instance reference";
  let description = [{
    This is a symbolic reference to a concrete type-value instance. Its type
    is the metatype of the type-value.
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "SymbolRefAttr":$symbol,
                                        "Type":$type), [{
      return get(type.getContext(), symbol, type);
    }]>
  ];
  let assemblyFormat = "`<` $symbol `>`";
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

/// This is a value of !kgen.dtype with a constant value from 0...255.
def KGEN_DTypeConstantAttr : KGEN_ExprAttr<"DTypeConstant", "dtype.constant",
    [DeclareAttrInterfaceMethods<ParameterAttrInterface, ["isLessThan"]>]> {
  let summary = "Constant !kgen.dtype value";
  let description = [{
    This is constant value for a dtype, whose elements correspond to DType.
  }];

  /// The value of a DTypeConstantAttr is always signless i8 constant value
  /// and the type is always !kgen.dtype.
  let parameters = (ins "KGENDType":$dType);

  let assemblyFormat = "`<` $dType `>`";

  let extraClassDeclaration = [{
    /// Check if the DType is convertible to the MLIR type.
    bool isConvertibleTo(Type type);
    /// Check if the MLIR type can be converted from the DType.
    bool isConvertibleFrom(Type type);
    /// Get the type of the attribute. It is always DTypeType.
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

def KGEN_SymbolConstantAttr : KGEN_ExprAttr<"SymbolConstant", "symbol.constant",
    [DeclareAttrInterfaceMethods<DeclRefAttrInterface>]> {
  let summary = "kgen.func or kgen.generator reference";
  let description = [{
    This is a value of FuncTypeGenerator type, which refers to a func or
    generator.  This may optionally bind the input parameter values at time of
    formation - when this happens, the result type is non-parametric.
  }];

  let parameters = (ins
    "SymbolRefAttr":$symbol,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "SymbolRefAttr":$symbol, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(type.getContext(), symbol, paramValues, type);
    }]>,
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(FlatSymbolRefAttr::get(name), type, paramValues);
    }]>,
    AttrBuilderWithInferredContext<(ins
        "GeneratorOp":$func, "FuncTypeGeneratorType":$type,
        CArg<"ArrayRef<TypedAttr>", "{}">:$paramValues), [{
      return get(func.getSymNameAttr(), type, paramValues);
    }]>,
    AttrBuilderWithInferredContext<(ins "GeneratorOp":$func), [{
      return get(func.getSymNameAttr(), func.getFuncTypeGenerator());
    }]>,
    AttrBuilderWithInferredContext<(ins "FuncOp":$func), [{
      return get(func.getSymNameAttr(), func.getFuncTypeGenerator());
    }]>,
  ];

  let assemblyFormat = [{
    `<` $symbol `` custom<ParameterValues>($paramValues) `` `>`
  }];
}

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

def KGEN_TargetParamAttr : KGEN_ExprAttr<"TargetParam", "target"> {
  let summary = "Target configuration parameter attribute.";
  let description = [{
    The `#kgen.target` is an attribute representing a target of type
    `!kgen.target`. It contains the target configuration information.

    Example:

    ```mlir
    #kgen.target<triple="triple", cpu="cpu", features="features",
                 data_layout="p:32:32", simd_bit_width=128>
    ```

    Target types can be manipulated using target operators.

    Example:

    ```mlir
    kgen.generator @target_host<t0: target>()
        constraints <[eq(:target
            #kgen.target<triple="triple", cpu="cpu", features="features",
                         data_layout="", simd_bit_width=128>,
            t0),
          "Must support the target!"
        ]> {
      kgen.return
    }
    ```
  }];
  let parameters = (ins "TargetInfoAttr":$target);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TargetInfoAttr":$target), [{
      return $_get(target.getContext(), target);
    }]>
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StructDefFieldAttr
//===----------------------------------------------------------------------===//

def KGEN_StructDefFieldAttr : KGENAttr<"StructDefField", "struct_def.field"> {
  let summary = "A single source struct field declaration.";
  let description = [{
    The `#kgen.struct_def.field` attribute represents a field declared in a
    mojo struct. It keeps track of the name and type of the field.

    Example:

    ```mlir
    #kgen.struct_def.field<"num" : Int>
    ```
  }];

  let parameters = (ins "StringAttr":$name, "Type":$type);
  let assemblyFormat = " `<` $name `:` $type `>` ";

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name, "Type":$type), [{
      return get(name.getContext(), name, type);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// StructAttr
//===----------------------------------------------------------------------===//

def KGEN_StructAttr : KGENAttr<"Struct", "struct", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A heterogeneous list of elements.";
  let description = [{
    The `#kgen.struct` attribute contains a heterogenous list of elements of
    struct type. It is used to represent constant struct values.

    Example:

    ```mlir
    #kgen.struct<3, 3.5> : !kgen.struct<(index, f32)>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "StructType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<StructElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values)>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "StructType":$type), [{
      return get(type.getContext(), values, type);
    }]>,
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractAttr
//===----------------------------------------------------------------------===//

def KGEN_StructExtractAttr : KGENAttr<"StructExtract", "struct.extract",
    [TypedAttrInterface]> {
  let summary = "Extract an element from a struct value";
  let description = [{
    The `#kgen.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #kgen.struct.extract<p, 4> : index
    ```
  }];

  let parameters = (ins "TypedAttr":$structValue, "unsigned":$fieldNo,
                        AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat =
    "`<` custom<ColonTypeParamValue>($structValue) `,` $fieldNo `>`";

  let skipDefaultBuilders = 1;

  let builders = [
    // Override the default builder used by the generated parser.  This computes
    // the result type correctly and performs inline folding.
    AttrBuilderWithInferredContext<
                (ins "TypedAttr":$structValue, "unsigned":$fieldNo),
                     "", "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$structValue, "unsigned":$fieldNo,
                     "Type":$resultType), "", "TypedAttr">
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static StructExtractAttr getFromBytecode(
        TypedAttr structValue, unsigned fieldNo, Type resultType);
  }];
}

//===----------------------------------------------------------------------===//
// PackAttr
//===----------------------------------------------------------------------===//

def KGEN_PackAttr : KGENAttr<"Pack", "pack", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "heterogeneous list of constant elements";
  let description = [{
    The `#kgen.pack` attribute contains a heterogenously typed list of constant
    elements. It can be used to represent constant pack values, and so is of
    pack type.

    Example:

    ```mlir
    // A pack of 3 elements.
    %0 = kgen.param.constant: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
    // An empty pack.
    %1 = kgen.param.constant: !kgen.pack<[]> = <<>>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "PackType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<PackElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "PackType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// VariantAttr
//===----------------------------------------------------------------------===//

def KGEN_VariantAttr : KGENAttr<"Variant", "variant", [
    TypedAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface>]> {
  let summary = "A variant constant value.";
  let description = [{
    The `#kgen.variant` attribute represents a constant `!kgen.variant` value.
    It contains a single typed attribute where the typed of the value is
    expected to equal one of the corresponding variant types.

    Example:

    ```mlir
    #kgen.variant<3 : index> : !kgen.variant<index, i32, f32>
    ```
  }];

  let parameters = (ins "TypedAttr":$value, "unsigned":$index,
                        AttributeSelfTypeParameter<"", "VariantType">:$type);

  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `,` $index `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "unsigned":$index,
                                        "VariantType":$type), [{
      return get(type.getContext(), value, index, type);
    }]>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// EnvAttr
//===----------------------------------------------------------------------===//

def KGEN_EnvAttr : KGENAttr<"Env", "env"> {
  let summary = "Environment variable attribute.";
  let description = [{
    The `#kgen.env` attribute defines a generic dictionary of environment
    parameters that can be accessed through parameter operators. The values
    contained can be integers, represented with the `index` type, unit, the
    presence of which indicates something, and strings, represented as
    `!kgen.string` types.

    Example:

    ```mlir
    #kgen<env{intVal = 1 : index, unitAttr, strVal = "hello" : !kgen.string}>
    ```
  }];

  let parameters = (ins "DictionaryAttr":$values);
  let assemblyFormat = "`<` $values `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "DictionaryAttr":$values), [{
      return $_get(values.getContext(), values);
    }]>
  ];

  let extraClassDeclaration = [{
    /// This is the attribute name of the `#kgen.env` attribute attached to the
    /// surrounding `ModuleOp`.
    static llvm::StringLiteral getEnvAttrName() { return "kgen.env"; }

    /// Parse a defines list and create the environment from that.
    static ErrorOr<EnvAttr> parseDefines(MLIRContext *ctx,
                                         ArrayRef<std::string> defines);

    /// Extends the EnvAttr with another one. If the keys conflict, then the
    /// extended value would take precedence.
    EnvAttr extend(EnvAttr attr);
  }];
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

// ParamOperatorAttr forms the core of the expression grammar for parameter
// expressions.  Its operators have been designed so that expression building
// force a deterministic canonical form on expression trees.  This ensures that
// `(x+1)*4` and `(x*4)+4` canonicalize into the same pointer value.
//
// For this reason, we should never have unary negate or subtract. `(x-y)` is
// modeled as `(x+-y)` => `(x + -1*y)`.  The ParamOperatorAttr class does
// provide convenience methods for these aggregate operations.
//
// The design of these operators assumes signful types like si64, and the fully
// associative operations are take arbitrary arity, also to make
// canonicalization simpler.
def KGEN_ParamOperatorAttr : KGEN_ExprAttr<"ParamOperator", "param.expr", [
    EvaluatableAttrInterface,
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["isLessThan", "validateForElaborator"]>
  ]> {
  let summary = "Parameter expression";
  let parameters = (ins "POC":$opcode,
                        OptionalArrayRefParameter<"TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"">:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser.
    AttrBuilder<(ins "POC":$opcode, "ArrayRef<TypedAttr>":$operands,
                     "Type":$type), "", "TypedAttr">,

    // Build a parameter expression.  This automatically canonicalizes and
    // folds, so it may not necessarily return a ParamOperatorAttr.
    AttrBuilderWithInferredContext<(ins
        "POC":$opcode, "ArrayRef<TypedAttr>":$operands), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "POC":$opcode,
        "ArrayRef<TypedAttr>":$operands, "Type":$type), [{
      return get(type.getContext(), opcode, operands, type);
    }], "TypedAttr">,

    // Build a binary parameter expression for convenience.
    AttrBuilderWithInferredContext<(ins "POC":$opcode, "TypedAttr":$lhs,
                                        "TypedAttr":$rhs), [{
      TypedAttr operands[] = { lhs, rhs };
      return get(opcode, operands);
    }], "TypedAttr">
  ];

  let assemblyFormat = "`<` $opcode (`,` $operands^)? `>`";
  let genVerifyDecl = 1;  // Generate getChecked/verify.

  let extraClassDeclaration = [{
    size_t getNumOperands() const {
      return getOperands().size();
    }
    TypedAttr getOperand(size_t i) const {
      return getOperands()[i];
    }

    /// Return (ne x, y) which is the same as (not (eq x, y)).
    static TypedAttr getNE(TypedAttr lhs, TypedAttr rhs) {
      return getNot(get(POC::EQ, lhs, rhs));
    }

    /// Return (not x) which is the same as (xor x, true).  The `operand` value
    /// must have type `i1`.
    static TypedAttr getNot(TypedAttr operand);

    /// Return (neg x) which is the same as (mul x, -1).  The `operand` value
    /// must have `index` type.
    static TypedAttr getNeg(TypedAttr operand);

    /// Return (x-y) which is the same as (add x, (neg y)).  The `operand` value
    /// must have `index` type.
    static TypedAttr getSub(TypedAttr lhs, TypedAttr rhs);

    /// If the specified attribute is a rebind, return its operand, otherwise
    /// return the rebind itself.
    static TypedAttr stripRebind(TypedAttr src);

    /// A specific builder used to build the attribute from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static ParamOperatorAttr getFromBytecode(
        POC opcode, ArrayRef<TypedAttr> operands, Type type);
  }];
}

//===----------------------------------------------------------------------===//
// MLIROpAttr
//===----------------------------------------------------------------------===//

def KGEN_MLIROpAttr : KGEN_ExprAttr<"MLIROp", "param.mlir_op"> {
  let summary = "an MLIR operation as a parameter expression";
  let description = [{
    The `#kgen.param.mlir_op` attribute represents an MLIR operation as a
    parameter expression. Its type is FuncTypeGeneratorType.

    Example:

    ```
    #kgen.param.mlir_op<"index.add", {}>
      : !kgen.generator<(index, index) -> index>
    ```

    Operation attributes can be specified using a dictionary attribute.

    Example:

    ```
    #kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>
      : !kgen.generator<(index, index) -> i1>
    ```

    The operation can be parameterized on any of its attributes that are
    parametric -- that is, which are `TypedAttr`. These attributes are omitted
    from the attribute dictionary and are present in the signature.

    Example:

    ```
    #kgen.param.mlir_op<"pop.array.get", {}> : !kgen.generator<
      <*"index">(!pop.array<size, type>) -> !kgen.param<type>
    >
    ```

    Parametric operations can be bound using `bind_signature`.
  }];

  let parameters = (ins
    "StringAttr":$name,
    "DictionaryAttr":$attrs,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );

  let assemblyFormat = "`<` $name `,` $attrs `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name, "DictionaryAttr":$attrs,
        "FuncTypeGeneratorType":$type), [{
      return $_get(name.getContext(), name, attrs, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// DecoratorsAttr
//===----------------------------------------------------------------------===//

def KGEN_DecoratorsAttr : ArrayOfAttr<KGENDialect, "Decorators", "decorators",
    "TypedAttr"> {
  let summary = "Compile-time decorator invocations.";
  let description = [{
    The `#kgen.decorators` attribute represents a list of decorator invocations
    attached to an operation. Decorators are closures where the first argument
    of the function will be the operation the decorator is attached to. The
    expected signature of the closure is:

    ```mlir
    (!pdl.operation) capturing -> !pdl.operation
    ```

    Decorators in the list are invoked on the operation from first to last.
    Decorators may be invoked at different points in KGEN pass pipeline. Each
    decorator contains a tag indicating when it should be invoked. Successive
    decorators must have later invocation points than previous ones.
    (TODO: Not implemented yet)
  }];
}

//===----------------------------------------------------------------------===//
// PreservedAttr
//===----------------------------------------------------------------------===//

def KGEN_PreservedAttr : KGENAttr<"Preserved", "preserved"> {
  let summary = "Preserved attribute that is not affected by rewrites.";
  let description = [{
    The `#kgen.preserved` attribute contains an attribute and isolates it from
    all rewrites and lowerings. This is useful for keeping higher-level
    information, like source information, around in the IR in case users need to
    inspect them.

    Example:

    ```mlir
    #kgen.preserved<!lit.generator<() -> ()>>
    ```
  }];

  let parameters = (ins "Attribute":$value);
  let assemblyFormat = "`<` $value `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return $_get(value.getContext(), value);
   }]>
  ];

  let extraClassDeclaration = [{
    static auto getWalkImmediateSubElementsFn() {
      return [](auto, function_ref<void(Attribute)>,
                      function_ref<void(Type)>) {};
    }
    static auto getReplaceImmediateSubElementsFn() {
      return [](auto instance, ArrayRef<Attribute>, ArrayRef<Type>) {
        return instance;
      };
    }
  }];
}

//===----------------------------------------------------------------------===//
// ExportKindAttr
//===----------------------------------------------------------------------===//

def KGEN_ExportKindAttr : EnumAttr<KGENDialect, KGEN_ExportKind, "export"> {
  let summary = "Defines the export semantics of a symbol.";
  let description = [{
    The `#kgen.export` attribute defines the export semantics of a symbol. A
    symbol can be:

    - Not exported: its linkage is internal and its visibility is hidden.
    - Exported: its linkage is external and its visibility is public.
    - C exported: like `exported`, but with a C-compatible name and ABI.
    - Package exported: like `exported`, but implicitly exported as part of a
                        package.

    Example:

    ```mlir
    #kgen.export_kind<not_exported>
    #kgen.export_kind<exported>
    #kgen.export_kind<c_exported>
    #kgen.export_kind<package_exported>
    ```
  }];

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// LinkDependencyArrayAttr
//===----------------------------------------------------------------------===//

def KGEN_LinkDependencyArrayAttr : ArrayOfAttr<KGENDialect,
  "LinkDependencyArray", "link.dependencies", "FlatSymbolRefAttr"> {
  let summary = "List of references to link dependencies.";
  let description = [{
    The `#kgen.link.dependencies` attribute represents a list of link
    dependencies, which are flat symbol references.
  }];
}

//===----------------------------------------------------------------------===//
// TailKindAttr
//===----------------------------------------------------------------------===//

def KGEN_TailKindAttr : EnumAttr<KGENDialect, KGEN_TailKind, "tailkind"> {
  let summary = "Defines the tail call type.";
  let description = [{
    The `#kgen.tailkind` attribute defines the export semantics of a symbol. A
    symbol can be:

    - None: Unspecified.
    - Musttail: Compilation will fail if this call cannot be tail call
                optimized.
    - NoTail: Do not tail call optimize this call.

    Example:

    ```mlir
    #kgen.tailkind<none>
    #kgen.tailkind<musttail>
    #kgen.tailkind<notail>
    ```
  }];

  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// ClosureMethodAttr
//===----------------------------------------------------------------------===//

def KGEN_ClosureMethodAttr : EnumAttr<KGENDialect, KGEN_ClosureMethod, "closure_method"> {
  let summary = "Reference to a closure method.";
  let description = [{
    The `#kgen.closure_method` attribute represents the symbol of a closure method.

    Example:

    ```mlir
    #kgen.closure_method<call>
    ```
  }];

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ClosureSymbolAttr
//===----------------------------------------------------------------------===//

def KGEN_ClosureSymbolAttr : KGENAttr<"ClosureSymbol", "closure.symbol",
  [TypedAttrInterface]> {
  let summary = "Reference to a yet to be synthesized closure method";
  let description = [{
    We want to model function calls to functions that have not been generated
    yet. These functions are with respect to a capture struct and a nested
    function. This attribute must contain enough information to pair it with
    the generated methods which includes the symbol of the enclosing method.

    Example:
    ```mlir
    #kgen.closure.symbol<@foo,
                         "fn",
                         #kgen.closure_method<call>,
                         <:!kgen.param_capture<@foo "fn"> ?>
                        >
    ```
  }];

  let parameters = (ins
    "SymbolRefAttr": $parentSymbol,
    "StringAttr": $nestedFuncName,
    "ClosureMethodAttr": $method,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    AttributeSelfTypeParameter<"", "FuncTypeGeneratorType">:$type
  );
  let assemblyFormat = [{
    `<` custom<ClosureSymbolValue>($parentSymbol,
                               $nestedFuncName,
                               $method, $paramValues) `>`
  }];
}

//===----------------------------------------------------------------------===//
// ClosureAttr
//===----------------------------------------------------------------------===//

def KGEN_ClosureAttr : KGENAttr<"Closure", "closure", [
    TypedAttrInterface]> {
  let summary = "A placeholder for the parameter captures";
  let description = [{
    The `#kgen.closure` attribute represents an uncomputed
    parametric closure. A parametric closure is a set of parametric
    values that are captured from the enclosing function. This abstraction
    is useful in the case where transformations are applied between the
    the closure definition site and the closure lowering pass that may alter
    the set of parameters captured.

    Example:

    In the following example we define a closure that captures the parameter
    value `C`. We want to bind this closure to the "x" parameter of the
    function `consume`. We must also pass the parameters that the closure
    depends on, which in this case is just `C` but could include `D` if we
    apply a transformation between now and when we lift the closure. To
    postpone the calculation of the captures, we bind an abstract value
    to the capture struct parameter of `consume` called
    `#kgen.closure<@foo "fn">`. This placeholder is of type ClosureAttr and
    represents the parameter captures of the "fn" closure.

    ```mlir

    kgen.generator @foo<C,D>() {
     %0 = kgen.closure.init()() -> index {
	         %1 = kgen.param.constant = <mul(C, C)>
	         kgen.return %1 : index
      } : (), !kgen.pointer<!kgen.closure<@foo, "fn" registerpassable>>
      %2 = kgen.call @consume<:type #type_value,
       :!kgen.param<!kgen.param_closure<@foo "fn">> #kgen.closure<@foo "fn">
       >(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
      kgen.return
    }

    kgen.generator @consume<x: type,
                        CAPTURE_INST: !kgen.param<get_vtable_entry(x, "CAPTURE_TYPE")>
                        >(%arg0: !kgen.param<x>) -> index {
        // BODY OMITTED FOR BREVITY
    }
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "ParamClosureType">:$type);

  let assemblyFormat = [{
    `<` custom<ClosureAttr>(ref($type)) `>`
  }];
}

//===----------------------------------------------------------------------===//
// MemSymbolTripleAttr
//===----------------------------------------------------------------------===//

def KGEN_MemSymbolTripleAttr : KGENAttr<"MemSymbolTriple", "mem_symbol_triple"> {
  let summary = "A tuple of symbols";
  let description = [{
    The `#kgen.mem_symbol_triple` attribute holds the symbols of a memory value.
    The symbols it holds are copy, move, and del. The copy symbol is optional.
    This attribute is useful in the context of abstracted operations whose
    lowering depends on these core symbols.

    Example:

    ```mlir
      #kgen.mem_symbol_triple<@bar_move<:type index, :type index>>,
                              @bar_del<:type index, :type index>>
                               : !kgen.pointer<struct<(index, index)>>
    ```
  }];

  let parameters = (ins OptionalParameter<"SymbolConstantAttr">:$copy,
                        "SymbolConstantAttr":$move,
                        "SymbolConstantAttr":$del);

 let assemblyFormat = [{
    `<` custom<MemSymbolTripleAttr>($copy, $move, $del) `>`
  }];
}

//===----------------------------------------------------------------------===//
// DeferredAttr
//===----------------------------------------------------------------------===//

def KGEN_DeferredAttr : KGENAttr<"Deferred", "deferred", [
    TypedAttrInterface]> {
  let summary = "A wrapper for non-typed attribute";
  let description = [{
    The `#kgen.deferred` attribute holds a non-typed attribute to allow it
    to be created later.

    Example:

    ```mlir
    #kgen.deferred #index<cmp_predicate sle>> : !kgen.deferred
    ```
  }];

  let parameters = (ins "Attribute":$attr);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
  let assemblyFormat = "$attr";

  let extraClassDeclaration = [{ Type getType() const; }];
  let genVerifyDecl = 1;  // Generate getChecked/verify.
}

#endif // KGEN_KGENDIALECT_KGENATTRS_TD
