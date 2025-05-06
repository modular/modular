//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_TD
#define KGEN_KGENDIALECT_KGENTYPES_TD

include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/Interpreter/InterpreterInterface.td"
include "Support/MDialect/MTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class KGENType<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<KGENDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ParamType
//===----------------------------------------------------------------------===//

def KGEN_ParamType : KGENType<"Param", "param"> {
  let summary = "Type specified by a parameter expression (type domain)";
  let description = [{
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted with a
    type constant, this ParamType is folded away and the MlirType inside
    the type constant is returned.
  }];

  let parameters = (ins "TypedAttr":$param);
  let assemblyFormat = "`<` custom<TypeParamValue>($param) `>`";

  // Our builders create ParamType but fold to the base type if not
  // parameterized.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$param), [{}], "Type">,
     // This is needed for ODS generated uses.
    TypeBuilderWithInferredContext<(ins "MLIRContext *":$context,
                                        "TypedAttr":$param),
                                   [{ return get(param); }], "Type">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static ParamType getFromBytecode(TypedAttr param);
  }];
}

//===----------------------------------------------------------------------===//
// TypeValueType
//===----------------------------------------------------------------------===//

def KGEN_TypeValueType : KGENType<"TypeValue", "typevalue"> {
  let summary = "Type specified by a parameter expression (value domain)";
  let description = [{
    This is the value domain counterpart to KGEN ParamType. It allows
    losslessly using a parameter expression as a MLIR Type: Even when its
    type value parameter is substituted with a type constant, it does not fold
    into the MlirType itself (unless the type constant is a trivial mlir type).
  }];

  let parameters = (ins "TypedAttr":$typeValue);
  let assemblyFormat = "`<` custom<TypeParamValue>($typeValue) `>`";

  // Our builders create TypeValueType but fold to the base type if not
  // parameterized.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$typeValue), [{}], "Type">,
     // This is needed for ODS generated uses.
    TypeBuilderWithInferredContext<(ins "MLIRContext *":$context,
                                        "TypedAttr":$typeValue),
                                   [{ return get(typeValue); }], "Type">,
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static TypeValueType getFromBytecode(TypedAttr typeValue);
  }];
}

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

def KGEN_TypeType : KGENType<"Type", "type", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Arbitrary type";
  let description = [{
    This kgen type represents an arbitrary type, used for parameterizing
    type-generic functions and structs.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def KGEN_DTypeType : KGENType<"DType", "dtype", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "8-bit integer DType type";
  let description = [{
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def KGEN_StringType : KGENType<"String", "string", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "String type for parameters";
  let description = [{
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// GeneratorType
//===----------------------------------------------------------------------===//

def KGEN_GeneratorType : KGENType<"Generator", "generator", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    ParameterScopeTypeInterface]> {
  let summary = "The type of a generator (i.e. a parameterized value)";
  let description = [{
    This type describes a generator (i.e. a parameterized value). It describes
    the generator's parameter signature (the types of input parameters), and the
    generator's output type (potentially in terms of the input parameters).
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$inputParamTypes,
    "Type":$body,
    "GeneratorMetadataAttrInterface":$metadata
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "ArrayRef<Type>":$inputParamTypes, "Type":$body,
        CArg<"Attribute", "{}">:$metadata), [{
      return $_get(body.getContext(), inputParamTypes, body,
                   ::cast_or_null<GeneratorMetadataAttrInterface>(metadata));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get this GeneratorType with a new body.
    GeneratorType getWithBody(Type newBody);

    /// Get this GeneratorType with some parameters bound.
    GeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      function_ref<InFlightDiagnostic()> emitErrorFn = {},
      ParameterEvaluationContext *evaluationContext = nullptr);
    GeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings, Location location,
      ParameterEvaluationContext *evaluationContext = nullptr);

    /// Returns true if the signature has no input params.
    bool isFullyBound() { return getInputParamTypes().empty(); };

    /// Returns the body while asserting that the generator is fully bound.
    Type getInstantiatedBody() {
      assert(isFullyBound() && "cannot instantiate parameterized body");
      return getBody();
    }
  }];
}

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//

def KGEN_FuncTypeGeneratorType : DialectType<KGENDialect,
    CPred<"llvm::isa<FuncTypeGeneratorType>($_self)">,
    "KGEN GeneratorType wrapping a FuncType", "FuncTypeGeneratorType">;

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def KGEN_FuncType : KGENType<"Func", "func", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "The type of a KGEN function";
  let description = [{
    This type describes the type of a function KGEN, which can have input/output
    value arguments and results.

    The metadata attribute is used to store additional information about a
    function, such as its argument calling conventions and the effects of the
    function itself. When the metadata only contains default values, it isn't
    printed in the textual MLIR format.
  }];

  let parameters = (ins
    "FunctionType":$values,
    ArrayRefParameter<"ArgConvention">:$argConventions,
    "FnEffects":$fnEffects,
    "FnMetadataAttrInterface":$metadata
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// This builder allows everything other than the function type to be null.
    TypeBuilderWithInferredContext<(ins
        "FunctionType":$values,
        CArg<"ArrayRef<ArgConvention>", "{}">:$argConvs,
        CArg<"FnEffects", "{}">:$effects,
        CArg<"Attribute", "{}">:$metadata), [{
      SmallVector<ArgConvention> convs = llvm::to_vector(argConvs);
      convs.resize(values.getNumInputs());
      return $_get(values.getContext(), values, convs, effects,
                   ::cast_or_null<FnMetadataAttrInterface>(metadata));
    }]>,
    TypeBuilder<(ins "TypeRange":$inputs, "TypeRange":$results)>
  ];

  let extraClassDeclaration = [{
    /// Helper to return the argument and result types.
    ArrayRef<Type> getArguments() const;
    ArrayRef<Type> getResults() const;

    /// Return true if this signature's last argument is a result from the
    /// function returned through memory.
    bool hasMemoryOnlyResult();

    bool isThrows() const { return getFnEffects().isThrows(); }
    bool isAsync() const { return getFnEffects().isAsync(); }
    bool isCapturing() const { return getFnEffects().isCapturing(); }
    bool isEscaping() const { return getFnEffects().isEscaping(); }
    bool isRefResult() const { return getFnEffects().isRefResult(); }

    /// Return the convention for the specified value argument.
    ArgConvention getArgConvention(size_t inputNo) {
      return getArgConventions()[inputNo];
    }

    /// Set a function convention.
    FuncType getWithFnEffects(FnEffects effect);
    /// Return this signature type with the value signature replaced.
    FuncType getWithValuesReplaced(FunctionType fnType);
    /// Return this signature with a new metadata attribute.
    FuncType getWithMetadata(FnMetadataAttrInterface metadata);

    /// Return the number of arguments.
    size_t getNumArguments() { return getArguments().size(); }
    /// Return the number of results.
    size_t getNumResults() { return getResults().size(); }

    /// Calls to async functions don't provide values for the `byref_error` and
    /// `byref_result` arguments. If this is an async function, return the
    /// number of result slots it has. Otherwise, return 0.
    size_t getNumAsyncReturnSlots();
  }];

  let genVerifyDecl = 1;
}

def CalleeArgumentTypes : TypesMatchWith<"operands match callee argument types",
    "callee", "operands",
    "cast<FuncTypeGeneratorType>($_self).getBody().getArguments()">;
def CalleeResultTypes : TypesMatchWith<"results match callee result types",
    "callee", "results",
    "cast<FuncTypeGeneratorType>($_self).getBody().getResults()">;

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def PointerType : KGENType<"Pointer", "pointer", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized pointer type";
  let description = [{
    This type represents a pointer. The pointee type is parameterized with a
    `!kgen.type` type. An optional `addressSpace` argument can be
    specified (default to 0).

    Example:

    ```mlir
    // A pointer to a scalar.
    !kgen.pointer<scalar<f32>>

    // A pointer to a SIMD vector.
    !kgen.pointer<simd<4, f32>>

    // A parameterized scalar pointer.
    !kgen.pointer<scalar<type>>

    // A completely parameterized pointer.
    !kgen.pointer<elementType>

    // The address space parameter is optional, but can be specified.
    !kgen.pointer<scalar<si32>, 2>

    // The address space also works on parametrized pointers.
    !kgen.pointer<elementType, 5>
    ```
  }];

  let parameters = (ins
    "Type":$elementType,
    DefaultValuedParameter<"TypedAttr",
                           "$_builder.getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<ParamType>($elementType) ``
    (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return llvm::cast<ElTy>(getElementType()); }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class PointerElementType<string lhs, string rhs> : TypesMatchWith<
    "pointer element type", lhs, rhs,
    "llvm::cast<PointerType>($_self).getElementType()">;

class PointerOf<Type type> : ConfinedType<PointerType,
    [CPred<"llvm::cast<PointerType>($_self).getElementType()">,
     SubstLeaves<"$_self",
      "llvm::cast<PointerType>($_self).getElementType()",
      type.predicate>],
    "pointer to " # type.summary> {
  let cppType = "::M::KGEN::PointerType";
  let builderCall = !if(!empty(type.builderCall), "",
                       "PointerType::get(" # type.builderCall #")");
}

//===----------------------------------------------------------------------===//
// VariadicType
//===----------------------------------------------------------------------===//

def KGEN_VariadicType : KGENType<"Variadic", "variadic", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized variadic type";
  let description = [{
    The `!kgen.variadic` type represents a homogeneously typed variadic sequence
    of zero or more elements.  It also includes the original argument convention
    so clients know if the input argument is supposed to be owned, read,
    mut, etc.

    Example:

    ```mlir
    // A variadic sequence of scalar floats.
    !kgen.variadic<scalar<f32>>

    // A parameterized variadic sequence.
    !kgen.variadic<type, owned_in_mem>
    ```
  }];

  let parameters = (ins "Type":$elementType,
                        "ArgConvention":$convention);
  let assemblyFormat = [{
    `<`custom<ParamType>($elementType)
      `` custom<VariadicConvention>($convention) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "Type":$elementType,
        CArg<"ArgConvention", "ArgConvention::ReadReg">:$convention), [{
      return $_get(elementType.getContext(), elementType, convention);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Get a variadic type with only the element type changed.
    VariadicType getWithElementType(Type type);
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class VariadicElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches variadic element type", lhs, rhs,
    "llvm::cast<KGEN::VariadicType>($_self).getElementType()",
    "typeRangeMatches">;

//===----------------------------------------------------------------------===//
// NoneType
//===----------------------------------------------------------------------===//

def KGEN_NoneType : KGENType<"None", "none",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>,
     DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Type used for 'void' function results";
  let description = [{
    The `!kgen.none` type represents a value with no contents. When used as a
    function result, it is equivalent to a void result or no results.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// TargetType
//===----------------------------------------------------------------------===//

def KGEN_TargetType : KGENType<"Target", "target"> {
  let summary = "The type of a target.";
  let description = [{
    A `!kgen.target` is the type of a target. It is used for parameterizing
    kernels by target.

    Example:
    ```mlir
    kgen.generator @target_params<t: !kgen.target>() {
      ...
    }
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// BuildInfoType
//===----------------------------------------------------------------------===//

def KGEN_BuildInfoType : KGENType<"BuildInfo", "build_info"> {
  let summary = "The type of a build info.";
  let description = [{
    A `!kgen.build_info` is the type of a build info. It is used for
    parameterizing kernels by how they are built.

    Example:
    ```mlir
    kgen.generator @target_params<bi: !kgen.build_info>() {
      ...
    }
    ```
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

def KGEN_StructType : KGENType<"Struct", "struct", [
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "parameterized struct type";
  let description = [{
    This type represents a struct. A struct contains element types arranged in
    their order of declaration and a flag indicating register-passability.

    This is the "anonymous" version of `!kgen.concrete_source_struct` that
    strips away all information except for those necessary for understanding the
    memory layout of the data it describes.

    Example:

    ```mlir
    // A struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>)>

    // A memory-only struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>) memoryOnly>

    // A struct with nested types.
    !kgen.struct<(
      !kgen.pointer<simd<4, si8>>,
      !kgen.array<24, scalar<si64>>,
      !kgen.struct<(
        !kgen.scalar<f32>,
        !kgen.scalar<f64>
      )>
    )>

    // A struct with parameterized element types.
    !kgen.struct<(type, array<size, scalar<dtype>>)>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"Type">:$elementTypes,
    "bool":$isMemoryOnly
  );
  let assemblyFormat = [{
    `<` `(` (`)`) : (custom<ParamTypes>($elementTypes)^ `)`)?
    `` custom<IsMemoryOnly>($isMemoryOnly) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types)>,
    TypeBuilder<(ins "ArrayRef<Type>":$types), [{
      return $_get($_ctxt, types, /*isMemoryOnly=*/false);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return true if the specified type is a NoneType or an empty struct.
    static bool isNoneOrEmpty(Type type);
    /// Get the number of struct elements.
    unsigned getNumElements() const { return getElementTypes().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// StructInstanceType
//===----------------------------------------------------------------------===//

def KGEN_StructInstanceType : KGENType<"StructInstance", "struct_inst"> {
  let summary = "concretized mojo struct type";
  let description = [{
    This type represents a concretized mojo source struct type. It is the result
    of flattening an AppliedStructType.

    Example:

    ```mlir
    // A trivially-concretized struct with no parameters and no fields.
    !kgen.struct_inst<"Foo">

    // A parameterized memory-only struct with primitive element types.
    !kgen.struct_inst<
      "Bar"
      [elemT]
      <:dtype f32>
      (first: !kgen.scalar<f32>, second: !kgen.scalar<f32>)
      memoryOnly
    >
    ```
  }];

  let parameters = (ins
    "StringAttr":$name,
    OptionalArrayRefParameter<"StringAttr">:$paramNames,
    OptionalArrayRefParameter<"TypedAttr">:$paramValues,
    OptionalArrayRefParameter<"StructDefFieldAttr">:$fields,
    "bool":$isMemoryOnly
  );

  let assemblyFormat = [{
    `<` $name
        (`[` custom<ParamNames>($paramNames)^ `]`)?
        `` custom<ParameterValues>($paramValues)
        (`(` custom<StructDefFields>($fields)^ `)`)?
        `` custom<IsMemoryOnly>($isMemoryOnly)
    `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name,
        "ArrayRef<StringAttr>":$paramNames,
        "ArrayRef<TypedAttr>":$paramValues,
        "ArrayRef<StructDefFieldAttr>":$fields,
        "bool":$isMemoryOnly)>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PackType
//===----------------------------------------------------------------------===//

def KGEN_PackType : KGENType<"Pack", "pack", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "parameterized pack type";
  let description = [{
    A `!kgen.pack` type represents a sequence of heterogeneously typed elements.
    This type can be used to represent a tuple of 0 or more elements.

    Example:

    ```mlir
    // A concrete pack type with no element types.
    !kgen.pack<[]>

    // A concrete pack type with two element types.
    !kgen.pack<[i32, i64]>

    kgen.generator @pack<Ts: variadic<!kgen.type>, T0: type, T1: type>(
      // A pack type parameterized on a variadic sequence of elements.
      %0: !kgen.pack<Ts>,
      // A pack type parameterized on two element types.
      %1: !kgen.pack<[T0, T1]>,
    ) { kgen.return }
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic);
  let assemblyFormat = "`<` custom<PackType>($variadic) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$variadic), [{
      return get(variadic.getContext(), variadic);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether this is a concrete pack type with no elements.
    bool isEmpty();

    /// If the pack type is concrete, returns the attribute representing the
    /// variadic sequence that the pack type is based upon. Otherwise, returns
    /// a null attribute.
    VariadicAttr getVariadicIfResolved() const;
  }];
}

//===----------------------------------------------------------------------===//
// VariantType
//===----------------------------------------------------------------------===//

def KGEN_VariantType : KGENType<"Variant", "variant", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>
]> {
  let summary = "A parametric variant type.";
  let description = [{
    A `!kgen.variant` type is a structural variant type that represents a value
    that is one of a list of types (discriminated union).

    Variants of zero types or one type are allowed, e.g. `!kgen.variant<i32>`,
    and all variant operations are defined on the even though they aren't
    particularly useful types.

    Example:

    ```mlir
    !kgen.variant<i32, scalar<dtype>, T>
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic);
  let assemblyFormat = "`<` custom<VariantTypes>($variadic) `>`";
  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilder<(ins "TypedAttr":$variadic)>,
    TypeBuilderWithInferredContext<(ins "ArrayRef<Type>":$types)>
  ];

  let extraClassDeclaration = [{
    /// A specific builder used to build the type from a bytecode context,
    /// providing a slightly different context from the normal builder.
    static VariantType getFromBytecode(TypedAttr variadic);

    /// Get an element type as a parameterized type.
    Type getType(unsigned index) const;

    /// Return the number of types in the variant.
    size_t getNumTypes() const;

    /// Get the number of bits allocated for the discriminator.
    size_t getDiscrSizeInBits() const;
    /// Get the size of the content section of the variant.
    std::optional<int64_t> getContentSize(TargetInfoAttr target) const;

    /// Get the concrete variant types.
    llvm::iterator_range<llvm::mapped_iterator<ArrayRef<TypedAttr>::iterator,
                                               Type (*)(TypedAttr)>>
    getTypes() const;
  }];
}

//===----------------------------------------------------------------------===//
// ClosureType
//===----------------------------------------------------------------------===//

def KGEN_ClosureType : KGENType<"Closure", "closure"> {
  let summary = "Represents a struct of captures";
  let description = [{
     A `!kgen.closure` type represents a struct of captures.
     Example:
     ```mlir
     kgen.generator @parent(%x: index) {
        %0 = kgen.closure.init(%x)(){
           kgen.call @foo(%x)
        } : (index), !kgen.closure<@parent, "closure", escaping>
        %1 = kgen.closure.init(%x)(){
           kgen.call @foo(%x)
        } : (index), !kgen.closure<@parent, "closure", nonescaping>
        %2 = kgen.closure.init(%x)(){
           kgen.call @foo(%x)
        } : (index), !kgen.closure<@parent, "closure", registerpassable>
     }
     }
     ```
  }];
  let parameters = (ins "SymbolRefAttr":$parentSymbol,
                        "StringAttr":$name,
                        "ClosureMemoryKind": $closureMemoryKind);
  let assemblyFormat = "`<` custom<ClosureTypes>($parentSymbol, $name, $closureMemoryKind) `>`";
}

//===----------------------------------------------------------------------===//
// ParamClosureType
//===----------------------------------------------------------------------===//

def KGEN_ParamClosureType : KGENType<"ParamClosure", "param_closure"> {
  let summary = "Represents a struct of parameter captures";
  let description = [{
     A `!kgen.param_closure` type represents a struct of  parameter
     captures. It is used as a placeholder before the captured parameters
     are calculated. Once calculated, the capture type is replaced with
     a !kgen.struct type with the capture types as the struct field types.

     Example:
     ```mlir
         #type_value = #kgen.type<!kgen.closure<@foo, "fn" registerpassable>,
          	{"__call__" :
          	  <!kgen.param_closure<@foo “fn”>>
              (!kgen.closure<@foo, "fn" registerpassable>) -> index =
              @foo_fn<:!kgen.param<!kgen.param_closure<@foo “fn”>> ?>
            }> : !kgen.type

        kgen.generator @foo<C>(%arg0 : index) {
          %1 = kgen.closure.init()(%arg1: index) -> index {
          %0 = kgen.param.constant = <add(C, C)>
          kgen.return %0 : index
          } : (), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
          %2 = kgen.call @consume<
            :type #type_value,
            :type !kgen.param_closure<@foo “fn”>,
            :!kgen.param<!kgen.param_closure<@foo “fn”>> #kgen.capture<@foo, “fn”>>
            (%3) :
            (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
          kgen.return
        }

        kgen.generator @consume<x: type,
                        CAPTURE_TYPE: type,
                        CAPTURE_INST: !kgen.param<CAPTURE_TYPE>
                        >(%arg0: !kgen.param<x>) -> index {
          %0 = kgen.call_param[(!kgen.param<x>) -> index:
                       bind_params(:<!kgen.param<CAPTURE_TYPE>>
                      (!kgen.none, index) -> index
                      get_vtable_entry(x, "__call__"), CAPTURE_INST)](%arg0, %arg1)
          kgen.return %0 : index
        }
     ```
  }];
  let parameters = (ins "SymbolRefAttr":$parentSymbol,
                        "StringAttr":$name
                   );
  let assemblyFormat = "`<` $parentSymbol $name `>`";
}

//===----------------------------------------------------------------------===//
// DeferredType
//===----------------------------------------------------------------------===//

def KGEN_DeferredType : KGENType<"Deferred", "deferred",
                          [ DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
  let summary = "Represents a deferred type";
  let description = [{
    A `!kgen.deferred` type represents a deferred attribute that will be
    concretized at elaborator

    Example:

    ```mlir
    #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred
    #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred
    ```
  }];
  let assemblyFormat = "";
}

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
