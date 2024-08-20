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
// ParamRefType
//===----------------------------------------------------------------------===//

def KGEN_ParamRefType : KGENType<"ParamRef", "paramref"> {
  let summary = "Type specified by a parameter expression (type domain)";
  let description = [{
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted with a
    type constant, this ParamRefType is folded away and the MlirType inside
    the type constant is returned.
  }];

  let parameters = (ins "TypedAttr":$param);
  let assemblyFormat = "`<` custom<TypeParamValue>($param) `>`";

  // Our builders create ParamRefType but fold to the base type if not
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
    static ParamRefType getFromBytecode(TypedAttr param);
  }];
}

//===----------------------------------------------------------------------===//
// TypeValueType
//===----------------------------------------------------------------------===//

def KGEN_TypeValueType : KGENType<"TypeValue", "typevalue"> {
  let summary = "Type specified by a parameter expression (value domain)";
  let description = [{
    This is the value domain counterpart to KGEN ParamRefType. It allows
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
// IntLiteralType
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralType : KGENType<"IntLiteral", "int_literal",[
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Arbitrary-precision integer type for literals.";
  let description = [{
    The `!kgen.int_literal` type represents an infinite precision
    integer value.  It can't be materialized, but allows for arbitrary
    integer size for compile time computations.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// FloatLiteralType
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralType : KGENType<"FloatLiteral", "float_literal", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>]> {
  let summary = "Arbitrary-precision float type for literals.";
  let description = [{
    The `!kgen.float_literal` type is for use in arbitrary precision
    compile time computations.  They are implemented as infinite precision
    rational number values.  It can't be materialized, but allows for
    arbitrary precision for compile time computations.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

def KGEN_SignatureType : KGENType<"Signature", "signature", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<MemoryableTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    ParameterScopeTypeInterface]> {
  let summary = "The signature for a parameterized declaration";
  let description = [{
    This type describes the signature for a parameterized declaration, which
    can have input/output parameters as well as input/output value arguments
    and results.

    The metadata attribute is used to store additional information about a
    function, such as its argument calling conventions and the effects of the
    function itself. When the metadata only contains default values, it isn't
    printed in the textual MLIR format.
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$inputParamTypes,
    ArrayRefParameter<"Type">:$resultParamTypes,
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
        CArg<"ArrayRef<Type>", "{}">:$inputParamTypes,
        CArg<"ArrayRef<Type>", "{}">:$resultParamTypes,
        CArg<"ArrayRef<ArgConvention>", "{}">:$argConvs,
        CArg<"FnEffects", "{}">:$effects,
        CArg<"Attribute", "{}">:$metadata), [{
      SmallVector<ArgConvention> convs = llvm::to_vector(argConvs);
      convs.resize(values.getNumInputs());
      return $_get(values.getContext(), inputParamTypes, resultParamTypes,
                   values, convs, effects,
                   ::cast_or_null<FnMetadataAttrInterface>(metadata));
    }]>,
    TypeBuilderWithInferredContext<(ins
        "FunctionType":$values,
        "ArrayRef<ArgConvention>":$argConvs,
        CArg<"FnEffects", "{}">:$effects,
        CArg<"Attribute", "{}">:$metadata), [{
      SmallVector<Type> inputParamTypes, resultParamTypes;
      return $_get(values.getContext(), inputParamTypes, resultParamTypes,
                   values, argConvs, effects,
                   ::cast_or_null<FnMetadataAttrInterface>(metadata));
    }]>,
    TypeBuilder<(ins "TypeRange":$inputs, "TypeRange":$results)>
  ];

  let extraClassDeclaration = [{
    /// Helper to return the argument and result types.
    ArrayRef<Type> getArguments() const;
    ArrayRef<Type> getResults() const;

    /// Return true if this signature's last argument is a result from the
    /// the function returned through memory.
    bool hasMemoryOnlyResult();
    /// Return true if this signature's first argument is an initializer result
    /// returned through memory.
    bool hasInitSelfArg();

    bool isThrows() const { return getFnEffects().isThrows(); }
    bool isAsync() const { return getFnEffects().isAsync(); }
    bool isCapturing() const { return getFnEffects().isCapturing(); }
    bool isEscaping() const { return getFnEffects().isEscaping(); }
    bool isRefResult() const { return getFnEffects().isRefResult(); }

    /// Return the convention for the specified value argument.
    ArgConvention getArgConvention(size_t inputNo) {
      return getArgConventions()[inputNo];
    }

    /// Returns true if the signature is a concrete callable.
    bool isConcrete();

    /// Set a function convention.
    SignatureType getWithFnEffects(FnEffects effect);
    /// Return this signature type with the value signature replaced.
    SignatureType getWithValuesReplaced(FunctionType fnType);

    /// Determine whether an argument with the given input convention expects to
    /// have a pointer or reference type.
    static bool hasAddress(ArgConvention conv);

    /// Determine whether an argument with the given input convention expects to
    /// have an implicit lifetime.
    static bool hasImplicitLifetime(ArgConvention conv) {
      return conv != ArgConvention::Ref && hasAddress(conv);
    }

    /// Return true if this is an automatically generated result slot.
    static bool isResultSlot(ArgConvention conv);

    /// Return the number of input parameters.
    size_t getNumInputParams() { return getInputParamTypes().size(); }
    /// Return the number of result parameters.
    size_t getNumResultParams() { return getResultParamTypes().size(); }
    /// Return the number of arguments.
    size_t getNumArguments() { return getArguments().size(); }
    /// Return the number of results.
    size_t getNumResults() { return getResults().size(); }

    /// Calls to async functions don't provide values for the `byref_error` and
    /// `byref_result` arguments. If this is an async function, return the
    /// number of result slots it has. Otherwise, return 0.
    size_t getNumAsyncReturnSlots();

    /// Return a signature with the specified parameter bindings substituted
    /// into it as happens in a call.  The types specified in the parameter
    /// bindings affects the type signature of the argument and results, and
    /// also can remap the signature in the parameter list itself.
    ///
    /// If an error occurs making the substitution, report it with emitErrorFn
    /// and return null. If `emitErrorFn` is not specified, the callee will
    /// assume that the substitution cannot fail. If a location is specified,
    /// a default diagnostic is created with it and used as emitErrorFn.
    SignatureType
    getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
      function_ref<InFlightDiagnostic()> emitErrorFn = {});
    SignatureType
    getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
      Location location);
    static SignatureType
    getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
      function_ref<InFlightDiagnostic()> emitErrorFn,
      ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
      FunctionType values, ArrayRef<ArgConvention> argConventions,
      FnEffects effects, FnMetadataAttrInterface metadata);

    /// Construct a signature from named parameter declarations, a function
    /// type, and metadata. This helper is used to convert between a named
    /// signature structure to a nameless `SignatureType` representation.
    static SignatureType
    remapToSignature(ArrayRef<ParamDeclAttr> inputParams,
                     ArrayRef<ParamDeclAttr> resultParams,
                     FunctionType functionType,
                     ArrayRef<ArgConvention> argConventions = {},
                     FnEffects effects = {}, Attribute metadata = {},
                     function_ref<InFlightDiagnostic()> emitError = {});
  }];

  let genVerifyDecl = 1;
}

def CalleeArgumentTypes : TypesMatchWith<"operands match callee argument types",
    "callee", "operands", "cast<SignatureType>($_self).getArguments()">;
def CalleeResultTypes : TypesMatchWith<"results match callee result types",
    "callee", "results", "cast<SignatureType>($_self).getResults()">;

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

    The `exclusive` parameter can be set to indicate that the underlying memory
    allocation of the pointer is known only to be accessed through this pointer
    and pointers derived from this pointer. Neither KGEN nor LLVM are guaranteed
    to do anything useful with this information, however.

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
                           "$_builder.getIndexAttr(0)">:$addressSpace,
    DefaultValuedParameter<"TypedAttr",
                           "$_builder.getBoolAttr(false)">:$exclusive);
  let assemblyFormat = [{
    `<` custom<ParamType>($elementType) ``
    (`,` custom<IndexParamValue>($addressSpace)^)?
    (` ` `exclusive` `(` custom<I1ParamValue>($exclusive)^ `)`)? `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace,
                                        CArg<"bool", "false">:$exclusive)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace,
                                        CArg<"bool", "false">:$exclusive)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace,
                                        "TypedAttr":$exclusive)>,
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
    so clients know if the input argument is supposed to be owned, borrowed,
    inout, etc.

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
        CArg<"ArgConvention", "ArgConvention::BorrowedInReg">:$convention), [{
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

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
