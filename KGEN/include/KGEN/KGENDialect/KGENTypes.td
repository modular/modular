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
  let summary = "Type specified by a parameter expression";
  let description = [{
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted, the
    type disappears.
  }];

  let parameters = (ins "mlir::TypedAttr":$param);
  let assemblyFormat = "`<` custom<TypeParamValue>($param) `>`";

  // Our builders create ParamRefType but fold to the base type if not
  // parameterized.
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::TypedAttr":$param),
                                   [{}], "Type">,
     // This is needed for ODS generated uses.
    TypeBuilderWithInferredContext<(ins "mlir::MLIRContext *":$context,
                                        "mlir::TypedAttr":$param),
                                   [{ return get(param); }], "Type">,
  ];
}

//===----------------------------------------------------------------------===//
// MLIRTypeType
//===----------------------------------------------------------------------===//

def KGEN_MLIRTypeType : KGENType<"MLIRType", "mlirtype",
    [DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "Arbitrary MLIR type";
  let description = [{
    This kgen type represents an arbitrary MLIR type, used for parameterizing
    type-generic operations.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def KGEN_DTypeType : KGENType<"DType", "dtype",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
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

def KGEN_StringType : KGENType<"String", "string",
    [DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
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

def KGEN_IntLiteralType : KGENType<"IntLiteral", "int_literal"> {
  let summary = "Arbitrary-precision integer type for literals.";
  let description = [{
    The `!kgen.int_literal` type represents an infinite precision
    integer value.  It can't be materialized, but allows for arbitrary
    integer size for compile time computations.
  }];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

def KGEN_SignatureType : KGENType<"Signature", "signature", [
    DeclareTypeInterfaceMethods<ParameterTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutInterface>]> {
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

  let parameters = (ins "TypeArrayAttr":$inputParamTypes,
                        "TypeArrayAttr":$resultParamTypes,
                        "FunctionType":$values,
                        "FnMetadataAttr":$metadata);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// This builder allows everything other than the function type to be null.
    TypeBuilderWithInferredContext<(ins
      "TypeArrayAttr":$inputParamTypes, "TypeArrayAttr":$resultParamTypes,
      "FunctionType":$values, "FnMetadataAttr":$metadata)>,
    TypeBuilder<(ins "TypeRange":$inputs, "TypeRange":$results)>,
    TypeBuilderWithInferredContext<(ins "FunctionType":$values)>
  ];

  let extraClassDeclaration = [{
    /// Helper to return the value input and result types.
    ArrayRef<Type> getValueInputs() const;
    ArrayRef<Type> getValueResults() const;

    /// Get the callable function effects.
    FnEffects getFnEffects() const { return getMetadata().getFnEffects(); }
    /// Get the function input argument conventions.
    ArrayRef<ValueInputConvention> getValueInputConventions() {
      return getMetadata().getInputConventions();
    }
    /// Get the function input argument names.
    StringArrayAttr getArgNames() { return getMetadata().getArgNames(); }
    /// Returns true if the argument at this index is a vararg.
    bool isVarArg(size_t index);
    /// Returns true if the argument at this index is a pack vararg.
    bool isPackVarArg(size_t index);
    /// Returns true if the argument at this index is a keyword vararg.
    bool isKWVarArg(size_t index);

    /// Return true if this signature's first argument is a result from the
    /// the function returned through memory.
    bool hasMemoryOnlyResult();
    /// Return true if this signature's first argument is an initializer result
    /// returned through memory.
    bool hasInitSelfResult();

    /// Get the function default arguments.
    ArrayRef<TypedAttr> getDefaultArguments() {
      return getMetadata().getDefaultArguments();
    }

    bool isThrows() const { return getFnEffects().isThrows(); }
    bool isAsync() const { return getFnEffects().isAsync(); }
    bool hasParamVarArgs() const { return getFnEffects().hasParamVarArgs(); }
    bool hasPackVarArgs() const { return getFnEffects().hasPackVarArgs(); }
    bool isCapturing() const { return getFnEffects().isCapturing(); }
    bool isEscaping() const { return getFnEffects().isEscaping(); }
    bool hasOwnedRegisterResult() const {
      return getFnEffects().hasOwnedRegisterResult();
    }

    /// Return the convention for the specified value input argument.
    ValueInputConvention getInputConvention(size_t inputNo) {
      return getValueInputConventions()[inputNo];
    }

    /// Return the name for the specified value input argument.
    StringAttr getArgName(size_t inputNo) { return getArgNames()[inputNo]; }

    /// Set a function convention.
    SignatureType getWithFnEffects(FnEffects effect);

    /// Return a signature with the specified parameter bindings substituted
    /// into it as happens in a call.  The types specified in the parameter
    /// bindings affects the type signature of the value input and outputs, and
    /// also can remap the signature in the parameter list itself.
    ///
    /// If an error occurs making the substitution, report it with emitErrorFn
    /// and return null.
    SignatureType
    getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
      function_ref<InFlightDiagnostic()> emitErrorFn);
    static SignatureType
    getSpecializedSignature(ArrayRef<TypedAttr> inputParamValues,
      function_ref<InFlightDiagnostic()> emitErrorFn,
      ArrayRef<Type> inputParamTypes, ArrayRef<Type> resultParamTypes,
      FunctionType values, FnMetadataAttr metadata);

    /// Return this signature type with the value signature replaced.
    SignatureType getWithValuesReplaced(FunctionType fnType);

    /// Return this signature with the input parameters dropped.
    SignatureType dropParamValues();

    /// Returns true if the signature is a concrete callable.
    bool isConcrete();

    /// Return the number of input parameters.
    size_t getNumInputParams() { return getInputParamTypes().size(); }
    /// Return the number of result parameters.
    size_t getNumResultParams() { return getResultParamTypes().size(); }
    /// Return the number of arguments.
    size_t getNumInputs() { return getValueInputs().size(); }
    /// Return the number of results.
    size_t getNumResults() { return getValueResults().size(); }
  }];

  let genVerifyDecl = 1;
}

def CalleeArgumentTypes : TypesMatchWith<"operands match callee argument types",
    "callee", "operands", "cast<SignatureType>($_self).getValueInputs()">;
def CalleeResultTypes : TypesMatchWith<"results match callee result types",
    "callee", "results", "cast<SignatureType>($_self).getValueResults()">;

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
    `!kgen.mlirtype` type. An optional `addressSpace` argument can be specified
    (default to 0).

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

  let parameters = (ins "TypedAttr":$elementType,
                        DefaultValuedParameter<"TypedAttr",
                          "Builder($_ctxt).getIndexAttr(0)">:$addressSpace);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($elementType) ``
    (`,` custom<IndexParamValue>($addressSpace)^)? `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        CArg<"unsigned", "0">:$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType,
                                        "TypedAttr":$addressSpace)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$addressSpace)>,
  ];

  let extraClassDeclaration = [{
    /// Return the elementType attribute as a type.
    Type getElementAsType() const;

    /// Get the element type as the provided type.
    template <typename ElTy>
    ElTy getElementAs() { return getElementAsType().cast<ElTy>(); }
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class PointerElementType<string lhs, string rhs> : TypesMatchWith<
    "pointer element type", lhs, rhs,
    "ParamRefType::get($_self.cast<PointerType>().getElementType())">;

class PointerOf<Type type> : ConfinedType<PointerType,
    [CPred<"$_self.cast<PointerType>().getElementType()">,
     SubstLeaves<"$_self",
      "$_self.cast<PointerType>().getElementAsType()",
      type.predicate>],
    "pointer to " # type.summary> {
  let cppClassName = "::M::KGEN::PointerType";
  let builderCall = !if(!empty(type.builderCall), "",
                       "PointerType::get(" # type.builderCall #")");
}

//===----------------------------------------------------------------------===//
// DeclRefType
//===----------------------------------------------------------------------===//

def KGEN_DeclRefType : KGENType<"DeclRef", "declref"> {
  let summary = "Reference to a type declaration.";
  let description = [{
    A `!kgen.declref` type references a type declaration and binds parameter values
    to the type.

    Example:

    ```mlir
    !kgen.declref<@MyStruct<size = 1, T:dtype = fooType>>
    ```
  }];

  let parameters = (ins "SymbolRefAttr":$symbol,
                        "ParamBindArrayAttr":$paramValues);
  let assemblyFormat = [{
    `<` $symbol `` custom<OptionalParamBindSpec>($paramValues) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name)>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "ParamBindArrayAttr":$paramValues)>,
    TypeBuilderWithInferredContext<(ins "SymbolRefAttr":$name,
                                        "ArrayRef<ParamBindAttr>":$paramValues)>
  ];

  let extraClassDeclaration = [{
    /// Get the name of the referenced type.
    StringAttr getName() { return getSymbol().getRootReference(); }

    /// Return an alias name if the DeclRefType should be printed with an alias.
    std::optional<StringRef> getAliasName();
  }];
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
// VariadicType
//===----------------------------------------------------------------------===//

def KGEN_VariadicType : KGENType<"Variadic", "variadic", [
    DeclareTypeInterfaceMethods<DataLayoutInterface>,
    DeclareTypeInterfaceMethods<ParameterTypeInterface>]> {
  let summary = "parameterized variadic type";
  let description = [{
    The `!kgen.variadic` type represents a homogeneously typed variadic sequence
    of zero or more elements.

    Example:

    ```mlir
    // A variadic sequence of scalar floats.
    !kgen.variadic<scalar<f32>>

    // A parameterized variadic sequence.
    !kgen.variadic<type>
    ```
  }];

  let parameters = (ins "TypedAttr":$elementType);
  let assemblyFormat = "`<` custom<TypeParamValue>($elementType) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,
  ];

  let extraClassDeclaration = [{
    /// Return the elementType attribute as a type.
    Type getElementAsType() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

class VariadicElementType<string lhs, string rhs> : TypesMatchWith<
    "value type matches variadic element type", lhs, rhs,
    "ParamRefType::get($_self.cast<KGEN::VariadicType>().getElementType())",
    "typeRangeMatches">;

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
