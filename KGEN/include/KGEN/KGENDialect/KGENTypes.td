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
    /// Returns true if the argument at this index is a vararg.
    bool isVararg(size_t index);
    /// Returns true if the argument at this index is a pack vararg.
    bool isPackVararg(size_t index);
    /// Returns true if the argument at this index is a keyword vararg.
    bool isKWVararg(size_t index);

    /// Return true if this signature has a first argument is a result from the
    /// function returned through memory.
    bool hasMemoryOnlyResult();

    /// Get the function default arguments.
    ArrayRef<TypedAttr> getDefaultArguments() {
      return getMetadata().getDefaultArguments();
    }

    /// Return true if the function throws.
    bool isThrows() {
      return bitEnumContainsAny(getFnEffects(), FnEffects::Throws);
    }

    /// Return true if the function is async.
    bool isAsync() {
      return bitEnumContainsAny(getFnEffects(), FnEffects::Async);
    }

    /// Return true if the function has parameter varargs.
    bool hasParamVarargs() {
      return bitEnumContainsAny(getFnEffects(), FnEffects::ParamVararg);
    }

    /// Return true if the function has pack varargs.
    bool hasPackVarargs() {
      return bitEnumContainsAny(getFnEffects(), FnEffects::PackVararg);
    }

    /// Return true if the function captures state, which means
    /// it requires a context pointer.
    bool isCapturing() const {
      return bitEnumContainsAny(getFnEffects(), FnEffects::Capturing);
    }

    bool isEscaping() const {
      return bitEnumContainsAny(getFnEffects(), FnEffects::Escaping);
    }

    /// Return true if the SSA result of this signature is a non-trivial owned
    /// register result.
    bool hasOwnedRegisterResult() {
      return bitEnumContainsAny(getFnEffects(), FnEffects::OwnedResult);
    }

    /// Return the convention for the specified value input parameter.
    ValueInputConvention getInputConvention(size_t inputNo) {
      return getValueInputConventions()[inputNo];
    }

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
    StringAttr getName() {
      return getSymbol().getRootReference(); }
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
  let assemblyFormat = "`<` custom<PrettyType>($elementType) `>`";
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
