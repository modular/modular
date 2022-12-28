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
include "Support/MDialect/MTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class KGENType<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<KGENDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ParamRefType
//===----------------------------------------------------------------------===//

def KGEN_ParamRefType : KGENType<"ParamRef", "paramref", [
    DeclareTypeInterfaceMethods<SubElementTypeInterface, ["replaceImmediateSubElements"]>
  ]> {
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

def KGEN_MLIRTypeType : KGENType<"MLIRType", "mlirtype"> {
  let summary = "Arbitrary MLIR type";
  let description = [{
    This kgen type represents an arbitrary MLIR type, used for parameterizing
    type-generic operations.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];
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
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def KGEN_StringType : KGENType<"String", "string"> {
  let summary = "String type for parameters";
  let description = [{
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

def KGEN_ListType : KGENType<"List", "list", [SubElementTypeInterface]> {
  let summary = "homogenous list type";
  let description = [{
    The `!kgen.list` type represents a homogenous sized list. The list type is
    the building block for variadic parameters. Lists have special behaviour in
    lowering. List operands to operations are expanded in-place during
    lowering, and consequently it has no runtime representation.

    Example:

    ```mlir
    %out = "foo.op"(%list) : (!kgen.list<index[N]>) -> !kgen.list<index[M]>
    ```

    Could become:

    ```mlir
    %out:2 = "foo.op"(%0, %1, %2) : (index, index, index) -> (index, index)
    ```

    List constants can be materialized with `kgen.param.constant`.

    Example:

    ```mlir
    %list = kgen.param.constant: list<index[3]> = <[1, 2, 3]>
    ```
  }];

  let parameters = (ins "TypedAttr":$elementType, "TypedAttr":$length);
  let assemblyFormat = [{
    `<` custom<TypeParamValue>($elementType) `[`
    custom<IndexParamValue>($length) `]` `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$elementType,
                                        "TypedAttr":$length)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "int64_t":$length)>,
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
                                        "TypedAttr":$length)>
  ];

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the SIMD type to an integer.
    Optional<int64_t> getResolvedLength() const;
    /// Try to narrow the type expression of the pointer type to an MLIR type.
    Type getResolvedElementType() const;
  }];
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

def KGEN_SignatureType : KGENType<"Signature", "signature",
    [SubElementTypeInterface]> {
  let summary = "The signature for a parameterized declaration";
  let description = [{
    This type describes the signature for a parameterized declaration, which
    can have input/output parameters as well as input/output value arguments
    and results.

    The first element of the conventions array specifies the FnConventions for
    the function and every additional element corresponds to a value input
    argument, indicating the ValueInputConvention parameter conventions.  When
    all elements are zero, it isn't printed in the textual MLIR format.
  }];

  let parameters = (ins "ParamDeclArrayAttr":$inputParams,
                        "TypeArrayAttr":$resultParamTypes,
                        "FunctionType":$values,
                        "ConventionsAttr":$conventions);
  let assemblyFormat = [{
    `<` $inputParams `,` $resultParamTypes `,`
    custom<ValuesAndOptionalConventions>($values, $conventions) `>`
  }];

  let builders = [
    /// This builder allows everything other than the function type to be null.
    TypeBuilderWithInferredContext<(ins
      "ParamDeclArrayAttr":$inputParams, "TypeArrayAttr":$resultParamTypes,
      "FunctionType":$values, "ConventionsAttr":$conventions)>,
    TypeBuilderWithInferredContext<(ins
      "ParamBindArrayAttr":$inputParams, "ParamDeclArrayAttr":$resultParams,
      "FunctionType":$values, "ConventionsAttr":$conventions)>,
    TypeBuilder<(ins "TypeRange":$inputs, "TypeRange":$results)>,
    TypeBuilderWithInferredContext<(ins "FunctionType":$values)>
  ];

  let extraClassDeclaration = [{
    /// Helper to return the value input and result types.
    ArrayRef<Type> getValueInputs();
    ArrayRef<Type> getValueResults();

    FnEffects getFnEffects() {
      return getConventions().getFnEffects();
    }

    ArrayRef<ValueInputConvention> getValueInputConventions() {
      return getConventions().getInputConventions();
    }

    /// Return the convention for the specified value input parameter.
    ValueInputConvention getInputConvention(size_t inputNo) {
      return ValueInputConvention(getValueInputConventions()[inputNo]);
    }

    /// Return a signature with the specified parameter bindings substituted
    /// into it as happens in a call.  The types specified in the parameter
    /// bindings affects the type signature of the value input and outputs, and
    /// also can remap the signature in the parameter list itself.
    ///
    /// If an error occurs making the substitution, report it with emitErrorFn
    /// and return null.
    SignatureType
    getSpecializedSignature(ArrayRef<ParamBindAttr> inputParams,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn);
    SignatureType
    getSpecializedSignature(ParamBindArrayAttr inputParams,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn);

    /// Return this signature type with the value signature replaced.
    SignatureType getWithValuesReplaced(FunctionType fnType);

    /// Return this signature with the input parameters dropped.
    SignatureType dropParamValues();

    /// Returns true if the signature is a concrete callable.
    bool isConcrete();
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// DeclRefType
//===----------------------------------------------------------------------===//

def KGEN_DeclRefType : KGENType<"DeclRef", "declref",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
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
}

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
