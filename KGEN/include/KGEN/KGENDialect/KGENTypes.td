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

def ParamRefType
  : KGENType<"ParamRef", "paramref",
             [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
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

def MLIRTypeType : KGENType<"MLIRType", "mlirtype"> {
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

def DTypeType : KGENType<"DType", "dtype"> {
  let summary = "8-bit integer DType type";
  let description = [{
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def StringType : KGENType<"String", "string"> {
  let summary = "String type for parameters";
  let description = [{
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

def SignatureType
  : KGENType<"Signature", "signature",
             [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "The signature for a parameterized declaration";
  let description = [{
    This type describes the signature for a parameterized declaration, which
    can have input/output parameters as well as input/output value arguments
    and results.  This is used for region parameters, among other things.
  }];

  let parameters = (ins "ParamDeclArrayAttr":$inputParams,
                        "TypeArrayAttr":$resultParamTypes,
                        "mlir::FunctionType":$values);
  let assemblyFormat = "`<` $inputParams `,` $resultParamTypes `,` $values `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "ParamDeclArrayAttr":$inputParams, "TypeArrayAttr":$resultParamTypes,
        "FunctionType":$values), [{
      return get(inputParams.getContext(), inputParams, resultParamTypes,
                 values);
    }]>,
    TypeBuilderWithInferredContext<(ins
        "ParamBindArrayAttr":$inputParams, "ParamDeclArrayAttr":$resultParams,
        "FunctionType":$values)>
  ];

  let extraClassDeclaration = [{
    /// Verify that this SignatureType is self-contained, all referenced
    /// parameters are defined within the signature.
    ErrorOrSuccess checkSelfContained();

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
  }];
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

def RefType : KGENType<"Ref", "ref",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "Reference to a type declaration.";
  let description = [{
    A `!kgen.ref` type references a type declaration and binds parameter values
    to the type.

    Example:

    ```mlir
    !kgen.ref<@MyStruct<size = 1, T:dtype = fooType>>
    ```
  }];

  let parameters = (ins "FlatSymbolRefAttr":$name,
                        "ParamBindArrayAttr":$paramValues);
  let assemblyFormat = "`<` $name custom<OptionalParamBindSpec>($paramValues) `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name,
                                        "ParamBindArrayAttr":$paramValues)>,
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name)>
  ];
}

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
