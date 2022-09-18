//===- KGENTypes.td -------------------------------------------------------===//
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
class KGENType<string name, list<Trait> traits = []>
  : TypeDef<KGENDialect, name, traits> { }

def ParamRefType
  : KGENType<"ParamRef",
             [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "Type specified by a parameter expression";
  let description = [{
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted, the
    type disappears.
  }];

  let mnemonic = "paramref";
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

def MLIRTypeType : KGENType<"MLIRType"> {
  let summary = "Arbitrary MLIR type";
  let description = [{
    This kgen type represents an arbitrary MLIR type, used for parameterizing
    type-generic operations.
    
    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];

  let mnemonic = "mlirtype";
  let parameters = (ins);
}

def DTypeType : KGENType<"DType"> {
  let summary = "8-bit integer DType type";
  let description = [{
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
  }];

  let mnemonic = "dtype";
  let parameters = (ins);
}

def StringType : KGENType<"String"> {
  let summary = "String type for parameters";
  let description = [{
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
  }];

  let mnemonic = "string";
  let parameters = (ins);
}

def SignatureType
  : KGENType<"Signature",
             [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "The signature for a parameterized declaration";
  let description = [{
    This type describes the signature for a parameterized declaration, which
    can have input/output parameters as well as input/output value arguments
    and results.  This is used for region parameters, among other things.
  }];

  let mnemonic = "signature";
  let parameters = (ins "ParamDeclArrayAttr":$inputParams,
                        "ParamDeclArrayAttr":$resultParams,
                        "mlir::FunctionType":$values);
  let assemblyFormat = "`<` $inputParams `,` $resultParams `,` $values `>`";

  // Not all signatures are valid, their may only refer to parameters defined
  // within its input parameter list and they must have the right types.
  let genVerifyDecl = 1;
}

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
