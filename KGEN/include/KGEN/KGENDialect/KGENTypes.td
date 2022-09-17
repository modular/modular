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

def RegionType
  : KGENType<"Region",
             [DeclareTypeInterfaceMethods<SubElementTypeInterface>]> {
  let summary = "The signature for a region/symbol parameter";
  let description = [{
    This is the type for a region parameter, describing the input and results
    values for the region or func, as well as parameter input and outputs.
  }];

  let mnemonic = "region";
  let parameters = (ins "mlir::FunctionType":$values,
                        "mlir::FunctionType":$params);
  let assemblyFormat = "`<` $values `,` $params `>`";


  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::FunctionType":$values,
                                        "mlir::FunctionType":$params), [{
      return get(values.getContext(), values, params);
    }]>,
  ];
}

#endif // KGEN_KGENDIALECT_KGENTYPES_TD
