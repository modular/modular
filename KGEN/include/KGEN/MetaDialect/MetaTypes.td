//===- MetaTypes.td -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the Meta dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class MetaType<string name, list<Trait> traits = []>
  : TypeDef<MetaDialect, name, traits> { }

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

def ScalarType : MetaType<"Scalar",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized scalar type";
  let description = [{
    This type is parameterized with a value of !kgen.dtype type.
  }];

  let mnemonic = "scalar";
  let parameters = (ins "::mlir::Attribute":$dtype);
  let assemblyFormat = "`<` custom<ParamDTypeValue>($dtype) `>`";
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$parameter), [{
      return $_get(parameter.getContext(), parameter);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

def SIMDType : MetaType<"SIMD",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized SIMD vector type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.
  }];

  let mnemonic = "simd";
  let parameters = (ins "::mlir::Attribute":$size, "::mlir::Attribute":$dtype);
  let assemblyFormat = [{
    `<` custom<IndexParamValue>($size) `,` ` ` custom<ParamDTypeValue>($dtype) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$size, "Attribute":$ty), [{
      return $_get(size.getContext(), size, ty);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def BufferType : MetaType<"Buffer",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is parameterized with a size and a !kgen.dtype type.  The size
    field is optional (may be null) in which case this is a buffer with an
    unknown size.  The dtype type is also optional, in which case this is a
    buffer with unknown element type.
  }];

  let mnemonic = "buffer";
  let parameters = (ins "::mlir::Attribute":$size, "::mlir::Attribute":$dtype);
  let assemblyFormat = [{
    `<` custom<OptionalIndexParamValue>($size) `,` ` `
        custom<OptionalParamDTypeValue>($dtype) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$size, "Attribute":$ty), [{
      return $_get(size.getContext(), size, ty);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return the DType of the dtype.
    /// Returns DType::invalid if dtype is null
    /// (indicating dtype is unknown)
    DType getDType() {
      DTypeConstantAttr buf_dtype =
          getDtype().dyn_cast_or_null<DTypeConstantAttr>();
      if (!buf_dtype)
        return DType::invalid;

      return buf_dtype.getDType();
    };
  }];
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//
def PointerType : MetaType<"Pointer",
      [DeclareTypeInterfaceMethods<SubElementTypeInterface,
                                   ["replaceImmediateSubAttribute"]>]> {
  let summary = "parameterized pointer type";
  let description = [{
    This type is parameterized with a !kgen.dtype type. The dtype type is also
    optional, in which case this is a pointer to a buffer with unknown element
    type.
  }];

  let mnemonic = "pointer";
  let parameters = (ins "::mlir::Attribute":$dtype);
  let assemblyFormat = [{
    `<` custom<OptionalParamDTypeValue>($dtype) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$ty), [{
      return $_get(ty.getContext(), ty);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return the DType of the dtype.
    /// Returns DType::invalid if dtype is null
    /// (indicating dtype is unknown)
    DType getDType() {
      auto ptr_dtype = getDtype().dyn_cast_or_null<DTypeConstantAttr>();
      if (!ptr_dtype)
        return DType::invalid;

      return ptr_dtype.getDType();
    };
  }];
}
