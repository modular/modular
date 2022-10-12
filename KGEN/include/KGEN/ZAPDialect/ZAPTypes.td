//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPTYPES_TD
#define ZAPTYPES_TD

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/ZAPDialect/ZAPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class ZAPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<ZAPDialect, typeName,
            [DeclareTypeInterfaceMethods<SubElementTypeInterface>] # traits> {
  let mnemonic = typeMnemonic;
}


//===----------------------------------------------------------------------===//
// MemoryLikeType
//===----------------------------------------------------------------------===//

class ZAP_MemoryLikeType<string typeName, string typeMnemonic,
        list<Trait> traits = []> : ZAPType<typeName, typeMnemonic,
        [DeclareTypeInterfaceMethods<OpaqueObjectInterface>] # traits> {
  string classDeclaration = "";
  string baseExtraClassDeclaration = [{
    /// Try to narrow the size expression of the type to an integer.
    Optional<int64_t> getResolvedSize() const;
    /// Try to narrow the dtype expression of the type to a dtype.
    Optional<DType> getResolvedDType() const;
    /// Get the pointer type for the object.
    POP::PointerType getPointerType() const;
    /// Get the scalar element type for the object.
    POP::ScalarType getElementType() const;
  }];
  let extraClassDeclaration = classDeclaration # baseExtraClassDeclaration;
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def ZAP_BufferType : ZAP_MemoryLikeType<"Buffer", "buffer", [
    DeclareTypeInterfaceMethods<OpaqueObjectInterface>
  ]> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is a parameterized buffer type with a size and a `!kgen.dtype`.
    The size field is optional (may be null) in which case this is a buffer
    with an unknown size. The dtype type is also optional, in which case this
    is a buffer with unknown element type.
  }];

  let parameters = (ins "TypedAttr":$size, "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalIndexParamValue>($size) `,` ` `
        custom<OptionalDTypeParamValue>($dType) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "TypedAttr":$size, "TypedAttr":$dtype)>,
    TypeBuilder<(ins "int64_t":$size, "DType":$dtype)>
  ];
}

//===----------------------------------------------------------------------===//
// BufferType Type Constraints
//===----------------------------------------------------------------------===//

def BufferKnownDType : ConfinedType<ZAP_BufferType,
    [CPred<"$_self.cast<ZAP::BufferType>().getDType()">],
    "buffer with concrete dtype">;

def ConcreteBuffer : ConfinedType<ZAP_BufferType,
    [CPred<"$_self.cast<ZAP::BufferType>().getDType() && " #
           "$_self.cast<ZAP::BufferType>().getSize()">],
    "buffer with concrete size and dtype">;

class BufferPointerType<string lhs, string rhs>
  : TypesMatchWith<"pointer to scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::BufferType>().getPointerType()">;

class BufferElementType<string lhs, string rhs>
  : TypesMatchWith<"scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::BufferType>().getElementType()">;


//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

def ZAP_TensorType : ZAP_MemoryLikeType<"Tensor", "tensor"> {
  let summary = "parameterized tensor type";
  let description = [{
    This type is a parameterized tensor type with both the shape and a
    `!kgen.dtype`. The rank is not parametrized however. A shape field is
    optional (may be null) in which case this tensor with a dynamic size at
    that shape axis and therefore is of an unknown size. The dtype type is also
    optional, in which case this is a tensor with unknown element type. None of
    the shape dimensions can be zero. Currently, the tensor cannot be of zero
    rank and cannot have more than 5 dimensions (i.e. cannot be of rank more
    than 5). For example:

    ```mlir
    // a rank-4 tensor with 4x4x4x3x1 shape and si32 dtype
    !zap.tensor<[4, 4, 3, 1], si32>
    // a rank-2 tensor with 2x2 shape and unknown dtype
    !zap.tensor<[2, 2], ?>
    // a rank-3 tensor with unknown shape and f32 dtype
    !zap.tensor<[?, ?, ?], f32>
    // an invalid tensor because it has shape dimension of 0
    !zap.tensor<[1, 0], f32>
    // an invalid tensor because it has rank greater than 5
    !zap.tensor<[2, 3, 3, 1, 1, 4], f32>
    ````
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$shape,
                        "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalIndicesParamValue>($shape) `,` ` `
        custom<OptionalDTypeParamValue>($dType) `>`
  }];

  let classDeclaration = [{
    /// Get the rank of the tensor.
    size_t getRank() const;
    /// Get the maximum rank of the tensor.
    static constexpr size_t getMaximumRank() { return 5; }
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$shape,
                                        "TypedAttr":$dtype)>,
    TypeBuilder<(ins "ArrayRef<TypedAttr>":$shape, "DType":$dtype)>,
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape, "DType":$dtype)>
  ];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Type Constraints
//===----------------------------------------------------------------------===//

def TensorKnownDType : ConfinedType<ZAP_TensorType,
    [CPred<"$_self.cast<ZAP::TensorType>().getDType()">],
    "tensor with concrete dtype">;

def ConcreteTensor : ConfinedType<ZAP_TensorType,
    [CPred<"$_self.cast<ZAP::TensorType>().getDType() && " #
           "$_self.cast<ZAP::TensorType>().getSize()">],
    "tensor with concrete size and dtype">;

class TensorPointerType<string lhs, string rhs>
  : TypesMatchWith<"pointer to scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::TensorType>().getPointerType()">;

class TensorElementType<string lhs, string rhs>
  : TypesMatchWith<"scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::TensorType>().getElementType()">;

#endif // ZAPTYPES_TD
