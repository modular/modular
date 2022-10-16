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
// NDBufferType
//===----------------------------------------------------------------------===//

def ZAP_NDBufferType : ZAP_MemoryLikeType<"NDBuffer", "ndbuffer"> {
  let summary = "parameterized NDBuffer type";
  let description = [{
    This type is a parameterized NDBuffer type with both the shape and a
    `!kgen.dtype`. The rank is not parametrized however. A shape field is
    optional (may be null) in which case this NDBuffer with a dynamic size at
    that shape axis and therefore is of an unknown size. The dtype type is also
    optional, in which case this is a NDBuffer with unknown element type. None
    of the shape dimensions can be zero. Currently, the NDBuffer cannot be of
    zero rank and cannot have more than 5 dimensions (i.e. cannot be of rank
    more than 5). For example:

    ```mlir
    // a rank-4 NDBuffer with 4x4x4x3x1 shape and si32 dtype
    !zap.ndbuffer<[4, 4, 3, 1], si32>
    // a rank-2 NDBuffer with 2x2 shape and unknown dtype
    !zap.ndbuffer<[2, 2], ?>
    // a rank-3 NDBuffer with unknown shape and f32 dtype
    !zap.ndbuffer<[?, ?, ?], f32>
    // an invalid NDBuffer because it has shape dimension of 0
    !zap.ndbuffer<[1, 0], f32>
    // an invalid NDBuffer because it has rank greater than 5
    !zap.ndbuffer<[2, 3, 3, 1, 1, 4], f32>
    ````
  }];

  let parameters = (ins ArrayRefParameter<"TypedAttr">:$shape,
                        "TypedAttr":$dType);
  let assemblyFormat = [{
    `<` custom<OptionalIndicesParamValue>($shape) `,` ` `
        custom<OptionalDTypeParamValue>($dType) `>`
  }];

  let classDeclaration = [{
    /// Get the rank of the NDBuffer.
    size_t getRank() const;
    /// Get the maximum rank of the NDBuffer.
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
// NDBuffer Type Constraints
//===----------------------------------------------------------------------===//

def NDBufferKnownDType : ConfinedType<ZAP_NDBufferType,
    [CPred<"$_self.cast<ZAP::NDBufferType>().getDType()">],
    "ndbuffer with concrete dtype">;

def ConcreteNDBuffer : ConfinedType<ZAP_NDBufferType,
    [CPred<"$_self.cast<ZAP::NDBufferType>().getDType() && " #
           "$_self.cast<ZAP::NDBufferType>().getSize()">],
    "ndbuffer with concrete size and dtype">;

class NDBufferPointerType<string lhs, string rhs>
  : TypesMatchWith<"pointer to scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::NDBufferType>().getPointerType()">;

class NDBufferElementType<string lhs, string rhs>
  : TypesMatchWith<"scalar of the same dtype", lhs, rhs,
                   "$_self.cast<ZAP::NDBufferType>().getElementType()">;

#endif // ZAPTYPES_TD
