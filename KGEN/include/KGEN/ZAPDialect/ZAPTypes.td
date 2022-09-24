//===- ZAPTypes.td --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ZAPTYPES_TD
#define ZAPTYPES_TD

include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "KGEN/ZAPDialect/ZAPDialect.td"
include "mlir/IR/AttrTypeBase.td"

class ZAPType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<ZAPDialect, typeName,
            [DeclareTypeInterfaceMethods<SubElementTypeInterface>] # traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

def ZAP_BufferType : ZAPType<"Buffer", "buffer",
    [DeclareTypeInterfaceMethods<OpaqueObjectInterface>]> {
  let summary = "parameterized buffer type";
  let description = [{
    This type is parameterized with a size and a `!kgen.dtype`. The size field
    is optional (may be null) in which case this is a buffer with an unknown
    size. The dtype type is also optional, in which case this is a buffer with
    unknown element type.
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

  let extraClassDeclaration = [{
    /// Try to narrow the size expression of the buffer type to an integer.
    Optional<int64_t> getResolvedSize() const;
    /// Try to narrow the dtype expression of the buffer type to a dtype.
    Optional<DType> getResolvedDType() const;
    /// Get the pointer type for the buffer.
    POP::PointerType getPointerType() const;
    /// Get the scalar element type for the buffer.
    POP::ScalarType getElementType() const;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints

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

#endif // ZAPTYPES_TD
