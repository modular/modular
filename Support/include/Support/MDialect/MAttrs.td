//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MATTRS_TD
#define MATTRS_TD

include "Support/MDialect/MDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class MAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// PrimitivesArrayAttr
//===----------------------------------------------------------------------===//

def M_PrimitivesArrayData : ArrayRefParameter<"uint8_t",
    "aligned storage for primitive elements"> {
  let allocator = [{
    $_dst = copyIntoAlignedBuffer($_allocator, $_self, elementType);
  }];
  let cppStorageType = "std::vector<uint8_t>";
}

def M_PrimitivesArrayAttr : MAttr<"PrimitiveArray", "primitives_array"> {
  let summary = "an array of primitive elements";
  let description = [{
    The `#M.primitives_array` attribute represents an array of primitive
    (boolean, integer, or floating point) data of equal size. The data is stored as
    a byte array with element types whose sizes are not multiples of bytes
    padded to the nearest byte. The underlying array is aligned to that byte
    size.

    Example:

    ```mlir
    // An array of integers.
    #M.primitives_array<si24: -2, 0, 2>

    // An array of floats.
    #M.primitives_array<bf16: 0.2, 1.2, 3.>

    // Boolean arrays use `i1` elements.
    #M.primitives_array<i1: true, false, true>
    ```
  }];

  let parameters = (ins M_PrimitivesArrayData:$data, "Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType custom<PrimitiveArray>($data, ref($elementType)) `>`
  }];
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<uint8_t>":$data,
                                        "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    /// Return the number of elements.
    int64_t size() const;
  }];
}

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

def M_ArrayElementsAttr : MAttr<"ArrayElements", "dense_array",
    [ElementsAttrInterface, TypedAttrInterface]> {
  let summary = "an elements attribute backed by an array";
  let description = [{
    The `#M.dense_array` attribute is an elements attribute backed by a
    primitive array. The attribute supports the full API expected by
    `ElementsAttr` and users thereof, including arbitrary shaped types and
    failable value iteration.

    Importantly, this attribute can only contain scalar integer and floating
    point types, does not bit-pack elements, and does not have special handling
    for splat elements.

    Example:

    ```mlir
    // An array of integers.
    #M.dense_array<-3, 0, 1, 42> : !M.array<4xi32>

    // A float tensor.
    #M.dense_array<3.4, 2.3, 5.2, 1.9> : tensor<2x2xf32>

    // A 0D vector.
    #M.dense_array<1> : vector<ui64>
    ```
  }];

  let parameters = (ins "PrimitiveArrayAttr":$data,
                        AttributeSelfTypeParameter<"", "ShapedType">:$type);
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<uint8_t>":$data,
                                        "ShapedType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Get the raw underlying data.
    ArrayRef<uint8_t> getRawData() const;

    /// `ElementsAttr` implementation. Support iteration with `Attribute` to be
    /// compatible with upstream dialects.
    using NonContiguousIterableTypesT = std::tuple<Attribute>;
    FailureOr<detail::AttrIterator>
    try_value_begin_impl(OverloadToken<Attribute>) const;
  }];
}

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

def IntArrayElementsAttr : Attr<CPred<"$_self.isa<IntArrayElementsAttr>()">,
    "Dense integer array"> {
  let storageType = "IntArrayElementsAttr";
  let returnType = "IntArrayElementsAttr";
}

def SI32ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "$_self.cast<IntArrayElementsAttr>().getElementType().isSignedInteger(32)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Signed)";
  let returnType = "ArrayRef<int32_t>";
  let convertFromStorage = "$_self.asArrayRef<int32_t>()";
}

#endif // MATTRS_TD
