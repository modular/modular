//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MATTRS_TD
#define MATTRS_TD

include "Support/MDialect/MDialect.td"
include "Support/MDialect/MAttrInterfaces.td"
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
    (boolean, integer, index, or floating point) data of equal size. The data is
    stored as a byte array with element types whose sizes are not multiples of
    bytes padded to the nearest byte. The underlying array is aligned to that
    byte size.

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
    `<` $elementType `` custom<PrimitiveArray>($data, ref($elementType)) `>`
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
    [ElementsAttrInterface, TypedAttrInterface,
     DeclareAttrInterfaceMethods<HasAlignedBytesInterface, ["getAlignedBytesType"]>]> {
  let summary = "an elements attribute backed by an array";
  let description = [{
    The `#M.dense_array` attribute is an elements attribute backed by a
    primitive array. The attribute supports the full API expected by
    `ElementsAttr` and users thereof, including arbitrary shaped types and
    failable value iteration.

    Importantly, this attribute can only contain scalar integer and floating
    point types, does not bit-pack elements, and does not have special handling
    for splat elements.

    This attribute implements the HasAlignedBytesInterface by returning a
    !M.aligned_bytes type capturing the byte size of the underlying element data
    array and an alignment of the next power-of-two of the byte size of the
    element type.

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
// AlignedBytesAttr
//===----------------------------------------------------------------------===//

// Make sure the attribute's data argument is interned with the requested
// alignment.
def M_AlignedBytesData : ArrayRefParameter<"uint8_t",
    "aligned storage for AlignedBytesAttr"> {
  let allocator = [{
    $_dst = copyIntoBytes($_allocator, $_self, align);
  }];
}

def M_AlignedBytesAttr : MAttr<"AlignedBytes", "aligned_bytes", [
    DeclareAttrInterfaceMethods<ElementsAttrInterface>,
    DeclareAttrInterfaceMethods<HasAlignedBytesInterface, ["getAlignedBytesType"]>]> {
  let summary = "A literal byte array with alignment.";
  let description = [{
    This attribute is the 'inline' equivalent of `dense_resource`, and describes
    a byte array with byte alignment. Nothing is implied about the internal
    structure of the bytes (rank, dimensions, underlying element type, padding,
    or element endianness).

    This attribute implements the ElementsAttrInterface as a rank-1 ShapedType
    over uint8_t elements.

    This attribute implements the HasAlignedBytesInterface by returning the
    obvious !M.aligned_bytes type.

    Generally this attribute should only be used for 'small' literals. Larger
    literals should be placed into the resources table and be referenced by the
    `dense_resources` attribute. Note that dense resources encode their
    alignment in their first four bytes using the little-endian convention,
    where as this attribute describes the alignment explicitly.

    Examples:
    ```mlir
    #M.aligned_bytes<"0x01020304", align 64> // 4 bytes, aligned on 64 bytes
    ```
  }];

  let parameters = (ins M_AlignedBytesData:$data, "uint64_t":$align);
  let assemblyFormat = [{
    `<` custom<AlignedBytesData>($data) `,` `align` $align `>`
  }];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// ElementsAttr interface implementation.
    using ContiguousIterableTypesT = std::tuple<uint8_t>;
    FailureOr<const uint8_t*> try_value_begin_impl(OverloadToken<uint8_t>) const;
    ShapedType getType() const;
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

def SI64ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "$_self.cast<IntArrayElementsAttr>().getElementType().isSignedInteger(64)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Signed)";
  let returnType = "ArrayRef<int64_t>";
  let convertFromStorage = "$_self.asArrayRef<int64_t>()";
}

def UI64ArrayElementsAttr : ConfinedAttr<IntArrayElementsAttr, [AttrConstraint<CPred<
    "$_self.cast<IntArrayElementsAttr>().getElementType().isUnsignedInteger(64)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<IntArrayElementsAttr>($0, IntegerType::Unsigned)";
  let returnType = "ArrayRef<uint64_t>";
  let convertFromStorage = "$_self.asArrayRef<uint64_t>()";
}

//===----------------------------------------------------------------------===//
// MultiLineString
//===----------------------------------------------------------------------===//

def M_MultiLineStringAttr
    : ArrayOfAttr<MDialect, "MultiLineString", "multiline", "StringAttr"> {
  let summary = "An array of strings which is encoded as a multi-line "
                "null-terminated string in MEF.";
  let description = [{
    An array of strings which is encoded as a multi-line null-terminated string
    in MEF.

    Example: The attribute
    ```mlir
      #m<multiline["line1",
                   "line2"]>
    ```
    will appear in MEF as the the string "line1\nline2\n\0".
  }];
}

//===----------------------------------------------------------------------===//
// PointerAttr
//===----------------------------------------------------------------------===//

def M_PointerAttr : MAttr<"Pointer", "pointer", [TypedAttrInterface]> {
  let summary = "an opaque pointer attribute";
  let description = [{
    The `#M.pointer` attribute represents an opaque pointer value as an address.
    Pointer constants have no meaning outside of in-memory IR, so the textual
    format of this attribute has no inherent meaning. It is meant to be used in
    operation folders or interpreters.
  }];

  let parameters = (ins "int64_t":$addr, AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "`<` $addr `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "int64_t":$addr, "Type":$type), [{
      return get(type.getContext(), addr, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

def MemRefAttr : MAttr<"MemRef", "memref", [TypedAttrInterface]> {
  let summary = "a reference to a block of memory";
  let description = [{
    The `#M.memref` attribute represents a reference to a block of memory. The
    block of memory lives in a dialect resource and describes the contents of
    the memory, and this attribute describes an offset into the block, which
    may be out-of-bounds, and how to interpret the memory.

    TODO: Add address space and write protection.

    Example:

    ```mlir
    #M.memref<my_blob, 24> : !M.pointer<index>
    ```
  }];

  let parameters = (ins "MemoryHandle":$memory, "int64_t":$offset,
                        "MemoryKind":$kind,
                        AttributeSelfTypeParameter<"">:$type);

  let assemblyFormat = [{
    `<` $memory `,` $offset `,` custom<MemoryKind>($kind) `>`
  }];
}

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

def OptionalStringRefParameter : StringRefParameter<""> {
  let defaultValue = "\"\"";
}

def M_TargetInfoAttr : MAttr<"TargetInfo", "target"> {
  let summary = "Target configuration attribute.";
  let description = [{
    The `#M.target` attribute represents a compilation target configuration. It
    contains the target triple, CPU type and features, and target data layout
    information.

    Example:

    ```mlir
    #M.target<triple="arm64-apple-darwin21.6.0", cpu="apple-m1", features="",
              data_layout="p:64:64-i64:64:64", simd_bit_width=128>
    ```
  }];

  let parameters = (ins
    "llvm::Triple":$triple,
    StringRefParameter<"">:$cpu,
    StringRefParameter<"">:$features,
    AttrParameter<"DataLayout", "data layout", "const DataLayout &">:
      $data_layout,
    "int64_t":$simd_bit_width,
    OptionalStringRefParameter:$tune_cpu
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Get the target triple as a string.
    std::string getTripleStr() const {
      return getTriple().normalize();
    }

    /// Check if the target supports a specific hardware feature.
    bool hasFeature(StringRef feature) const {
      return getFeatures().contains(feature);
    }

    /// Returns the operating system for the triple.
    StringRef getOS() const {
      return llvm::Triple::getOSTypeName(getTriple().getOS());
    }
  }];
}

//===----------------------------------------------------------------------===//
// BuildInfoAttr
//===----------------------------------------------------------------------===//

def KGEN_BuildInfoAttr : MAttr<"BuildInfo", "build_info"> {
  let summary = "Build configuration attribute.";
  let description = [{
    The `#M.build_info` attribute represents a compilation build configuration.
    It contains information how the compiler was built (in debug mode, with
    debug symbols, etc.).

    Example:

    ```mlir
    #M.build<type="debug", kernels_type="debug", profile_level=2>
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$buildType,
                        StringRefParameter<"">:$kernelsBuildType,
                        "int64_t":$lLCLMaxProfilingLevel);

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Get the build configuration attribute for the current build.
    static BuildInfoAttr getForCurrentBuild(MLIRContext *ctx);
  }];
}

#endif // MATTRS_TD
