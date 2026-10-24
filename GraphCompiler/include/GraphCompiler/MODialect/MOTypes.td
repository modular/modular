//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines types in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD

include "GraphCompiler/MODialect/MODialect.td"
include "GraphCompiler/MODialect/MOTypeInterfaces.td"
include "Support/Compiler/TensorUtils.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class MO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MODialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_Tensor
//===----------------------------------------------------------------------===//

def MO_Tensor : MO_Type<"Tensor", "tensor", [MO_HasDeviceTypeInterface, MO_ShapedTypeInterface,
  DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    A tensor's shape and element type, an optional device ref, and an optional
    metadata dictionary (e.g. layout).

    `shape` is a `MOSH::ShapeAttr` (e.g. `[D0, 42, N]`) with static rank;
    dimensions may be integers or parametric. The element type is an M::DType,
    with `invalid` meaning unknown.

    Examples:
    ```mlir
    !mo.tensor<[4, 16], f32>           // static shape
    !mo.tensor<[N, N, 6], i32>         // parameterized shape
    !mo.tensor<[1, M, N], i32>         // partially known and parameterized shape
    !mo.tensor<[4, 16], f32, gpu:0>    // optional device
    ```
  }];

  let parameters = (ins
    "::M::MOSH::ShapeAttr" : $shape,
    "DType" : $dtype,
    DefaultValuedParameter<"::M::DeviceRefAttr", "::M::DeviceRefAttr::getDefault($_ctxt)"> : $deviceRef,
    DefaultValuedParameter<"DictionaryAttr", "{}"> : $metadata);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    // Base builder which all others call.
    TypeBuilder<(ins
      "MOSH::ShapeAttr" : $shape,
      "DType" : $dtype,
      "DeviceRefAttr" : $deviceRef,
      "DictionaryAttr" : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "MOSH::ShapeAttr" : $shape,
      "DType" : $dtype,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "MOSH::ShapeAttr" : $shape,
      "Type" : $elementType,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>" : $shape,
      "Type" : $elementType,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>" : $shape,
      "Type" : $elementType,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilder<(ins
      "ArrayRef<int64_t>" : $shape,
      "DType" : $dtype,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilder<(ins
      "ArrayRef<TypedAttr>" : $shape,
      "DType" : $dtype,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>];

  let extraClassDeclaration = [{
    TensorType cloneWith(
      ArrayRef<int64_t> shape, Type dtype) const;
    TensorType cloneWith(
      ArrayRef<TypedAttr> shapeDims) const;
    TensorType cloneWith(MOSH::ShapeAttr shape) const;
    TensorType cloneWith(DType dtype) const;
    TensorType cloneWith(LayoutAttr layoutAttr) const;
    TensorType cloneWithoutLayout() const;
    TensorType cloneWith(IOKindAttr ioKindAttr) const;
    TensorType cloneWithoutIOKind() const;
    TensorType cloneWith(DictionaryAttr metadataAttr) const;
    TensorType cloneWith(DeviceRefAttr device) const;

    bool hasRank() const;

    /// Fails if not statically ranked.
    size_t getRank() const;

    /// Like getRank, but scalars are rank 1.
    size_t getRankNonZero() const;

    bool hasStaticShape() const;

    Type getElementType() const;

    /// Returns the referenced parameter names (shape or dimension).
    SmallPtrSet<StringAttr, 4> getUsedParamNames() const;

    /// Fails if the shape is unknown.
    ArrayRef<TypedAttr> getDims() const;

    /// Fails if the shape is unknown.
    TypedAttr getDim(size_t n) const;

    /// True for parameters and parametric expressions (e.g. A + 2), not `?`.
    /// Fails if dynamically ranked.
    bool isParameterizedDim(size_t n) const;

    /// Fails if dynamically ranked.
    bool isIntegerDim(size_t n) const;

    static bool isParameterizedDimParam(TypedAttr attr);
    static bool isIntegerDimParam(TypedAttr attr);

    /// Unknown dimensions are mlir::ShapedType::kDynamic. Fails if the shape is
    /// unknown.
    SmallVector<int64_t> getIntDims() const;

    /// Unknown dimensions are mlir::ShapedType::kDynamic. Fails if the shape is
    /// unknown.
    int64_t getIntDim(size_t n) const;

    /// Fails if not statically ranked.
    TypedAttr getSymbolicNumElements() const;

    /// Returns no value unless all dimensions are known integers >= 0.
    std::optional<int64_t> getNumElements() const;

    static TensorType getScalarType(Type elementType,
      DeviceRefAttr deviceRef = {}, DictionaryAttr metadata = {});

    static TensorType getVectorSI64Type(MLIRContext *ctx, int64_t size,
      DeviceRefAttr deviceRef = {}, DictionaryAttr metadata = {});

    /// Returns a null `LayoutAttr{}` if there is none.
    MO::LayoutAttr getLayoutAttr();

    /// Returns a null `IOKindAttr{}` if there is none.
    MO::IOKindAttr getIOKindAttr();

    BoolAttr getIsMutableAttr();

    TensorType cloneWithMut(BoolAttr isMutable) const;

    TensorType cloneWithNewDim(uint64_t dim, uint64_t size);

    /// Returns a clone marked stable.
    TensorType cloneWithStable(bool isStable = true) const;

    /// Returns true if the tensor is stable: its data pointer is a stable
    /// address slot rather than the data itself, so a kernel must read
    /// through it at run time (see `!mgp.stable_addr`).
    bool isStable() const;
  }];
}

//===----------------------------------------------------------------------===//
// MO_List
//===----------------------------------------------------------------------===//

def MO_List : MO_Type<"List", "list", [MO_HasDeviceTypeInterface,
  DeclareTypeInterfaceMethods<MO_HasDeviceTypeInterface, ["getDeviceRef", "cloneWithNewDevice"]>]>
{
  let summary = "Type of immutable list container.";
  let description = [{
    Elements are currently restricted to `!mo.tensor`.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// MO_Bundle
//===----------------------------------------------------------------------===//

def MO_Bundle : MO_Type<"Bundle", "bundle"> {
  let summary = "Ordered collection of per-device tensors.";
  let description = [{
    Bundles per-device `!mo.tensor`s into one SSA value. Elements may differ in
    device, shape, or dtype.

    The verbose syntax lists each element and is used for non-uniform
    elements:

    ```mlir
    !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
    ```

    The compact syntax is used when all elements share shape, dtype, device
    label, and metadata, with contiguous device IDs (`ids[i] == ids[0] + i`):

    ```mlir
    !mo.bundle<[3], f32, gpu:0-1>
    ```

    A single element omits the range, and on the default host device omits
    the device too:

    ```mlir
    !mo.bundle<[3], f32, gpu:0>   // single element, explicit device
    !mo.bundle<[3], f32>          // single element, default host device
    ```

    Both forms store the same `elementTypes`.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$elementTypes);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    size_t getNumElements() const { return getElementTypes().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// MO_Scalar
//===----------------------------------------------------------------------===//

def MO_Scalar : MO_Type<"Scalar", "scalar",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "Type of scalars.";
  let parameters = (ins
    "DType" : $dtype);

  let assemblyFormat = "`<` $dtype `>`";

  let extraClassDeclaration = [{
    Type getDType() const;
  }];
}

//===----------------------------------------------------------------------===//
// MO_Buffer
//===----------------------------------------------------------------------===//

def MO_Buffer : MO_Type<"Buffer", "buffer", [MO_HasDeviceTypeInterface, MO_ShapedTypeInterface,
  DeclareTypeInterfaceMethods<MO_ShapedTypeInterface>,
  DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "Mutable tensor type.";
  let description = [{
    A mutable analogue of `!mo.tensor`; with mo.mutable.load and
    mo.mutable.store it models in-place operations. `shape` is a statically
    ranked `MOSH::ShapeAttr`; dtype `invalid` means unknown.

    Examples:
    ```mlir
    !mo.buffer<[4, 16], f32>    // static shape
    !mo.buffer<[N, N, 6], i32>  // parameterized shape
    ```
  }];

  let parameters = (ins
    "::M::MOSH::ShapeAttr" : $shape,
    "DType" : $dtype,
    DefaultValuedParameter<"::M::DeviceRefAttr", "::M::DeviceRefAttr::getDefault($_ctxt)"> : $deviceRef,
    DefaultValuedParameter<"DictionaryAttr", "{}"> : $metadata);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    // Base builder which all others call
    TypeBuilder<(ins
      "MOSH::ShapeAttr" : $shape,
      "DType" : $dtype,
      "DeviceRefAttr" : $deviceRef,
      "DictionaryAttr" : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "MOSH::ShapeAttr" : $shape,
      "DType" : $dtype,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "MOSH::ShapeAttr" : $shape,
      "Type" : $elementType,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<TypedAttr>" : $shape,
      "Type" : $elementType,
      CArg<"DeviceRefAttr", "{}"> : $deviceRef,
      CArg<"DictionaryAttr", "{}"> : $metadata)>,
    TypeBuilderWithInferredContext<(ins
      "TensorType" : $tensorType)>,
  ];

  let extraClassDeclaration = [{
    /// The type must not already have a device annotation.
    BufferType cloneWith(DeviceRefAttr deviceRef) const;

    static TensorType toTensorType(BufferType type);
    TensorType toTensorType();
  }];
}

//===----------------------------------------------------------------------===//
// MO_Chain
//===----------------------------------------------------------------------===//

def MO_Chain : MO_Type<"Chain", "chain"> {
  let summary = "Controls sequencing of side-effecting operations.";
  let description = [{
    Every side-effecting MO op consumes and produces a chain.
  }];
}

//===----------------------------------------------------------------------===//
// MO_Opaque
//===----------------------------------------------------------------------===//

def MO_Opaque : MO_Type<"Opaque", "opaque",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface,
                               ["getKernelOperandOpaqueName"]>]> {
  let summary = "Opaque MO type.";
  let description = [{
    A custom user-defined type.
      Example:
      ```mlir
        !mo.opaque<"my_list">
        !mo.opaque<"my_list", {foo = 42}>
      ```
  }];

  let parameters = (ins
    "StringAttr": $symbol,
    DefaultValuedParameter<"::mlir::DictionaryAttr",
                           "::mlir::DictionaryAttr::get($_ctxt)">:$parameters
  );

  let builders = [
    TypeBuilder<(ins "StringAttr": $symbol, "DictionaryAttr": $parameters)>,
    TypeBuilderWithInferredContext<(ins "StringAttr": $symbol)>,
    TypeBuilderWithInferredContext<(ins "StringAttr": $symbol,
                                        "DictionaryAttr": $parameters)>,
  ];

  let assemblyFormat = "`<` $symbol (`,` $parameters^)? `>`";
  let skipDefaultBuilders = true;

  let extraClassDeclaration = [{
    /// For converting to MGP::OpaqueType; std::nullopt if empty.
    std::optional<DictionaryAttr> getOptionalParameters() const {
      DictionaryAttr params = getParameters();
      if (params && !params.empty())
        return params;
      return std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ODS constraints to inject verifier logic via type annotation.
//===----------------------------------------------------------------------===//

class MO_HasAnyRankOfPred<list<int> ranks> :
  Or<!foreach(rank, ranks,
    CPred<[{cast<MO::TensorType>($_self).getRank() == }] # rank>)>;

class MO_BufferHasAnyRankOfPred<list<int> ranks> :
  Or<!foreach(rank, ranks,
    CPred<[{cast<MO::BufferType>($_self).toTensorType().getRank() == }] # rank>)>;

def FloatElementTypePred : CPred<
  "llvm::isa<FloatType>(cast<::M::MO::TensorType>($_self).getElementType())">;

def IntOrIndexElementTypePred : CPred<
  "cast<::M::MO::TensorType>($_self).getElementType().isInteger()">;

def SignedElementTypePred : CPred<
  "!cast<::M::MO::TensorType>($_self).getElementType().isUnsignedInteger() && "
  "!cast<::M::MO::TensorType>($_self).getDtype().isBool()">;

def HasSI64ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI64.predicate>;

def HasSI32ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI32.predicate>;

def HasBoolElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", I1.predicate>;

class MO_ConfinedBuffer<list<Pred> predicates, string summary = ""> :
  ConfinedType<MO_Buffer, predicates, summary, "::M::MO::BufferType">;

def MO_Rank1Buffer: MO_ConfinedBuffer<
  [MO_BufferHasAnyRankOfPred<[1]>], "rank-1 buffer">;

class MO_ConfinedTensor<list<Pred> predicates, string summary = ""> :
  ConfinedType<MO_Tensor, predicates, summary, "::M::MO::TensorType">;

class MO_ConfinedScalar<list<Pred> predicates, string summary = ""> :
  ConfinedType<MO_Scalar, predicates, summary, "::M::MO::ScalarType">;

def BoolScalarPred : CPred<
  "cast<::M::MO::ScalarType>($_self).getDtype().isBool()">;

def MO_BoolScalar : MO_ConfinedScalar<[BoolScalarPred], "boolean scalar">;

/// Excludes !mo.buffer because there is no alias analysis for ops that
/// forward their inputs (e.g. mo.guard, control flow with mo.yield).
/// TODO(GEX-1576): Implement alias analysis
def MO_AnyNonBufferType : Type<
  CPred<"!llvm::isa<::M::MO::BufferType>($_self)">,
  "any type but !mo.buffer">;

def MO_RankedScalarTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>], "ranked scalar tensor">;

def MO_ScalarTensorIfRanked : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>], "scalar or unranked tensor">;

def MO_RankedScalarIndexTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>, IntOrIndexElementTypePred],
  "ranked scalar index tensor">;

def MO_RankedScalarFloatTensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[0]>, FloatElementTypePred],
  "ranked scalar floating point tensor">;

def MO_FloatScalarIfRanked : MO_ConfinedTensor<
  [FloatElementTypePred, MO_HasAnyRankOfPred<[0]>],
  "rank 0 or unranked with float elements">;

def MO_FloatTensor : MO_ConfinedTensor<[FloatElementTypePred], "floating point tensor">;

def MO_NonScalarIfRanked: MO_ConfinedTensor<
  [Neg<MO_HasAnyRankOfPred<[0]>>], "unranked or non-scalar tensor">;

def MO_Rank1FloatTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, FloatElementTypePred], "rank-1 tensor with floats">;

def MO_Rank1IndexTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, IntOrIndexElementTypePred], "rank-1 tensor with indices">;

def MO_Rank2IndexTensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>, IntOrIndexElementTypePred], "rank-2 index tensor">;

def MO_Rank2SI32Tensor : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>, HasSI32ElementTypePred],
  "rank-2 tensor with signed 32-bit integer elements">;

def MO_Rank1Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>], "rank-1 tensor">;

def MO_Rank2Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>], "rank-2 tensor">;

def MO_Rank3Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[3]>], "rank-3 tensor">;

def MO_Rank4Tensor: MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[4]>], "rank-4 tensor">;

def MO_IndicesTensor : MO_ConfinedTensor<
  [IntOrIndexElementTypePred], "tensor with integer or index elements">;

def MO_SignedTensor : MO_ConfinedTensor<
  [SignedElementTypePred], "tensor with signed elements">;

def MO_ShapeLike : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred,
                                 HasSI32ElementTypePred]>],
  "rank 1 and signed 64 or signed 32 integer elements">;

def MO_PaddingLike : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred, HasSI32ElementTypePred]>],
  "shape [N] and signed 64 or signed 32 integer elements">;

def MO_PaddingLikeIfRanked : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred, HasSI32ElementTypePred]>],
  "unranked or shape [N] and signed 64 or signed 32 integer elements">;

def MO_ShapeLikeIfRanked : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>, Or<[HasSI64ElementTypePred,
                               HasSI32ElementTypePred]>],
  "rank 1 or unranked with signed 64 or signed 32 integer elements">;

def MO_VectorIfRanked : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[1]>], "rank 1 or unranked">;

def MO_IndicesVectorIfRanked : MO_ConfinedTensor<
  [IntOrIndexElementTypePred, MO_HasAnyRankOfPred<[1]>],
  "rank 1 or unranked with integer or index elements">;

def MO_IndexScalarIfRanked : MO_ConfinedTensor<
  [IntOrIndexElementTypePred, MO_HasAnyRankOfPred<[0]>],
  "rank 0 or unranked with integer or index elements">;

def MO_MatrixIfRanked : MO_ConfinedTensor<
  [MO_HasAnyRankOfPred<[2]>], "rank 2 or unranked">;

def MO_ScalarOfAnyRank : MO_ConfinedTensor<
  [CPred<"cast<MO::TensorType>($_self).getNumElements().value_or(0) == 1">],
  "Has exactly one element">;

def MO_BooleanMask : MO_ConfinedTensor<
    [HasBoolElementTypePred], "boolean element type">;

def MO_CondLike :
  ConfinedType<MO_Tensor,
    [MO_HasAnyRankOfPred<[0]>, HasBoolElementTypePred],
    "rank 0 and boolean element type"> {
  let cppType = "::M::MO::TensorType";

  let builderCall = [{
    ::MO::TensorType::get(::llvm::ArrayRef<::mlir::TypedAttr>{},
                          $_builder.getI1Type())
  }];
}

def MO_CondLikeIfRanked : ConfinedType<MO_Tensor,
    [MO_HasAnyRankOfPred<[0]>, HasBoolElementTypePred],
    "rank 0 or unranked with boolean element type"> {
  let cppType = "::M::MO::TensorType";
}

def MO_I64Tensor : ConfinedType<MO_Tensor,
    [HasSI64ElementTypePred],
    "signed 64 integer element type"> {
  let cppType = "::M::MO::TensorType";
}

def MO_Rank2I64Tensor : ConfinedType<MO_Tensor,
    [HasSI64ElementTypePred, MO_HasAnyRankOfPred<[2]>],
    "rank 2 signed 64 integer tensor"> {
  let cppType = "::M::MO::TensorType";
}

class DTypeStrFunc<string name> : StrFunc<"::M::MO::getDTypeOrInvalid($" # name # ".getType())">;

class ShapeAttrStrFunc<string name> : StrFunc<"::M::MO::getShape($" # name # ".getType())">;

/// e.g. `AllDTypesMatch<["input", "result"]>`. The builtin
/// `AllElementTypesMatch` doesn't work with MO::TensorType.
class AllDTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, DTypeStrFunc<"_self">.result,
                              "dtype">;

class AllShapeAttrsMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, ShapeAttrStrFunc<"_self">.result,
                              "shape">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPES_TD
