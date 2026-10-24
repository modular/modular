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
// This defines operations in the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD

include "Mojo/KGENDialect/KGENTypes.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/RMODialect/RMODialect.td"
include "GraphCompiler/RMODialect/RMOInterfaces.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MODialect/MOEnums.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for ops in this dialect.
class RMO_Op<string mnemonic, list<Trait> traits = []> :
  Op<RMODialect, mnemonic, traits # [
    RMO_TensorOpInterface,
    MO_ParamDeclaration,
  ]> {

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMO Asserting Rebinds
//===----------------------------------------------------------------------===//
def RMO_RebindTensorShapeOp : RMO_Op<"rebind_tensor_shape", [RMO_TensorSameDTypeOperandsAndResults]> {
  let summary = "Rebind the input tensor's shape with runtime check.";
  let description = [{
    Unlike `mo.rebind`, this also checks at runtime that the shapes are
    compatible. The check is modeled as a side effect because our stack cannot
    yet model assertions.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ParamDecls:$outputParamDecls,
    DefaultValuedAttr<StrAttr, "\"RMO rebind shape check failed.\"">:$message
  );
  let results = (outs MO_Tensor:$output);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    (`[` `message` `=` $message^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];

  let builders = [
    /// Inferred return type builders
    OpBuilder<(ins
      "TensorValue":$input,
      "ShapeAttr":$newShape,
      CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls
    )>,

    OpBuilder<(ins
      "TensorValue":$input,
      "ShapeAttr":$newShape,
      "StringRef":$message,
      CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls
    )>,
  ];

  // Folds away when the rebind provably keeps the same shape.
  let hasFolder = 1;
}


class RMO_PureBinaryElementWiseOp<string name, list<Trait> traits = []> :
  RMO_Op<name, traits # [
    InferTypeOpAdaptor,
    Pure,
  ]> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                $cppClass::Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryElemwiseImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_AddOp : RMO_PureBinaryElementWiseOp<"add", [Commutative]> {
  let summary = "Flexible binary add op";
  let description = [{
    Elementwise add with implicit broadcasting and dtype promotion.
  }];
}

def RMO_SubOp : RMO_PureBinaryElementWiseOp<"sub", []> {
  let summary = "Flexible binary sub op";
  let description = [{
    Elementwise sub with implicit broadcasting and dtype promotion.
  }];
}

def RMO_MulOp : RMO_PureBinaryElementWiseOp<"mul", [Commutative]> {
  let summary = "Flexible binary mul op";
  let description = [{
    Elementwise mul with implicit broadcasting and dtype promotion.
  }];
}

def RMO_DivOp : RMO_PureBinaryElementWiseOp<"div", []> {
  let summary = "Flexible binary div op";
  let description = [{
    Elementwise div with implicit broadcasting and dtype promotion.
  }];
}

def RMO_ModOp : RMO_PureBinaryElementWiseOp<"mod", []> {
  let summary = "Flexible binary mod op";
  let description = [{
    Elementwise mod with implicit broadcasting and dtype promotion.
  }];
}

// TODO(GEX-568): The MO pow op accepts mixed input dtypes (e.g. float ** int);
// update the lowering to avoid unnecessary casts.
def RMO_PowOp : RMO_PureBinaryElementWiseOp<"pow", []> {
  let summary = "Flexible binary pow op";
  let description = [{
    Elementwise pow with implicit broadcasting and dtype promotion.
  }];
}

def RMO_MinOp : RMO_PureBinaryElementWiseOp<"min", [Commutative]> {
  let summary = "Flexible binary min op";
  let description = [{
    Elementwise min with implicit broadcasting and dtype promotion.
  }];
}

def RMO_MaxOp : RMO_PureBinaryElementWiseOp<"max", [Commutative]> {
  let summary = "Flexible binary max op";
  let description = [{
    Elementwise max with implicit broadcasting and dtype promotion.
  }];
}

class RMO_PureBinaryComparisonOp<string name, list<Trait> traits = []> :
  RMO_Op<name, traits # [
    InferTypeOpAdaptor,
    MO_SingleDevice,
    Pure,
  ]> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_BooleanMask:$result);

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryComparisonImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_EqualOp : RMO_PureBinaryComparisonOp<"equal", [Commutative]> {
  let summary = "Flexible equality comparison op";
  let description = [{
    Elementwise equality comparison with implicit broadcasting and dtype promotion.

    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_GreaterOp : RMO_PureBinaryComparisonOp<"greater"> {
  let summary = "Flexible greater than comparison op";
  let description = [{
    Elementwise greater than comparison with implicit broadcasting and dtype promotion.

    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_GreaterEqualOp : RMO_PureBinaryComparisonOp<"greater_equal"> {
  let summary = "Flexible greater than or equal to comparison op";
  let description = [{
    Elementwise greater than or equal comparison with implicit broadcasting and dtype promotion.

    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_NotEqualOp : RMO_PureBinaryComparisonOp<"not_equal", [Commutative]> {
  let summary = "Flexible not equal to op";
  let description = [{
    Elementwise not equal comparison with implicit broadcasting and dtype promotion.

    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

class RMO_PureBinaryLogicalOp<string name, list<Trait> traits = []> :
  RMO_Op<name, traits # [
    InferTypeOpAdaptor,
    MO_SingleDevice,
    Pure,
  ]> {
  let arguments = (ins
    MO_BooleanMask:$input_x,
    MO_BooleanMask:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_BooleanMask:$result);

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryLogicalImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_AndOp : RMO_PureBinaryLogicalOp<"and", [Commutative]> {
  let summary = "Flexible and op";
  let description = [{
    Elementwise logical and with implicit broadcasting.

    Returns elementwise `x & y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.and(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_OrOp : RMO_PureBinaryLogicalOp<"or", [Commutative]> {
  let summary = "Flexible or op";
  let description = [{
    Elementwise logical or with implicit broadcasting.

    Returns elementwise `x | y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.or(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_XorOp : RMO_PureBinaryLogicalOp<"xor", [Commutative]> {
  let summary = "Flexible xor op";
  let description = [{
    Elementwise logical xor with implicit broadcasting.

    Returns elementwise `x ^ y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.xor(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_BroadcastLikeOp : RMO_Op<"broadcast_like", [Pure]> {
  let summary = "Broadcast the `input` tensor to be the same shape as `like`.";
  let description = [{
    Equivalent to the following:

      ```
      %shape_of = rmo.shape_of(%like)
      %result = rmo.broadcast_to(%broadcast, %shape_of)
      ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$shapeLike,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$shapeLike)>,
  ];

  let hasVerifier = 1;
}

def RMO_ConcatOp : RMO_Op<"concat", [
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates one or more `inputs` along `axis` (numpy semantics, so -1 is
    the last axis). The output's `axis` dimension is the sum of the inputs'.

    Shape restrictions:
    1. All inputs must have known and equal ranks.
    2. Input dimensions must match along non-concat-axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %res = rmo.concat(%arg0, %arg1) [axis = 1]  : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Builtin_IntegerAttr:$axis,
    OptionalAttr<AnyAttrOf<[Builtin_IntegerAttr, KGEN_ParamDeclRefAttr]>>:$out_dim,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $inputs `)`
    `[` `axis` `=` $axis (`,` `out_dim` `=` $out_dim^)? `]` attr-dict `:`
    functional-type($inputs, results)
  }];
}

def RMO_ReshapeOp : RMO_Op<"reshape", [
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Reshape a tensor";
  let description = [{
    Returns `input` reshaped to `newShape`. A `-1` dimension is inferred.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeAttr:$newShape);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `newShape` `=` qualified($newShape) `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_BroadcastToOp : RMO_Op<"broadcast_to", [
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Broadcast a tensor";
  let description = [{
    Broadcasts the input tensor to the specified shape.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeAttr:$newShape);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `newShape` `=` qualified($newShape) `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_ShapeToTensorOp : RMO_Op<"shape_to_tensor", [
  InferTypeOpAdaptorWithIsCompatible,
  Pure,
]> {
  let summary = "Converts a shape to a tensor";
  let description = [{
    Converts a static shape attr into a 1d tensor.
  }];
  let arguments = (ins MOSH_ShapeAttr:$shape);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `shape` `=` qualified($shape) `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MatmulOp : RMO_Op<"matmul", [
  InferTypeOpAdaptorWithIsCompatible,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs general matrix multiplication with broadcasting.

    The innermost 2 dimensions are the matrices (`MxK @ KxN -> MxN`); outer
    dimensions broadcast. A 1d lhs is treated as `1xK` and a 1d rhs as `Kx1`,
    and the added `1` dimension is dropped from the output.

    Example shapes with outputs:
    [K] @ [K] -> []
    [5, K] @ [K] -> [5]
    [K] @ [K, 6] -> [6]
    [4, K] @ [K, 6] -> [4, 6]
    [8, 10, 4, K] @ [8, 1, K, 6] -> [8, 10, 4, 6]
    [10, 4, K] @ [K, 6] -> [10, 4, 6]
    [10, 4, K] @ [K] -> [10, 4]
    [K] @ [10, 4, K, 7] -> [10, 4, 7]

    Only the result shape is inferred. The creator picks the result dtype,
    and the inputs keep their own, which may differ from each other and from
    the result (for example bf16 x bf16 -> f32).
  }];
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_ConvOp : RMO_Op<"conv", [
  InferTypeOpAdaptor,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Convolution operation";
  let description = [{
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    Supports 1D-3D convolution. The input is channel-last (NHWC for 2D). The
    filter layout comes from the filter tensor type's layout attribute: FCRS
    (default, `(out_channels, in_channels / num_groups, height, width)`), RSCF
    (legacy), or packed variants like FRSCf.

    `strides` and `dilations` have `input_rank - 2` elements, one per spatial
    dimension. `paddings` has `2 * (input_rank - 2)` elements of the form
    (dim1_before, dim1_after, dim2_before, ...) and zero-pads the spatial
    dimensions of `input` (dim1 is H, dim2 is W in 2D). For example, padding a
    2x3 spatial `input` with [0, 1, 2, 1] yields:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter ranks must match, and input and output
    channels must both be divisible by `num_groups`.

    Example:

    ```mlir
    %input: !mo.tensor<[10, 5, 5, 32], f32>,
    %filter: !mo.tensor<[64, 32, 2, 2], f32>

    %result = rmo.conv(%input, %filter) [
      strides = #mosh<ape[2, 2]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      num_groups = 1
    ] : (
      !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[64, 32, 2, 2], f32>
    ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MOSH_ShapeAttr:$strides,
    MOSH_ShapeAttr:$dilations,
    MOSH_ShapeAttr:$paddings,
    SI64Attr:$num_groups,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `strides` `=` qualified($strides) `,`
        `dilations` `=` qualified($dilations) `,`
        `paddings` `=` qualified($paddings) `,`
        `num_groups` `=` $num_groups
        (`,` `input_layout` `=` $input_layout^)? `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_ConvTransposeOp : RMO_Op<"conv_transpose", [
  InferTypeOpAdaptor,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Transposed convolution operation";
  let description = [{
    Computes the gradient of a convolution with the same filter and
    hyperparameters with respect to its input. See
    https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    Supports 1D-3D spatial dimensions (`out_channels` is w.r.t. the original
    convolution):
    - input is channel-last (NHWC for 2D)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters are rank 1 with `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (dim1_before, dim1_after, dim2_before, ...) over the output's spatial
    dimensions (dim1 is H_out, dim2 is W_out in 2D). For example,
    padding a 2x4 spatial `output` with [0, 1, 2, 1] yields:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    `output_paddings` disambiguates the output shape when a stride is greater
    than 1 by appending `output_paddings[i]` zeros to output axis `i`. Only 0 is
    supported.

    The input, output and filter ranks must match if statically known.

    Example:

    ```mlir
    %res = rmo.conv_transpose(%input, %filter) [
      strides = #mosh<ape[1, 1]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      output_paddings = #mosh<ape[0, 0]> : !mosh.ape
    ] : (
      !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
    ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MOSH_ShapeAttr:$strides,
    MOSH_ShapeAttr:$dilations,
    MOSH_ShapeAttr:$paddings,
    MOSH_ShapeAttr:$output_paddings,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `strides` `=` qualified($strides) `,`
        `dilations` `=` qualified($dilations) `,`
        `paddings` `=` qualified($paddings) `,`
        `output_paddings` `=` qualified($output_paddings)
        (`,` `input_layout` `=` $input_layout^)? `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

class RMO_PoolOp<string name> : RMO_Op<name, [
  InferTypeOpAdaptor,
  Pure,
]> {
  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeAttr:$filter_shape,
    MOSH_ShapeAttr:$strides,
    MOSH_ShapeAttr:$dilations,
    MOSH_ShapeAttr:$paddings,
    BoolAttr:$ceil_mode
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)`
    `[` `filter_shape` `=` qualified($filter_shape) `,`
        `strides` `=` qualified($strides) `,`
        `dilations` `=` qualified($dilations) `,`
        `paddings` `=` qualified($paddings) `,`
        `ceil_mode` `=` $ceil_mode `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_AvgPoolOp : RMO_PoolOp<"avg_pool"> {
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 with `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after).

    Example:

    ```mlir
    %res = rmo.avg_pool(%input) [
      filter_shape = #mosh<ape[2, 3]> : !mosh.ape,
      strides = #mosh<ape[2, 3]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      ceil_mode = false,
      count_boundary = true
    ] : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeAttr:$filter_shape,
    MOSH_ShapeAttr:$strides,
    MOSH_ShapeAttr:$dilations,
    MOSH_ShapeAttr:$paddings,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_boundary
  );

  let assemblyFormat = [{
    `(` operands `)`
    `[` `filter_shape` `=` qualified($filter_shape) `,`
        `strides` `=` qualified($strides) `,`
        `dilations` `=` qualified($dilations) `,`
        `paddings` `=` qualified($paddings) `,`
        `ceil_mode` `=` $ceil_mode `,`
        `count_boundary` `=` $count_boundary `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MaxPoolOp : RMO_PoolOp<"max_pool"> {
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 with `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after).

    Example:

    ```mlir
    %res = rmo.max_pool(%input) [
      filter_shape = #mosh<ape[2, 3]> : !mosh.ape,
      strides = #mosh<ape[2, 3]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      ceil_mode = false
    ] : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
  }];
}

def RMO_SelectOp : RMO_Op<"select", [
  InferTypeOpAdaptor,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Select operation with broadcasting and dtype casting";
  let description = [{
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    All inputs broadcast to the common shape of `cond`, `x` and `y`. Dtype
    promotion applies only to `x` and `y`; `cond` must be bool.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[1, 3], f32>
      %y: !mo.tensor<[2, 3], i8>
      %res = rmo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins
    MO_BooleanMask:$cond,
    MO_Tensor:$input_x,
    MO_Tensor:$input_y
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_TopKOp : RMO_Op<"top_k", [
  InferTypeOpAdaptor,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Top-K operation";
  let description = [{
    Returns the `k` largest values along `axis` (in [-rank, rank)), sorted,
    and their indices.

    Example:
    ```mlir
      %input: !mo.tensor<[2, 6], si64>
      %values, %indices = rmo.top_k(%input) [k = 3, axis = 1] :
        (!mo.tensor<[2, 6], si64>) ->
        (!mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>)
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    SI64Attr:$k,
    SI64Attr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs
    MO_Tensor:$values,
    MO_I64Tensor:$indices
  );
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `k` `=` $k `,` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_SliceOp : RMO_Op<"slice", [
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Slice operation with type inference.""";
  let description = [{
    Slices an N-dimensional `input` with numpy semantics: dimension `i` is
    indexed by `starts[i]:stops[i]:steps[i]`, each attribute having N elements.
    - Negative `starts`/`stops` are supported; -1 is the last index.
    - Unlike `mo.slice`, `starts` and `stops` must be in [-dim, dim].
    - `steps` must be nonzero; negative steps are supported.

    As in Python, negative indices are normalized before clipping, so
    `l[:-1:-1]` is empty; reverse with `l[:-N-1:-1]` where N is the dimension
    size.

    Example:
    ```mlir
      %input: !mo.tensor<[10, 10], f32>

      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.slice(%input) [starts = #mosh<ape[1, -6]> : !mosh.ape, stops = #mosh<ape[-3, 6]> : !mosh.ape, steps = #mosh<ape[5, 1]> : !mosh.ape] : (
        !mo.tensor<[10, 10], f32>,
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeAttr:$starts,
    MOSH_ShapeAttr:$stops,
    MOSH_ShapeAttr:$steps,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `starts` `=` qualified($starts) `,`
        `stops` `=` qualified($stops) `,`
        `steps` `=` qualified($steps) `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
