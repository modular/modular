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
// This defines MO analogue operations in the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD

include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/RMODialect/RMOOps.td"
include "GraphCompiler/RMODialect/RMODialect.td"
include "GraphCompiler/RMODialect/RMOInterfaces.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MODialect/MOEnums.td"
include "mlir/IR/BuiltinAttributes.td"

class RMO_MOAnalogueOp<string mnemonic, list<Trait> traits = []> :
  RMO_Op<"mo." # mnemonic, traits # [
    DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>,
    MO_BufferOpConformance,
  ]> {
  let builders = [
    /// Infers result types via `inferMOTypes`.
    OpBuilder<
      (ins "ValueRange":$inputValues, "MO::GraphOp":$graphOp),
      [{
        SmallVector<Type> _moOperandTypes, moResultTypes;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;

        inferMOTypesImpl(odsState.getContext(), graphOp, inputValues,
                         std::nullopt, _moOperandTypes, moResultTypes,
                         moInputParamDecls, moOutputParamDecls);

        SmallVector<KGEN::ParamDeclAttr> moOutputParamDeclsFlattened;
        for (const auto &resultDecls : moOutputParamDecls) {
          moOutputParamDeclsFlattened.append(resultDecls);
        }

        $_state.addOperands(inputValues);
        populateDefaultProperties($_state.name,
                                  $_state.getOrAddProperties<Properties>());
        $_state.addTypes(mlir::TypeRange{moResultTypes});

        $_state.addAttribute(
            getOutputParamDeclsAttrName($_state.name),
            $_builder.getAttr<ParamDeclArrayAttr>(moOutputParamDeclsFlattened));
      }]
    >,
  ];

  let extraClassDeclaration = [{

    /// Adaptor-based implementation of `inferMOTypes`.
    static void inferMOTypesImpl(
      MLIRContext* context,
      MO::GraphOp graphOp,
      Adaptor operands,
      std::optional<ValueRange> rmoResults,
      SmallVectorImpl<Type>& moOperandTypes,
      SmallVectorImpl<Type>& moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moOutputParamDecls
    );
  }];
}

// Base class for ops that map directly to a MO tensor op
class RMO_MOAnaloguePureTensorOp<string mnemonic, list<Trait> traits = []> :
  RMO_MOAnalogueOp<mnemonic, traits # [Pure]>;


//===----------------------------------------------------------------------===//
// 1-1 MO Op Analogues
//===----------------------------------------------------------------------===//

class RMO_MOAnaloguePureElementWiseOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureTensorOp<name, traits> {
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureBinaryElementWiseOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls
  );

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        MO::TensorType lhsType = cast<MO::TensorType>(operands.getInputX().getType());
        MO::TensorType rhsType = cast<MO::TensorType>(operands.getInputY().getType());

        DType rhsDType = rhsType.getDtype();

        MO::TensorType refinedShape = refineShape(lhsType, rhsType);
        moOperandTypes.push_back(refinedShape);
        moOperandTypes.push_back(refinedShape.cloneWith(rhsDType));
        moInputParamDecls.push_back({});
        moInputParamDecls.push_back({});

        moResultTypes.push_back(refinedShape);
        moOutputParamDecls.push_back({});
    }
  }];
}

class RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureBinaryElementWiseOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits>;

def RMO_MOAddOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"add", [Commutative]> {
  let summary = "Binary addition";
  let description = [{
    Does a non-broadcasted elementwise addition.
  }];
}

def RMO_MOShapeOfOp : RMO_MOAnaloguePureTensorOp<"shape_of"> {
  let summary = "Get shape of tensor";
  let description = [{
    Calculates the runtime shape of the given tensor as a tensor.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_IndicesTensor:$shape);
}

def RMO_MOBroadcastShapeOp : RMO_MOAnaloguePureTensorOp<"broadcast_shape"> {
  let summary = "Apply broadcasting rules";
  let description = [{
    Returns the broadcast of two shape tensors.
  }];
  let arguments = (ins
    MO_ShapeLikeIfRanked:$input_x,
    MO_ShapeLikeIfRanked:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_ShapeLikeIfRanked:$shape);
}

def RMO_MOReshapeOp : RMO_MOAnaloguePureTensorOp<"reshape"> {
  let summary = "Reshape a tensor";
  let description = [{
    Reshape the `input` tensor to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);
}

def RMO_MOBroadcastToOp : RMO_MOAnaloguePureTensorOp<"broadcast_to"> {
  let summary = "Broadcast a tensor";
  let description = [{
    Broadcast the given `input` to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MOCastOp : RMO_MOAnaloguePureTensorOp<"cast"> {
  let summary = "Cast a tensor to another dtype";
  let description = [{
    Casts `input` to another dtype.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

def RMO_MOMulOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"mul", [Commutative]> {
  let summary = "Binary multiplication";
  let description = [{
    Does a non-broadcasted elementwise multiplication.
  }];
}

def RMO_MOModOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"mod"> {
  let summary = "Binary modulus op";
  let description = [{
    Does a non-broadcasted elementwise modulus.
  }];
}

def RMO_MOMinOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"min", [Commutative]> {
  let summary = "Binary minimum op";
  let description = [{
    Does a non-broadcasted elementwise minimum.
  }];
}

def RMO_MOMaxOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"max", [Commutative]> {
  let summary = "Binary maximum op";
  let description = [{
    Does a non-broadcasted elementwise maximum.
  }];
}

def RMO_MODivOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"div"> {
  let summary = "Binary division op";
  let description = [{
    Does a non-broadcasted elementwise division.
  }];
}

def RMO_MOPowOp : RMO_MOAnaloguePureBinaryElementWiseOp<"pow"> {
  let summary = "Binary power op";
  let description = [{
    Does a non-broadcasted elementwise power. Allows mixed precision operands.
  }];
}

def RMO_MOSubOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"sub"> {
  let summary = "Binary subtraction op";
  let description = [{
    Does a non-broadcasted elementwise subtraction.
  }];
}

def RMO_MOConvOp : RMO_MOAnaloguePureTensorOp<"conv"> {
  let summary = "Convolution operation";
  let description = [{
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    Supports 1D-3D convolution with a channel-last input (NHWC for 2D) and an
    RSCF filter, i.e. (height, width, in_channels / num_groups, out_channels).

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. With a
    static input rank, statically shaped hyperparameters have `input_rank - 2`
    elements, one per spatial dimension, except `paddings`, which has
    `2 * (input_rank - 2)` elements of the form (dim1_before, dim1_after,
    dim2_before, ...) and zero-pads the spatial dimensions of `input` (dim1 is
    H, dim2 is W in 2D). For example, padding a 2x3 spatial `input` with
    [0, 1, 2, 1] yields:

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

    The input, output and filter ranks must match if statically known.
    `num_groups` is a ranked scalar that divides both input and output
    channels. Strides and padding apply only to the input.

    Example:

    ```mlir
      %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant <#M.dense_array<1> : tensor<si64>> {device = #M.device_ref<"cpu", 0>}
        : %!mo.tensor<[], si64>
      %res = rmo.mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_Rank1IndexTensor:$strides,
    MO_Rank1IndexTensor:$dilations,
    MO_Rank1IndexTensor:$paddings,
    MO_IndicesTensor:$num_groups,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    ` ` `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
                    `paddings` `=` $paddings `,`
                    `num_groups` `=` $num_groups `,`
                    `filter_layout` `=` $filter_layout
                    (`,` `input_layout` `=` $input_layout^)? `]` `:`
    functional-type(operands, results)
  }];
}

def RMO_MOConvTransposeOp : RMO_MOAnaloguePureTensorOp<"conv_transpose"> {
  let summary = "Transposed convolution operation";
  let description = [{
    Computes the gradient of a convolution with the same filter and
    hyperparameters with respect to its input. See
    https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    Supports 1D-3D spatial dimensions (`out_channels` is w.r.t. the original
    convolution):
    - input is channel-last (NHWC for 2D)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters are rank 1 or unranked. With a static input rank,
    statically shaped hyperparameters have `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (dim1_before, dim1_after, dim2_before, ...) over the output's spatial
    dimensions (dim1 is H_out, dim2 is W_out in 2D). For example, padding a
    2x4 spatial `output` with [0, 1, 2, 1] yields:

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
      %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[4], si64>
      %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %res = rmo.mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_Rank1IndexTensor:$strides,
    MO_Rank1IndexTensor:$dilations,
    MO_Rank1IndexTensor:$paddings,
    MO_Rank1IndexTensor:$output_paddings,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
        `paddings` `=` $paddings `,` `output_paddings` `=` $output_paddings `,`
        `filter_layout` `=` $filter_layout
        (`,` `input_layout` `=` $input_layout^)? `]` `:`
    functional-type(operands, results)
  }];
}

class RMO_MOAnaloguePurePoolOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureTensorOp<name, traits # [AllDTypesMatch<["input", "result"]>]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_Rank1IndexTensor:$filter_shape,
    MO_Rank1IndexTensor:$strides,
    MO_Rank1IndexTensor:$dilations,
    MO_Rank1IndexTensor:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOAvgPoolOp : RMO_MOAnaloguePurePoolOp<"avg_pool"> {
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 or unranked. With a static input rank,
    statically shaped hyperparameters have `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after). Strides and dilations apply only to
    the filter.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Rank1IndexTensor:$filter_shape,
    MO_Rank1IndexTensor:$strides,
    MO_Rank1IndexTensor:$dilations,
    MO_Rank1IndexTensor:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls
  );

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `,`
        `count_boundary` `=` $count_boundary `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOAvgPoolCeilOp : RMO_MOAnaloguePurePoolOp<"avg_pool_ceil_mode_true"> {
  let summary = "Ceiling-mode average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 or unranked. With a static input rank,
    statically shaped hyperparameters have `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after). Strides and dilations apply only to
    the filter.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Rank1IndexTensor:$filter_shape,
    MO_Rank1IndexTensor:$strides,
    MO_Rank1IndexTensor:$dilations,
    MO_Rank1IndexTensor:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls
  );

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `,`
        `count_boundary` `=` $count_boundary `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOMaxPoolOp : RMO_MOAnaloguePurePoolOp<"max_pool"> {
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 or unranked. With a static input rank,
    statically shaped hyperparameters have `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after). Strides and dilations apply only to
    the filter.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];
}

def RMO_MOMaxPoolCeilOp : RMO_MOAnaloguePurePoolOp<"max_pool_ceil_mode_true"> {
  let summary = "Ceiling-mode maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    Only 2D pooling on a 4D NHWC input is supported.

    All hyperparameters are rank 1 or unranked. With a static input rank,
    statically shaped hyperparameters have `input_rank - 2` elements, except
    `paddings`, which has `2 * (input_rank - 2)` elements of the form
    (H_before, H_after, W_before, W_after). Strides and dilations apply only to
    the filter.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];
}

class RMO_MOAnaloguePureComparisonOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureTensorOp<name, traits> {
  let results = (outs MO_BooleanMask:$result);
}

class RMO_MOAnaloguePureBinaryComparisonOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureComparisonOp<name, traits # [AllDTypesMatch<["input_x", "input_y"]>]> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls
  );

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        MO::TensorType lhsType = cast<MO::TensorType>(operands.getInputX().getType());
        MO::TensorType rhsType = cast<MO::TensorType>(operands.getInputY().getType());

        MO::TensorType refinedShape = refineShape(lhsType, rhsType);
        moOperandTypes.push_back(refinedShape);
        moOperandTypes.push_back(refinedShape);
        moInputParamDecls.push_back({});
        moInputParamDecls.push_back({});

        moResultTypes.push_back(refinedShape.cloneWith(DType::kBool));
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_MOEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"equal", [Commutative]> {
  let summary = "Equality test operation";
  let description = [{
    Returns elementwise `x == y`.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGreaterOp : RMO_MOAnaloguePureBinaryComparisonOp<"greater"> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x > y`.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                          !mo.tensor<[2, 3], f32>
                                          ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGreaterEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"greater_equal"> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x >= y`.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                                !mo.tensor<[2, 3], f32>
                                                ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MONotEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"not_equal", [Commutative]> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x != y`.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

class RMO_MOAnaloguePureLogicalBinaryOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureBinaryElementWiseOp<name, traits # [Commutative]> {
  let arguments = (ins
    MO_BooleanMask:$input_x,
    MO_BooleanMask:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_BooleanMask:$result);
}

def RMO_MOAndOp : RMO_MOAnaloguePureLogicalBinaryOp<"and"> {
  let summary = "Conjunction operation";
  let description = [{
    Returns elementwise `x and y` on boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOOrOp : RMO_MOAnaloguePureLogicalBinaryOp<"or"> {
  let summary = "Disjunction operation";
  let description = [{
    Returns elementwise `x or y` on boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                     !mo.tensor<[2, 3], bool>
                                     ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOXorOp : RMO_MOAnaloguePureLogicalBinaryOp<"xor"> {
  let summary = "Exclusive disjunction operation";
  let description = [{
    Returns elementwise `x xor y` on boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOSelectOp : RMO_MOAnaloguePureTensorOp<"select", [
  AllDTypesMatch<["x", "y", "result"]>,
]> {
  let summary = "Select operation";
  let description = [{
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins
    MO_BooleanMask:$cond,
    MO_Tensor:$x,
    MO_Tensor:$y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MOSqueezeShapeOp : RMO_MOAnaloguePureTensorOp<"squeeze_shape", [
  AllDTypesMatch<["input_shape", "result"]>,
]> {
  let summary = "Squeeze-shape operation";
  let description = [{
    Removes the `M` dimensions at `remove_indices` from a rank-`N` shape,
    returning a rank `N - M` shape. Each removed dimension must have size 1,
    otherwise it errors. Indices are in [-N, N), with Python-style negative
    indexing.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = rmo.mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLikeIfRanked:$input_shape,
    MO_ShapeLikeIfRanked:$remove_indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_ShapeLikeIfRanked:$result);
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD
