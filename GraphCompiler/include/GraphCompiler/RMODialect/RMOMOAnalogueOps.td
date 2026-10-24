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

class RMO_MOAnaloguePureUnaryElementWiseOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesElemwiseUnaryImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}
class RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureUnaryElementWiseOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits>;

class RMO_MOAnaloguePureUnaryExplicitOutputOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesUnaryExplicitOutputTypeImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}
class RMO_MOAnaloguePureUnaryExplicitOutputMonoTypeOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureUnaryExplicitOutputOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits>;

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

def RMO_MOAbsOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"abs"> {
  let summary = "Absolute value operation";
  let description = [{
    Returns `abs(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOCosOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"cos"> {
  let summary = "Cosine operation";
  let description = [{
    Returns `cos(x)`.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOErfOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"erf"> {
  let summary = "(Gaussian) error function operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOExpOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"exp"> {
  let summary = "Exponential function operation";
  let description = [{
    Returns `exp(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOCeilOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"ceil"> {
  let summary = "Ceil operation";
  let description = [{
    Returns the elementwise smallest integer not less than `x`.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOFloorOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"floor"> {
  let summary = "Floor operation";
  let description = [{
    Returns the elementwise largest integer not greater than `x`.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}


def RMO_MOLogOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"log"> {
  let summary = "Natural logarithm operation";
  let description = [{
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOLog1pOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"log1p"> {
  let summary = "Natural logarithm of `1 + x` operation ";
  let description = [{
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

class RMO_MOSoftmaxLikeOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureUnaryElementWiseOp<name, traits # [
    AllDTypesMatch<["input", "result"]>,
    Pure,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        // `input` type passes through; `axis` is a compile-time `index`
        // attribute and contributes no MO operand.
        MO::TensorType inputTy = cast<MO::TensorType>(operands.getInput().getType());
        moOperandTypes.push_back(inputTy);
        moInputParamDecls.push_back({});

        moResultTypes.push_back(inputTy);
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_MOLogSoftmaxOp : RMO_MOSoftmaxLikeOp<"reduce.logsoftmax"> {
  let summary = "Logarithmic softmax operation";
  let description = [{
    Returns `log(softmax(x))` along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MONegativeOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"negative"> {
  let summary = "Negation operation";
  let description = [{
    Returns `-x`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MONotOp : RMO_MOAnaloguePureTensorOp<"not"> {
  let summary = "Logical negation operation";
  let description = [{
    Returns elementwise `not x` for a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesElemwiseUnaryImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}

def RMO_MOReluOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"relu"> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns `max(0, x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOGeluOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"gelu"> {
  let summary = "Gaussian error linear unit operation";
  let description = [{
    Returns the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOGeluTanhOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"gelu_tanh"> {
  let summary = "Tanh approximation of the GELU operation";
  let description = [{
    Returns the tanh approximation of GELU
    `0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu_tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOGeluQuickOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"gelu_quick"> {
  let summary = "Quick (sigmoid) approximation of the GELU operation";
  let description = [{
    Returns the quick approximation of GELU `x * sigmoid(1.702 * x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu_quick(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSigmoidOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"sigmoid"> {
  let summary = "Sigmoid operation";
  let description = [{
    Returns the sigmoid activation `1 / (1 + exp(-x))`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSiluOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"silu"> {
  let summary = "SiLU (Swish) operation";
  let description = [{
    Returns the SiLU (Swish) activation `x * sigmoid(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.silu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MORoundOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"round"> {
  let summary = "Rounding operation";
  let description = [{
    Returns the elementwise nearest integer, with ties going away from zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MORsqrtOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"rsqrt"> {
  let summary = "Reciprocal square root operation";
  let description = [{
    Returns `1/sqrt(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSinOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"sin"> {
  let summary = "Sine operation";
  let description = [{
    Returns `sin(x)`.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSqrtOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"sqrt"> {
  let summary = "Square root operation";
  let description = [{
    Returns `sqrt(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTanhOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"tanh"> {
  let summary = "Hyperbolic tangent operation";
  let description = [{
    Computes `tanh(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOATanhOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"atanh"> {
  let summary = "AHyperbolic tangent operation";
  let description = [{
    Computes `atanh(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTruncOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"trunc"> {
  let summary = "Truncation operation";
  let description = [{
    Returns the elementwise round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

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

class RMO_MOAnaloguePureReductionOp<
    string name,
    bit input_result_same_dtype = 1,
    list<Trait> traits = []> :
  RMO_MOAnaloguePureTensorOp<
    name,
    traits # !if(input_result_same_dtype, [AllDTypesMatch<["input", "result"]>], [])
  > {
  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def RMO_MOReduceMulOp : RMO_MOAnaloguePureReductionOp<"reduce.mul"> {
  let summary = "Product reduction operation";
  let description = [{
    Reduces `input` across `axis` (numpy semantics, -1 is the last axis) to
    its product, setting that dimension to 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.mul(%arg) [axis = 1] : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def RMO_MOReduceAddOp : RMO_MOAnaloguePureReductionOp<"reduce.add"> {
  let summary = "Sum reduction operation";
  let description = [{
    Reduces `input` across `axis` (numpy semantics, -1 is the last axis) to
    its sum, setting that dimension to 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.add(%arg) [axis = 1] : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_MOReduceMinOp : RMO_MOAnaloguePureReductionOp<"reduce.min"> {
  let summary = "Minimum reduction operation";
  let description = [{
    Reduces `input` across `axis` (numpy semantics, -1 is the last axis) to
    its minimum, setting that dimension to 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.min(%arg) [axis = 1] : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_MOReduceMaxOp : RMO_MOAnaloguePureReductionOp<"reduce.max"> {
  let summary = "Maximum reduction operation";
  let description = [{
    Reduces `input` across `axis` (numpy semantics, -1 is the last axis) to
    its maximum, setting that dimension to 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.max(%arg) [axis = 1] : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

class RMO_MOAnalogueArgMinMaxOp<string mnemonic> :
  RMO_MOAnaloguePureReductionOp<mnemonic, /*input_result_same_dtype=*/0> {
  let results = (outs MO_IndicesTensor:$result);
}

def MO_MOArgMaxOp : RMO_MOAnalogueArgMinMaxOp<"reduce.arg_max"> {
  let summary = "Maximum-value indices operation";
  let description = [{
    Like `reduce_max`, but returns indices instead of values. Ties return the
    lowest index.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
      %1 = rmo.mo.reduce.arg_max(%0) [axis = 1] : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_MOArgMinOp : RMO_MOAnalogueArgMinMaxOp<"reduce.arg_min"> {
  let summary = "Minimum-value indices operation";
  let description = [{
    Like `reduce_min`, but returns indices instead of values. Ties return the
    lowest index.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
      %axis = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
      %1 = rmo.mo.reduce.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

class RMO_MOAnaloguePureScatterOp<string name> :
  RMO_MOAnaloguePureTensorOp<name, [AllDTypesMatch<["input", "updates", "result"]>]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    MO_IndexScalarIfRanked:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesScatterImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getUpdates()),
          cast<TensorValue>(operands.getIndices()),
          cast<TensorValue>(operands.getAxis()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

// Unlike the `scatter.*` reduction variants, `scatter` takes `axis` as an
// attribute, so it does not use `RMO_MOAnaloguePureScatterOp`.
def RMO_MOScatterOp : RMO_MOAnaloguePureTensorOp<"scatter",
  [AllDTypesMatch<["input", "updates", "result"]>]> {
  let summary = "`scatter` operation";
  let description = [{
    Returns a copy of `input` with elements from `updates` scattered at
    `indices` along `axis`. `input`, `updates` and `indices` have the same
    rank. Each entry's target index is its own index with the `axis` component
    replaced by the `indices` value. In 2D:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter(%inputs, %updates, %indices) [axis = 0] : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       IndexAttr:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesScatterImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getUpdates()),
          cast<TensorValue>(operands.getIndices()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOScatterAddOp : RMO_MOAnaloguePureScatterOp<"scatter.add"> {
  let summary = "`scatter-add` operation";
  let description = [{
    Like `scatter`, but takes a scalar `axis` operand and combines each update
    with the existing element via addition. In 2D:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMaxOp : RMO_MOAnaloguePureScatterOp<"scatter.max"> {
  let summary = "scatter-max operation";
  let description = [{
    Like `scatter`, but takes a scalar `axis` operand and combines each update
    with the existing element via maximum. In 2D:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMinOp : RMO_MOAnaloguePureScatterOp<"scatter.min"> {
  let summary = "`scatter-min` operation";
  let description = [{
    Like `scatter`, but takes a scalar `axis` operand and combines each update
    with the existing element via minimum. In 2D:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMulOp : RMO_MOAnaloguePureScatterOp<"scatter.mul"> {
  let summary = "scatter-mul operation";
  let description = [{
    Like `scatter`, but takes a scalar `axis` operand and combines each update
    with the existing element via multiplication. In 2D:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
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

def RMO_MORangeOp : RMO_MOAnaloguePureTensorOp<"range", [
  AllDTypesMatch<["start", "limit", "step", "result"]>,
]> {
  let summary = "Range operation";
  let description = [{
    Returns the sequence from `start` up to (excluding) `limit` in increments
    of `step`. `step` must be non-zero, and `limit - start` must be zero or
    have the same sign as `step`.

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
      %step = mo.constant <#M.dense_array<1.5> : tensor<f32>> : !mo.tensor<[], f32>
      %res = rmo.mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
      %stepInt = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[], si32>
      %limitInt = mo.constant <#M.dense_array<11> : tensor<si32>> : !mo.tensor<[], si32>
      %oddNumbersBelowTen = rmo.mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$start,
    MO_Tensor:$limit,
    MO_Tensor:$step,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOResizeLinearOp : RMO_MOAnaloguePureTensorOp<"resize.linear"> {
  let summary = "Linear resize operation";
  let description = [{
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    `antialias` applies an antialiasing filter when downscaling.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    BoolAttr:$antialias,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `coordinate_transform_mode` `=`
            qualified($coordinate_transform_mode) `,`
        `antialias` `=` $antialias `]` attr-dict `:`
    functional-type(operands, results)
  }];
}


def RMO_MOResizeNearestOp : RMO_MOAnaloguePureTensorOp<"resize.nearest"> {
  let summary = "Nearest-neighbor resize operation";
  let description = [{
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    DefaultValuedAttr<I64Attr, "0">:$round_mode,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `coordinate_transform_mode` `=`
            qualified($coordinate_transform_mode)
        (`,` `round_mode` `=` $round_mode^)? `]` attr-dict `:`
    functional-type(operands, results)
  }];
}


def RMO_MOResizeBicubicOp : RMO_MOAnaloguePureTensorOp<"resize.bicubic"> {
  let summary = "Bicubic resize operation";
  let description = [{
    Resizes a tensor to a new shape using the bicubic interpolation algorithm.

    Uses Keys' cubic convolution (a = -0.5) over a 4x4 pixel neighborhood.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant <#M.dense_array<1, 3, 448, 448> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = rmo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$size,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);
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

class RMO_MOAnaloguePureUnaryComparisonOp<string name, list<Trait> traits = []> :
  RMO_MOAnaloguePureComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_ParamDecls:$outputParamDecls);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        auto inputTy = cast<MO::TensorType>(operands.getInputX().getType());

        moOperandTypes.push_back(inputTy);
        moInputParamDecls.push_back({});

        moResultTypes.push_back(inputTy.cloneWith(DType::kBool));
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_MOIsNaNOp : RMO_MOAnaloguePureUnaryComparisonOp<"is_nan"> {
  let summary = "NaN test operation";
  let description = [{
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOIsInfOp : RMO_MOAnaloguePureUnaryComparisonOp<"is_inf"> {
  let summary = "Infinity test operation";
  let description = [{
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGatherOp : RMO_MOAnaloguePureTensorOp<"gather", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "`gather` operation";
  let description = [{
    Gathers slices from input's axis according to indices.

    The output rank is `inputRank - 1 + indicesRank`, and:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    `axis` and `indices` follow numpy semantics (-1 is the last).

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %res = rmo.mo.gather(%input, %indices) [axis = 0] : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$indices,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOGatherNdOp : RMO_MOAnaloguePureTensorOp<"gather_nd", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Multi-dimensional `gather` operation";
  let description = [{
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension of `indices` is an index of length at most the input
    rank; outer dimensions act as batch dimensions. A shorter index gathers
    slices over the leading input dimensions.

    ```
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )
    ```

    ```mlir
      %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<2x2x4xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> {device = #M.device_ref<"cpu", 0>} :
        !mo.tensor<[3], si64>

      %result = rmo.mo.gather_nd(%input, %indices) [batchDims = 0] :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$indices,
    // TODO(#21324): Support dynamic batchDims
    IndexAttr:$batchDims,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `batchDims` `=` $batchDims `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOSoftmaxOp : RMO_MOSoftmaxLikeOp<"reduce.softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Returns `exp(input) / sum(exp(input))`, with the sum along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.softmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTileOp : RMO_MOAnaloguePureTensorOp<"tile", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Tiling operation";
  let description = [{
    Repeats `input` `repeats[i]` times along each dimension `i`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = rmo.mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$repeats,
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

def RMO_MOTransposeOp : RMO_MOAnaloguePureTensorOp<"transpose", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Transposition operation";
  let description = [{
    Permutes the dimensions of `input` by `perm`, which must hold unique values
    in `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$perm,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureTopKOp<string name> :
  RMO_MOAnaloguePureTensorOp<name, [AllDTypesMatch<["input", "values"]>]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndexScalarIfRanked:$K,
    MO_IndexScalarIfRanked:$axis,
    MO_CondLikeIfRanked:$sorted,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs
    MO_Tensor:$values,
    MO_I64Tensor:$indices
  );

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesTopKImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getK()),
          cast<TensorValue>(operands.getAxis()),
          cast<TensorValue>(operands.getSorted()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOBottomKOp : RMO_MOAnaloguePureTopKOp<"bottom_k"> {
  let summary = "Bottom-K operation";
  let description = [{
    Returns the `K` lowest values along `axis` and their indices, always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>> : !mo.tensor<[2, 6], si64>
      %k = mo.constant <#M.dense_array<3> : tensor<si64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %axis = mo.constant <#M.dense_array<1> : tensor<si64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %sorted = mo.constant <#M.dense_array<1> : tensor<1xi1>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], bool>
      %values, %indices = rmo.mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

class RMO_MOAnaloguePureLinalgOp<string mnemonic, list<Trait> traits = []> :
  RMO_MOAnaloguePureTensorOp<"linalg." # mnemonic, traits> {
  string cppNamespace = RMODialect.cppNamespace # "::LinAlg";
}

def RMO_MOBandPartOp : RMO_MOAnaloguePureLinalgOp<"band_part", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Matrix band part operation";
  let description = [{
    Zeroes everything outside the central band of each matrix formed by the
    last two axes; leading axes are batches. For input [I, J, ..., M, N]:

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    A negative `num_lower`/`num_upper` keeps the whole lower/upper triangle.
    `exclude` inverts the selection, zeroing the band instead.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[], si64>
      %num_upper = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
      %exclude = mo.constant <#M.dense_array<0> : tensor<1xui8>> : !mo.tensor<[], bool>
      %res = rmo.mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndexScalarIfRanked:$num_lower,
    MO_IndexScalarIfRanked:$num_upper,
    MO_CondLikeIfRanked:$exclude,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);
}

def RMO_MOPadConstantOp : RMO_MOAnaloguePureTensorOp<"pad.constant", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Constant-mode padding operation";
  let description = [{
    Pads a rank-`N` `input` with the scalar `constant`. `paddings` has `2 * N`
    elements, a `[before, after]` pair per dimension, so output dimension D is
    `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ScalarTensorIfRanked:$constant,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MOPadReflectOp : RMO_MOAnaloguePureTensorOp<"pad.reflect", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Reflection-mode padding operation";
  let description = [{
    Pads a rank-`N` `input` by reflection. `paddings` is as in `pad.constant`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MOPadRepeatOp : RMO_MOAnaloguePureTensorOp<"pad.repeat", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Repeat-mode padding operation";
  let description = [{
    Pads a rank-`N` `input` by repeating border values. `paddings` is as in
    `pad.constant`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MOSliceOp : RMO_MOAnaloguePureTensorOp<"slice", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Slice operation";
  let description = [{
    Slices an N-dimensional `input` with numpy semantics: dimension `i` is
    indexed by `start[i]:stop[i]:step[i]`, each a 1D index tensor of N elements.
    - Negative `start`/`stop` are supported; -1 is the last index.
    - Out-of-bound `start`/`stop` are clamped to [-dim, dim].
    - `step` must be nonzero; negative steps are supported.

    As in Python, negative indices are normalized before clipping, so
    `l[:-1:-1]` is empty; reverse with `l[:-N-1:-1]` where N is the dimension
    size.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$start,
    MO_IndicesVectorIfRanked:$stop,
    MO_IndicesVectorIfRanked:$step,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
}

def RMO_MORoiAlignOp : RMO_MOAnaloguePureTensorOp<"roi_align"> {
  let summary = "ROI alignment operation";
  let description = [{
    Pools `input` over the regions of interest `rois`.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant <#M.dense_array<5> : tensor<1xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], f32>

      %res = rmo.mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"]
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```

  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$rois,
    MO_IndexScalarIfRanked:$output_height,
    MO_IndexScalarIfRanked:$output_width,
    MO_FloatScalarIfRanked:$spatial_scale,
    MO_FloatScalarIfRanked:$sampling_ratio,
    BoolAttr:$aligned,
    Builtin_StringAttr:$mode,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `aligned` `=` $aligned `,` `mode` `=` $mode `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MORandomNormalOp : RMO_MOAnaloguePureTensorOp<"random.normal", [
  AllDTypesMatch<["mean", "variance"]>,
]> {
  let summary = "Normal distribution operation";
  let description = [{
    Returns a `shape` tensor sampled from a normal distribution with mean
    `mean` and standard deviation `variance`.

  Example:
    ```mlir
      %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %mean = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
      %variance = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
      %seed = mo.constant <#M.dense_array<1> : tensor<1xui64>> : !mo.tensor<[1], ui64>
      %res = rmo.mo.random.normal(%size, %mean, %variance, %seed) :
            (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
            !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesVectorIfRanked:$shape,
    MO_FloatScalarIfRanked:$mean,
    MO_FloatScalarIfRanked:$variance,
    MO_IndicesVectorIfRanked:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);
}

def RMO_MORandomUniformOp : RMO_MOAnaloguePureTensorOp<"random.uniform", [
  AllDTypesMatch<["lowerBound", "upperBound", "result"]>,
]> {
  let summary = "Uniform distribution operation";
  let description = [{
    Returns a `shape` tensor sampled uniformly from [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %lowerBound = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %upperBound = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
    %seed = mo.constant <#M.dense_array<1> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %res = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$shape,
    MO_ScalarTensorIfRanked:$lowerBound,
    MO_ScalarTensorIfRanked:$upperBound,
    MO_IndicesVectorIfRanked:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureScatterNdOp<string name> :
  RMO_MOAnaloguePureTensorOp<name, [AllDTypesMatch<["input", "updates", "result"]>]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let extraClassDefinition = [{
    void $cppClass::inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesScatterNdImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getUpdates()),
          cast<TensorValue>(operands.getIndices()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOScatterNdOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd"> {
  let summary = "Multi-dimensional `scatter` operation";
  let description = [{
    Returns a copy of `input` with slices from `updates` scattered at
    `indices`. The last dimension of `indices` indexes into `input`; an index
    shorter than the input rank addresses a slice of the leading dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdAddOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.add"> {
  let summary = "Multi-dimensional `scatter-add` operation";
  let description = [{
    Like `scatter_nd`, but stores the sum for duplicate indices.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMaxOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.max"> {
  let summary = "Multi-dimensional `scatter-max` operation";
  let description = [{
    Like `scatter_nd`, but stores the maximum for duplicate indices.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMinOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.min"> {
  let summary = "Multi-dimensional `scatter-min` operation";
  let description = [{
    Like `scatter_nd`, but stores the minimum for duplicate indices.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMulOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.mul"> {
  let summary = "Multi-dimensional `scatter-mul` operation";
  let description = [{
    Like `scatter_nd`, but stores the product for duplicate indices.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MONonMaximumSuppressionOp : RMO_MOAnaloguePureTensorOp<"non_maximum_suppression"> {
  let summary = "Non-max suppression operation";
  let description = [{
   Filters out boxes with high intersection-over-union (IOU).

   `boxes` are [y1, x1, y2, x2] for any diagonal pair of corners, normalized
   to [0, 1] or absolute.

    Example:

    ```mlir
      %boxes : !mo.tensor<[1, 6, 4], f32>
      %scores : !mo.tensor<[1, 1, 6], f32>
      %maxOutputBoxesPerClass : !mo.tensor<[], si64>
      %iouThreshold : !mo.tensor<[], f32>
      %scoreThreshold : !mo.tensor<[], f32>
      %res = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold)
        : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
        -> !mo.tensor<[?, ?], si64>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$boxes,
    MO_Tensor:$scores,
    MO_IndexScalarIfRanked:$maxOutputBoxesPerClass,
    MO_FloatScalarIfRanked:$iouThreshold,
    MO_FloatScalarIfRanked:$scoreThreshold,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_IndicesTensor:$output);
}

def RMO_MOArgNonzeroOp : RMO_MOAnaloguePureTensorOp<"arg_nonzero"> {
  let summary = "Nonzero-value indices operation";
  let description = [{
    Returns the row-major coordinates of the nonzero values of `input` as a
    [nnz, rank] tensor.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>> : !mo.tensor<[3, 3], si32>
      %1 = rmo.mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si64>
    ```
    }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Rank2I64Tensor:$result);
}

def RMO_MOCumSumOp : RMO_MOAnaloguePureTensorOp<"cumsum", [
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "Cumulative summation operation";
  let description = [{
    Returns the cumulative sum along `axis` (numpy semantics). `exclusive = 1`
    excludes the current element; `reverse = 1` sums in the opposite direction.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = rmo.mo.cumsum(%arg) [axis = 0, exclusive = 1, reverse = 0] : (
      !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    IndexAttr:$exclusive,
    IndexAttr:$reverse,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `,` `exclusive` `=` $exclusive `,`
        `reverse` `=` $reverse `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOMutableLoadOp : RMO_MOAnalogueOp<"mutable.load", [
  AllDTypesMatch<["inBuffer", "outTensor"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>,
]> {
  let arguments = (ins
    MO_Buffer:$inBuffer,
    MO_ParamDecls:$outputParamDecls,
    MO_Chain:$inChain
  );
  let results = (outs MO_Tensor:$outTensor, MO_Chain:$outChain);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOMutableStoreOp : RMO_MOAnalogueOp<"mutable.store", [
  AllDTypesMatch<["inBuffer", "inTensor"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>,
]> {
  let arguments = (ins
    MO_Buffer:$inBuffer,
    MO_Tensor:$inTensor,
    MO_ParamDecls:$outputParamDecls,
    MO_Chain:$inChain
  );

  let results = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOMutableStoreSliceOp : RMO_MOAnalogueOp<"mutable.store.slice", [
  AllDTypesMatch<["inBuffer", "slice"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
]> {
  let arguments = (ins
    MO_Buffer:$inBuffer,
    MO_Tensor:$slice,
    MO_IndicesVectorIfRanked:$start,
    MO_IndicesVectorIfRanked:$stop,
    MO_IndicesVectorIfRanked:$step,
    MO_ParamDecls:$outputParamDecls,
    MO_Chain:$inChain
  );
  let results = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOTransferOp : RMO_MOAnalogueOp<"transfer"> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ParamDecls:$outputParamDecls,
    MO_Chain:$inChain
  );
  let results = (outs MO_Tensor:$result, MO_Chain:$outChain);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $inChain `]` $input `:` qualified(type($input)) `to`
    custom<TransferResult>(ref(type($input)), type($result)) attr-dict
  }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD
