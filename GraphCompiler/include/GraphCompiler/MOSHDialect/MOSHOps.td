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
// This defines shape operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_TD

include "GraphCompiler/MOSHDialect/MOSHTypes.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class MOSH_Op<string mnemonic, list<Trait> traits = []> :
  Op<MOSHDialect, mnemonic, traits>;

def MOSH_ParamToValueOp : MOSH_Op<"param.to_value", [
  AllTypesMatch<["value", "result"]>,
  DeclareOpInterfaceMethods<ParamOpInterface>,
  Pure,
]>, NoParamDecls {
  let summary = "create an SSA value from a parameter value";
  let description = [{
    The `mo.param.to_value` operation materializes the value of a parameter
    expression as an SSA value.

    Example:

    ```mlir
    %idx   = mo.param.to_value = <D0>
    %shape = mo.param.to_value: !mosh.ape = <[10, D1, 20]>
    ```

    The output type annotation may be omitted if it's `index`.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `` custom<ParamToValueOpValue>($value, type($result)) attr-dict
  }];
}

def MOSH_ParamFromValueOp : MOSH_Op<"param.from_value", [
  AllTypesMatch<["paramDecl", "value"]>,
  DeclareOpInterfaceMethods<ParamOpInterface>,
]>, OneParamDecl {
  let summary = "Create a parameter from an SSA value";

  let arguments = (ins
    KGEN_ParamDeclAttr:$paramDecl,
    AnyType:$value
  );

  let assemblyFormat = [{
    $paramDecl `=` $value attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def MOSH_BroadcastOp : MOSH_Op<"broadcast", [
  Commutative,
  Pure,
]> {
  let summary = "Shape broadcasting operation";
  let description = [{
    Returns the shape both inputs broadcast to under numpy rules. Same as
    `mo.broadcast_shape`, but on `!mosh.ape` values instead of shape-like
    tensors.

    Example:

    ```mlir
    %sh1: !mosh.ape
    %sh2: !mosh.ape
    %sh = mosh.broadcast(%sh1, %sh2) : !mosh.ape
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input_x, MOSH_ShapeType:$input_y);
  let results = (outs MOSH_ShapeType:$shape);

  let assemblyFormat = "`(` operands `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_ConcatOp : MOSH_Op<"concat", [Pure]> {
  let summary = "Concatenate a variadic number of shapes.";
  let description = [{
    Return a new shape by concatenating individual dimensions in the input
    shapes in order.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %sh2 : !mosh.ape
    %sh3 = mosh.concat(%sh0, %sh1, %sh2, %sh1)
    ```
  }];

  let arguments = (ins Variadic<MOSH_ShapeType>:$inputs);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $inputs `)` attr-dict";

  let builders = [
    /// Convenience method that works well with `rewriter.create(...)`
    OpBuilder<(ins
      "TypedValue<ShapeType>":$input1,
      "TypedValue<ShapeType>":$input2
    )>,
  ];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_EqualOp : MOSH_Op<"eq", [
  Commutative,
  Pure,
]> {
  let summary = "Check if two shapes are equal.";
  let description = [{
    Compare individual dimensions of the input shapes, and return `true` if they
    are all equal, or `false` otherwise.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %b = mosh.equal(%sh0, %sh1)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input1, MOSH_ShapeType:$input2);
  let results = (outs BoolScalar:$result);

  let assemblyFormat = "`(` $input1 `,` $input2 `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    /// Return as parameter expression (the two shapes must have equal ranks).
    static TypedAttr asExpression(MOSH::ShapeAttr shape1,
                                  MOSH::ShapeAttr shape2);
  }];
}

def MOSH_GetDimOp : MOSH_Op<"get_dim", [Pure]> {
  let summary = "Operation to extract a dimension from a shape.";
  let description = [{
    Return the size of the specified dimension in the given shape.

    The dimension must be a compile-time constant. Negative values index from
    the end, as in Python (e.g. -2 is the second to last dimension).

    Example:

    ```mlir
    %sh : !mosh.ape
    %dim : si64
    %firstDim = mosh.get_dim(%sh)[%dim]
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input, SI64Scalar:$dim);
  let results = (outs SI64Scalar:$result);

  let assemblyFormat = "`(` $input `)` `` `[` $dim `]` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    /// Create a static call to apply(mosh.get_dim, inputShape, axis)
    static TypedAttr emitStaticCall(TypedAttr inputShape, int64_t axis);
  }];
}

def MOSH_GetRankOp : MOSH_Op<"get_rank", [Pure]> {
  let summary = "Operation to get the rank of a shape.";
  let description = [{
    Return the rank of the given shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %r = mosh.get_rank(%sh)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs SI64Scalar:$result);

  let assemblyFormat = "`(` $input `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_NewOp : MOSH_Op<"new", [Pure]> {
  let summary = "Create a new shape from dimension values.";
  let description = [{
    Return a new shape whose dimensions equal to the given values in order.

    Example:

    ```mlir
    %d0 : !kgen.scalar<si64>
    %d1 : !kgen.scalar<si64>
    %sh0 = mosh.new(%d0, %d1)

    %d2 : !kgen.scalar<si64>
    %sh1 = mosh.new(%d0, %d1, %d2)
    ```
  }];

  let arguments = (ins Variadic<SI64Scalar>:$dims);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $dims `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_NumElementsOp : MOSH_Op<"num_elements", [Pure]> {
  let summary = "Returns the number of elements in a tensor.";
  let description = [{
    The op computes the product of the dimensions in a shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %numEl = mosh.num_elements(%sh)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs SI64Scalar:$result);

  let assemblyFormat = "`(` $input `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_SliceOp : MOSH_Op<"slice", [Pure]> {
  let summary = "Slice out a new shape from an existing shape.";
  let description = [{
    `start` and `end` follow Python slice semantics; omitting `end` gives
    `shape[start:]`.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %zero = mosh.param.to_value = <0>
    %two = mosh.param.to_value = <-2>
    %negOne = mosh.param.to_value = <-2>
    %negTwo = mosh.param.to_value = <-2>

    %sh1 = mosh.slice(%sh0)[%zero, %two]      // first two dims
    %sh2 = mosh.slice(%sh0)[%negTwo]          // last two dims
    %sh3 = mosh.slice(%sh0)[%zero, %negTwo]   // all but last two dims
    %sh4 = mosh.slice(%sh0)[%negTwo, %negOne] // second to last dim
    ```
  }];

  let arguments = (ins
    MOSH_ShapeType:$input,
    SI64Scalar:$start,
    SI64Scalar:$end
  );
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $input `)` `` `[` $start `,` $end `]` attr-dict";

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOSHAPEOPS_TD
