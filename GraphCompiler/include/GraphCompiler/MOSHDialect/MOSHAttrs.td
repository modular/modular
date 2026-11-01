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
// This defines attributes in the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD

include "GraphCompiler/MOSHDialect/MOSHDialect.td"
include "Mojo/KGENDialect/KGENAttrs.td"

def MOSH_ShapeAttr : KGEN_ExprAttr<"Shape", "ape"> {
  let summary = "Shape expression attribute";
  let description = [{
    The `#mosh.ape` attribute contains a variable number of `TypedAttr`s, each
    of which has a kgen.scalar<si64> type. The type of this attribute is
    always `!mosh.ape`.

    Each dimension `TypedAttr` inside `values` is one of:
    1. `KGEN::ParamDeclRefAttr` for a dimension parameter, e.g., `D0`.
    2. `KGEN::ParamOperatorAttr` for a dimension parameter expression, e.g.,
    `add(D1, 2)`.
    3. `KGEN::SIMDAttr` for a concrete integer dimension, e.g., `42` (typed as
    `kgen.scalar<si64>`).

    -1 denotes a dimension inferred from the others and the element count. At
    most one dimension can be -1.

    Example:

    ```mlir
    kgen.param.declare N = <3>
    #mosh<ape[1, ?, N]> : !mosh.ape
    #mosh<ape[3, -1, 42]> : !mosh.ape
    ```
  }];
  let dialect = MOSHDialect;

  let parameters = (ins
    OptionalArrayRefParameter<"TypedAttr">:$values,
    AttributeSelfTypeParameter<"", "MOSH::ShapeType">:$type
  );

  let assemblyFormat = [{
    `[` (`]`) : (custom<ShapeValue>($values, ref($type))^ `]`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "MOSH::ShapeType":$type), [{
      return $_get(type.getContext(), values, type);
    }]>,
    AttrBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$values,
                                        "MOSH::ShapeType":$type)>,
    AttrBuilderWithInferredContext<(ins "IntArrayElementsAttr":$intDims,
                                        "MOSH::ShapeType":$type)>
  ];

  let extraClassDeclaration = [{
    /// Preferred type for shape dimensions.
    static KGEN::SIMDType getDimType(MLIRContext* ctx);

    static KGEN::SIMDAttr getDim(MLIRContext* ctx, int64_t value);

    /// Is the given type a legal dimension (scalar `si64` SIMD lane)?
    static bool isDimType(::mlir::Type type);

    static bool isDim(TypedAttr attr) {
      return isDimType(attr.getType());
    }

    /// If `attr` is a compile-time constant dimension literal for this shape
    /// representation, return its signed value.
    static std::optional<int64_t> getStaticDim(Attribute attr);

    /// Returns the rank (i.e. size) of the shape.
    size_t getRank() const { return getSize(); }

    /// Returns the rank with rank 0 reported as 1, matching
    /// `MOShapedTypeInterface::getRankNonZero` and the emitters, which
    /// represent a rank-0 shape as rank-1 extent-1.
    size_t getRankNonZero() const { return getRank() == 0 ? 1 : getRank(); }

    /// Same as `getRank`, for when `ShapeAttr` holds a list of ints rather
    /// than a shape (e.g., slice bounds).
    size_t getSize() const { return getValues().size(); }

    TypedAttr operator[](int axis) const { return getValues()[axis]; }

    /// Whether the shape is static, i.e. all dims are known constants.
    bool isStatic() const { return isConstant(); }

    /// Returns the shape as integer values, using mlir::ShapedType::kDynamic
    /// for unknown dimensions.
    SmallVector<int64_t> getIntDims() const;

    /// Returns a dimension as an integer, using mlir::ShapedType::kDynamic for
    /// unknown dimensions.
    int64_t getIntDim(size_t n) const;

    /// Returns the number of elements as a symbolic expression.
    TypedAttr getSymbolicNumElements() const;

    /// Computes the common broadcast shape of the given shapes. Supports
    /// parametric and unknown shapes, as long as they can be broadcasted
    /// without constraints.
    static ErrorOr<ShapeAttr> broadcastShapes(ShapeAttr xSh, ShapeAttr ySh);

    /// Calculate the minimal common shape that all the input shape attributes
    /// can be broadcasted to. If not possible, returns a null attribute.
    static TypedAttr getCommonBroadcastShape(ArrayRef<TypedAttr> shapeParams);

    /// Returns the sorted axes of `bcastedShape` produced by broadcasting:
    /// newly introduced dims and 1 --> non-1 dims. Errors when two parameters
    /// A, B broadcast to each other, since that depends on runtime values.
    static ErrorOr<SmallVector<int64_t>> getBroadcastedAxes(ShapeAttr beforeShape, ShapeAttr bcastedShape);
  }];

  let genVerifyDecl = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHATTRS_TD
