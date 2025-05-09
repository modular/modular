# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from max.graph import Dim, Graph, Symbol, TensorType, Type, _testing
from max.graph.ops.slicing import concat
from max.tensor import Tensor, TensorShape
from testing import assert_raises, assert_true


fn test_concat() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 2),
            TensorType(DType.int32, 3, 4),
            TensorType(DType.int32, 3, Dim("x")),
        ),
    )

    #      concat axis
    #              |
    #              v
    # shape_0: [3, 2]
    # shape_1: [3, 4]
    var cc1 = concat(List[Symbol](g[0], g[1]), axis=1)
    # TODO(GEX-552): Enable comparison of shape directly instead of printing full mlir
    assert_true(String(cc1).endswith("!mo.tensor<[3, 6], si32>"))

    #      concat axis
    #              |
    #              v
    # shape_0: [3, 4]
    # shape_1: [3, x]
    var cc2 = concat(List[Symbol](g[1], g[2]), axis=1, out_dim=Dim("y"))
    # TODO(GEX-552): Enable comparison of shape directly instead of printing full mlir
    assert_true(String(cc2).endswith("!mo.tensor<[3, y], si32>"))


fn test_concat_error() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, Dim("x")),
            TensorType(DType.int32, 5, 6),
            TensorType(DType.int32, 3, 2),
        ),
    )
    #      concat axis
    #              |
    #              v
    # shape_0: [3, x]
    # shape_1: [5, 6]
    #
    # NOTE error because `3 != 5`, user made a mistake here.
    with assert_raises(
        contains="[concat] input shapes must match except at concat axis"
    ):
        _ = concat(List[Symbol](g[0], g[1]), axis=1)

    #      concat axis
    #           |
    #           v
    # shape_0: [3, x]
    # shape_1: [5, 6]
    #
    # NOTE error because we can't prove `x == 6` at compile time; user must use
    # rebind to "cast" the `x` dimension to 6.
    with assert_raises(
        contains="[concat] input shapes must match except at concat axis"
    ):
        _ = concat(List[Symbol](g[0], g[1]), axis=0)

    #      concat axis
    #              |
    #              v
    # shape_0: [3, x]
    # shape_1: [3, 2]
    #
    # NOTE error because the graph api does not support algebraic expressions in dimensions.
    # So `x + 2` is not valid. `out_dim` must be set.
    with assert_raises(contains="Please set out_dim"):
        _ = concat(List[Symbol](g[0], g[2]), axis=1)


fn test_concat_rebind() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, Dim("x")),
            TensorType(DType.int32, 3, 2),
        ),
    )

    g.output(concat(List[Symbol](g[0], g[1]), axis=1, out_dim=Dim("y")))

    var x = Tensor[DType.int32](TensorShape(3, 1), 7)
    var y = Tensor[DType.int32](TensorShape(3, 2), -1)

    var expected = Tensor[DType.int32](
        TensorShape(3, 3), 7, -1, -1, 7, -1, -1, 7, -1, -1
    )

    var res = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_equal(expected, res)


fn test_concat_invalid_rebind() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, Dim("x")),
            TensorType(DType.int32, 3, 2),
        ),
    )

    # This is incorrect, `x+2 != x`. A new name is required here.
    g.output(concat(List[Symbol](g[0], g[1]), axis=1, out_dim=Dim("x")))

    var x = Tensor[DType.float32](TensorShape(3, 8), 7)
    var y = Tensor[DType.float32](TensorShape(3, 2), -1)

    with assert_raises():
        _ = _testing.execute_binary(g, x, y)


def main():
    test_concat()
    test_concat_error()
    test_concat_rebind()
    test_concat_invalid_rebind()
