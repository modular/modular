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

from max.graph import Graph, TensorType, _testing
from max.graph.ops.slicing import select
from max.tensor import Tensor, TensorShape


fn test_select() raises:
    var g = Graph(TensorType(DType.bool, 1, 2, 2))

    var x = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            -1.5,
            2.5,
            -3.5,
            4.5,
        )
    )

    var y = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            -4.5,
            3.5,
            -2.5,
            1.5,
        )
    )

    g.output(
        select(
            g[0],
            x,
            y,
        )
    )
    g.verify()

    # fmt: off
    var input = Tensor[DType.bool](
        TensorShape(1, 2, 2),
        True, False,
        False, True,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -1.5, 3.5,
        -2.5, 4.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_select_broadcast() raises:
    var g = Graph(TensorType(DType.bool, 2))

    var x = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 1),
            -3.5,
            4.5,
        )
    )

    var y = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 1, 2),
            -2.5,
            1.5,
        )
    )

    g.output(
        select(
            g[0],
            x,
            y,
        )
    )
    g.verify()

    # fmt: off
    var input = Tensor[DType.bool](
        TensorShape(2),
        True, False,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -3.5, 1.5,
        4.5, 1.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_select_dtype_cast() raises:
    var g = Graph(TensorType(DType.bool, 1, 2, 2))

    var x = g.constant(
        Tensor[DType.int8](
            TensorShape(1, 2, 2),
            -1,
            2,
            -3,
            4,
        )
    )

    var y = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 2, 2),
            -4.5,
            3.5,
            -2.5,
            1.5,
        )
    )

    g.output(
        select(
            g[0],
            x,
            y,
        )
    )
    g.verify()

    # fmt: off
    var input = Tensor[DType.bool](
        TensorShape(1, 2, 2),
        True, False,
        False, True,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -1, 3.5,
        -2.5, 4,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, input)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_select()
    test_select_broadcast()
    test_select_dtype_cast()
