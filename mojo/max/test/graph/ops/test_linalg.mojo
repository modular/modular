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
from max.graph.ops.linalg import layer_norm, tile
from max.tensor import Tensor, TensorShape


fn test_layer_norm() raises:
    var g = Graph(TensorType(DType.float32, 4, 4))

    var gamma = g.constant(Tensor[DType.float32](TensorShape(4), 0.1))

    var beta = g.constant(Tensor[DType.float32](TensorShape(4), 0.2))

    g.output(
        layer_norm[DType.float32](
            g[0],
            gamma=gamma,
            beta=beta,
            epsilon=1e-05,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(4, 4),
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
        -1.0, 0.0, 1.0, -2.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(4, 4),
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
        0.155278, 0.244721, 0.334163, 0.065836,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_tile() raises:
    var g = Graph(TensorType(DType.float32, 2, 3))

    g.output(
        tile(
            g[0],
            List[Int64](3, 2),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(2, 3),
        1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(6, 6),
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
        1.0, 0.5, 3.0, 1.0, 0.5, 3.0,
        -1.0, 2.0, 4.0, -1.0, 2.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_layer_norm()
    test_tile()
