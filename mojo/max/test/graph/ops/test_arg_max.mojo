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

from max.graph import Graph, TensorType, _testing, ops
from max.tensor import Tensor, TensorShape


fn test_arg_max() raises:
    var g = Graph(TensorType(DType.float32, 2, 3))
    g.output(ops.arg_max(g[0], 1))
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 3.0, 2.0, 1.0
    )
    var expected = Tensor[DType.int64](TensorShape(2, 1), 2, 0)

    var actual = _testing.execute_unary[outtype = DType.int64](g, x)
    _testing.assert_tensors_equal(expected, actual)


fn test_arg_max_neg_axis() raises:
    var g = Graph(TensorType(DType.float32, 2, 3))
    g.output(ops.arg_max(g[0], -1))
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 3.0, 2.0, 1.0
    )
    var expected = Tensor[DType.int64](TensorShape(2, 1), 2, 0)

    var actual = _testing.execute_unary[outtype = DType.int64](g, x)
    _testing.assert_tensors_equal(expected, actual)


def main():
    test_arg_max()
    test_arg_max_neg_axis()
