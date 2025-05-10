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
# RUN: mojo %s

from max.graph import Graph, Symbol, TensorType, Type, _testing
from max.tensor import Tensor, TensorShape


fn test_insert_transform() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64), TensorType(DType.int64)),
    )
    g.output(g[0] + g[1])

    fn transform(arg: Symbol) raises -> Symbol:
        return arg + 3

    g[0].insert_transformation(transform)

    var x = Tensor[DType.int64](TensorShape(), 1)
    var y = Tensor[DType.int64](TensorShape(), 2)
    var expected = Tensor[DType.int64](TensorShape(), 6)

    var actual = _testing.execute_binary[DType.int64](g, x, y)

    _testing.assert_tensors_equal[DType.int64](expected, actual)


def main():
    test_insert_transform()
