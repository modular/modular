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

from max.graph import Graph, TensorType, Type, _testing, ops
from max.graph.quantization import Q4_0Encoding
from max.tensor import Tensor, TensorShape


def test_qmatmul():
    g = Graph(
        List[Type](
            TensorType(DType.float32, 5, 32), TensorType(DType.uint8, 32, 18)
        ),
    )
    g.output(ops.qmatmul[Q4_0Encoding](g[0], g[1]))
    g.verify()

    # This is a pretty bad test -- the inputs and outputs here are all zeroes.
    # But it's better than nothing -- at least we don't crash.  Also qmatmul
    # does not validate its tensor shapes all (except that the second input's
    # first dimension is a multiple of 32) so even if this were wrong we would
    # not be able to tell.
    x = Tensor[DType.float32](TensorShape(5, 32), 0)
    y = Tensor[DType.uint8](TensorShape(32, 18), 0)
    expected = Tensor[DType.float32](TensorShape(5, 32), 0)

    actual = _testing.execute_binary[DType.float32, DType.uint8, DType.float32](
        g, x, y
    )
    _testing.assert_tensors_equal(expected, actual)


def main():
    test_qmatmul()
