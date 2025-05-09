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
from max.graph.ops.repeat_interleave import repeat_interleave
from max.tensor import Tensor, TensorShape


fn test_repeat_interleave() raises:
    var g = Graph(TensorType(DType.float32, 2, 2))

    g.output(
        repeat_interleave(
            g[0],
            repeats=2,
            dim=1,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(2, 2),
        1.0, 2.0,
        3.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(2, 4),
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_equal[DType.float32](actual, expected)


fn test_repeat_interleave_dim_none() raises:
    var g = Graph(TensorType(DType.float32, 2, 2))

    g.output(
        repeat_interleave(
            g[0],
            repeats=2,
            dim=None,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(2, 2),
        1.0, 2.0,
        3.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(8),
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_equal(actual, expected)


def main():
    test_repeat_interleave()
    test_repeat_interleave_dim_none()
