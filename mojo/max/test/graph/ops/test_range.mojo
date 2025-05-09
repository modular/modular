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

from max.graph import Graph, TensorType, Type, _testing
from max.tensor import Tensor, TensorShape


fn test_range() raises:
    var g = Graph(List[Type]())

    g.output(
        g.range[DType.int32](
            0,
            12,
            1,
        )
    )
    g.verify()

    # fmt: off
    var expected = Tensor[DType.int32](
        TensorShape(12),
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    )
    # fmt: on

    var actual = _testing.execute_nullary[DType.int32](g)
    _testing.assert_tensors_equal(expected, actual)


fn test_range_symbol() raises:
    var g = Graph(List[Type]())

    g.output(
        g.range(
            g.scalar(0, DType.int32),
            g.scalar(12, DType.int32),
            g.scalar(1, DType.int32),
            out_dim=12,
        )
    )
    g.verify()

    # fmt: off
    var expected = Tensor[DType.int32](
        TensorShape(12),
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    )
    # fmt: on

    var actual = _testing.execute_nullary[DType.int32](g)
    _testing.assert_tensors_equal(expected, actual)


def main():
    # TODO(GEX-1784): MEF deserialization causes a segfault
    # test_range()
    test_range_symbol()
