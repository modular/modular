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

import testing
from max.graph import Graph, Type
from max.tensor import Tensor, TensorShape


fn test_basic_tensor() raises:
    var g = Graph(List[Type]())
    var y = g.constant(Tensor[DType.int32](TensorShape(1, 2), 1, 2))
    g.output(y)

    testing.assert_true(
        "value = #M.dense_array<1, 2> : tensor<1x2xsi32>" in String(y)
    )
    testing.assert_true("!mo.tensor<[1, 2], si32>" in String(y))


fn test_scalar_basic() raises:
    var g = Graph(List[Type]())
    var y = g.scalar[DType.bool](True)
    g.output(y)

    testing.assert_true(
        "value = #M.dense_array<true> : tensor<i1>" in String(y)
    )
    testing.assert_true(": !mo.tensor<[], bool>" in String(y))


fn test_scalar_high_rank() raises:
    var g = Graph(List[Type]())
    var y = g.scalar[DType.bool](True, 3)
    g.output(y)

    testing.assert_true(
        "value = #M.dense_array<true> : tensor<1x1x1xi1>" in String(y)
    )
    testing.assert_true(": !mo.tensor<[1, 1, 1], bool>" in String(y))


fn test_basic_i64() raises:
    var g = Graph(List[Type]())
    var y = g.scalar(Int64(1), rank=1)
    _ = g.output(y)

    testing.assert_true(
        "value = #M.dense_array<1> : tensor<1xsi64>" in String(y)
    )
    testing.assert_true(": !mo.tensor<[1], si64>" in String(y))


fn main() raises:
    test_basic_tensor()
    test_scalar_basic()
    test_scalar_high_rank()
    test_basic_i64()
