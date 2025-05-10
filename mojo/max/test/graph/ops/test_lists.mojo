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
from max.graph import Graph, Symbol, TensorType, Type, _testing, ops
from max.tensor import Tensor, TensorShape


fn test_list_empty() raises:
    var g = Graph(List[Type]())
    var l = ops.list(TensorType(DType.float32, 1), g)
    g.output(l)

    g.verify()
    testing.assert_true("mo.list.create" in String(g))

    var lst = _testing.execute_nullary_list(g)
    testing.assert_equal(len(lst), 0)


fn test_list_nonempty() raises:
    var g = Graph(TensorType(DType.float32, 1))
    var l = ops.list(List[Symbol](g[0]))
    g.output(l)

    var x = Tensor[DType.float32](TensorShape(1), 1.0)

    g.verify()
    testing.assert_true("mo.list.create" in String(g))
    print(g)

    var lst = _testing.execute_unary_list(g, x)
    testing.assert_equal(len(lst), 1)
    _testing.assert_tensors_equal(lst[0], x)


fn test_list_empty_wrong_overload() raises:
    with testing.assert_raises():
        _ = ops.list(List[Symbol]())


fn test_list_mismatched_types() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 1), TensorType(DType.float32, 1, 2)
        ),
    )
    with testing.assert_raises():
        _ = ops.list(List[Symbol](g[0], g[1]))


fn main() raises:
    test_list_empty()
    test_list_nonempty()
    test_list_empty_wrong_overload()
    test_list_mismatched_types()
