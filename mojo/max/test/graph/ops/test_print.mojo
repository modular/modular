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
# RUN: mojo "%s" | FileCheck %s

from max.engine import InferenceSession, PrintStyle, TensorMap
from max.graph import Dim, Graph, Symbol, TensorType, _testing
from max.tensor import Tensor, TensorShape


fn test_print() raises:
    var g = Graph(TensorType(DType.float32, Dim.dynamic(), Dim.dynamic(), 2))
    g[0].print()
    g.output(List[Symbol]())
    g.verify()

    var x = Tensor[DType.float32](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    # CHECK{LITERAL}: debug_tensor = tensor([[[1.0000, 2.0000],
    # CHECK{LITERAL}: [3.0000, 4.0000]],
    # CHECK{LITERAL}: [[5.0000, 6.0000],
    # CHECK{LITERAL}: [7.0000, 8.0000]]], dtype=f32, shape=[2,2,2])
    _ = _testing.execute_base(g, x)


fn test_print_with_config() raises:
    var g = Graph(TensorType(DType.float32, Dim.dynamic(), Dim.dynamic(), 2))
    g[0].print()
    g.output(List[Symbol]())
    g.verify()

    var x = Tensor[DType.float32](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    var session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.FULL, precision=2)

    var model = session.load(g)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", x)

    # CHECK: debug_tensor = tensor<2x2x2xf32> [1.00e+00, 2.00e+00, 3.00e+00, 4.00e+00, 5.00e+00, 6.00e+00, 7.00e+00, 8.00e+00]
    _ = model.execute(input_map)
    _ = x^


fn test_print_none() raises:
    var g = Graph(TensorType(DType.float32, Dim.dynamic(), Dim.dynamic(), 2))
    g[0].print("does_not_print")
    g.output(List[Symbol]())
    g.verify()

    var x = Tensor[DType.float32](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    var session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.NONE)

    var model = session.load(g)

    var input_map = session.new_tensor_map()
    input_map.borrow("input0", x)

    # CHECK-NOT: does_not_print
    _ = model.execute(input_map)
    _ = x^


def main():
    test_print()
    test_print_with_config()
    test_print_none()
