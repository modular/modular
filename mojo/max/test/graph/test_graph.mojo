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
# RUN: mojo %s | FileCheck %s

from pathlib.path import Path

from max.graph import Graph, Symbol, TensorType, Type
from testing import *


fn test_identity_graph() raises:
    var g = Graph(
        name="identity_graph",
        in_types=List[Type](TensorType(DType.int32)),
    )
    g.output(g[0])

    g.verify()
    # CHECK: module {
    # CHECK:   mo.graph @identity_graph([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:     mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK:   }
    # CHECK: }
    print(g)


fn test_identity_graph_no_module() raises:
    var g = Graph(
        name="identity_graph_no_module",
        in_types=List[Type](TensorType(DType.int32)),
    )
    g.output(g[0])

    g.verify()
    # CHECK: mo.graph @identity_graph_no_module([[arg:.*]]: !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
    # CHECK:   mo.output [[arg]] : !mo.tensor<[], si32>
    # CHECK: }
    print(g)


fn test_basic_add() raises:
    var g = Graph(
        "basic_add",
        List[Type](TensorType(DType.int32), TensorType(DType.int32)),
    )
    var y = g.op("mo.add", List[Symbol](g[0], g[1]), TensorType(DType.int32))
    g.output(y)

    g.verify()
    # CHECK-LABEL: @basic_add
    # CHECK: mo.add
    print(g)


fn test_add_constant() raises:
    var g = Graph("add_constant", List[Type](TensorType(DType.float32, 1)))
    var x = g.scalar[DType.float32](1.0, rank=1)
    var y = g.op("mo.add", List[Symbol](g[0], x), TensorType(DType.float32, 1))
    g.output(y)

    g.verify()
    # CHECK-LABEL: @add_constant
    # CHECK: mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>}
    # CHECK: mo.add
    print(g)


fn test_symbolic_dim() raises:
    var g = Graph(
        "symbolic_dim",
        List[Type](TensorType(DType.float32, "batch", "x")),
    )
    var x = g.scalar[DType.float32](1.0, rank=1)
    var y = g[0] + x
    g.output(y)

    g.verify()
    # CHECK-LABEL: mo.graph @symbolic_dim<batch, x>(%arg0: !mo.tensor<[batch, x], f32>) -> !mo.tensor<[batch, x], f32>
    print(g)


def test_layer():
    g = Graph(List[Type]())
    testing.assert_equal(g.current_layer(), "")
    with g.layer("foo"):
        with g.layer("bar"):
            x = g.constant[DType.int64](1)
            g.output(x)

    g.verify()
    # CHECK: loc("foo.bar":0:0)
    print(g.debug_str())


def _layer_context_return(g: Graph) -> Symbol:
    with g.layer("cheese"):
        with g.layer("wheel"):
            return g.constant[DType.int64](1)


def test_layer_context_return():
    g = Graph(List[Type]())
    testing.assert_equal(g.current_layer(), "")
    x = _layer_context_return(g)
    g.output(x)
    g.verify()
    # CHECK: loc("cheese.wheel":0:0)
    print(g.debug_str())


def test_current_layer():
    g = Graph(List[Type]())
    testing.assert_equal(g.current_layer(), "")
    with g.layer("foo"):
        testing.assert_equal(g.current_layer(), "foo")
        with g.layer("bar"):
            testing.assert_equal(g.current_layer(), "foo.bar")
        testing.assert_equal(g.current_layer(), "foo")
    testing.assert_equal(g.current_layer(), "")


fn main() raises:
    test_identity_graph()
    test_identity_graph_no_module()
    test_basic_add()
    test_add_constant()
    test_symbolic_dim()
    test_layer()
    test_current_layer()
    test_layer_context_return()
