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

from max.graph import Graph, Symbol, TensorType, Type
from max.graph._attributes import _tensor_attr
from max.tensor import Tensor, TensorShape

# Based on the MNIST example at https://www.tensorflow.org/datasets/keras_example:
#
#   func.func @serving_default(%arg0: tensor<1x28x28x1xf32> {tf_saved_model.index_path = ["inputs"]}) -> (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_inputs:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
#     %cst = "tf.Const"() {value = ...
#     %cst_0 = "tf.Const"() {value = dense<[-0.0675942451, 0.0063267909, 7.43086217E-4, -0.0126994187, 0.0148473661, 0.108896509, -0.0398316309, 0.0461452715, -0.0281771384, -0.0431172103]> : tensor<10xf32>} : () -> tensor<10xf32>
#     %cst_1 = "tf.Const"() {value = ...
#     %cst_2 = "tf.Const"() {value = ...
#     %cst_3 = "tf.Const"() {value = dense<[-1, 784]> : tensor<2xi32>} : () -> tensor<2xi32>
#     %0 = "tf.Squeeze"(%arg0) {device = "", squeeze_dims = [-1]} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28xf32>
#     %1 = "tf.Reshape"(%0, %cst_3) {device = ""} : (tensor<1x28x28xf32>, tensor<2xi32>) -> tensor<1x784xf32>
#     %2 = "tf.MatMul"(%1, %cst_1) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
#     %3 = "tf.BiasAdd"(%2, %cst_2) {data_format = "NHWC", device = ""} : (tensor<1x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
#     %4 = "tf.Relu"(%3) {device = ""} : (tensor<1x128xf32>) -> tensor<1x128xf32>
#     %5 = "tf.MatMul"(%4, %cst) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
#     %6 = "tf.BiasAdd"(%5, %cst_0) {data_format = "NHWC", device = ""} : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
#     return %6 : tensor<1x10xf32>
#   }


def test_mnist_low_level():
    var g = Graph(
        "test_mnist_low_level",
        List[Type](TensorType(DType.float32, 1, 28, 28, 1)),
        out_types=List[Type](TensorType(DType.float32, 1, 10)),
    )
    var cst_data = Tensor[DType.float32](128, 10)
    cst_data._to_buffer().fill(0.5)
    var cst = g.op(
        "mo.constant",
        TensorType(DType.float32, 128, 10),
        List(_tensor_attr(g._context(), "value", cst_data)),
    )
    var cst_0 = g.op(
        "mo.constant",
        TensorType(DType.float32, 1, 10),
        List(
            _tensor_attr(
                g._context(),
                "value",
                Tensor[DType.float32](
                    TensorShape(1, 10),
                    -0.0675942451,
                    0.0063267909,
                    7.43086217e-4,
                    -0.0126994187,
                    0.0148473661,
                    0.108896509,
                    -0.0398316309,
                    0.0461452715,
                    -0.0281771384,
                    -0.0431172103,
                ),
            )
        ),
    )

    var cst_1_data = Tensor[DType.float32](784, 128)
    cst_1_data._to_buffer().fill(0.5)
    var cst_1 = g.op(
        "mo.constant",
        TensorType(DType.float32, 784, 128),
        List(_tensor_attr(g._context(), "value", cst_1_data)),
    )
    var cst_2_data = Tensor[DType.float32](1, 128)
    cst_2_data._to_buffer().fill(0.5)
    var cst_2 = g.op(
        "mo.constant",
        TensorType(DType.float32, 1, 128),
        List(_tensor_attr(g._context(), "value", cst_2_data)),
    )
    var cst_3 = g.op(
        "mo.constant",
        TensorType(DType.int32, 2),
        List(
            _tensor_attr(
                g._context(),
                "value",
                Tensor[DType.int32](TensorShape(2), -1, 784),
            )
        ),
    )
    var p1 = g.op(
        "mo.reshape",
        List[Symbol](g[0], cst_3),
        TensorType(DType.float32, 1, 784),
    )
    var p2 = g.op(
        "mo.matmul", List[Symbol](p1, cst_1), TensorType(DType.float32, 1, 128)
    )
    var p3 = g.op(
        "mo.add", List[Symbol](p2, cst_2), TensorType(DType.float32, 1, 128)
    )
    var p4 = g.op("mo.relu", p3, TensorType(DType.float32, 1, 128))
    var p5 = g.op(
        "mo.matmul", List[Symbol](p4, cst), TensorType(DType.float32, 1, 10)
    )
    var p6 = g.op(
        "mo.add", List[Symbol](p5, cst_0), TensorType(DType.float32, 1, 10)
    )
    _ = g.nvop("mo.output", List(p6))

    # CHECK: test_mnist_low_level
    g.verify()
    print(g)


def test_mnist_helpers():
    var g = Graph(
        "test_mnist_helpers",
        List[Type](TensorType(DType.float32, 1, 28, 28, 1)),
    )
    var cst_data = Tensor[DType.float32](128, 10)
    cst_data._to_buffer().fill(0.5)
    var cst = g.constant(cst_data)

    var cst_0 = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 10),
            -0.0675942451,
            0.0063267909,
            7.43086217e-4,
            -0.0126994187,
            0.0148473661,
            0.108896509,
            -0.0398316309,
            0.0461452715,
            -0.0281771384,
            -0.0431172103,
        )
    )

    var cst_1_data = Tensor[DType.float32](784, 128)
    cst_1_data._to_buffer().fill(0.5)
    var cst_1 = g.constant(cst_1_data)

    var cst_2_data = Tensor[DType.float32](1, 128)
    cst_2_data._to_buffer().fill(0.5)
    var cst_2 = g.constant(cst_2_data)

    var p1 = g[0].reshape(1, 784)
    var p2 = p1 @ cst_1
    var p3 = p2 + cst_2
    var p4 = g.op("mo.relu", p3, TensorType(DType.float32, 1, 128))
    var p5 = p4 @ cst
    var p6 = p5 + cst_0
    _ = g.output(p6)

    # CHECK: test_mnist_helpers
    g.verify()
    print(g)


def main():
    test_mnist_low_level()
    test_mnist_helpers()
