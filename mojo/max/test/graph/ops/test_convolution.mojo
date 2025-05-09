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
from max.graph.ops.convolution import avg_pool, conv2d, conv3d, max_pool
from max.tensor import Tensor, TensorShape


fn test_avg_pool_padded() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    g.output(
        avg_pool(
            g[0],
            filter_shape=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -0.66666, 0.0, 0.66666, -1.33333, 0.0, 1.33333, -2.0, 0.0, 2.0, -1.55555, 0.0, 1.55555,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -2.33333, 0.0, 2.33333,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -2.33333, 0.0, 2.33333,
        -0.66666, 0.0, 0.66666, -1.33333, 0.0, 1.33333, -2.0, 0.0, 2.0, -1.55555, 0.0, 1.55555,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_avg_pool_padded_no_boundary() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    g.output(
        avg_pool(
            g[0],
            filter_shape=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            count_boundary=False,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.5, 0.0, 1.5, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -3.5, 0.0, 3.5,
        -1.5, 0.0, 1.5, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -3.5, 0.0, 3.5,
        -1.5, 0.0, 1.5, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -3.5, 0.0, 3.5,
        -1.5, 0.0, 1.5, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -3.5, 0.0, 3.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_avg_pool_no_padding() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    g.output(
        avg_pool(
            g[0],
            filter_shape=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2, 3),
        -2.0, 0.0, 2.0, -3.0, 0.0, 3.0,
        -2.0, 0.0, 2.0, -3.0, 0.0, 3.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_conv2d_padded() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    # fmt: off
    var filter = Tensor[DType.float32](
        TensorShape(3, 3, 3, 1),
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    )
    # fmt: on
    var filter_constant = g.constant(filter)

    g.output(
        conv2d(
            g[0],
            filter_constant,
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            groups=1,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 4, 4, 1),
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_conv2d_no_padding() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    # fmt: off
    var filter = Tensor[DType.float32](
        TensorShape(3, 3, 3, 1),
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    )
    # fmt: on
    var filter_constant = g.constant(filter)

    g.output(
        conv2d(
            g[0],
            filter_constant,
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0, 0, 0),
            groups=1,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2, 1),
        3.0, 4.5,
        3.0, 4.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_conv2d_stride() raises:
    var g = Graph(TensorType(DType.float32, 1, 8, 8, 3))

    var filter = Tensor[DType.float32](TensorShape(3, 5, 3, 1), 0.5)
    var filter_constant = g.constant(filter)

    g.output(
        conv2d(
            g[0],
            filter_constant,
            stride=(3, 4),
            dilation=(1, 1),
            padding=(0, 0, 1, 1),
            groups=1,
        )
    )
    g.verify()

    var x = Tensor[DType.float32](TensorShape(1, 8, 8, 3), 0.5)
    # fmt: off
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2, 1),
        9.0, 11.25,
        9.0, 11.25,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_conv3d_padded() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 4, 3))

    # fmt: off
    var filter = Tensor[DType.float32](
        TensorShape(3, 3, 3, 3, 1),
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    )
    # fmt: on
    var filter_constant = g.constant(filter)

    g.output(
        conv3d(
            g[0],
            filter_constant,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            padding=(1, 1, 1, 1, 1, 1),
            groups=1,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 4, 4, 4, 1),
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
        1.5, 3.0, 4.5, 6.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_conv3d_no_padding() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 4, 3))

    # fmt: off
    var filter = Tensor[DType.float32](
        TensorShape(3, 3, 3, 3, 1),
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    )
    # fmt: on
    var filter_constant = g.constant(filter)

    g.output(
        conv3d(
            g[0],
            filter_constant,
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            padding=(0, 0, 0, 0, 0, 0),
            groups=1,
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2, 2, 1),
        3.0, 4.5,
        3.0, 4.5,
        3.0, 4.5,
        3.0, 4.5,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_max_pool_padded() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    g.output(
        max_pool(
            g[0],
            filter_shape=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 2.0, -1.0, 0.0, 3.0, -2.0, 0.0, 4.0, -3.0, 0.0, 4.0,
        -1.0, 0.0, 2.0, -1.0, 0.0, 3.0, -2.0, 0.0, 4.0, -3.0, 0.0, 4.0,
        -1.0, 0.0, 2.0, -1.0, 0.0, 3.0, -2.0, 0.0, 4.0, -3.0, 0.0, 4.0,
        -1.0, 0.0, 2.0, -1.0, 0.0, 3.0, -2.0, 0.0, 4.0, -3.0, 0.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_max_pool_no_padding() raises:
    var g = Graph(TensorType(DType.float32, 1, 4, 4, 3))

    g.output(
        max_pool(
            g[0],
            filter_shape=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0, 0, 0),
        )
    )
    g.verify()

    # fmt: off
    var x = Tensor[DType.float32](
        TensorShape(1, 4, 4, 3),
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -3.0, 0.0, 3.0, -4.0, 0.0, 4.0,
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 2, 3),
        -1.0, 0.0, 3.0, -2.0, 0.0, 4.0,
        -1.0, 0.0, 3.0, -2.0, 0.0, 4.0,
    )
    # fmt: on

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_avg_pool_padded()
    test_avg_pool_padded_no_boundary()
    test_avg_pool_no_padding()
    test_conv2d_padded()
    test_conv2d_no_padding()
    test_conv3d_padded()
    test_conv3d_no_padding()
    test_max_pool_padded()
    test_max_pool_no_padding()
