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

from max.graph import Dim, Graph, TensorType, Type, _testing
from max.tensor import Tensor, TensorShape
from testing import assert_raises, assert_true


fn test_reshape() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 4),
            TensorType(DType.int32, 2, Dim("x"), 2),
        ),
    )

    # [3, 4] -> [6, 2]
    var reshape1 = g[0].reshape(6, 2)
    assert_true(reshape1.shape() == List[Dim](6, 2))

    # [3, 4] -> [2, 3, -1]
    var reshape2 = g[0].reshape(2, 3, -1)
    assert_true(reshape2.shape() == List[Dim](2, 3, 2))

    # [2, x, 2] -> [-1, x]
    var reshape3 = g[1].reshape(Dim(-1), Dim("x"))
    assert_true(reshape3.shape() == List[Dim](4, "x"))


fn test_reshape_error() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 3, 4),
        ),
    )

    # [3, 4] -> [6, 1]
    # NOTE error because `12 != 6`, user made a mistake here.
    with assert_raises(
        contains="[reshape] input and output number of elements must match"
    ):
        _ = g[0].reshape(6, 1)

    # [3, 4] -> [12, 0]
    # NOTE error because `12 != 0`, user made a mistake here.
    with assert_raises(
        contains="[reshape] input and output number of elements must match"
    ):
        _ = g[0].reshape(12, 0)

    # [3, 4] -> [-1, -1]
    # NOTE error because multiple `-1`, user made a mistake here.
    with assert_raises(
        contains="[reshape] multiple -1 detected in target shape"
    ):
        _ = g[0].reshape(-1, -1)

    # [3, 4] -> [0, -1]
    # NOTE error because `-1` can not be used with `0`, user made a mistake here.
    with assert_raises(
        contains=(
            "[reshape] cannot infer dimension when a specified output dimension"
            " is 0"
        )
    ):
        _ = g[0].reshape(0, -1)


fn test_reshape_runtime_zero() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 2, Dim("x"), 2),
        ),
    )

    # [2, x, 2] -> [-1, x]
    var reshape = g[0].reshape(Dim(-1), Dim("x"))

    g.output(reshape)
    g.verify()

    var x = Tensor[DType.int16](TensorShape(2, 0, 2))

    with assert_raises(
        contains=(
            "[reshape] cannot infer dimension when a specified output dimension"
            " is 0"
        )
    ):
        _ = _testing.execute_unary(g, x)


fn test_broadcast_to() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 2, 1, 2),
            TensorType(DType.int32, Dim("x"), 1),
        ),
    )

    var broadcast1 = g[0].broadcast_to(2, 7, 2)
    assert_true(broadcast1.shape() == List[Dim](2, 7, 2))

    var broadcast2 = g[0].broadcast_to(2, "x", 2)
    assert_true(broadcast2.shape() == List[Dim](2, "x", 2))

    var broadcast3 = g[1].broadcast_to("x", 7)
    assert_true(broadcast3.shape() == List[Dim]("x", 7))


fn test_broadcast_to_error() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.int32, 2, 1, 2),
            TensorType(DType.int32, Dim("x"), 1),
        ),
    )

    with assert_raises(contains="must be either 1 or equal to"):
        _ = g[0].broadcast_to(3, 7, 2)

    with assert_raises(contains="must be either 1 or equal to"):
        _ = g[0].broadcast_to(1, 1, 2)

    with assert_raises(contains="must be either 1 or equal to"):
        _ = g[1].broadcast_to("y", 1)


fn test_rebind() raises:
    var g = Graph(TensorType(DType.float32, "x", "y", "x"))

    with assert_raises(
        contains="rebind out_dims statically known to be incorrect"
    ):
        _ = g[0].rebind(List[Dim](1, 1, 3))

    g = Graph(TensorType(DType.float32, 1, 1, 1, 6))

    with assert_raises(
        contains="rebind out_dims statically known to be incorrect"
    ):
        _ = g[0].rebind(List[Dim](1, 1, 1, 3))


def main():
    test_reshape()
    test_reshape_error()
    # TODO(GEX-578): Once we have dim expression in, test a runtime negative number.
    test_reshape_runtime_zero()
    test_broadcast_to()
    test_broadcast_to_error()
    test_rebind()
