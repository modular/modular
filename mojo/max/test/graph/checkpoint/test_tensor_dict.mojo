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


import testing
from max.graph import _testing
from max.graph.checkpoint import TensorDict
from max.tensor import Tensor, TensorShape


fn test_simple() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))
    _testing.assert_equal(2, len(tensors))
    var x = tensors.get[DType.int32]("x")
    _testing.assert_tensors_equal(
        x,
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )

    # Test getting the same key again.
    _testing.assert_tensors_equal(
        tensors.get[DType.int32]("x"),
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )

    var y = tensors.pop[DType.float32]("y")
    _testing.assert_tensors_equal(
        y,
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )
    _testing.assert_equal(1, len(tensors))

    with testing.assert_raises():
        _ = tensors.get[DType.float32]("z")


fn test_overwrite() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("x", Tensor[DType.float32](TensorShape(10, 5), -1.23))
    _testing.assert_equal(1, len(tensors))

    var x = tensors.pop[DType.float32]("x")
    _testing.assert_tensors_equal(
        x,
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )

    with testing.assert_raises():
        _ = tensors.get[DType.float32]("y")


fn test_empty_key() raises:
    var tensors = TensorDict()
    tensors.set("", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))

    var t = tensors.get[DType.int32]("")
    _testing.assert_tensors_equal(
        t,
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )


def main():
    test_simple()
    test_overwrite()
    test_empty_key()
