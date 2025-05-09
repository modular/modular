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
# RUN: %mojo %s

from pathlib import Path
from tempfile import NamedTemporaryFile

from max.graph import _testing
from max.graph.checkpoint import TensorDict, load, save
from max.tensor import Tensor, TensorShape


fn test_simple() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))

    with NamedTemporaryFile(name=String("test_simple")) as TEMP_FILE:
        save(tensors, TEMP_FILE.name)

        var loaded = load(TEMP_FILE.name)
        _testing.assert_equal(2, len(loaded))
        var x = loaded.get[DType.int32]("x")
        _testing.assert_tensors_equal(
            x,
            Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
        )

        var y = loaded.get[DType.float32]("y")
        _testing.assert_tensors_equal(
            y,
            Tensor[DType.float32](TensorShape(10, 5), -1.23),
        )


fn test_weird_keys() raises:
    var tensors = TensorDict()
    tensors.set("", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set(
        "key that's very long",
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )
    tensors.set(
        "another/key.with.some.symbols!",
        Tensor[DType.uint64](TensorShape(2), 123, 4567),
    )

    with NamedTemporaryFile(name=String("test_weird_keys")) as TEMP_FILE:
        save(tensors, TEMP_FILE.name)

        var loaded = load(TEMP_FILE.name)
        _testing.assert_equal(3, len(loaded))
        _testing.assert_tensors_equal(
            loaded.get[DType.int32](""),
            Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
        )

        _testing.assert_tensors_equal(
            loaded.get[DType.float32]("key that's very long"),
            Tensor[DType.float32](TensorShape(10, 5), -1.23),
        )

        _testing.assert_tensors_equal(
            loaded.get[DType.uint64]("another/key.with.some.symbols!"),
            Tensor[DType.uint64](TensorShape(2), 123, 4567),
        )


fn test_large_tensor() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(100, 200, 200), 567))

    with NamedTemporaryFile(name=String("test_large_tensor")) as TEMP_FILE:
        save(tensors, TEMP_FILE.name)

        var loaded = load(TEMP_FILE.name)
        _testing.assert_equal(1, len(loaded))
        var t = loaded.get[DType.int32]("x")
        _testing.assert_tensors_equal(
            t,
            Tensor[DType.int32](TensorShape(100, 200, 200), 567),
        )


def main():
    test_simple()
    test_weird_keys()
    test_large_tensor()
