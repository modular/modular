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

import compiler
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)


@compiler.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        x = in_place_tensor._ptr.load(0)
        x += 1
        in_place_tensor._ptr.store(0, x)


@compiler.register("foo")
struct FooKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


@compiler.register("bar")
struct BarKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


@compiler.register("baz")
struct BazKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)
