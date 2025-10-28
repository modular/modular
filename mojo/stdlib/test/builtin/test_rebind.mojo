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

from testing import TestSuite, assert_equal
from test_utils import MoveCopyCounter, DelCounter


# ===----------------------------------------------------------------------=== #
# test_rebind_register
# ===----------------------------------------------------------------------=== #


fn indirect_rebind_reg[X: Int](a: SIMD[DType.int32, X]) -> String:
    return String(rebind[SIMD[DType.int32, 4]](a))


def test_rebind_register():
    var value = SIMD[DType.int32, 4](17)

    var string = indirect_rebind_reg(value)
    assert_equal(string, "[17, 17, 17, 17]")


# ===----------------------------------------------------------------------=== #
# test_rebind_memory
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct MyMemStruct[size: Int](Movable):
    var value: Int

    fn sizes(self) -> Tuple[Int, Int]:
        return (size, self.value)


fn indirect_with_rebind[X: Int](a: MyMemStruct[X]) -> Tuple[Int, Int]:
    return rebind[MyMemStruct[4]](a).sizes()


def test_rebind_memory():
    var mem = MyMemStruct[4](17)

    var size, value = indirect_with_rebind(mem)
    assert_equal(size, 4)
    assert_equal(value, 17)


# ===----------------------------------------------------------------------=== #
# rebind_var
# ===----------------------------------------------------------------------=== #


fn indirect_with_rebind_var[x: Int](var a: MyMemStruct[x]) -> MyMemStruct[4]:
    return rebind_var[MyMemStruct[4]](a^)


def test_rebind_var():
    var value = MyMemStruct[4](17)
    var rebound = indirect_with_rebind_var(value^)
    assert_equal(rebound.size, 4)
    assert_equal(rebound.value, 17)


def test_rebind_var_does_not_copy_only_moves():
    var counter = MoveCopyCounter()
    var rebound = rebind_var[MoveCopyCounter](counter^)
    assert_equal(rebound.copied, 0)
    assert_equal(rebound.moved, 1)


def test_rebind_does_not_call_del():
    var n_dels = 0
    var counter = DelCounter(UnsafePointer(to=n_dels))
    var rebound = rebind_var[type_of(counter)](counter^)
    assert_equal(n_dels, 0)
    _ = rebound
    assert_equal(n_dels, 1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
