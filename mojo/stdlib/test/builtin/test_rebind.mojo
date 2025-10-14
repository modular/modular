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

from testing import TestSuite


# ===----------------------------------------------------------------------=== #
# test_rebind_register
# ===----------------------------------------------------------------------=== #


fn test_rebind_reg[X: Int](a: SIMD[DType.int32, X]):
    print("there: ", rebind[SIMD[DType.int32, 4]](a))


def test_rebind_register():
    # CHECK-LABEL: test_rebind_memory
    print("test_rebind_memory")

    value = SIMD[DType.int32, 4](17)
    # CHECK-NEXT: here: [17, 17, 17, 17]
    print("here:", value)
    # CHECK-NEXT: there: [17, 17, 17, 17]
    test_rebind_reg(value)

    # CHECK-NEXT: done
    print("done")


# ===----------------------------------------------------------------------=== #
# test_rebind_memory
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct MyMemStruct[size: Int]:
    var value: Int

    fn __copyinit__(out self, existing: Self):
        # Make sure no copy is made due to the rebind.
        print("Should not copy this!")
        self.value = existing.value

    fn speak(self):
        print("hello, I am", size, "I hold", self.value)


fn indirect_with_rebind[X: Int](a: MyMemStruct[X]):
    rebind[MyMemStruct[4]](a).speak()


def test_rebind_memory():
    # CHECK-LABEL: test_rebind_memory
    print("test_rebind_memory")

    value = MyMemStruct[4](17)

    # CHECK-NEXT: hello, I am 4 I hold 17
    value.speak()
    # CHECK-NEXT: hello, I am 4 I hold 17
    indirect_with_rebind(value)

    # CHECK-NEXT: done
    print("done")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
