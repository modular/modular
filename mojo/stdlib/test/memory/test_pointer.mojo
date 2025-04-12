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
from testing import assert_equal, assert_not_equal, assert_true


def test_copy_reference_explicitly():
    var a = List[Int](1, 2, 3)

    var b = Pointer(to=a)
    var c = b.copy()

    c[][0] = 4
    assert_equal(a[0], 4)
    assert_equal(b[][0], 4)
    assert_equal(c[][0], 4)


def test_equality():
    var a = List[Int](1, 2, 3)
    var b = List[Int](4, 5, 6)

    assert_true(Pointer(to=a) == Pointer(to=a))
    assert_true(Pointer(to=b) == Pointer(to=b))
    assert_true(Pointer(to=a) != Pointer(to=b))


def test_str():
    var a = Int(42)
    var a_ref = Pointer(to=a)
    assert_true(String(a_ref).startswith("0x"))


def test_pointer_to():
    var local = 1
    assert_not_equal(0, Pointer(to=local)[])


# We don't actually need to run this,
# but Mojo's exclusivity check shouldn't complain
def test_get_immutable() -> Int:
    fn foo(x: Pointer[mut=False, Int], y: Pointer[mut=False, Int]) -> Int:
        return x[]

    var x = Int(0)
    return foo(Pointer(to=x), Pointer(to=x))


def main():
    test_copy_reference_explicitly()
    test_equality()
    test_str()
    test_pointer_to()
