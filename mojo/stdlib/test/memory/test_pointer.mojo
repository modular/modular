# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from memory import AddressSpace
from test_utils import check_write_to
from testing import TestSuite
from testing import assert_equal, assert_not_equal, assert_true


def test_copy_reference_explicitly():
    var a = [1, 2, 3]

    var b = Pointer(to=a)
    var c = b.copy()

    c[][0] = 4
    assert_equal(a[0], 4)
    assert_equal(b[][0], 4)
    assert_equal(c[][0], 4)


def test_equality():
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    assert_true(Pointer(to=a) == Pointer(to=a))
    assert_true(Pointer(to=b) == Pointer(to=b))
    assert_true(Pointer(to=a) != Pointer(to=b))


def test_str():
    var a = Int(42)
    var a_ref = Pointer(to=a)
    assert_true(String(a_ref).startswith("0x"))


def test_write_to():
    var a = Int(42)
    check_write_to(Pointer(to=a), contains="0x", is_repr=False)
    var s = String("hello")
    check_write_to(Pointer(to=s), contains="0x", is_repr=False)


def test_write_repr_to():
    var n = Int(42)
    check_write_to(
        Pointer(to=n),
        contains=(
            "Pointer[mut=True, Int, address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        Pointer(to=n).get_immutable(),
        contains=(
            "Pointer[mut=False, Int, address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )

    var s = String("hello")
    check_write_to(
        Pointer(to=s),
        contains=(
            "Pointer[mut=True, String, address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )


comptime ADDRESS_SPACE_STRINGS = [
    (AddressSpace.GENERIC, "AddressSpace.GENERIC"),
    (AddressSpace.GLOBAL, "AddressSpace.GLOBAL"),
    (AddressSpace.SHARED, "AddressSpace.SHARED"),
    (AddressSpace.CONSTANT, "AddressSpace.CONSTANT"),
    (AddressSpace.LOCAL, "AddressSpace.LOCAL"),
    (AddressSpace.SHARED_CLUSTER, "AddressSpace.SHARED_CLUSTER"),
    (AddressSpace(42), "AddressSpace(42)"),
]


def test_address_space_write_to():
    for address_space, expected in materialize[ADDRESS_SPACE_STRINGS]():
        check_write_to(address_space, expected=expected, is_repr=False)


def test_address_space_write_repr_to():
    for address_space, expected in materialize[ADDRESS_SPACE_STRINGS]():
        check_write_to(address_space, expected=expected, is_repr=True)


def test_pointer_to():
    var local = 1
    assert_not_equal(0, Pointer(to=local)[])


# Test pointer merging with ternary operation.
def test_merge():
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    fn inner(cond: Bool, x: Int, mut a: List[Int], mut b: List[Int]):
        var either = Pointer(to=a) if cond else Pointer(to=b)
        either[].append(x)

    inner(True, 7, a, b)
    inner(False, 8, a, b)

    assert_equal(a, [1, 2, 3, 7])
    assert_equal(b, [4, 5, 6, 8])


# We don't actually need to run this,
# but Mojo's exclusivity check shouldn't complain
def _test_get_immutable() -> Int:
    fn foo(x: Pointer[mut=False, Int], y: Pointer[mut=False, Int]) -> Int:
        return x[]

    var x = Int(0)
    return foo(Pointer(to=x), Pointer(to=x))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
