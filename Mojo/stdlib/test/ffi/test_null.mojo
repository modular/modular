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

from std.ffi import ImmNull, MutNull, Null, external_call
from std.testing import TestSuite, assert_equal, assert_false, assert_true


def _takes_mut(ptr: Optional[MutPointer[Int, _]]) -> Bool:
    return Bool(ptr)


def _takes_imm(ptr: Optional[ImmPointer[Int, _]]) -> Bool:
    return Bool(ptr)


def _takes_any_mut(ptr: Optional[Pointer[Int, _]]) -> Bool:
    return Bool(ptr)


def _takes_void(ptr: Optional[MutOpaquePointer[_]]) -> Bool:
    return Bool(ptr)


def _takes_any_void(ptr: Optional[Pointer[NoneType, _]]) -> Bool:
    return Bool(ptr)


def test_types() raises:
    comptime assert (
        type_of(MutNull[Int]) == Optional[Pointer[Int, MutUntrackedOrigin]]
    )
    comptime assert (
        type_of(ImmNull[Int]) == Optional[Pointer[Int, ImmUntrackedOrigin]]
    )
    comptime assert type_of(Null[mut=True, Int]) == type_of(MutNull[Int])
    comptime assert type_of(Null[mut=False, Int]) == type_of(ImmNull[Int])
    comptime assert (
        type_of(Null[Int, MutAnyOrigin]) == Optional[Pointer[Int, MutAnyOrigin]]
    )
    comptime assert (
        type_of(MutNull[Int, address_space=AddressSpace.SHARED])
        == Optional[
            Pointer[Int, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
        ]
    )


def test_mut_pointer_param() raises:
    assert_false(_takes_mut(MutNull[Int]))


def test_imm_pointer_param() raises:
    assert_false(_takes_imm(ImmNull[Int]))


def test_any_mut_pointer_param() raises:
    assert_false(_takes_any_mut(MutNull[Int]))
    assert_false(_takes_any_mut(ImmNull[Int]))
    assert_false(_takes_any_mut(Null[mut=True, Int]))
    assert_false(_takes_any_mut(Null[Int, ImmutAnyOrigin]))


def test_void_pointer_param() raises:
    assert_false(_takes_void(MutNull[NoneType]))
    assert_false(_takes_any_void(MutNull[NoneType]))
    assert_false(_takes_any_void(ImmNull[NoneType]))


def test_is_empty() raises:
    var mut_ptr = MutNull[Int]
    assert_true(mut_ptr is None)
    assert_false(mut_ptr is not None)
    assert_true(not mut_ptr)
    assert_false(Bool(mut_ptr))

    var imm_ptr = ImmNull[Int]
    assert_true(imm_ptr is None)
    assert_true(not imm_ptr)

    comptime assert not MutNull[Int]
    comptime assert not ImmNull[Int]


def test_is_zero_address() raises:
    # A null `Optional[Pointer]` is a zero address, which is what C reads.
    var ptr = MutNull[NoneType]
    assert_equal(Pointer(to=ptr).unsafe_bitcast[Int]()[], 0)


def test_declared_variable() raises:
    var value = 42
    var ptr: Optional[MutPointer[Int, origin_of(value)]] = Null[
        Int, origin_of(value)
    ]
    assert_true(ptr is None)
    ptr = Pointer(to=value)
    assert_equal(ptr.value()[], 42)


def test_external_call() raises:
    # `time_t time(time_t *tloc)` only stores the time when `tloc` isn't null.
    var now = external_call["time", Int](MutNull[NoneType])
    assert_true(now > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
