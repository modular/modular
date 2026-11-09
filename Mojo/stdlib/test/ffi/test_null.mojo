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

from std.ffi import _null
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def take_mut(ptr: Optional[MutPointer[Int, _]]) -> Bool:
    return Bool(ptr)


def take_imm(ptr: Optional[ImmPointer[Int, _]]) -> Bool:
    return Bool(ptr)


def take_unbound(ptr: Optional[Pointer[Int, _]]) -> Bool:
    return Bool(ptr)


def take_opaque(ptr: Optional[MutOpaquePointer[_]]) -> Bool:
    return Bool(ptr)


def take_generic[T: AnyType](ptr: Optional[MutPointer[T, _]]) -> Bool:
    return Bool(ptr)


def write_if_present(ptr: Optional[MutPointer[Int, _]]):
    if case .Some(data) = ptr:
        data[] = 42


def test_null_into_origin_inferring_params() raises:
    assert_false(take_mut(_null[Int]))
    assert_false(take_imm(_null[Int]))
    assert_false(take_unbound(_null[Int]))
    assert_false(take_opaque(_null[NoneType]))


def test_null_with_explicit_origin() raises:
    assert_false(take_imm(_null[Int, ImmUntrackedOrigin]))
    assert_false(take_unbound(_null[Int, ImmUntrackedOrigin]))


def test_null_with_dependent_pointee_type() raises:
    assert_false(take_generic(_null[Float64]))


def test_null_and_real_pointer_share_param() raises:
    var x = 5
    write_if_present(Pointer(to=x))
    assert_equal(x, 42)
    write_if_present(_null[Int])
    assert_equal(x, 42)


def test_mut_optional_pointer_weakens_to_imm() raises:
    var x = 7
    var mut_opt = Optional(Pointer(to=x))
    assert_true(take_imm(mut_opt))
    var weak: Optional[ImmPointer[Int, ImmOrigin(origin_of(x))]] = mut_opt
    assert_equal(weak.copied(), 7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
