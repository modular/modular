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

from memory import ArcPointer
from test_utils import ObservableDel
from testing import assert_equal, assert_false, assert_true, TestSuite


def test_basic():
    var p = ArcPointer(4)
    var p2 = p
    p2[] = 3
    assert_equal(3, p[])


def test_is():
    var p = ArcPointer(3)
    var p2 = p
    var p3 = ArcPointer(3)
    assert_true(p is p2)
    assert_false(p is not p2)
    assert_false(p is p3)
    assert_true(p is not p3)


def test_deleter_not_called_until_no_references():
    var deleted = False
    var p = ArcPointer(ObservableDel(UnsafePointer(to=deleted).as_immutable()))
    var p2 = p
    _ = p^
    assert_false(deleted)

    var vec = List[__type_of(p)]()
    vec.append(p2)
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_deleter_not_called_until_no_references_explicit_copy():
    var deleted = False
    var p = ArcPointer(ObservableDel(UnsafePointer(to=deleted).as_immutable()))
    var p2 = p.copy()
    _ = p^
    assert_false(deleted)

    var vec = List[__type_of(p)]()
    vec.append(p2.copy())
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_count():
    var a = ArcPointer(10)
    var b = a.copy()
    var c = a
    assert_equal(UInt64(3), a.count())
    _ = b^
    assert_equal(UInt64(2), a.count())
    _ = c
    assert_equal(UInt64(1), a.count())


def test_steal_data_and_construct_from_raw_ptr():
    var deleted = False
    var leaked = ArcPointer(
        ObservableDel(UnsafePointer(to=deleted).as_immutable())
    )

    var raw = leaked^.steal_data()
    assert_false(deleted)

    var p = ArcPointer(unsafe_from_raw_pointer=raw)
    _ = p^
    assert_true(deleted)


def test_steal_data_does_not_decrement_refcount():
    var leaked = ArcPointer(42)
    var copy = leaked.copy()

    assert_equal(UInt64(2), copy.count())
    var raw = leaked^.steal_data()

    # since we leaked the data, the refcount was not decremented
    assert_equal(UInt64(2), copy.count())

    _ = copy^

    var p = ArcPointer(unsafe_from_raw_pointer=raw)
    assert_equal(UInt64(1), p.count())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
