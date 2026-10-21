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
"""Tests for reading `@__annotation` values back through `reflect[T]`."""

from std.testing import TestSuite, assert_equal


@__annotation("a")
@__annotation("b", 2)
struct Target:
    @__annotation(3)
    var value: String
    var plain: Int


@__annotation(Self.N, Self.N + 1)
struct Param[N: Int]:
    @__annotation(Self.N * 2)
    var value: Int


struct Plain:
    var value: Int


@fieldwise_init
struct Tag(Deinitable, Movable):
    var name: StaticString


@fieldwise_init
struct Rename(Deinitable, Movable):
    var to: StaticString


@__annotation(1)
@__annotation(Tag("first"), Tag("second"))
struct Tagged:
    @__annotation(Rename("renamed"), 7)
    var value: String


# ===----------------------------------------------------------------------=== #
# Struct annotations
# ===----------------------------------------------------------------------=== #


def test_annotation_types_are_materialized() raises:
    comptime types = type_of(reflect[Target].annotations()).Ts
    assert_equal(types.length, 3)
    comptime assert types[0] == String
    comptime assert types[1] == String
    comptime assert types[2] == Int


def test_annotations_empty() raises:
    # A struct with no annotations reads as an empty tuple rather than failing.
    assert_equal(len(reflect[Plain].annotations()), 0)
    assert_equal(type_of(reflect[Plain].annotations()).Ts.length, 0)


def test_annotations_tuple() raises:
    var values = reflect[Target].annotations()
    assert_equal(len(values), 3)
    assert_equal(values[0], "a")
    assert_equal(values[1], "b")
    assert_equal(values[2], 2)


# ===----------------------------------------------------------------------=== #
# Field annotations
# ===----------------------------------------------------------------------=== #


def test_field_annotation_types() raises:
    comptime types = type_of(reflect[Target].field_annotations[0]()).Ts
    assert_equal(types.length, 1)
    comptime assert types[0] == Int


def test_field_annotations_tuple() raises:
    var values = reflect[Target].field_annotations[0]()
    assert_equal(len(values), 1)
    assert_equal(values[0], 3)
    # An undecorated field reads as an empty tuple rather than failing.
    assert_equal(len(reflect[Target].field_annotations[1]()), 0)
    assert_equal(len(reflect[Plain].field_annotations[0]()), 0)


def test_user_struct_annotation_values() raises:
    # A constructor call is stored unevaluated and folded on a concrete read.
    assert_equal(reflect[Tagged].annotations()[1].name, "first")
    assert_equal(reflect[Tagged].field_annotations[0]()[0].to, "renamed")


# ===----------------------------------------------------------------------=== #
# Struct parameters and generic code
# ===----------------------------------------------------------------------=== #


def test_annotation_binds_struct_parameters() raises:
    # Annotations are written in the struct's own scope, so each
    # instantiation reads back its own parameter values.
    assert_equal(reflect[Param[3]].annotations()[0], 3)
    assert_equal(reflect[Param[3]].annotations()[1], 4)
    assert_equal(reflect[Param[5]].annotations()[0], 5)
    assert_equal(reflect[Param[5]].field_annotations[0]()[0], 10)


def _first_of[T: AnyType]() -> Int:
    # The element's type depends on `T`, so generic code names it explicitly.
    return rebind[Int](reflect[T].annotations()[0])


def _field_first_of[T: AnyType, field_index: Int]() -> Int:
    return rebind[Int](reflect[T].field_annotations[field_index]()[0])


def _tuple_len[T: AnyType]() -> Int:
    var values = reflect[T].annotations()
    return len(values)


def test_annotation_through_generic() raises:
    assert_equal(_first_of[Param[7]](), 7)
    assert_equal(_field_first_of[Param[7], 0](), 14)
    assert_equal(_field_first_of[Target, 0](), 3)
    assert_equal(_tuple_len[Param[3]](), 2)
    assert_equal(_tuple_len[Plain](), 0)


def test_annotation_tuple_through_generic() raises:
    assert_equal(_tuple_len[Target](), 3)


def test_annotation_iteration() raises:
    comptime types = type_of(reflect[Target].annotations()).Ts
    var names = List[StaticString]()
    comptime for i in range(types.length):
        names.append(reflect[types[i]].name())
    assert_equal(len(names), 3)
    assert_equal(names[0], reflect[String].name())
    assert_equal(names[1], reflect[String].name())
    assert_equal(names[2], reflect[Int].name())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
