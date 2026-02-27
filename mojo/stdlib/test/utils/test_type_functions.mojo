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

from std.sys.intrinsics import _type_is_eq

from std.utils.type_functions import ConditionalType
from std.testing import assert_true, assert_false, TestSuite


def test_conditional_type_with_bool_literal() raises:
    comptime IsInt = ConditionalType[
        Trait=AnyType, If=True, Then=Int, Else=String
    ]
    comptime IsString = ConditionalType[
        Trait=AnyType, If=False, Then=Int, Else=String
    ]

    assert_true(_type_is_eq[IsInt, Int]())
    assert_true(_type_is_eq[IsString, String]())


def test_conditional_type_with_bool_function() raises:
    fn bool[b: Bool]() -> Bool:
        return b

    comptime IsInt = ConditionalType[
        Trait=AnyType, If = bool[True](), Then=Int, Else=String
    ]
    comptime IsString = ConditionalType[
        Trait=AnyType, If = bool[False](), Then=Int, Else=String
    ]

    assert_true(_type_is_eq[IsInt, Int]())
    assert_true(_type_is_eq[IsString, String]())


def test_conditional_type_nested() raises:
    comptime IsInt = ConditionalType[
        Trait=AnyType, If=True, Then=Int, Else=String
    ]
    comptime IsString = ConditionalType[
        Trait=AnyType, If=False, Then=Int, Else=String
    ]
    comptime FinalType = ConditionalType[
        Trait=AnyType, If=True, Then=IsInt, Else=IsString
    ]

    assert_true(_type_is_eq[FinalType, Int]())


def test_conditional_type_same_types() raises:
    comptime Int1 = ConditionalType[Trait=AnyType, If=True, Then=Int, Else=Int]
    comptime Int2 = ConditionalType[Trait=AnyType, If=False, Then=Int, Else=Int]

    assert_true(_type_is_eq[Int1, Int]())
    assert_true(_type_is_eq[Int2, Int]())


def test_conditional_type_ternary_tree() raises:
    comptime PickAType[outer: Bool, inner: Bool] = ConditionalType[
        Trait=AnyType,
        If=outer,
        Then = ConditionalType[Trait=AnyType, If=inner, Then=Int, Else=Float64],
        Else = ConditionalType[
            Trait=AnyType, If=inner, Then=String, Else = List[Int]
        ],
    ]

    assert_true(_type_is_eq[PickAType[True, True], Int]())
    assert_true(_type_is_eq[PickAType[True, False], Float64]())
    assert_true(_type_is_eq[PickAType[False, True], String]())
    assert_true(_type_is_eq[PickAType[False, False], List[Int]]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
