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
# RUN: %mojo -debug-level full %s

from max.engine import ShapeElement
from testing import assert_equal, assert_false, assert_true


fn test_shape_element() raises:
    var static3: ShapeElement = 3
    var static5: ShapeElement = 5
    var unnamed: ShapeElement = None
    var name_a: ShapeElement = "a"
    var name_b: ShapeElement = "b"

    # Testing is_* methods.
    assert_true(static3.is_static())
    assert_false(static3.is_dynamic())
    assert_false(static3.is_unnamed_dynamic())
    assert_false(static3.is_named_dynamic())

    assert_false(unnamed.is_static())
    assert_true(unnamed.is_dynamic())
    assert_true(unnamed.is_unnamed_dynamic())
    assert_false(unnamed.is_named_dynamic())

    assert_false(name_a.is_static())
    assert_true(name_a.is_dynamic())
    assert_false(name_a.is_unnamed_dynamic())
    assert_true(name_a.is_named_dynamic())

    # Testing accessors.
    assert_equal(static3.static_value(), 3)
    assert_equal(static5.static_value(), 5)
    assert_equal(unnamed.static_value(), 0)
    assert_equal(name_a.static_value(), 0)

    assert_equal(static3.name(), "")
    assert_equal(unnamed.name(), "")
    assert_equal(name_a.name(), "a")
    assert_equal(name_b.name(), "b")

    # Testing __eq__/__ne__.
    assert_true(static3 == static3)
    assert_false(static3 != static3)
    assert_false(static3 == static5)
    assert_false(static3 == unnamed)
    assert_false(static3 == name_a)

    assert_true(unnamed == unnamed)
    assert_true(name_a == name_a)
    assert_false(name_a == name_b)

    # Testing copy & move init.
    var name_a_copy = name_a
    assert_true(name_a == name_a_copy)
    var name_b_moved = name_b^
    assert_true(name_b_moved.is_named_dynamic())
    assert_equal(name_b_moved.name(), "b")


fn main() raises:
    test_shape_element()
