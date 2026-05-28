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

from std.testing import TestSuite, assert_equal, assert_raises
from std.iter import once_with


def _create_value() -> Int:
    return 5


def test_once_with() raises:
    var it = once_with[_create_value]()
    assert_equal(next(it), 5)
    with assert_raises():
        _ = next(it)
<<<<<<< HEAD


def test_once_with_bounds() raises:
    var it = once_with[_create_value]()
=======
>>>>>>> 15cf47cd52 (Made function pointer a compile-time parameter)

    var lower, upper = it.bounds()
    assert_equal(lower, 1)
    assert_equal(upper, Optional(1))

<<<<<<< HEAD
    var _ = next(it)
=======
def test_once_with_iter() raises:
    var it = once_with[_create_value]()
    var it_copy = iter(it)
>>>>>>> 15cf47cd52 (Made function pointer a compile-time parameter)

    lower, upper = it.bounds()
    assert_equal(lower, 0)
    assert_equal(upper, Optional(0))


def test_once_with_bounds() raises:
    var it = once_with[_create_value]()

    var lower, upper = it.bounds()
    assert_equal(lower, 1)
    assert_equal(upper, Optional(1))

    var _ = next(it)

    lower, upper = it.bounds()
    assert_equal(lower, 0)
    assert_equal(upper, Optional(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
