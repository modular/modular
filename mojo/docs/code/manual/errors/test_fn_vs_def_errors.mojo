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
from testing import assert_equal, assert_raises, TestSuite

from fn_vs_def_errors import validate_def, validate_fn, validated_operation


def test_validate_def_pass():
    assert_equal(validate_def(5), 5)


def test_validate_def_fail():
    with assert_raises(contains="cannot be negative"):
        _ = validate_def(-1)


def test_validate_fn_pass():
    assert_equal(validate_fn(5), 5)


def test_validate_fn_fail():
    with assert_raises(contains="cannot be negative"):
        _ = validate_fn(-1)


def test_validate_fn_field_access():
    try:
        _ = validate_fn(-1)
    except e:
        assert_equal(e.field, "value")
        assert_equal(e.reason, "cannot be negative")


def test_wrapper_pass():
    assert_equal(validated_operation(5), 5)


def test_wrapper_fail():
    with assert_raises(contains="cannot be negative"):
        _ = validated_operation(-1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
