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

from testing import TestSuite, assert_equal
from utils import ReadOnly


fn get_runtime_value() -> String:
    # simulate a function that gets a runtime value
    return "user input"


def test_read_only_default_usage():
    ref value = ReadOnly(get_runtime_value())[]
    # value = ""  # compile-time error

    assert_equal(
        value,
        "user input",
        msg="failed to use runtime constant value",
    )


def test_read_only_non_mutable():
    var runtime_value = get_runtime_value()
    ref value = ReadOnly(runtime_value)[]
    runtime_value = ""  # does not affect value
    # value = ""  # compile-time error

    assert_equal(
        value,
        "user input",
        msg="failed to use runtime constant value",
    )
    assert_equal(
        runtime_value,
        "",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
