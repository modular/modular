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
from testing import assert_equal, assert_raises, assert_true, TestSuite

from enumerated_errors import FileError, open_file


def test_open_file_success():
    assert_equal(open_file("/data"), "Contents of /data")


def test_not_found():
    with assert_raises(contains="not_found"):
        _ = open_file("")


def test_permission_denied():
    with assert_raises(contains="permission_denied"):
        _ = open_file("/secret")


def test_already_exists():
    with assert_raises(contains="already_exists"):
        _ = open_file("/existing")


def test_variant_equality():
    assert_true(FileError.not_found == FileError.not_found)
    assert_true(FileError.permission_denied == FileError.permission_denied)


def test_variant_inequality():
    assert_true(FileError.not_found != FileError.permission_denied)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
