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

from os.path import isfile
from pathlib import Path

from builtin._location import __source_location
from testing import TestSuite, assert_false, assert_true


def test_isfile():
    assert_true(isfile(__source_location().file_name))
    assert_false(isfile("this/file/does/not/exist"))

    assert_false(isfile(Path()))


def main():
    var suite = TestSuite()

    suite.test[test_isfile]()

    suite^.run()
