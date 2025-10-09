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

from os import stat
from stat import S_ISREG

from builtin._location import __source_location
from testing import TestSuite, assert_not_equal, assert_true


def test_stat():
    var st = stat(__source_location().file_name)
    assert_not_equal(String(st), "")
    assert_true(S_ISREG(st.st_mode))


def main():
    var suite = TestSuite()

    suite.test[test_stat]()

    suite^.run()
