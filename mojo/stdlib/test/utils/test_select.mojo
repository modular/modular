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

from utils._select import _select_register_value


def test_select_register_value():
    assert_equal(_select_register_value(True, 42, 100), 42)
    assert_equal(_select_register_value(False, 42, 100), 100)


def main():
    var suite = TestSuite()

    suite.test[test_select_register_value]()

    suite^.run()
