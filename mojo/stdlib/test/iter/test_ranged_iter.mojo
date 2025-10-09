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


from testing import TestSuite, assert_true, assert_equal
from stdlib.collections.string.string_slice import (
    _SplitlinesIter,
    _RangedIteratorWrapper,
    _to_string_list,
)


fn test_ranged_iter() raises:
    var data = "0\n1\n2\n3\n4\n5"
    var iterator = data.splitlines()
    assert_equal(
        _to_string_list(iterator.copy()[0:6:1]),
        ["0", "1", "2", "3", "4", "5"],
    )
    assert_equal(
        _to_string_list(iterator.copy()[::]),
        ["0", "1", "2", "3", "4", "5"],
    )
    assert_equal(_to_string_list(iterator.copy()[0:2]), ["0", "1"])
    assert_equal(_to_string_list(iterator.copy()[1:3]), ["1", "2"])
    assert_equal(
        _to_string_list(iterator.copy()[1:6]), ["1", "2", "3", "4", "5"]
    )
    assert_equal(_to_string_list(iterator.copy()[1:6:2]), ["1", "3", "5"])
    assert_equal(_to_string_list(iterator.copy()[0:6:2]), ["0", "2", "4"])
    assert_equal(_to_string_list(iterator.copy()[0:6:3]), ["0", "3"])
    assert_equal(_to_string_list(iterator.copy()[1:6:3]), ["1", "4"])
    # once we make it general enough so that the inner_iter type can be itself
    # assert_equal(_to_string_list(iterator.copy()[1:2:3][:]), ["1"])
    # assert_equal(_to_string_list(iterator.copy()[1:6:3][:1]), ["1"])
    # assert_equal(_to_string_list(iterator.copy()[1:6:2][::2]), ["0", "4"])


def main():
    var suite = TestSuite()

    suite.test[test_ranged_iter]()

    suite^.run()
