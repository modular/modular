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

from testing import assert_equal


fn test_map() raises:
    var l = [1, 2, 3]

    fn add_one(x: Int) -> Int:
        return x + 1

    var m = map[add_one](l)
    assert_equal(next(m), 2)
    assert_equal(next(m), 3)
    assert_equal(next(m), 4)
    assert_equal(m.__has_next__(), False)

    fn to_str(x: Int) -> String:
        return String(x)

    var m2 = map[to_str](l)
    assert_equal(next(m2), "1")
    assert_equal(next(m2), "2")
    assert_equal(next(m2), "3")
    assert_equal(m2.__has_next__(), False)


fn test_map_captures():
    pass


def main():
    test_map()
