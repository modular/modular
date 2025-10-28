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


from testing import assert_equal, assert_false, assert_true, TestSuite


fn test_iterator() raises:
    fn helper(*args: Int) raises:
        var n = 5
        var count = 0

        for i, e in enumerate(args):
            assert_equal(e, n)
            assert_equal(i, count)
            count += 1
            n -= 1

    helper(5, 4, 3, 2, 1)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
