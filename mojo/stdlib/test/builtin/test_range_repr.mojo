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

from testing import assert_equal, TestSuite


def test_range_repr_normal():
    r = range(2, 5)
    assert_equal(repr(r), "range(2, 5)")


def test_range_repr_empty():
    r = range(3, 3)
    assert_equal(repr(r), "range(3, 3)")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
