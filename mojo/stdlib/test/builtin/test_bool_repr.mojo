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

# ===----------------------------------------------------------------------=== #
# Focused tests for Bool repr
# ===----------------------------------------------------------------------=== #

from testing import assert_equal, TestSuite


def test_bool_repr_true():
    assert_equal(repr(True), "True")


def test_bool_repr_false():
    assert_equal(repr(False), "False")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
