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

from os import abort
from testing import assert_raises, TestSuite


def test_nonconforming_signature(x: Int):
    abort("should not be run")


def nonconforming_name():
    abort("should not be run")


def test_failing():
    raise Error("should be raised")


def main():
    # TODO(MSTDL-1916): Capture the output of TestSuite.run() and test that it
    # contains the expected error message, and change the above functions to
    # raise instead of aborting.
    with assert_raises():
        TestSuite.discover_tests[__functions_in_module()]().run()
