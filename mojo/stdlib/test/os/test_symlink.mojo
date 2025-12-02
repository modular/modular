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

import os
from os.path import exists
from pathlib import Path

from testing import TestSuite, assert_equal


def test_create_symlink():
    os.symlink("test_create_symlink", "test_create_symlink_symlink")
    with open("test_create_symlink", "w") as f:
        f.write("test_create_symlink")
    with open("test_create_symlink_symlink", "r") as f:
        assert_equal(f.read(), "test_create_symlink")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
