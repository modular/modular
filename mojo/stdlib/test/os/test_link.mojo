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
from pathlib import Path


from testing import TestSuite, assert_equal


def test_create_hardlink():
    with open("test_create_link", "w") as f:
        f.write("test_create_link")
    os.link("test_create_link", "test_create_link_link")
    with open("test_create_link_link", "r") as f:
        assert_equal(f.read(), "test_create_link")
    var oldstat = os.stat("test_create_link")
    var newstat = os.stat("test_create_link_link")
    assert_equal(oldstat.st_ino, newstat.st_ino)
    assert_equal(oldstat.st_nlink, 2)
    assert_equal(newstat.st_nlink, 2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
