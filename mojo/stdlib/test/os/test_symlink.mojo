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

import os
from os import remove
from os.path import exists
from pathlib import Path
from tempfile import gettempdir
from testing import TestSuite, assert_equal


def test_create_symlink():
    var tempdir = Path(gettempdir().value())
    var src = tempdir / "test_create_symlink"
    var link = tempdir / "test_create_symlink_symlink"

    # Clean up any leftover files from previous runs
    try:
        remove(link)
    except:
        pass
    try:
        remove(src)
    except:
        pass

    os.symlink(src, link)
    with open(src, "w") as f:
        f.write("test_create_symlink")
    with open(link, "r") as f:
        assert_equal(f.read(), "test_create_symlink")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
