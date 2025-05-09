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
from os.path import isdir, islink
from pathlib import Path
from sys import env_get_string

from testing import assert_false, assert_true

alias TEMP_DIR = os.getenv("TEST_TMPDIR")


def main():
    assert_true(isdir(Path(TEMP_DIR)))
    assert_true(isdir(TEMP_DIR))
    assert_true(islink(TEMP_DIR))
    assert_false(islink(String(Path(TEMP_DIR) / "nonexistent")))
