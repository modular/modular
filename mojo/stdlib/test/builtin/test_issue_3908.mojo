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
# RUN: echo -n | %mojo %s

import sys

from builtin.io import _fdopen
from testing import testing


fn test_read_until_delimiter_raises_eof() raises:
    var stdin = _fdopen["r"](sys.stdin)
    with testing.assert_raises(contains="EOF"):
        # Assign to a variable to silence a warning about unused String value
        # if an error wasn't raised.
        var unused = stdin.read_until_delimiter("\n")


fn main() raises:
    test_read_until_delimiter_raises_eof()
