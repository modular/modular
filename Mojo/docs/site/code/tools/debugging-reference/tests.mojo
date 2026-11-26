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
# tests.mojo
# Tests for debugging-reference.mdx code examples.
#
# Not tested here (tool behavior; see test_debug_reference.sh):
#   - The mojo debug command, debug information levels, stack traces,
#     MODULAR_DEBUG, LLDB with Mojo, VS Code, environment variables, and
#     platform notes.
from std.sys import is_defined
from std.testing import assert_equal, assert_false

# --- Programmatic breakpoints ---


def bin_index(value: UInt8, bin_count: Int) -> Int:
    var index = Int(value) * bin_count // 256
    comptime if is_defined["BREAK_ON_BAD_BIN"]():
        if index >= bin_count:
            breakpoint()
    return index


def test_guarded_breakpoint_not_compiled() raises:
    # Without -D BREAK_ON_BAD_BIN, the guard compiles to nothing.
    assert_false(is_defined["BREAK_ON_BAD_BIN"]())
    assert_equal(bin_index(255, 4), 3)
    assert_equal(bin_index(0, 4), 0)


def main() raises:
    test_guarded_breakpoint_not_compiled()
    print("debug reference tests passed")
