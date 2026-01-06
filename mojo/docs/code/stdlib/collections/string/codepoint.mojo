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


def summary():
    from collections.string import Codepoint
    from testing import assert_true

    # Create a codepoint from a character
    var c = Codepoint.ord("A")

    # Check properties
    assert_true(c.is_ascii())
    assert_true(c.is_ascii_upper())

    # Convert to string
    var s = String(c)  # "A"


def run_is_python_space():
    from testing import assert_true, assert_false

    # ASCII space characters
    assert_true(Codepoint.ord(" ").is_python_space())
    assert_true(Codepoint.ord("	").is_python_space())

    # Unicode paragraph separator:
    assert_true(Codepoint.from_u32(0x2029).value().is_python_space())

    # Letters are not space characters
    assert_false(Codepoint.ord("a").is_python_space())


def main():
    summary()
    run_is_python_space()
