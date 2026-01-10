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


def string_slice_summary():
    from testing import assert_equal, assert_raises

    # Create a string slice
    var text = StringSlice("Hello, 世界")

    # Zero-copy slicing
    var hello = text[0:5]  # Hello

    # Unicode-aware operations
    var world = text[7:13]  # "世界"

    # String comparison
    if text.startswith("Hello"):
        print("Found greeting")

    # String formatting
    var format_string = StaticString("{}: {}")
    print(format_string.format("bats", 6))  # bats: 6

    # --------
    # Not in docstring:
    assert_equal(world, "世界")
    assert_equal(format_string.format("bats", 6), "bats: 6")
    # --------


def main():
    string_slice_summary()
