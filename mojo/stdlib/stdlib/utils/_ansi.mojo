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


@fieldwise_init
struct Color(ImplicitlyCopyable, Movable, Writable):
    """ANSI colors for terminal output."""

    var color: StaticString

    alias RED = Self("\033[91m")
    alias GREEN = Self("\033[92m")
    alias YELLOW = Self("\033[93m")
    alias BLUE = Self("\033[94m")
    alias MAGENTA = Self("\033[95m")
    alias CYAN = Self("\033[96m")
    alias BOLD_WHITE = Self("\033[1;97m")
    alias END = Self("\033[0m")

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)


@fieldwise_init
struct Text[W: Writable, origin: ImmutableOrigin, //, color: Color](Writable):
    """Colors the given writable with the given `Color`."""

    var writable: Pointer[W, origin]

    fn __init__(out self, ref [origin]w: W):
        self.writable = Pointer(to=w)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)
        self.writable[].write_to(writer)
        writer.write(Color.END)
