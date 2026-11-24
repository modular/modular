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
# Tests "Call file-based libraries" in interop/libraries.mdx: the program
# loads ./libcustom from the working directory, and reports a failure it can
# catch when the library isn't there.
#
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: %if system-darwin %{ cp %S/libcustom.dylib %t.dir/ %} %else %{ cp %S/libcustom.so %t.dir/ %}
# RUN: rm -rf %t.empty && mkdir -p %t.empty
# RUN: %mojo-build %s -o %t.dir/main
# RUN: cd %t.dir && ./main | FileCheck %s
# RUN: cd %t.empty && %t.dir/main | FileCheck %s --check-prefix=MISSING

# CHECK: custom_abs returned 65: True
# MISSING: Failed to load library or missing function

from std.ffi import OwnedDLHandle, c_int
from std.sys.info import platform_map

# Use the full or relative (to the working directory) path to
# each library file
comptime LIBCUSTOM = platform_map[
    "libcustom", linux="./libcustom.so", macos="./libcustom.dylib"
]()


def main():
    try:
        var lib = OwnedDLHandle(LIBCUSTOM)  # returns the library handle
        var custom_abs = lib.get_function[c_int]("custom_abs")
        var n = custom_abs(c_int(-65))
        print(t"custom_abs returned 65: {n == 65}")  # True
    except:
        print("Failed to load library or missing function")
        return
