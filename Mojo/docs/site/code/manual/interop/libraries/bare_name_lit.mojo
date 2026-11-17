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
# Tests the bare-name rules in "Library handles and file paths" in
# interop/libraries.mdx. With libcustom in the working directory, a bare name
# finds it on macOS but not on Linux. Adding the directory to
# LD_LIBRARY_PATH or DYLD_LIBRARY_PATH makes the bare name work on both.
#
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: %if system-darwin %{ cp %S/libcustom.dylib %t.dir/ %} %else %{ cp %S/libcustom.so %t.dir/ %}
# RUN: %mojo-build %s -o %t.dir/main
# RUN: cd %t.dir && %if system-darwin %{ ./main | FileCheck %s --check-prefix=FOUND %} %else %{ ./main | FileCheck %s --check-prefix=NOTFOUND %}
# RUN: cd %t.dir && %if system-darwin %{ env DYLD_LIBRARY_PATH=. ./main | FileCheck %s --check-prefix=FOUND %} %else %{ env LD_LIBRARY_PATH=. ./main | FileCheck %s --check-prefix=FOUND %}

# FOUND: opened libcustom
# NOTFOUND: libcustom not found

from std.ffi import OwnedDLHandle
from std.sys.info import platform_map

comptime BARE_NAME = platform_map[
    "libcustom", linux="libcustom.so", macos="libcustom.dylib"
]()


def main():
    try:
        _ = OwnedDLHandle(BARE_NAME)
        print("opened libcustom")
    except:
        print("libcustom not found")
