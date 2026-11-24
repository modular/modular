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
# Tests "Call runtime lookups" in interop/libraries.mdx: link libcustom with
# the page's options (including --no-as-needed on Linux), then look up
# custom_abs() through a handle to the process. Without linking, the lookup
# raises an error the program catches.
#
# COM: Not tested: a Linux linker that defaults to --as-needed dropping the
# COM:   library. That default depends on the distribution's toolchain; the
# COM:   hermetic test linker doesn't use it.
#
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: %if system-darwin %{ cp %S/libcustom.dylib %t.dir/ %} %else %{ cp %S/libcustom.so %t.dir/ %}
# RUN: cd %t.dir && %if system-darwin %{ %mojo-build %s -o main -Xlinker -L. -Xlinker -lcustom -Xlinker -rpath -Xlinker @loader_path %} %else %{ %mojo-build %s -o main -Xlinker -L. -Xlinker --no-as-needed -Xlinker -lcustom -Xlinker -rpath -Xlinker '$ORIGIN' %}
# RUN: cd / && %t.dir/main | FileCheck %s
# RUN: %mojo-build %s -o %t.dir/unlinked
# RUN: %t.dir/unlinked | FileCheck %s --check-prefix=MISSING

# CHECK: custom_abs returned 65: True
# MISSING: Failed to find symbol 'custom_abs'

from std.ffi import OwnedDLHandle, c_int


def main():
    try:
        # No path: a handle to the libraries already in the process
        var process = (
            OwnedDLHandle()
        )  # Open a handle to the process's libraries
        var custom_abs = process.get_function[c_int](
            "custom_abs"
        )  # Fetch by name
        var n = custom_abs(c_int(-65))  # Call
        print(t"custom_abs returned 65: {n == 65}")  # True
    except:
        print("Failed to find symbol 'custom_abs'")
        return
