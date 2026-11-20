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
# Tests "Call build-time libraries" in interop/libraries.mdx: link libcustom
# with the page's -Xlinker options, then run the program from another
# directory, so only the rpath (@loader_path or $ORIGIN) can find the library.
#
# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: %if system-darwin %{ cp %S/libcustom.dylib %t.dir/ %} %else %{ cp %S/libcustom.so %t.dir/ %}
# RUN: cd %t.dir && %if system-darwin %{ %mojo-build %s -o main -Xlinker -L. -Xlinker -lcustom -Xlinker -rpath -Xlinker @loader_path %} %else %{ %mojo-build %s -o main -Xlinker -L. -Xlinker -lcustom -Xlinker -rpath -Xlinker '$ORIGIN' %}
# RUN: cd / && %t.dir/main | FileCheck %s
#
# COM: `mojo run` takes the same options before the file name.
# RUN: %mojo -Xlinker -L%t.dir -Xlinker -lcustom %s | FileCheck %s
#
# COM: After the file name, the options go to the program, not the linker.
# RUN: not %mojo %s -Xlinker -L%t.dir -Xlinker -lcustom 2>&1 \
# RUN:   | FileCheck %s --check-prefix=AFTER

# CHECK: custom_abs returned 65: True
# AFTER: Symbols not found: {{.*}}custom_abs

from std.ffi import external_call, c_int


def main():
    # int custom_abs(int x);
    var n = external_call["custom_abs", c_int](c_int(-65))
    print(t"custom_abs returned 65: {n == 65}")  # True
