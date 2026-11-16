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
# Tests the interop/interoperability.mdx example: a C program calls the
# exported Mojo `add()` from a `--emit shared-lib` library.
#
# COM: Uses `.dylib` on every platform, as shared-lib-c-host does. dlopen
# COM:   takes a literal path, so the extension is cosmetic.
# RUN: mkdir -p %t.dir
# RUN: %mojo-build %s -o %t.dir/libmylib.dylib --emit shared-lib
# RUN: $(dirname %s)/c_host %t.dir/libmylib.dylib | FileCheck %s

# CHECK: sum = 108

# start-mylib
from std.ffi import c_long  # 64-bit on macOS and Linux
from std.runtime import initialize_runtime


@export("add")
def add(x: c_long, y: c_long) abi("C") -> c_long:
    initialize_runtime()  # call first in every exported function
    return x + y


# end-mylib
