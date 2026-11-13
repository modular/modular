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
# Tests "Handle missing functions and unsafe signatures" in interop/libraries.mdx: external_call()
# with a name no library defines fails when you build, and `mojo run` fails
# before the program starts. Neither failure reaches code the program could
# catch.
#
# COM: Sanitizer builds link with unresolved symbols allowed, so the
# COM:   missing name fails at launch instead of at build.
# UNSUPPORTED: ubsan
#
# RUN: not %mojo-build %s -o %t 2>&1 | FileCheck %s --check-prefix=BUILD
# RUN: not %mojo %s 2>&1 | FileCheck %s --check-prefix=JIT

# BUILD: no_such_function
# JIT-NOT: main started
# JIT: Symbols not found: {{.*}}no_such_function

from std.ffi import external_call, c_int


def main():
    print("main started")
    _ = external_call["no_such_function", c_int](c_int(-65))
