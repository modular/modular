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
# Tests "Handle missing functions and unsafe signatures" in
# interop/libraries.mdx: Mojo checks only the functions the program calls. A
# function that nothing calls can use a missing symbol, and the program still
# builds and runs.
#
# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s
# RUN: %mojo %s | FileCheck %s

# CHECK: built without calling optional_reading

from std.ffi import external_call, c_int


def optional_reading() -> c_int:
    return external_call["no_such_function", c_int]()


def main():
    print("built without calling optional_reading")
