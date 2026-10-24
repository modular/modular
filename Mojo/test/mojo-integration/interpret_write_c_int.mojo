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

# The comptime interpreter accepts the C ABI types of `write`'s arguments.

# RUN: %mojo %s | FileCheck %s

from std.ffi import external_call, c_int, c_size_t, c_ssize_t


@no_inline
def comp_write() -> Int:
    var bytes = String("hi\n").as_bytes()
    return Int(
        external_call["write", c_ssize_t](
            c_int(1), bytes.unsafe_ptr(), c_size_t(len(bytes))
        )
    )


def main():
    # CHECK: hi
    comptime written = comp_write()
    # CHECK-NEXT: 3
    print(written)
