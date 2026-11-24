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
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: error: cannot slice out exported symbol
# CHECK-SAME: scale{{.*}}n=3
# CHECK-NOT: Assertion
# CHECK-NOT: PLEASE submit a bug report

from std.compile import compile_info

comptime Fn = def(Float32) thin -> Float32


def scale[n: Int](x: Float32) -> Float32:
    return x * Float32(n)


def get_fn[b: Bool]() -> Fn:
    if b:
        return scale[2]
    return scale[3]


def kernel[f: Fn](p: Pointer[Float32, MutAnyOrigin]):
    p[] = f(p[])


def main() raises:
    # `get_fn[False]()` returns a runtime function pointer which is not a
    # symbol that the elaborator can refer to for offload compilation.
    # Slicing the kernel functions should properly error this instead of assert.
    print(compile_info[kernel[get_fn[False]()]]())
