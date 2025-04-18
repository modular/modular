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
# Can only run with asan version of amd drivers installed.
# UNSUPPORTED: asan
# REQUIRES: AMD-GPU
# RUN: mojo build --sanitize=address --external-libasan=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so --target-accelerator=mi300x -g %s -o %t
# RUN: export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib/clang/18/lib/linux:/opt/rocm/lib/asan
# RUN: export LD_PRELOAD=/opt/rocm/lib/llvm/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so:/opt/rocm/lib/asan/libamdhip64.so
# RUN: export ASAN_OPTIONS=detect_leaks=0
# RUN: export HSA_XNACK=1
# RUN: not %t 5 2>&1 | FileCheck %s

# CHECK: AddressSanitizer: heap-buffer-overflow on amdgpu device
# CHECK: at {{.*}}test_amd_asan_oob.mojo:27

from sys import argv

from gpu.host import DeviceContext
from memory import UnsafePointer


fn bad_func(ptr: UnsafePointer[Int32], i: Int):
    # Potential out of bounds access
    ptr[i] = 42


fn test(ctx: DeviceContext, i: Int) raises:
    alias n = 4
    var buf = ctx.enqueue_create_buffer[DType.int32](n)

    ctx.enqueue_function[bad_func](buf, i, grid_dim=(1), block_dim=(1))
    ctx.synchronize()


fn main() raises:
    i = atol(argv()[1])
    with DeviceContext() as ctx:
        test(ctx, i)
