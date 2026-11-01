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
#
# Regression test: a GPU kernel whose @__name starts with a digit must be
# correctly sanitized to a valid PTX identifier ("_1kernel") in the kernel
# entry point.
#
# Cross-compiled to NVPTX without requiring GPU hardware.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-accelerator nvidia:sm_90a \
# RUN:   -o %t/out.s %s
# RUN: cat %t/*.ptx > %t/all_kernels.ptx
# RUN: FileCheck %s --check-prefix=CHECK-PTX --input-file=%t/all_kernels.ptx

from max.gpu import global_idx
from max.gpu.host import DeviceContext


# Linkage name starts with a digit — requires PTX sanitization to "_1kernel".
@__name("1kernel")
def digit_prefix_kernel(data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + 1.0


# Same linkage name for two different instantiations.
# The compiler mangling should save us here and produce two distinct suffixes
@__name("name_conflict_42")
def name_conflict_kernel[
    x: Int
](data: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = global_idx.x
    if idx < Int(n):
        data[unsafe_offset=idx] = data[unsafe_offset=idx] + Float32(x)


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf_f32 = ctx.enqueue_create_buffer[.float32](n)
        buf_f32.enqueue_fill(0.0)

        comptime k_digit = digit_prefix_kernel
        ctx.enqueue_function[k_digit](
            buf_f32, Int32(n), grid_dim=1, block_dim=n
        )

        comptime nc_1 = name_conflict_kernel[5]
        comptime nc_2 = name_conflict_kernel[6]
        ctx.enqueue_function[nc_1](buf_f32, Int32(n), grid_dim=1, block_dim=n)
        ctx.enqueue_function[nc_2](buf_f32, Int32(n), grid_dim=1, block_dim=n)
        ctx.synchronize()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------
# The digit-prefix kernel must be sanitized: "1kernel" -> "_1kernel...".
# CHECK-PTX: .visible .entry _1kernel_{{[0-9a-f]+}}(

# The two name_conflict_kernel instantiations share the same @__name literal
# but differ in their `x` parameter. The compiler appends a hash derived from
# the mangled sym_name (which encodes the parameter value), so each
# instantiation gets a unique suffix. Two distinct DAG matches ensure both
# entries exist and that FileCheck binds them to different input lines.
# CHECK-PTX-DAG: .visible .entry name_conflict_42_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry name_conflict_42_{{[0-9a-f]+}}(
