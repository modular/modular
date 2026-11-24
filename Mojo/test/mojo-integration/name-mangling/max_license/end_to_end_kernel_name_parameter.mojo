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
# Test that `enqueue_function` with a callable parameter produces correct PTX
# kernel entries (one per instantiation, no crash).
#
# The kernel is parametric by `DType`, `simd_width`, and a compile-time
# function parameter `func`. Two instantiations are enqueued:
#   - elementwise_kernel[DType.float32, 4, add_one]
#   - elementwise_kernel[DType.float16, 8, scale_two]
#
# Cross-compiled to NVPTX without requiring GPU hardware. The generated PTX
# files are checked for `.visible .entry` lines.
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


# TODO: once all kernel names are preserved, have two
# versions of the kernel, one with @__name and one without it,
# and lock down both names.
# For now, only @__name version is used, because non-named
# kernels produce unreadable names.
@__name(t"elementwise_{dtype}_{simd_width}")
def elementwise_kernel[
    dtype: DType,
    simd_width: Int,
    func: def(SIMD[dtype, simd_width]) thin -> SIMD[dtype, simd_width],
](data: Pointer[Scalar[dtype], MutAnyOrigin], n: Int32):
    var idx = global_idx.x * simd_width
    if idx < Int(n):
        var v = data.unsafe_load[width=simd_width](idx)
        data.unsafe_store(idx, func(v))


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf_f32 = ctx.enqueue_create_buffer[.float32](n)
        var buf_f16 = ctx.enqueue_create_buffer[.float16](n)
        buf_f32.enqueue_fill(0.0)
        buf_f16.enqueue_fill(0.0)

        def add_one(v: SIMD[.float32, 4]) -> SIMD[.float32, 4]:
            return v + 1

        def scale_two(v: SIMD[.float16, 8]) -> SIMD[.float16, 8]:
            return v * 2

        comptime kernel_f32 = elementwise_kernel[.float32, 4, add_one]
        ctx.enqueue_function[kernel_f32](
            buf_f32, Int32(n), grid_dim=1, block_dim=n // 4
        )
        comptime kernel_f16 = elementwise_kernel[.float16, 8, scale_two]
        ctx.enqueue_function[kernel_f16](
            buf_f16, Int32(n), grid_dim=1, block_dim=n // 8
        )
        ctx.synchronize()


# ---------------------------------------------------------------------------
# FileCheck patterns
# ---------------------------------------------------------------------------

# Two distinct PTX kernels must be produced, one per instantiation.
# The non-experimental enqueue_function path compiles the kernel directly
# (no conversion thunk wrapper), so the PTX entry name is the @__name value.
# CHECK-PTX-DAG: .visible .entry elementwise_float32_4_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry elementwise_float16_8_{{[0-9a-f]+}}(
