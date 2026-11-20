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
# Test that `enqueue_function` with a `@__parameter` closure kernel (not a
# top-level def) respects the `@__name` decorator.
#
# The kernel is defined as a `@__parameter` closure inside `launch_kernel`,
# analogous to how `elementwise` defines `_kernel`. Two instantiations are
# launched with different dtypes.
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
from max.gpu.host import DeviceContext, DeviceBuffer


def launch_kernel[
    dtype: DType, kernel_name: StaticString, width: Int
](ctx: DeviceContext, mut data: DeviceBuffer[dtype], n: Int) raises:
    var buf = data.unsafe_ptr()
    var len = Int32(n)

    @__name(kernel_name + "_" + String(width))
    def _kernel() {var buf, var len}:
        var idx = global_idx.x * width
        if idx < Int(len):
            buf[unsafe_offset=idx] = buf[unsafe_offset=idx] + 1

    ctx.enqueue_function(_kernel, grid_dim=1, block_dim=n // width)


def main() raises:
    with DeviceContext() as ctx:
        comptime n = 64
        var buf_f32 = ctx.enqueue_create_buffer[.float32](n)
        var buf_f16 = ctx.enqueue_create_buffer[.float16](n)
        buf_f32.enqueue_fill(0.0)
        buf_f16.enqueue_fill(0.0)

        launch_kernel[.float32, "closure_kernel_float32", 4](ctx, buf_f32, n)
        launch_kernel[.float32, "closure_kernel_float32", 8](ctx, buf_f32, n)
        launch_kernel[.float16, "closure_kernel_float16", 4](ctx, buf_f16, n)
        launch_kernel[.float16, "closure_kernel_float16", 8](ctx, buf_f16, n)
        ctx.synchronize()


# Four distinct PTX kernels must be produced — one per (dtype, width) pair —
# each named after the @__name value with the width appended.
# CHECK-PTX-DAG: .visible .entry closure_kernel_float32_4_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry closure_kernel_float32_8_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry closure_kernel_float16_4_{{[0-9a-f]+}}(
# CHECK-PTX-DAG: .visible .entry closure_kernel_float16_8_{{[0-9a-f]+}}(
