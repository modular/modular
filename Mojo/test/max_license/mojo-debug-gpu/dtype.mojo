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
# Check that local variable can be inspected.

# REQUIRES: NVIDIA-GPU

# FIXME: KERN-1468
# UNSUPPORTED: H100-GPU
# UNSUPPORTED: B200-GPU

# GDB-COMMAND: b %breakpoint1:location
# GDB-COMMAND: c
# GDB-COMMAND: cuda thread 2
# GDB-COMMAND: p val_bf16
# GDB-COMMAND: ptype val_bf16
# GDB-COMMAND: p val_f16
# GDB-COMMAND: ptype val_f16
# GDB-COMMAND: p val_f64
# GDB-COMMAND: ptype val_f64

# RUN: %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB

# CHECK-GDB: Switching focus
# CHECK-GDB: = {15.38}
# CHECK-GDB: __nv_bfloat16 __attribute__ ((vector_size(1)))
# CHECK-GDB: = {15.398}
# CHECK-GDB: __half __attribute__ ((vector_size(1)))
# CHECK-GDB: = {15.4}
# CHECK-GDB: double __attribute__ ((vector_size(1)))

from max.gpu import *
from max.gpu.host import DeviceContext


struct Point:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def vec_func(
    out_bf16: Pointer[BFloat16, MutAnyOrigin],
    out_f16: Pointer[Float16, MutAnyOrigin],
    out_f64: Pointer[Float64, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    var val_bf16 = BFloat16(tid) * BFloat16(7.7)
    var val_f16 = Float16(tid) * Float16(7.7)
    var val_f64 = Float64(tid) * Float64(7.7)
    if tid < len:
        out_bf16[tid] = val_bf16  # breakpoint1
        out_f16[tid] = val_f16
        out_f64[tid] = val_f64


def run_vec_func(ctx: DeviceContext) raises:
    comptime length = 1024
    var out_bf16_device = ctx.enqueue_create_buffer[.bfloat16](length)
    var out_f16_device = ctx.enqueue_create_buffer[.float16](length)
    var out_f64_device = ctx.enqueue_create_buffer[.float64](length)
    var block_dim = 32

    ctx.enqueue_function[vec_func](
        out_bf16_device,
        out_f16_device,
        out_f64_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Wait for the kernel to finish running, and give the debugger opportunity to connect.
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        run_vec_func(ctx)
