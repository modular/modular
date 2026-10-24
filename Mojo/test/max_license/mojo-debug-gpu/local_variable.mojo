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
# GDB-COMMAND: p tid
# GDB-COMMAND: cuda thread 3
# GDB-COMMAND: p tid
# GDB-COMMAND: p val
# GDB-COMMAND: ptype val
# GDB-COMMAND: p point

# RUN: %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB

# CHECK-GDB: = 0
# CHECK-GDB: Switching focus
# CHECK-GDB: = 3
# CHECK-GDB: = {23.0999985}
# CHECK-GDB: float
# CHECK-GDB: {x = 3, y = 3}

from max.gpu import *
from max.gpu.host import DeviceContext


struct Point:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def vec_func(
    output: Pointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    var point = Point(Int(thread_idx.x), Int(tid))
    var val = Float32(tid) * 7.7
    if tid < len:
        output[tid] = val  # breakpoint1
    else:
        output[tid] = Float32(point.x) + Float32(point.y)


def run_vec_func(ctx: DeviceContext) raises:
    comptime length = 1024
    var out_device = ctx.enqueue_create_buffer[.float32](length)
    var block_dim = 32

    ctx.enqueue_function[vec_func](
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Wait for the kernel to finish running, and give the debugger opportunity to connect.
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        run_vec_func(ctx)
