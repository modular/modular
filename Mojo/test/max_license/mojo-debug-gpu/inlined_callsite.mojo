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


# Check that breakpoints can be set on inlined callsites.

# REQUIRES: NVIDIA-GPU

# FIXME: KERN-1468
# UNSUPPORTED: H100-GPU
# UNSUPPORTED: B200-GPU


# GDB-COMMAND: b %breakpoint1:location
# GDB-COMMAND: c

# RUN: %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB

# CHECK-GDB: hit Breakpoint
# CHECK-GDB: var tid2 = get_tid()

from max.gpu import *
from max.gpu.host import DeviceContext


@always_inline
def get_tid() -> Int:
    return Int(global_idx.x)


def vec_func(
    output: Pointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = get_tid()
    var tid2 = get_tid()  # breakpoint1
    if tid < len:
        output[tid] = Float32(tid2)


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
