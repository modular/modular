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
# REQUIRES: NVIDIA-GPU

# FIXME: KERN-1468
# UNSUPPORTED: H100-GPU
# UNSUPPORTED: B200-GPU

# RUN: %if !debugging-test %{ %mojo-no-debug %s | FileCheck %s %}

# ===----------------------------------------------------------------------=== #
# Debugging tests:
# Run them with `./bazelw test KGEN/test/mojo-debug-gpu/test_vec_add.mojo.test`

# compile:
# RUN: %if debugging-test %{ %mojo-build-no-debug -debug-level=line-tables -O0 %s %}

# execute:
# GDB-COMMAND: b %breakpoint1:location
# GDB-COMMAND: c
# GDB-COMMAND: cuda thread 0
# GDB-COMMAND: info locals
# GDB-COMMAND: info args

# RUN: %if debugging-test %{ %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB %}

# CHECK-GDB: hit Breakpoint
# CHECK-GDB: tid = 0
# CHECK-GDB: in0 = 0x
# CHECK-GDB: in1 = 0x
# CHECK-GDB: output = 0x
# CHECK-GDB: len = 1024
# ===----------------------------------------------------------------------=== #


from max.gpu import *
from max.gpu.host import DeviceContext


def vec_func(
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = in0[tid] + in1[tid]  # breakpoint1


# CHECK-LABEL: run_vec_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
def run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    comptime length = 1024

    var in0_device = ctx.enqueue_create_buffer[.float32](length)
    var in1_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    with in0_device.map_to_host() as in0_host, in1_device.map_to_host() as in1_host, out_device.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = 2

    var block_dim = 32

    var kernel_func = ctx.compile_function[vec_func]()
    ctx.enqueue_function(
        kernel_func,
        in0_device,
        in1_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.synchronize()

    # CHECK: at index 0 the value is 2.0
    # CHECK: at index 1 the value is 3.0
    # CHECK: at index 2 the value is 4.0
    # CHECK: at index 3 the value is 5.0
    # CHECK: at index 4 the value is 6.0
    # CHECK: at index 5 the value is 7.0
    # CHECK: at index 6 the value is 8.0
    # CHECK: at index 7 the value is 9.0
    # CHECK: at index 8 the value is 10.0
    # CHECK: at index 9 the value is 11.0
    with out_device.map_to_host() as out_host:
        for i in range(10):
            print("at index", i, "the value is", out_host[i])


def main() raises:
    with DeviceContext() as ctx:
        run_vec_add(ctx)
