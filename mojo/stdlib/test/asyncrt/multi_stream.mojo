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

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]


fn test_concurrent_copy(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    print("-------")
    print(
        "Running test_concurrent_copy("
        + ctx1.name()
        + ", "
        + ctx2.name()
        + "):"
    )

    alias length = 1 * 1024 * 1024
    alias T = DType.float32

    var in0_dev1 = ctx1.enqueue_create_buffer[T](length)
    var in0_dev2 = ctx1.enqueue_create_buffer[T](length)
    var in0_dev3 = ctx1.enqueue_create_buffer[T](length)

    # Initialize the variable inputs with known values.
    with in0_dev1.map_to_host() as in_host1, in0_dev2.map_to_host() as in_host2, in0_dev3.map_to_host() as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = index
            in_host2[i] = 2 * index
            in_host3[i] = 3 * index

    # Initialize the fixed (right) inputs.
    in1_dev1 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(1.0)
    in1_dev2 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(2.0)
    in1_dev3 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(3.0)

    # Initialize the device outputs with known bad values.
    out_dev1 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(101.0)
    out_dev2 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(102.0)
    out_dev3 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(103.0)
    # Initialize the result buffer on a second queue with known bad values.
    var out_host1 = ctx2.enqueue_create_host_buffer[T](length).enqueue_fill(0.1)
    var out_host2 = ctx2.enqueue_create_host_buffer[T](length).enqueue_fill(0.2)
    var out_host3 = ctx2.enqueue_create_host_buffer[T](length).enqueue_fill(0.3)

    for i in range(10):
        print(out_host1[i])
        print(out_host2[i])
        print(out_host3[i])

    # Pre-compile and pre-register the device function
    var dev_func = ctx1.compile_function_checked[vec_func, vec_func]()

    # Make sure both queues are ready to run at this point.
    ctx1.synchronize()
    ctx2.synchronize()

    var block_dim = 1

    ctx1.enqueue_function_checked(
        dev_func,
        in0_dev1,
        in1_dev1,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev1.reassign_ownership_to(ctx2)
    out_dev1.enqueue_copy_to(out_host1)
    ctx1.enqueue_function_checked(
        dev_func,
        in0_dev2,
        in1_dev2,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev2.reassign_ownership_to(ctx2)
    out_dev2.enqueue_copy_to(out_host2)
    ctx1.enqueue_function_checked(
        dev_func,
        in0_dev3,
        in1_dev3,
        out_dev3,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    out_dev3.reassign_ownership_to(ctx2)
    out_dev3.enqueue_copy_to(out_host3)

    # Wait for the copies to be completed.
    ctx2.synchronize()

    for i in range(length):
        var index = i % 2048
        if i < 10:
            print("at index", i, "the value is", out_host1[i])
            print("at index", i, "the value is", out_host2[i])
            print("at index", i, "the value is", out_host3[i])
        expect_eq(
            out_host1[i],
            Float32(index + 1.0),
            "out_host1[",
            i,
            "] is ",
            out_host1[i],
        )
        expect_eq(
            out_host2[i],
            Float32(2.0 * index + 2.0),
            "out_host2[",
            i,
            "] is ",
            out_host2[i],
        )
        expect_eq(
            out_host3[i],
            Float32(3.0 * index + 3.0),
            "out_host3[",
            i,
            "] is ",
            out_host3[i],
        )

    print("Done.")


fn test_concurrent_func(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    print("-------")
    print(
        "Running test_concurrent_func("
        + ctx1.name()
        + ", "
        + ctx2.name()
        + "):"
    )

    alias length = 20 * 1024 * 1024
    alias T = DType.float32

    # Initialize the variable inputs with known values.
    var in_dev1 = ctx1.enqueue_create_buffer[T](length)
    var in_dev2 = ctx1.enqueue_create_buffer[T](length)
    var in_dev3 = ctx1.enqueue_create_buffer[T](length)
    with in_dev1.map_to_host() as in_host1, in_dev2.map_to_host() as in_host2, in_dev3.map_to_host() as in_host3:
        for i in range(length):
            var index = i % 2048
            in_host1[i] = index
            in_host2[i] = 2 * index
            in_host3[i] = 3 * index

    # Initialize the fixed (right) inputs.
    var in_dev4 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(1.0)
    var in_dev5 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(2.0)

    # Initialize the outputs with known bad values
    var out_dev1 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(101.0)
    var out_dev2 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(102.0)
    var out_dev3 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(103.0)
    var out_dev4 = ctx1.enqueue_create_buffer[T](length).enqueue_fill(104.0)

    var out_host = ctx2.enqueue_create_host_buffer[T](length).enqueue_fill(0.5)

    # Pre-compile and pre-register the device function
    var dev_func1 = ctx1.compile_function_checked[vec_func, vec_func]()
    var dev_func2 = ctx2.compile_function_checked[vec_func, vec_func]()

    # Ensure the setup has completed.
    ctx1.synchronize()
    ctx2.synchronize()

    var block_dim = 1

    ctx1.enqueue_function_checked(
        dev_func1,
        in_dev1,
        in_dev4,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_wait_for(ctx1)
    ctx2.enqueue_function_checked(
        dev_func2,
        in_dev2,
        out_dev1,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx1.enqueue_function_checked(
        dev_func1,
        in_dev3,
        in_dev5,
        out_dev3,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    ctx2.enqueue_wait_for(ctx1)
    ctx2.enqueue_function_checked(
        dev_func2,
        out_dev2,
        out_dev3,
        out_dev4,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    # Wait for ctx2 to be done with running the function to make sure `out_dev4` writes
    # have settled since `out_dev4` is associated with ctx1.
    ctx1.enqueue_wait_for(ctx2)
    # Schedule a cross-context copy `out_dev4`@ctx1->`out_host`@ctx2
    out_dev4.enqueue_copy_to(out_host)
    # Reassign ownership of `out_host` to ctx1, then synchronize on ctx1 for the
    #  copies to be completed.
    out_host.reassign_ownership_to(ctx1)
    ctx1.synchronize()

    for i in range(length):
        var index = i % 2048
        var o1 = index + 1
        var o2 = 2 * index + o1
        var o3 = 3 * index + 2
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            Float32(o2 + o3),
            "out_host[",
            i,
            "] is ",
            out_host[i],
            " (expected=",
            Float32(o2 + o3),
            ")",
        )

    print("Done.")


fn main() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    test_concurrent_copy(ctx1, ctx2)
    test_concurrent_func(ctx1, ctx2)
