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
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer


fn _run_badbuf(ctx: DeviceContext) raises:
    print("-")
    print("_run_badbuf()")

    alias alloc_size = 256

    # Construct a bad buffer by adopting the host pointer.
    var host_ptr = UnsafePointer[Int8].alloc(alloc_size)
    var bad_buf = DeviceBuffer(ctx, host_ptr, alloc_size, owning=True)

    # Make a call that should succeed even with a bad buffer having been constructed.
    ctx.synchronize()

    # Free the pointer now to avoid leaking. This does not change the test.
    host_ptr.free()

    try:
        # Release the bad buffer, which should raise an execption in Mojo instead of crashing.
        _ = bad_buf^
        ctx.synchronize()

    except e:
        print("Correctly raised an exception: ", e)
        return

    raise "Test failed: Should not reach here."


fn main() raises:
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_buffer(" + ctx.name() + "):")

    _run_badbuf(ctx)

    print("Done.")
