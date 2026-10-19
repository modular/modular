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
# Legacy closures: the original motivation for `DevicePassable`.
# See max/docs/internal/max-serve/device-passable-closures.md.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime N = 256
comptime SCALE: Float32 = 3.0


def scale_kernel[
    epilogue: def(Int, Float32) capturing -> None
](src: MutPointer[Float32, MutAnyOrigin], n: Int32):
    """Named kernel. `src` is an explicit argument: `DevicePassable` rewrites
    the `DeviceBuffer` the host passes into this device pointer. `epilogue` is
    a comptime parameter, so it is part of the kernel's code, not its
    arguments."""
    var i = global_idx.x
    if i >= Int(n):
        return
    epilogue(i, src[unsafe_offset=i] * SCALE)


def main() raises:
    with DeviceContext() as ctx:
        var src_buf = ctx.enqueue_create_buffer[.float32](N)
        var out_buf = ctx.enqueue_create_buffer[.float32](N)
        with src_buf.map_to_host() as h:
            for i in range(N):
                h[i] = Float32(i)
        out_buf.enqueue_fill(Float32(0))

        # Legacy capture: a raw device address, bit-copied by `populate`.
        # Capturing `out_buf` (or `out_buf.device_ptr()`) here would copy host
        # bytes onto the device, because legacy captures are never encoded.
        var out_ptr = out_buf.unsafe_ptr()

        @__parameter
        @__copy_capture(out_ptr)
        def epilogue(i: Int, val: Float32) capturing -> None:
            out_ptr[unsafe_offset=i] = val

        # `src_buf` is a `DeviceBuffer` argument. `DevicePassable` encodes it
        # to `DeviceBuffer.device_type` (a device `Pointer`), which is exactly
        # the declared `src` parameter type.
        ctx.enqueue_function[scale_kernel[epilogue]](
            src_buf, Int32(N), grid_dim=1, block_dim=N
        )

        with out_buf.map_to_host() as h:
            for i in range(N):
                assert_equal(h[i], Float32(i) * SCALE)
        print("legacy: OK")
