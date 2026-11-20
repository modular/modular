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
# Unified closure as the kernel: the sound `DevicePassable` launch.
# See max/docs/internal/max-serve/device-passable-closures.md.

from layout import TileTensor, row_major
from layout.tensor_engine import DevicePointerEngine
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.builtin.device_passable import DevicePassable
from std.testing import assert_equal

comptime N = 256
comptime SCALE: Float32 = 3.0


def assert_closure_needs_encoding[ClosureType: DevicePassable]():
    # The host closure type is not its own device type: launching its host
    # bytes verbatim would be wrong, so encoding is required.
    comptime assert ClosureType != ClosureType.device_type
    comptime assert not ClosureType._is_convertible_to_device_type[
        ClosureType
    ]()


def main() raises:
    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        with buf.map_to_host() as h:
            for i in range(N):
                h[i] = Float32(i)

        # Host handle: the engine stores a `DevicePointer`, not a device address.
        var data = TileTensor(buf.device_ptr(), row_major[N]())
        comptime assert data.Engine == DevicePointerEngine[element_width=1]

        # The closure is the kernel. No named kernel exists, so no signature can
        # disagree with the synthesized `device_type`.
        def scale_kernel() {var data}:
            var i = global_idx.x
            if i >= N:
                return
            data[i] = data[i] * SCALE

        assert_closure_needs_encoding[type_of(scale_kernel)]()

        # Compiles `scale_kernel.__call__` over `type_of(scale_kernel).device_type`,
        # then encodes the host closure into that same layout.
        ctx.enqueue_function(scale_kernel, grid_dim=1, block_dim=N)

        with buf.map_to_host() as h:
            for i in range(N):
                assert_equal(h[i], Float32(i) * SCALE)
        print("correct: OK")
