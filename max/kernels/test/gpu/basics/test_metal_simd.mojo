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

from std.collections import Array
from std.testing import assert_equal

from max.gpu import thread_idx
from max.gpu.host import DeviceContext

comptime _W = 8
comptime _V = SIMD[DType.float32, _W]
comptime _NUM_OPS = 6


def _ops(v: _V) -> Array[_V, _NUM_OPS]:
    var high = v.slice[_W // 2, offset=_W // 2]()
    return [
        high.join(high),
        v.insert[offset=2](SIMD[DType.float32, 2](-1, -2)),
        v.rotate_left[3](),
        v.rotate_right[3](),
        v.shift_left[3](),
        v.shift_right[3](),
    ]


def _input(base: Int) -> _V:
    # Lanes come from a run-time value so the operations aren't folded.
    var v = _V()
    for i in range(_W):
        v[i] = Float32(base + i)
    return v


def _kernel(output: MutPointer[Float32, MutAnyOrigin]):
    var results = _ops(_input(thread_idx.x + 1))
    for op in range(_NUM_OPS):
        for i in range(_W):
            output.unsafe_offset(op * _W + i).unsafe_store(results[op][i])


def main() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[DType.float32](_NUM_OPS * _W)
        ctx.enqueue_function[_kernel](out_device, grid_dim=1, block_dim=1)
        var expected = _ops(_input(1))
        with out_device.map_to_host() as out_host:
            for op in range(_NUM_OPS):
                for i in range(_W):
                    assert_equal(
                        out_host[op * _W + i],
                        expected[op][i],
                        msg=String("op ", op, " lane ", i),
                    )
