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
"""Independent signed-value checks of both CDNA4 mixed MFMA operand orders.

Uniform fragments make every accumulator equal to a hand-computed dot product.
Distinct packed E8M0 words and nonzero byte selectors expose scale-order bugs.
Both 16x16x128 and 32x32x64 shapes execute on hardware.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from std.builtin._closure import __ownership_keepalive
from std.testing import assert_true
from std.utils import StaticTuple

from layout import TensorLayout, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _mixed_mma[
    Layout: TensorLayout,
    accum_width: Int,
    reverse: Bool,
](out_tt: TileTensor[.float32, Layout, MutAnyOrigin]):
    comptime fp8 = CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3
    comptime fp4 = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1
    var a = SIMD[.uint8, fp8.simd_width()](0x3C)  # E4M3 +1.5
    var b = SIMD[.uint8, fp4.simd_width()](0xCC)  # Two E2M1 -2 values
    var sa = Int32(0x807F7E7D)  # [.25, .5, 1, 2], select byte 3
    var sb = Int32(0x7E7F8081)  # [4, 2, 1, .5], select byte 1
    var acc = SIMD[.float32, accum_width](0.0)
    comptime if reverse:
        cdna4_block_scaled_mfma[1, 3, fp4, fp8](acc, b, a, sb, sa)
    else:
        cdna4_block_scaled_mfma[3, 1, fp8, fp4](acc, a, b, sa, sb)
    out_tt.store((lane_id(), 0), acc)


def _check[accum_width: Int, reverse: Bool](ctx: DeviceContext) raises:
    comptime count = WARP_SIZE * accum_width
    var device = ctx.enqueue_create_buffer[.float32](count)
    var host = ctx.enqueue_create_host_buffer[.float32](count)
    var tt = TileTensor(device, row_major[WARP_SIZE, accum_width]())
    comptime kernel = _mixed_mma[type_of(tt).LayoutType, accum_width, reverse]
    ctx.enqueue_function[kernel](
        tt.as_unsafe_any_origin(), grid_dim=1, block_dim=WARP_SIZE
    )
    ctx.enqueue_copy(host, device)
    ctx.synchronize()
    # (+1.5 * 2) * (-2 * 2), accumulated over logical K.
    comptime logical_k = 128 if accum_width == 4 else 64
    for i in range(count):
        assert_true(host[i] == Float32(-12 * logical_k))
    __ownership_keepalive(device)
    print(
        "mixed MFMA: accum_width=", accum_width, " reverse=", reverse, " PASS"
    )


def main() raises:
    with DeviceContext() as ctx:
        _check[4, False](ctx)
        _check[4, True](ctx)
        _check[16, False](ctx)
        _check[16, True](ctx)
