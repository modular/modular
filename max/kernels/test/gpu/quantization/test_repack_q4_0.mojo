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

from std.memory.unsafe import bitcast
from std.testing import assert_equal

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0


def test_repack_q4_0[N: Int, K: Int](ctx: DeviceContext) raises:
    comptime group_size = 32
    comptime group_bytes = 18
    comptime groups = K // group_size
    comptime packed_bytes = N * groups * group_bytes
    comptime guard_bytes = 64

    var input_host = ctx.enqueue_create_host_buffer[.uint8](packed_bytes)
    var output_host = ctx.enqueue_create_host_buffer[.uint8](
        packed_bytes + guard_bytes
    )
    var input_tensor = TileTensor(
        input_host, row_major[N, groups * group_bytes]()
    )
    var output_tensor = TileTensor(
        output_host, row_major[packed_bytes + guard_bytes]()
    )
    _ = output_tensor.fill(0xA5)

    for row in range(N):
        for group in range(groups):
            var scale = Float32((row % 31) - 15) * Float32(group % 7 + 1) / 128
            input_tensor.store[2](
                (row, group * group_bytes),
                bitcast[.uint8, 2](bitcast[.uint16, 1](scale.cast[.float16]())),
            )
            for column in range(16):
                input_tensor[row, group * group_bytes + 2 + column] = UInt8(
                    row * 37 + group * 13 + column * 11
                )

    var input_device = ctx.enqueue_create_buffer[.uint8](packed_bytes)
    var output_device = ctx.enqueue_create_buffer[.uint8](
        packed_bytes + guard_bytes
    )
    ctx.enqueue_copy(input_device, input_host)
    ctx.enqueue_copy(output_device, output_host)
    gpu_qint4_repack_Q4_0["gpu"](
        TileTensor(input_device, row_major[N, groups * group_bytes]()).as_imm(),
        TileTensor(output_device, row_major[N, groups * group_bytes]()),
        ctx,
    )
    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()

    var packed_weights = TileTensor(
        output_host.unsafe_ptr().bitcast[UInt32](), row_major[N // 64, K * 8]()
    )
    var packed_scales = TileTensor(
        (output_host.unsafe_ptr() + N * K // 2).bitcast[UInt16](),
        row_major[groups, N](),
    )
    for row_tile in range(N // 64):
        for group in range(groups):
            for half in range(2):
                for lane in range(32):
                    for word in range(4):
                        var expected = UInt32(0)
                        for nibble in range(8):
                            var row = (
                                row_tile * 64
                                + lane // 4
                                + word * 16
                                + (nibble % 4 // 2) * 8
                            )
                            var column = (
                                lane % 4 * 2 + (nibble % 2) * 8 + nibble // 4
                            )
                            var source = input_tensor[
                                row, group * group_bytes + 2 + column
                            ]
                            var value = (source >> UInt8(half * 4)) & 0xF
                            expected |= value.cast[.uint32]() << UInt32(
                                nibble * 4
                            )
                        assert_equal(
                            packed_weights[
                                row_tile,
                                (group * 2 + half) * 128 + lane * 4 + word,
                            ],
                            expected,
                        )

                    var scale_row = (
                        row_tile * 64 + lane % 4 * 16 + lane // 4 + half * 8
                    )
                    var scale_bits = input_tensor.load[2](
                        (scale_row, group * group_bytes)
                    )
                    var scale = bitcast[.float16, 1](
                        bitcast[.uint16, 1](scale_bits)
                    ).cast[.float32]()
                    var expected_scale = (
                        bitcast[.uint32, 1](scale) >> 16
                    ).cast[.uint16]()
                    assert_equal(
                        packed_scales[group, row_tile * 64 + lane * 2 + half],
                        expected_scale,
                    )
    for i in range(packed_bytes, packed_bytes + guard_bytes):
        assert_equal(output_tensor[i], UInt8(0xA5))


def main() raises:
    with DeviceContext() as ctx:
        test_repack_q4_0[128, 1024](ctx)
        test_repack_q4_0[256, 2048](ctx)
