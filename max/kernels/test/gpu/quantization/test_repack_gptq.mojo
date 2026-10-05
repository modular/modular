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
from quantization.qmatmul_gpu import gpu_qint4_repack_GPTQ


def test_repack_gptq[
    N: Int, K: Int, group_size: Int, has_perm: Bool
](ctx: DeviceContext) raises:
    comptime groups = K // group_size
    comptime packed_bytes = N * groups * (group_size // 2 + 2)
    comptime guard_bytes = 64

    var input_host = ctx.enqueue_create_host_buffer[.uint8](packed_bytes)
    var output_host = ctx.enqueue_create_host_buffer[.uint8](
        packed_bytes + guard_bytes
    )
    var perm_host = ctx.enqueue_create_host_buffer[.int32](K)
    var input_weights = TileTensor(
        input_host.unsafe_ptr().bitcast[UInt32](), row_major[K // 8, N]()
    )
    var input_scales = TileTensor(
        (input_host.unsafe_ptr() + N * K // 2).bitcast[Float16](),
        row_major[groups, N](),
    )
    var output_tensor = TileTensor(
        output_host, row_major[packed_bytes + guard_bytes]()
    )
    _ = output_tensor.fill(0xA5)

    for column in range(K // 8):
        for row in range(N):
            input_weights[column, row] = (
                UInt32(column * 131 + row * 37) * 0x01020307
            )
    for group in range(groups):
        for row in range(N):
            var scale = Float32((row % 31) - 15) * Float32(group % 7 + 1) / 128
            input_scales[group, row] = scale.cast[.float16]()
    for i in range(K):
        perm_host[i] = Int32(K - 1 - i)

    var input_device = ctx.enqueue_create_buffer[.uint8](packed_bytes)
    var output_device = ctx.enqueue_create_buffer[.uint8](
        packed_bytes + guard_bytes
    )
    var perm_device = ctx.enqueue_create_buffer[.int32](K)
    ctx.enqueue_copy(input_device, input_host)
    ctx.enqueue_copy(output_device, output_host)
    ctx.enqueue_copy(perm_device, perm_host)
    var input_tensor = TileTensor(
        input_device, row_major[groups * (group_size // 2 + 2), N]()
    )
    var result_tensor = TileTensor(
        output_device, row_major[N, groups * (group_size // 2 + 2)]()
    )
    comptime if has_perm:
        var perm_tensor = TileTensor(
            perm_device.unsafe_ptr(), row_major(Int(K))
        )
        gpu_qint4_repack_GPTQ[group_size, "gpu"](
            input_tensor.as_imm(),
            result_tensor,
            perm_tensor.as_imm().as_unsafe_any_origin(),
            ctx,
        )
    else:
        gpu_qint4_repack_GPTQ[group_size, "gpu"](
            input_tensor.as_imm(), result_tensor, ctx=ctx
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
        for column_tile in range(K // 16):
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
                            column_tile * 16
                            + lane % 4 * 2
                            + (nibble % 2) * 8
                            + nibble // 4
                        )
                        comptime if has_perm:
                            column = Int(perm_host[column])
                        var source = input_weights[column // 8, row]
                        var value = (source >> UInt32(column % 8 * 4)) & 0xF
                        expected |= value << UInt32(nibble * 4)
                    assert_equal(
                        packed_weights[
                            row_tile, column_tile * 128 + lane * 4 + word
                        ],
                        expected,
                    )
        for group in range(groups):
            for lane in range(32):
                for half in range(2):
                    var row = (
                        row_tile * 64 + lane % 4 * 16 + lane // 4 + half * 8
                    )
                    var scale = input_scales[group, row].cast[.float32]()
                    var expected = (bitcast[.uint32, 1](scale) >> 16).cast[
                        .uint16
                    ]()
                    assert_equal(
                        packed_scales[group, row_tile * 64 + lane * 2 + half],
                        expected,
                    )
    for i in range(packed_bytes, packed_bytes + guard_bytes):
        assert_equal(output_tensor[i], UInt8(0xA5))


def main() raises:
    with DeviceContext() as ctx:
        test_repack_gptq[128, 1024, 32, False](ctx)
        test_repack_gptq[128, 1024, 32, True](ctx)
        test_repack_gptq[256, 2048, 128, False](ctx)
        test_repack_gptq[256, 2048, 128, True](ctx)
