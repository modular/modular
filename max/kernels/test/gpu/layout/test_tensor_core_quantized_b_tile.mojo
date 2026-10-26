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

from std.testing import assert_equal
from std.memory.unsafe import bitcast
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tensor_core import TensorCore


@inline(.always)
def nibble_value(kg: Int, lane: Int, word: Int, nibble: Int) -> Int:
    return (kg * 11 + lane * 5 + word * 7 + nibble * 3) % 16


@inline(.always)
def packed_word(kg: Int, lane: Int, word: Int) -> UInt32:
    var packed = UInt32(0)
    comptime for nibble in range(8):
        packed |= UInt32(nibble_value(kg, lane, word, nibble)) << UInt32(
            4 * nibble
        )
    return packed


@inline(.always)
def scale_value(row: Int, lane: Int) -> BFloat16:
    return (Float32(row + 1) * 0.37 + Float32(lane % 7) * 0.03).cast[
        .bfloat16
    ]()


def quantized_b_kernel[
    packed_dtype: DType, NF: Int, output_layout: TensorLayout
](output: TileTensor[.bfloat16, output_layout, MutAnyOrigin],):
    comptime assert output.rank == output.flat_rank == 4
    var source = stack_allocation[packed_dtype, address_space=.SHARED](
        row_major[1, 388]()
    )
    comptime for kg in range(3):
        comptime for word in range(4):
            source[0, 4 + kg * 128 + thread_idx.x * 4 + word] = bitcast[
                packed_dtype, 1
            ](packed_word(kg, thread_idx.x, word))
    barrier()

    # A 16-byte source prefix and scalar-aligned padded registers retain the
    # real packed-word load contract without relying on contiguous destinations.
    var warp_tile = source.slice[0:1, 4:388]()
    var storage = stack_allocation[.bfloat16, address_space=.LOCAL](
        row_major[NF, 5]()
    )
    var registers = storage.slice[0:NF, 0:4]()
    var scale_storage = stack_allocation[.bfloat16, address_space=.LOCAL](
        row_major[NF, 3]()
    )
    var scales = scale_storage.slice[0:NF, 1:2]()
    var tc = TensorCore[.float32, .bfloat16, (16, 8, 16)]()
    comptime for row in range(NF):
        scale_storage[row, 0] = -29
        scale_storage[row, 1] = scale_value(row, thread_idx.x)
        scale_storage[row, 2] = -30
    comptime for kg in range(3):
        comptime for row in range(NF):
            comptime for reg in range(5):
                storage[row, reg] = -31
        tc.load_b(warp_tile, registers, scales, kg)
        comptime for row in range(NF):
            comptime for reg in range(4):
                output[kg, row, thread_idx.x, reg] = registers[row, reg]
            output[kg, row, thread_idx.x, 4] = storage[row, 4]
            output[kg, row, thread_idx.x, 5] = scale_storage[row, 0]
            output[kg, row, thread_idx.x, 6] = scales[row, 0]
            output[kg, row, thread_idx.x, 7] = scale_storage[row, 2]


def test_quantized_b[packed_dtype: DType, NF: Int](ctx: DeviceContext) raises:
    comptime warp_size = ctx.default_device_info.warp_size
    comptime layout = row_major[3, NF, warp_size, 8]()
    var output = HostDeviceTileTensor[.bfloat16](layout, ctx)
    _ = output.host_tensor().fill(-32)
    output.to_device()
    ctx.enqueue_function[quantized_b_kernel[packed_dtype, NF, type_of(layout)]](
        output.device_tensor().as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=warp_size,
    )
    output.to_host()
    var actual = output.host_tensor()
    for kg in range(3):
        for row in range(NF):
            for lane in range(warp_size):
                var scale = scale_value(row, lane)
                for reg in range(4):
                    # Each packed word supplies adjacent B fragments. Within
                    # each row the BF16 pairs interleave low and high nibbles.
                    var nibble = (row % 2) * 2 + reg // 2 + (reg % 2) * 4
                    var value = nibble_value(kg, lane, row // 2, nibble) - 8
                    var expected = (
                        Float32(value) * scale.cast[.float32]()
                    ).cast[.bfloat16]()
                    assert_equal(actual[kg, row, lane, reg], expected)
                assert_equal(actual[kg, row, lane, 4], BFloat16(-31))
                assert_equal(actual[kg, row, lane, 5], BFloat16(-29))
                assert_equal(actual[kg, row, lane, 6], scale)
                assert_equal(actual[kg, row, lane, 7], BFloat16(-30))


def main() raises:
    var ctx = DeviceContext()
    test_quantized_b[.int32, 2](ctx)
    test_quantized_b[.uint32, 4](ctx)
    test_quantized_b[.int32, 6](ctx)
    test_quantized_b[.uint32, 8](ctx)
