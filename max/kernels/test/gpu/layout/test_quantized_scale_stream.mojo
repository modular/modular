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
from max.gpu import WARP_SIZE, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import async_copy_commit_group, async_copy_wait_group
from layout import (
    TileTensor,
    TensorLayout,
    coord,
    row_major,
    stack_allocation,
)
from layout.tile_layout import Layout
from layout.layout_tensor import LayoutTensorIter
from layout.layout import Layout as LegacyLayout
from layout._host_device_tile_tensor import HostDeviceTileTensor
from quantization.qmatmul_gpu import (
    _quantized_scale_stage,
    _copy_quantized_scale_tile,
    _load_quantized_scale_registers,
)


@inline(.always)
def scale_value(group: Int, column: Int) -> BFloat16:
    return (Float32(group * 193 + column + 1) * 0.007).cast[.bfloat16]()


def scale_stream_kernel[
    CADENCE: Int,
    STAGES: Int,
    input_layout: TensorLayout,
    output_layout: TensorLayout,
](
    source: TileTensor[.bfloat16, input_layout, ImmutAnyOrigin],
    output: TileTensor[.bfloat16, output_layout, MutAnyOrigin],
):
    comptime assert output.rank == output.flat_rank == 4
    comptime assert source.rank == source.flat_rank == 2
    comptime BN = 128
    comptime N = 3 * BN
    comptime GROUPS = 16
    comptime ITERS = GROUPS * CADENCE
    comptime RING = (STAGES - 1 + CADENCE - 1) // CADENCE + 1
    var lane = Int(thread_idx.x % WARP_SIZE)
    var partition = Int(thread_idx.x // WARP_SIZE)
    # The BF16 suffix is reached through a byte pointer, as in packed weights.
    var suffix = source.unsafe_ptr().bitcast[UInt8]() + 16
    var global_scales = TileTensor(
        suffix.bitcast[BFloat16](), row_major[2 * GROUPS, N]()
    )
    var storage = stack_allocation[.bfloat16, address_space=.SHARED](
        row_major[2, RING, BN]()
    )
    var shared = TileTensor(
        storage.unsafe_ptr() + partition * RING * BN,
        row_major[RING, BN](),
    )
    comptime LegacyIter = LayoutTensorIter[
        .bfloat16,
        LegacyLayout.row_major(1, BN),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var previous = LegacyIter(
        shared.unsafe_ptr(), LegacyIter.linear_uint_type(RING * BN)
    )
    var legacy_global = global_scales.to_layout_tensor().tiled_iterator[
        1, BN, axis=0
    ](partition * GROUPS, 1)
    var registers_storage = stack_allocation[.bfloat16, address_space=.LOCAL](
        row_major[8, 3]()
    )
    var registers = TileTensor(
        registers_storage.unsafe_ptr() + 1,
        Layout(coord[8, 1], coord[3, 1]),
    )
    comptime for i in range(8):
        registers_storage[i, 0] = -29
        registers_storage[i, 2] = -30
    var legacy_registers = stack_allocation[.bfloat16, address_space=.LOCAL](
        row_major[8, 1]()
    )
    var consumer = 0
    var producer = 0
    comptime for stage in range(STAGES - 1):
        comptime if stage % CADENCE == 0:
            var dst = shared.tile[1, BN]((stage // CADENCE, 0))
            var src = global_scales.tile[1, BN](
                (partition * GROUPS + producer, 1)
            ).address_space_cast[.GENERIC]()
            _copy_quantized_scale_tile(dst, src)
            producer += 1
            legacy_global._incr()
    async_copy_commit_group()
    async_copy_wait_group(0)
    barrier()
    for k in range(ITERS):
        var stage = shared.tile[1, BN]((consumer, 0))
        var warp_scales = stage.tile[1, 64]((0, 1))
        _load_quantized_scale_registers(registers, warp_scales, lane)
        var legacy_scales = previous[].tile[1, 64](0, 1)
        legacy_registers.to_layout_tensor().vectorize[8, 1]().copy_from(
            legacy_scales.vectorize[1, 8]().distribute[
                LegacyLayout.row_major(8, 4), axis=0
            ](lane)
        )
        comptime for i in range(8):
            output[partition, k, lane, i] = registers[i, 0]
            output[partition, k, lane, 8 + i] = legacy_registers[i, 0]
            output[partition, k, lane, 16 + i] = registers_storage[i, 0]
            output[partition, k, lane, 24 + i] = registers_storage[i, 2]
        output[partition, k, lane, 32] = BFloat16(1)
        output[partition, k, lane, 33] = BFloat16(1)
        # The real loop issues this copy before advancing the scale consumer.
        if k + STAGES - 1 < ITERS and (k + STAGES - 1) % CADENCE == 0:
            var next_stage = _quantized_scale_stage(consumer, RING - 1, RING)
            var dst = shared.tile[1, BN]((next_stage, 0))
            var old_dst = previous.next_unsafe(
                LegacyIter.linear_uint_type(RING - 1)
            )[]
            output[partition, k, lane, 32] = BFloat16(
                1 if dst.unsafe_ptr() == old_dst.ptr else 0
            )
            var src = global_scales.tile[1, BN](
                (partition * GROUPS + producer, 1)
            ).address_space_cast[.GENERIC]()
            output[partition, k, lane, 33] = BFloat16(
                1 if src.unsafe_ptr() == legacy_global[].ptr else 0
            )
            _copy_quantized_scale_tile(dst, src)
            producer += 1
            legacy_global._incr()
        async_copy_commit_group()
        async_copy_wait_group(0)
        barrier()
        if (k + 1) % CADENCE == 0:
            consumer = _quantized_scale_stage(consumer, 1, RING)
            previous._incr()


def test_scale_stream[CADENCE: Int, STAGES: Int](ctx: DeviceContext) raises:
    comptime N = 384
    comptime GROUPS = 16
    comptime input_layout = row_major[1, 8 + 2 * GROUPS * N]()
    comptime output_layout = row_major[2, GROUPS * CADENCE, WARP_SIZE, 34]()
    var source = HostDeviceTileTensor[.bfloat16](input_layout, ctx)
    var output = HostDeviceTileTensor[.bfloat16](output_layout, ctx)
    var host = source.host_tensor()
    _ = host.fill(-31)
    for group in range(2 * GROUPS):
        for column in range(N):
            host[0, 8 + group * N + column] = scale_value(group, column)
    source.to_device()
    ctx.enqueue_function[
        scale_stream_kernel[
            CADENCE, STAGES, type_of(input_layout), type_of(output_layout)
        ]
    ](
        source.device_tensor().as_unsafe_any_origin(),
        output.device_tensor().as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=2 * WARP_SIZE,
    )
    output.to_host()
    var actual = output.host_tensor()
    for partition in range(2):
        for k in range(GROUPS * CADENCE):
            for lane in range(WARP_SIZE):
                assert_equal(actual[partition, k, lane, 32], BFloat16(1))
                assert_equal(actual[partition, k, lane, 33], BFloat16(1))
                for i in range(8):
                    var expected = scale_value(
                        partition * GROUPS + k // CADENCE,
                        128 + 64 + 8 * (lane // 4) + i,
                    )
                    assert_equal(actual[partition, k, lane, i], expected)
                    assert_equal(actual[partition, k, lane, 8 + i], expected)
                    assert_equal(
                        actual[partition, k, lane, 16 + i], BFloat16(-29)
                    )
                    assert_equal(
                        actual[partition, k, lane, 24 + i], BFloat16(-30)
                    )


def main() raises:
    var ctx = DeviceContext()
    test_scale_stream[1, 3](ctx)
    test_scale_stream[2, 4](ctx)
    test_scale_stream[4, 4](ctx)
