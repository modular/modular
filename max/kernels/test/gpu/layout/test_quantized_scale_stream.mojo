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
    row_major,
    stack_allocation,
)
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
    var lane = thread_idx.x % WARP_SIZE
    var partition = thread_idx.x // WARP_SIZE
    # The BF16 suffix is reached through a byte pointer, as in packed weights.
    var suffix = source.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_offset(16)
    var global_scales = TileTensor(
        suffix.unsafe_bitcast[BFloat16](), row_major[2 * GROUPS, N]()
    )
    var storage = stack_allocation[.bfloat16, address_space=.SHARED](
        row_major[2, RING, BN]()
    )
    var shared = storage.tile[1, RING, BN]((partition, 0, 0)).reshape(
        row_major[RING, BN]()
    )
    var registers_storage = stack_allocation[.bfloat16, address_space=.LOCAL](
        row_major[8, 3]()
    )
    var registers = registers_storage.tile[8, 1]((0, 1))
    comptime for i in range(8):
        registers_storage[i, 0] = -29
        registers_storage[i, 2] = -30
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
    async_copy_commit_group()
    async_copy_wait_group(0)
    barrier()
    for k in range(ITERS):
        var stage = shared.tile[1, BN]((consumer, 0))
        var warp_scales = stage.tile[1, 64]((0, 1))
        _load_quantized_scale_registers(registers, warp_scales, lane)
        comptime for i in range(8):
            output[partition, k, lane, i] = registers[i, 0]
            output[partition, k, lane, 8 + i] = registers_storage[i, 0]
            output[partition, k, lane, 16 + i] = registers_storage[i, 2]
        # The real loop issues this copy before advancing the scale consumer.
        if k + STAGES - 1 < ITERS and (k + STAGES - 1) % CADENCE == 0:
            var next_stage = _quantized_scale_stage(consumer, RING - 1, RING)
            var dst = shared.tile[1, BN]((next_stage, 0))
            var src = global_scales.tile[1, BN](
                (partition * GROUPS + producer, 1)
            ).address_space_cast[.GENERIC]()
            _copy_quantized_scale_tile(dst, src)
            producer += 1
        async_copy_commit_group()
        async_copy_wait_group(0)
        barrier()
        if (k + 1) % CADENCE == 0:
            consumer = _quantized_scale_stage(consumer, 1, RING)


def test_scale_stream[CADENCE: Int, STAGES: Int](ctx: DeviceContext) raises:
    comptime warp_size = ctx.default_device_info.warp_size
    comptime N = 384
    comptime GROUPS = 16
    comptime input_layout = row_major[1, 8 + 2 * GROUPS * N]()
    comptime output_layout = row_major[2, GROUPS * CADENCE, warp_size, 24]()
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
        block_dim=2 * warp_size,
    )
    output.to_host()
    var actual = output.host_tensor()
    for partition in range(2):
        for k in range(GROUPS * CADENCE):
            for lane in range(warp_size):
                for i in range(8):
                    var expected = scale_value(
                        partition * GROUPS + k // CADENCE,
                        128 + 64 + 8 * (lane // 4) + i,
                    )
                    assert_equal(actual[partition, k, lane, i], expected)
                    assert_equal(
                        actual[partition, k, lane, 8 + i], BFloat16(-29)
                    )
                    assert_equal(
                        actual[partition, k, lane, 16 + i], BFloat16(-30)
                    )


def main() raises:
    var ctx = DeviceContext()
    test_scale_stream[1, 3](ctx)
    test_scale_stream[2, 4](ctx)
    test_scale_stream[4, 4](ctx)
