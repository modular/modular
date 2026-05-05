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
"""Minimal MI355 smoke test for the CDNA4 `f8f6f4` block-scaled MFMA wrappers.

This test intentionally validates the wrappers at the raw fragment level rather
than reconstructing a logical output matrix. It launches a single warp, runs
the same MFMA twice with identical packed raw fragments, and checks that
doubling one real E8M0 scale word doubles every accumulator lane for each
supported CDNA4 `f8f6f4` operand format and wrapper shape.
"""

from std.builtin.simd import _convert_f32_to_float8_ue8m0
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, lane_id
from std.gpu.host import DeviceContext
from std.gpu.host.info import MI355X
from std.memory import bitcast
from std.testing import assert_true
from std.utils import StaticTuple

from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)


@always_inline
def _pack_e8m0_scale_word(value: Float32) -> Int32:
    """Packs one exact E8M0 scale into all four bytes of the MFMA scale word."""
    var scale = _convert_f32_to_float8_ue8m0[target=DType.float8_e8m0fnu](value)
    var scale_byte = bitcast[DType.uint8](scale)
    return Int32(
        UInt32(scale_byte)
        | (UInt32(scale_byte) << 8)
        | (UInt32(scale_byte) << 16)
        | (UInt32(scale_byte) << 24)
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def _block_scaled_mma_smoke_kernel[
    BaselineLayout: TensorLayout,
    ScaledLayout: TensorLayout,
    accum_width: Int,
    matrix_format: CDNA4F8F6F4MatrixFormat,
](
    baseline_out: TileTensor[DType.float32, BaselineLayout, MutAnyOrigin],
    scaled_out: TileTensor[DType.float32, ScaledLayout, MutAnyOrigin],
):
    comptime assert accum_width == 4 or accum_width == 16, (
        "AMD block-scaled MMA smoke test only supports 4- or 16-lane"
        " accumulators"
    )
    var lane = Int(lane_id())
    var a_frag = SIMD[DType.uint8, 32](UInt8(0x21))
    var b_frag = SIMD[DType.uint8, 32](UInt8(0x12))

    var one = _pack_e8m0_scale_word(Float32(1.0))
    var two = _pack_e8m0_scale_word(Float32(2.0))

    comptime if accum_width == 4:
        var baseline_acc = SIMD[DType.float32, 4](0.0)
        var scaled_acc = SIMD[DType.float32, 4](0.0)
        cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
            baseline_acc,
            a_frag,
            b_frag,
            one,
            one,
        )
        cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
            scaled_acc,
            a_frag,
            b_frag,
            two,
            one,
        )
        baseline_out.store[width=4](
            Coord(Idx(lane), Idx[0]()),
            baseline_acc,
        )
        scaled_out.store[width=4](
            Coord(Idx(lane), Idx[0]()),
            scaled_acc,
        )
    else:
        var baseline_acc = SIMD[DType.float32, 16](0.0)
        var scaled_acc = SIMD[DType.float32, 16](0.0)
        cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
            baseline_acc,
            a_frag,
            b_frag,
            one,
            one,
        )
        cdna4_block_scaled_mfma[0, 0, matrix_format, matrix_format](
            scaled_acc,
            a_frag,
            b_frag,
            two,
            one,
        )
        baseline_out.store[width=16](
            Coord(Idx(lane), Idx[0]()),
            baseline_acc,
        )
        scaled_out.store[width=16](
            Coord(Idx(lane), Idx[0]()),
            scaled_acc,
        )


def _run_block_scaled_mma_amd_smoke[
    matrix_format: CDNA4F8F6F4MatrixFormat,
    accum_width: Int,
](ctx: DeviceContext) raises:
    comptime num_values = WARP_SIZE * accum_width

    var baseline_device = ctx.enqueue_create_buffer[DType.float32](num_values)
    var scaled_device = ctx.enqueue_create_buffer[DType.float32](num_values)

    var baseline_tt = TileTensor(
        baseline_device,
        row_major(Coord(Idx[WARP_SIZE](), Idx[accum_width]())),
    )
    var scaled_tt = TileTensor(
        scaled_device,
        row_major(Coord(Idx[WARP_SIZE](), Idx[accum_width]())),
    )

    comptime kernel = _block_scaled_mma_smoke_kernel[
        type_of(baseline_tt).LayoutType,
        type_of(scaled_tt).LayoutType,
        accum_width,
        matrix_format,
    ]

    ctx.enqueue_function[kernel](
        baseline_tt.as_any_origin(),
        scaled_tt.as_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()

    var baseline_host = alloc[Scalar[DType.float32]](num_values)
    var scaled_host = alloc[Scalar[DType.float32]](num_values)
    ctx.enqueue_copy(baseline_host, baseline_device)
    ctx.enqueue_copy(scaled_host, scaled_device)
    ctx.synchronize()

    var saw_nonzero = False
    for i in range(num_values):
        var baseline = baseline_host[i]
        var scaled = scaled_host[i]
        if abs(baseline) > Float32(1e-6):
            saw_nonzero = True
        var diff = abs(scaled - baseline * Float32(2.0))
        assert_true(
            diff <= Float32(1e-5),
            "scaled MFMA output should double when scale_a doubles",
        )
    assert_true(
        saw_nonzero, "wrapper smoke test should produce non-zero output"
    )


def test_block_scaled_mma_amd_smoke(ctx: DeviceContext) raises:
    comptime for matrix_format in [
        CDNA4F8F6F4MatrixFormat.FLOAT8_E4M3,
        CDNA4F8F6F4MatrixFormat.FLOAT8_E5M2,
        CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3,
        CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2,
        CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
    ]:
        _run_block_scaled_mma_amd_smoke[matrix_format, 4](ctx)
        _run_block_scaled_mma_amd_smoke[matrix_format, 16](ctx)


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "AMD block-scaled MMA smoke test requires MI355X"

    print("== test_block_scaled_mma_amd_smoke")
    test_block_scaled_mma_amd_smoke(ctx)
    print("PASS")
