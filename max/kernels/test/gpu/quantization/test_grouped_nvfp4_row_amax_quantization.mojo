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

from std.math import ceildiv
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_equal
from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import Layout
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
)
from linalg.block_scaled_quantization import (
    grouped_quantize_dynamic_scaled_fp4_async,
    grouped_quantize_nvfp4_row_amax,
)
from nn.activations import relu

comptime _RANGE = Float32(6.0 * 448.0)
# Fills the row-scale buffer, so a slot the kernel must skip is checkable.
comptime _ROW_SCALE_SENTINEL = Scalar[DType.bfloat16](-3.0)


def _scales_tile_gap(row_scales_stride: Int) -> Int:
    """Unused scale bytes between M tiles in the strided-output cases."""
    return 0 if row_scales_stride == 1 else 512


def _next_random(mut state: UInt64) -> Float32:
    """A deterministic uniform value in [0, 1)."""
    state = state * 6364136223846793005 + 1442695040888963407
    return Float32(Int(state >> 40)) / Float32(1 << 24)


def _fill_row(
    row: MutPointer[Scalar[DType.bfloat16], _],
    n: Int,
    kind: Int,
    mut state: UInt64,
):
    """Fills a row with one of the cases the quantizer must get right."""
    for c in range(n):
        var u = _next_random(state) * 2.0 - 1.0
        var v = u
        match kind:
            case 1:
                v = 0.0
            case 2:
                # One huge outlier among small values.
                v = u * 1e-3
                if c == (n // 3):
                    v = 3.0e4
            case 3:
                # A tiny row whose amax / 2688 falls below the 1e-30 floor.
                v = u * 1e-33
            case 4:
                # A single non-zero element.
                v = 0.0
                if c == n - 1:
                    v = u + 2.0
            case 5:
                # Every value at a quantization tie and the largest scale.
                v = 1.25 if c % 2 == 0 else -1.5
            case 6:
                # Wide dynamic range across 16-element blocks.
                v = u * Float32(10.0) ** Float32(((c // 16) % 9) - 4)
            case 7:
                # Mostly negative: relu2 zeros most of it.
                v = u - 0.8
            case _:
                pass
        row[c] = v.cast[DType.bfloat16]()


def test_row_amax_quantization[
    N: Int, apply_relu2: Bool, row_scales_stride: Int = 1
](
    ctx: DeviceContext, expert_counts: List[Int], top_k: Int, gather: Bool
) raises:
    """Compares the fused kernel with the chain it replaces.

    The reference rounds on the host exactly as the graph chain does, then
    quantizes with the existing grouped kernel and a tensor-wise scale of 1.
    A `row_scales_stride` above 1 writes the row scales through a strided
    view, and the block scales through a view with a gap between M tiles;
    the skipped slots and the gaps must keep their contents.
    """
    var num_experts = len(expert_counts)
    comptime out_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime output_N = N // 2
    comptime scales_per_m_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    comptime tile_gap = _scales_tile_gap(row_scales_stride)
    comptime tile_stride = scales_per_m_tile + tile_gap

    var row_offsets_host = unsafe_alloc[UInt32](num_experts + 1)
    var scales_offsets_host = unsafe_alloc[UInt32](num_experts)
    var expert_ids_host = unsafe_alloc[Int32](num_experts)
    var ones_host = unsafe_alloc[Float32](num_experts)
    var tile_starts = unsafe_alloc[Int](num_experts + 1)
    row_offsets_host[0] = 0
    tile_starts[0] = 0
    for i in range(num_experts):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(expert_counts[i])
        tile_starts[i + 1] = tile_starts[i] + ceildiv(
            expert_counts[i], SF_MN_GROUP_SIZE
        )
        scales_offsets_host[i] = UInt32(
            tile_starts[i] - Int(row_offsets_host[i]) // SF_MN_GROUP_SIZE
        )
        expert_ids_host[i] = Int32(i)
        ones_host[i] = 1.0
    var total = Int(row_offsets_host[num_experts])
    var total_m_tiles = tile_starts[num_experts]
    var total_scales = total_m_tiles * scales_per_m_tile

    # Source rows, and the source row each output row reads.
    var src_rows = max(1, total // top_k) if gather else total
    var src_host = unsafe_alloc[Scalar[DType.bfloat16]](src_rows * N)
    var state = UInt64(12345 + N)
    for r in range(src_rows):
        _fill_row(src_host.unsafe_offset(r * N), N, r % 8, state)
    var indices_host = unsafe_alloc[UInt32](max(total, 1))
    var source_of = unsafe_alloc[Int](max(total, 1))
    for i in range(total):
        if gather:
            var token = (i * 7 + 3) % src_rows
            source_of[i] = token
            indices_host[i] = UInt32(token * top_k + i % top_k)
        else:
            source_of[i] = i
            indices_host[i] = UInt32(i)

    # Host reference of the graph chain.
    var scaled_host = unsafe_alloc[Scalar[DType.bfloat16]](max(total * N, 1))
    var ref_row_scales = unsafe_alloc[Scalar[DType.bfloat16]](max(total, 1))
    for i in range(total):
        var amax = Float32(0)
        var row = unsafe_alloc[Scalar[DType.bfloat16]](N)
        for c in range(N):
            var x = src_host[source_of[i] * N + c]
            comptime if apply_relu2:
                var r = max(x, Scalar[DType.bfloat16](0)).cast[DType.float32]()
                x = (r * r).cast[DType.bfloat16]()
            row[c] = x
            amax = max(amax, abs(x.cast[DType.float32]()))
        var rs = max(amax / _RANGE, Float32(1e-30)).cast[DType.bfloat16]()
        ref_row_scales[i] = rs
        var inv = Float32(1.0) / rs.cast[DType.float32]()
        for c in range(N):
            scaled_host[i * N + c] = (row[c].cast[DType.float32]() * inv).cast[
                DType.bfloat16
            ]()
        row.unsafe_free()

    var dev_src = ctx.enqueue_create_buffer[DType.bfloat16](src_rows * N)
    var dev_scaled = ctx.enqueue_create_buffer[DType.bfloat16](
        max(total * N, 1)
    )
    var dev_indices = ctx.enqueue_create_buffer[DType.uint32](max(total, 1))
    var dev_row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var dev_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var dev_expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var dev_ones = ctx.enqueue_create_buffer[.float32](num_experts)
    ctx.enqueue_copy(dev_src, src_host)
    ctx.enqueue_copy(dev_scaled, scaled_host)
    ctx.enqueue_copy(dev_indices, indices_host)
    ctx.enqueue_copy(dev_row_offsets, row_offsets_host)
    ctx.enqueue_copy(dev_scales_offsets, scales_offsets_host)
    ctx.enqueue_copy(dev_expert_ids, expert_ids_host)
    ctx.enqueue_copy(dev_ones, ones_host)

    # One slack row, and every scale preset to a non-zero value, so a missing
    # payload write, a write past the rows, or a padding slot left unzeroed
    # all show.
    var dev_out_ref = ctx.enqueue_create_buffer[out_dtype](
        max(total * output_N, 1) + output_N
    )
    var dev_out = ctx.enqueue_create_buffer[out_dtype](
        max(total * output_N, 1) + output_N
    )
    var dev_scales_ref = ctx.enqueue_create_buffer[scales_dtype](
        max(total_scales, 1)
    )
    var dev_scales = ctx.enqueue_create_buffer[scales_dtype](
        max(total_m_tiles * tile_stride, 1)
    )
    var dev_row_scales = ctx.enqueue_create_buffer[DType.bfloat16](
        max(total * row_scales_stride, 1)
    )
    ctx.enqueue_memset(dev_out_ref, Scalar[out_dtype](0xEE))
    ctx.enqueue_memset(dev_out, Scalar[out_dtype](0xEE))
    ctx.enqueue_memset(dev_scales_ref, Scalar[scales_dtype](1.0))
    ctx.enqueue_memset(dev_scales, Scalar[scales_dtype](1.0))
    ctx.enqueue_memset(dev_row_scales, _ROW_SCALE_SENTINEL)

    ctx.synchronize()
    # The tensors carry the expert count statically, as the graph ops do.
    comptime for experts in [2, 3, 128]:
        if num_experts == experts:
            _run_case[N, experts, apply_relu2, row_scales_stride](
                ctx,
                dev_src,
                dev_scaled,
                dev_indices,
                dev_row_offsets,
                dev_scales_offsets,
                dev_expert_ids,
                dev_ones,
                dev_out_ref,
                dev_out,
                dev_scales_ref,
                dev_scales,
                dev_row_scales,
                total,
                src_rows,
                total_m_tiles,
                top_k,
                gather,
            )

    var host_out_ref = unsafe_alloc[Scalar[out_dtype]](
        total * output_N + output_N
    )
    var host_out = unsafe_alloc[Scalar[out_dtype]](total * output_N + output_N)
    var host_scales_ref = unsafe_alloc[Scalar[scales_dtype]](
        max(total_scales, 1)
    )
    var host_scales = unsafe_alloc[Scalar[scales_dtype]](
        max(total_m_tiles * tile_stride, 1)
    )
    var host_row_scales = unsafe_alloc[Scalar[DType.bfloat16]](
        max(total * row_scales_stride, 1)
    )
    ctx.enqueue_copy(host_out_ref, dev_out_ref)
    ctx.enqueue_copy(host_out, dev_out)
    ctx.enqueue_copy(host_scales_ref, dev_scales_ref)
    ctx.enqueue_copy(host_scales, dev_scales)
    ctx.enqueue_copy(host_row_scales, dev_row_scales)
    ctx.synchronize()

    for i in range(total * output_N + output_N):
        assert_equal(Int(host_out[i]), Int(host_out_ref[i]))
    for t in range(total_m_tiles):
        for i in range(scales_per_m_tile):
            assert_equal(
                Int(host_scales.unsafe_bitcast[UInt8]()[t * tile_stride + i]),
                Int(
                    host_scales_ref.unsafe_bitcast[UInt8]()[
                        t * scales_per_m_tile + i
                    ]
                ),
            )
        for i in range(scales_per_m_tile, tile_stride):
            assert_equal(
                Int(host_scales.unsafe_bitcast[UInt8]()[t * tile_stride + i]),
                Int(bitcast[DType.uint8, 1](Scalar[scales_dtype](1.0))),
            )
    for i in range(total):
        assert_equal(
            Int(
                host_row_scales.unsafe_bitcast[UInt16]()[i * row_scales_stride]
            ),
            Int(ref_row_scales.unsafe_bitcast[UInt16]()[i]),
        )
        for j in range(1, row_scales_stride):
            assert_equal(
                Int(
                    host_row_scales.unsafe_bitcast[UInt16]()[
                        i * row_scales_stride + j
                    ]
                ),
                Int(bitcast[DType.uint16, 1](_ROW_SCALE_SENTINEL)),
            )

    host_out_ref.unsafe_free()
    host_out.unsafe_free()
    host_scales_ref.unsafe_free()
    host_scales.unsafe_free()
    host_row_scales.unsafe_free()
    src_host.unsafe_free()
    indices_host.unsafe_free()
    source_of.unsafe_free()
    scaled_host.unsafe_free()
    ref_row_scales.unsafe_free()
    row_offsets_host.unsafe_free()
    scales_offsets_host.unsafe_free()
    expert_ids_host.unsafe_free()
    ones_host.unsafe_free()
    tile_starts.unsafe_free()


def _run_case[
    N: Int, num_experts: Int, apply_relu2: Bool, row_scales_stride: Int
](
    ctx: DeviceContext,
    dev_src: DeviceBuffer[DType.bfloat16],
    dev_scaled: DeviceBuffer[DType.bfloat16],
    dev_indices: DeviceBuffer[DType.uint32],
    dev_row_offsets: DeviceBuffer[DType.uint32],
    dev_scales_offsets: DeviceBuffer[DType.uint32],
    dev_expert_ids: DeviceBuffer[DType.int32],
    dev_ones: DeviceBuffer[DType.float32],
    mut dev_out_ref: DeviceBuffer[DType.uint8],
    mut dev_out: DeviceBuffer[DType.uint8],
    mut dev_scales_ref: DeviceBuffer[NVFP4_SF_DTYPE],
    mut dev_scales: DeviceBuffer[NVFP4_SF_DTYPE],
    mut dev_row_scales: DeviceBuffer[DType.bfloat16],
    total: Int,
    src_rows: Int,
    total_m_tiles: Int,
    top_k: Int,
    gather: Bool,
) raises:
    comptime K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime output_N = N // 2
    var scales_shape = Coord(
        total_m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var row_offsets_t = TileTensor(
        dev_row_offsets, row_major(Idx[num_experts + 1])
    )
    var scales_offsets_t = TileTensor(
        dev_scales_offsets, row_major(Idx[num_experts])
    )
    var expert_ids_t = TileTensor(dev_expert_ids, row_major(Idx[num_experts]))
    var ones_t = TileTensor(dev_ones, row_major(Idx[num_experts]))

    # Reference: the existing kernel on the host-rounded rows.
    grouped_quantize_dynamic_scaled_fp4_async(
        TileTensor(dev_out_ref, row_major(Coord(total, Idx[output_N]))),
        TileTensor(dev_scales_ref, row_major(scales_shape)),
        TileTensor(dev_scaled, row_major(Coord(total, Idx[N]))),
        row_offsets_t,
        scales_offsets_t,
        expert_ids_t,
        ones_t,
        ctx,
    )

    comptime if row_scales_stride == 1:
        _launch_fused[N, num_experts, apply_relu2, row_scales_stride](
            ctx,
            TileTensor(dev_row_scales, row_major(Coord(total))),
            dev_out,
            dev_scales,
            dev_src,
            dev_indices,
            dev_row_offsets,
            dev_scales_offsets,
            total,
            src_rows,
            total_m_tiles,
            top_k,
            gather,
        )
    else:
        _launch_fused[N, num_experts, apply_relu2, row_scales_stride](
            ctx,
            TileTensor(
                dev_row_scales,
                Layout(Coord(total), Coord(Idx[row_scales_stride])),
            ),
            dev_out,
            dev_scales,
            dev_src,
            dev_indices,
            dev_row_offsets,
            dev_scales_offsets,
            total,
            src_rows,
            total_m_tiles,
            top_k,
            gather,
        )
    ctx.synchronize()


def _launch_fused[
    N: Int, num_experts: Int, apply_relu2: Bool, row_scales_stride: Int
](
    ctx: DeviceContext,
    row_scales_t: TileTensor[
        mut=True, DType.bfloat16, address_space=.GENERIC, ...
    ],
    mut dev_out: DeviceBuffer[DType.uint8],
    mut dev_scales: DeviceBuffer[NVFP4_SF_DTYPE],
    dev_src: DeviceBuffer[DType.bfloat16],
    dev_indices: DeviceBuffer[DType.uint32],
    dev_row_offsets: DeviceBuffer[DType.uint32],
    dev_scales_offsets: DeviceBuffer[DType.uint32],
    total: Int,
    src_rows: Int,
    total_m_tiles: Int,
    top_k: Int,
    gather: Bool,
) raises:
    """Runs the fused kernel as the graph op does, relu2 included."""
    comptime K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime scales_per_m_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    comptime output_N = N // 2

    @inline(.always)
    def input_fn[
        dtype: DType, width: SIMDLength
    ](v: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime if apply_relu2:
            var r = relu(v)
            return r * r
        else:
            return v

    var scales_shape = Coord(
        total_m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var row_offsets_t = TileTensor(
        dev_row_offsets, row_major(Idx[num_experts + 1])
    )
    var scales_offsets_t = TileTensor(
        dev_scales_offsets, row_major(Idx[num_experts])
    )
    var out_t = TileTensor(dev_out, row_major(Coord(total, Idx[output_N])))
    # Row-major within an M tile; the M-tile stride may leave a gap.
    var scales_t = TileTensor(
        dev_scales,
        Layout(
            scales_shape,
            Coord(
                scales_per_m_tile + _scales_tile_gap(row_scales_stride),
                Idx[SF_MN_GROUP_SIZE * SF_ATOM_K],
                Idx[SF_ATOM_M[1] * SF_ATOM_K],
                Idx[SF_ATOM_K],
                Idx[1],
            ),
        ),
    )
    var src_t = TileTensor(dev_src, row_major(Coord(src_rows, Idx[N])))
    if gather:
        var idx_t = (
            TileTensor(dev_indices, row_major(Coord(total)))
            .as_unsafe_any_origin()
            .as_imm()
        )
        if top_k == 6:
            grouped_quantize_nvfp4_row_amax[
                input_fn=input_fn,
                row_indices_dtype=DType.uint32,
                row_index_divisor=6,
            ](
                out_t,
                scales_t,
                row_scales_t,
                src_t,
                row_offsets_t,
                scales_offsets_t,
                ctx,
                idx_t,
            )
        else:
            grouped_quantize_nvfp4_row_amax[
                input_fn=input_fn,
                row_indices_dtype=DType.uint32,
                row_index_divisor=1,
            ](
                out_t,
                scales_t,
                row_scales_t,
                src_t,
                row_offsets_t,
                scales_offsets_t,
                ctx,
                idx_t,
            )
    else:
        grouped_quantize_nvfp4_row_amax[input_fn=input_fn](
            out_t,
            scales_t,
            row_scales_t,
            src_t,
            row_offsets_t,
            scales_offsets_t,
            ctx,
        )


def test_empty_batch(ctx: DeviceContext) raises:
    """An empty batch (no rows, no scale tiles) returns without launching."""
    var dev_out = ctx.enqueue_create_buffer[DType.uint8](1)
    var dev_scales = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](1)
    var dev_row_scales = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var dev_src = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var dev_indices = ctx.enqueue_create_buffer[DType.uint32](1)
    var dev_row_offsets = ctx.enqueue_create_buffer[DType.uint32](3)
    var dev_scales_offsets = ctx.enqueue_create_buffer[DType.uint32](2)
    ctx.enqueue_memset(dev_row_offsets, UInt32(0))
    ctx.enqueue_memset(dev_scales_offsets, UInt32(0))
    for gather in [False, True]:
        _launch_fused[352, 2, True, 1](
            ctx,
            TileTensor(dev_row_scales, row_major(Coord(0))),
            dev_out,
            dev_scales,
            dev_src,
            dev_indices,
            dev_row_offsets,
            dev_scales_offsets,
            0,
            0,
            0,
            6,
            gather,
        )


def main() raises:
    with DeviceContext() as ctx:
        # The Nemotron-H shapes: the hidden size, and the expert width that
        # the down projection reads under relu2.
        comptime for relu2 in [False, True]:
            test_row_amax_quantization[2688, relu2](
                ctx, [256, 128, 100], 0, False
            )
            test_row_amax_quantization[1856, relu2](ctx, [13, 200], 0, False)
            # Source rows gathered through the routing slots.
            test_row_amax_quantization[2688, relu2](ctx, [300, 90, 1], 6, True)
            test_row_amax_quantization[1856, relu2](ctx, [40, 40, 40], 1, True)
            # A width that does not fill its last 64-column scale tile.
            test_row_amax_quantization[352, relu2](ctx, [100, 60], 0, False)
            # A row too wide for the kernel's register cache, read twice.
            test_row_amax_quantization[3264, relu2](ctx, [70, 3], 0, False)

        # Decode routing: many experts holding a few rows each.
        var decode = List[Int]()
        for i in range(128):
            decode.append(1 + (i % 5) * 7 if i % 2 == 0 else 0)
        comptime for relu2 in [False, True]:
            test_row_amax_quantization[2688, relu2](ctx, decode, 6, True)

        # Row scales written through a strided view (stride 4), and block
        # scales through a view with a 512-byte gap between M tiles.
        test_row_amax_quantization[352, False, row_scales_stride=4](
            ctx, [100, 60], 0, False
        )
        test_row_amax_quantization[1856, True, row_scales_stride=4](
            ctx, [40, 40, 40], 1, True
        )

        # No rows at all, as an empty batch or an empty gather gives.
        test_empty_batch(ctx)
