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

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    size_of,
    get_defined_bool,
    align_of,
)
from std.math import ceildiv, divmod
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from std.memory.alloc import unsafe_alloc
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout.tile_layout import TensorLayout
from layout._fillers import random
from max.gpu.host.info import MI355X, _is_sm10x_gpu

from std.utils.index import IndexList
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
)
from linalg.block_scaled_quantization import (
    quantize_dynamic_block_scaled,
    quantize_dynamic_scaled_fp4fp8,
    quantize_dynamic_scaled_fp4_async,
    grouped_quantize_dynamic_scaled_fp4_async,
    swiglu_quantize_dynamic_block_scaled,
)
from linalg.swiglu_activation import apply_swiglu_activation, swiglu
from internal_utils import arg_parse


def bench_1d1d_quantization[
    in_dtype: DType, cols: Int, use_async: Bool, is_fp4: Bool
](ctx: DeviceContext, mut b: Bench, fn_name: String, rows: Int) raises:
    comptime out_dtype = DType.uint8 if is_fp4 else DType.float8_e4m3fn
    comptime scales_dtype = NVFP4_SF_DTYPE if is_fp4 else MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE if is_fp4 else MXFP8_SF_VECTOR_SIZE
    comptime K_tiles = ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K)

    var in_device = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var input_tensor = TileTensor(in_device, row_major(rows, Idx[cols]))

    # Packed FP4 holds two values a byte; MXFP8 one.
    comptime out_cols = ceildiv(cols, 2) if is_fp4 else cols
    var out_device = ctx.enqueue_create_buffer[out_dtype](rows * out_cols)
    var output_tensor = TileTensor(out_device, row_major(rows, Idx[out_cols]))

    var scales_shape = IndexList[5](
        ceildiv(rows, SF_MN_GROUP_SIZE),
        K_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_shape.flattened_length()
    )
    var scales_tensor = TileTensor(
        scales_device,
        row_major(
            ceildiv(rows, SF_MN_GROUP_SIZE),
            Idx[K_tiles],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = TileTensor(in_host, row_major(rows, Idx[cols]))
        random(in_host_tensor)

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {var input_tensor, var output_tensor, var scales_tensor, imm,}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            # Run the quantization kernel
            comptime if use_async:
                quantize_dynamic_scaled_fp4_async[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE
                ](
                    ctx,
                    output_tensor.as_unsafe_any_origin(),
                    scales_tensor.as_unsafe_any_origin(),
                    input_tensor.as_unsafe_any_origin(),
                )
            else:
                quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    ctx,
                    output_tensor.as_unsafe_any_origin(),
                    scales_tensor.as_unsafe_any_origin(),
                    input_tensor.as_unsafe_any_origin(),
                    num_cols=cols,
                    num_cols_padded=cols,
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    var bytes = ThroughputMeasure(
        BenchMetric.bytes,
        (rows * cols) * size_of[in_dtype]()
        + (rows * cols // (2 if is_fp4 else 1)) * size_of[out_dtype]()
        + (ceildiv(rows, SF_MN_GROUP_SIZE) * K_tiles)
        * (SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K)
        * size_of[scales_dtype](),
    )

    b.bench_function(
        bench_fn,
        BenchId(
            "1d1d_quantization",
            input_id=String(
                fn_name,
                in_dtype,
                out_dtype,
                scales_dtype,
                SF_VECTOR_SIZE,
                rows,
                cols,
                sep="/",
            ),
        ),
        [bytes],
        # fixed_iterations=1,
    )

    ctx.synchronize()


def bench_grouped_quantization[
    in_dtype: DType, cols: Int, num_experts: Int, spread: Bool = True
](ctx: DeviceContext, mut b: Bench, rows: Int) raises:
    """Times the grouped NVFP4 quantize across expert-occupancy shapes.

    Cost here tracks occupied scale tiles rather than rows, because a block
    covers a whole 128-row tile whatever the tile holds. Sweeping
    `num_experts` at a fixed `rows` moves the tile count while the payload
    stays put, and sweeping `rows` at a fixed expert count does the reverse,
    so the two separate. Expect the expert sweep to move the number and the
    row sweep to barely register.
    """
    comptime out_dtype = DType.uint8
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime K_tiles = ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K)

    # `spread` shares the rows evenly across experts, which is what decode
    # looks like: top_k x batch routed pairs land on nearly that many distinct
    # experts, each holding a row or two and so each needing its own scale
    # tile. That makes the occupied tile count track the expert count rather
    # than the row count. The `spread=False` layout parks every row on one
    # expert and is the easy case, kept only as a contrast.
    var host_row_offsets = unsafe_alloc[UInt32](num_experts + 1)
    var host_scales_offsets = unsafe_alloc[UInt32](num_experts)
    var host_expert_ids = unsafe_alloc[Int32](num_experts)
    var host_sf = unsafe_alloc[Float32](num_experts)
    var tile_starts = unsafe_alloc[Int](num_experts + 1)

    host_row_offsets[0] = 0
    tile_starts[0] = 0
    var base_count, remainder = divmod(rows, num_experts)
    for i in range(num_experts):
        var count: Int
        comptime if spread:
            count = base_count + (1 if i < remainder else 0)
        else:
            count = rows if i == 0 else 0
        host_row_offsets[i + 1] = host_row_offsets[i] + UInt32(count)
        tile_starts[i + 1] = tile_starts[i] + ceildiv(count, SF_MN_GROUP_SIZE)
        host_scales_offsets[i] = UInt32(
            tile_starts[i] - Int(host_row_offsets[i]) // SF_MN_GROUP_SIZE
        )
        host_expert_ids[i] = Int32(i)
        host_sf[i] = 1.0

    # Occupied tiles, and the worst case the host must allocate for.
    var occupied_tiles = tile_starts[num_experts]
    var total_m_tiles = ceildiv(rows, SF_MN_GROUP_SIZE) + min(num_experts, rows)

    var dev_in = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var dev_out = ctx.enqueue_create_buffer[out_dtype](rows * ceildiv(cols, 2))
    var dev_scales = ctx.enqueue_create_buffer[scales_dtype](
        total_m_tiles * K_tiles * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var dev_row_offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var dev_scales_offsets = ctx.enqueue_create_buffer[.uint32](num_experts)
    var dev_expert_ids = ctx.enqueue_create_buffer[.int32](num_experts)
    var dev_sf = ctx.enqueue_create_buffer[.float32](num_experts)

    # The group max feeds a data-dependent branch, so the input has to be
    # defined: an all-zero buffer skips the reciprocal path that real
    # activations always take.
    var host_input = unsafe_alloc[Scalar[in_dtype]](rows * cols)
    var host_input_tensor = TileTensor(host_input, row_major(rows, Idx[cols]))
    random(host_input_tensor, min=-1.0, max=1.0)
    ctx.enqueue_copy(dev_in, host_input)

    ctx.enqueue_copy(dev_row_offsets, host_row_offsets)
    ctx.enqueue_copy(dev_scales_offsets, host_scales_offsets)
    ctx.enqueue_copy(dev_expert_ids, host_expert_ids)
    ctx.enqueue_copy(dev_sf, host_sf)

    var in_tensor = TileTensor(dev_in, row_major(rows, Idx[cols]))
    var out_tensor = TileTensor(dev_out, row_major(rows, Idx[ceildiv(cols, 2)]))
    var scales_tensor = TileTensor(
        dev_scales,
        row_major(
            total_m_tiles,
            Idx[K_tiles],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    var row_offsets_t = TileTensor(
        dev_row_offsets, row_major(Idx[num_experts + 1])
    )
    var scales_offsets_t = TileTensor(
        dev_scales_offsets, row_major(Idx[num_experts])
    )
    var expert_ids_t = TileTensor(dev_expert_ids, row_major(Idx[num_experts]))
    var sf_t = TileTensor(dev_sf, row_major(Idx[num_experts]))

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var out_tensor,
        var scales_tensor,
        var in_tensor,
        var row_offsets_t,
        var scales_offsets_t,
        var expert_ids_t,
        var sf_t,
        imm,
    }:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            grouped_quantize_dynamic_scaled_fp4_async(
                out_tensor,
                scales_tensor,
                in_tensor,
                row_offsets_t,
                scales_offsets_t,
                expert_ids_t,
                sf_t,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "grouped_quantization",
            input_id=String(
                "rows",
                rows,
                "cols",
                cols,
                "experts",
                num_experts,
                "spread" if spread else "single",
                "occupied",
                occupied_tiles,
                "alloc",
                total_m_tiles,
                sep="/",
            ),
        ),
    )
    ctx.synchronize()


@__name("bench_silu_mul_kernel")
def _bench_silu_mul_kernel[
    N: Int, h_layout: TensorLayout, gate_up_layout: TensorLayout
](
    h: TileTensor[.bfloat16, h_layout, MutAnyOrigin],
    gate_up: TileTensor[.bfloat16, gate_up_layout, ImmutAnyOrigin],
    num_rows: Int32,
):
    """Elementwise `h = swiglu(gate_up)`, bfloat16 rounded: the arm the
    fused gated MX quantize replaces on AMD.

    The 8-wide accesses claim only element alignment, not the 16-byte
    vector width that GPU loads and stores default to.
    """
    comptime alignment = align_of[BFloat16]()
    var i = global_idx.x
    if i >= Int(num_rows) * (N // 8):
        return
    var row, c = divmod(i, N // 8)
    var h_vec = h.vectorize[1, 8]()
    var gate_up_vec = gate_up.vectorize[1, 8]()
    h_vec.store[alignment=alignment](
        (row, c),
        apply_swiglu_activation[swiglu[0.0]](
            gate_up_vec.load[alignment=alignment]((row, c)),
            gate_up_vec.load[alignment=alignment]((row, c + N // 8)),
        ),
    )


def bench_mx_amd_quantization[
    cols: Int, is_fp4: Bool
](ctx: DeviceContext, mut b: Bench, rows: Int) raises:
    """Times the ungated AMD CDNA4 MX quantize: `[rows, cols]` bf16 in,
    nibble-packed MXFP4 or byte-per-element MXFP8 out."""
    comptime out_dtype = DType.uint8 if is_fp4 else DType.float8_e4m3fn
    comptime packed_cols = cols // 2 if is_fp4 else cols
    comptime scale_cols = cols // MXFP8_SF_VECTOR_SIZE

    var in_device = ctx.enqueue_create_buffer[.bfloat16](rows * cols)
    var input_tensor = TileTensor(in_device, row_major(rows, Idx[cols]))
    with in_device.map_to_host() as in_host:
        random(TileTensor(in_host, row_major(rows, Idx[cols])))

    var out_device = ctx.enqueue_create_buffer[out_dtype](rows * packed_cols)
    var output_tensor = TileTensor(
        out_device, row_major(rows, Idx[packed_cols])
    )
    var scales_device = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
        rows * scale_cols
    )
    var scales_tensor = TileTensor(
        scales_device, row_major(rows, Idx[scale_cols])
    )

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {var input_tensor, var output_tensor, var scales_tensor, imm}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            quantize_dynamic_block_scaled[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                output_tensor.as_unsafe_any_origin(),
                scales_tensor.as_unsafe_any_origin(),
                input_tensor.as_unsafe_any_origin(),
                Float32(1.0),
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    var bytes = ThroughputMeasure(
        BenchMetric.bytes,
        rows * cols * size_of[BFloat16]()
        + rows * packed_cols * size_of[out_dtype]()
        + rows * scale_cols * size_of[MXFP8_SF_DTYPE](),
    )

    b.bench_function(
        bench_fn,
        BenchId(
            "quant_amd",
            input_id=String("mx4" if is_fp4 else "mx8", rows, cols, sep="/"),
        ),
        [bytes],
    )
    ctx.synchronize()


def bench_mx_amd_gated[
    gate_cols: Int, is_fp4: Bool, fused: Bool
](ctx: DeviceContext, mut b: Bench, rows: Int) raises:
    """Times the gated MX quantize over a `[rows, 2 * gate_cols]` bf16
    `[gate | up]` tile.

    `fused=False` times the two kernels main runs instead on AMD: the
    swiglu elementwise kernel writing a bf16 `[rows, gate_cols]`
    activation, then the ungated quantize over it. `fused=True` times the
    single kernel. Both arms report the same BenchId, so the plot pairs
    them.
    """
    comptime cols = 2 * gate_cols
    comptime out_dtype = DType.uint8 if is_fp4 else DType.float8_e4m3fn
    comptime packed_cols = gate_cols // 2 if is_fp4 else gate_cols
    comptime scale_cols = gate_cols // MXFP8_SF_VECTOR_SIZE

    var in_device = ctx.enqueue_create_buffer[.bfloat16](rows * cols)
    var input_tensor = TileTensor(in_device, row_major(rows, Idx[cols]))
    with in_device.map_to_host() as in_host:
        random(TileTensor(in_host, row_major(rows, Idx[cols])))

    var out_device = ctx.enqueue_create_buffer[out_dtype](rows * packed_cols)
    var output_tensor = TileTensor(
        out_device, row_major(rows, Idx[packed_cols])
    )
    var scales_device = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
        rows * scale_cols
    )
    var scales_tensor = TileTensor(
        scales_device, row_major(rows, Idx[scale_cols])
    )
    # The unfused arm stages the bf16 activation here; the fused arm never
    # touches it, so it holds one element rather than the whole tile.
    var h_elems = 1
    comptime if not fused:
        h_elems = rows * gate_cols
    var h_device = ctx.enqueue_create_buffer[.bfloat16](h_elems)

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var input_tensor,
        var output_tensor,
        var scales_tensor,
        var h_device,
        imm,
    }:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            comptime if fused:
                # The op the graph calls; on AMD it routes MXFP4 and MXFP8
                # to the gated kernel.
                swiglu_quantize_dynamic_block_scaled[
                    SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
                    activation=swiglu[0.0],
                ](
                    output_tensor.as_unsafe_any_origin(),
                    scales_tensor.as_unsafe_any_origin(),
                    input_tensor.as_unsafe_any_origin(),
                    Float32(1.0),
                    ctx,
                )
            else:
                var h_tensor = TileTensor(
                    h_device, row_major(rows, Idx[gate_cols])
                )
                ctx.enqueue_function[
                    _bench_silu_mul_kernel[
                        gate_cols,
                        type_of(h_tensor).LayoutType,
                        type_of(input_tensor).LayoutType,
                    ]
                ](
                    h_tensor,
                    input_tensor.as_imm(),
                    Int32(rows),
                    grid_dim=ceildiv(rows * gate_cols, 8 * 256),
                    block_dim=256,
                )
                quantize_dynamic_block_scaled[
                    SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE
                ](
                    output_tensor.as_unsafe_any_origin(),
                    scales_tensor.as_unsafe_any_origin(),
                    h_tensor.as_unsafe_any_origin(),
                    Float32(1.0),
                    ctx,
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    # The unfused arm also writes the [rows, gate_cols] bf16 activation and
    # reads it back to quantize; charging both arms only the input and the
    # final outputs would overstate that arm's bandwidth.
    var moved_bytes = (
        rows * cols * size_of[BFloat16]()
        + rows * packed_cols * size_of[out_dtype]()
        + rows * scale_cols * size_of[MXFP8_SF_DTYPE]()
    )
    comptime if not fused:
        moved_bytes += 2 * rows * gate_cols * size_of[BFloat16]()
    var bytes = ThroughputMeasure(BenchMetric.bytes, moved_bytes)

    b.bench_function(
        bench_fn,
        BenchId(
            "gated_amd",
            input_id=String(
                "mx4" if is_fp4 else "mx8", rows, gate_cols, sep="/"
            ),
        ),
        [bytes],
    )
    ctx.synchronize()


def main() raises:
    comptime in_dtype = get_defined_dtype["dtype", .bfloat16]()

    var rows = Int(arg_parse("M", 1))
    comptime cols = get_defined_int["N", 4096]()
    comptime use_async = get_defined_bool["use_async", True]()
    comptime is_fp4 = get_defined_bool["is_fp4", True]()
    # The grouped cases carry their own row counts, so they ignore `M` and
    # would repeat identically across an `M` sweep. Opt in with -D grouped=1.
    comptime grouped = get_defined_bool["grouped", False]()
    # Which AMD MI355X case to time; see the dispatch below. The output
    # format stays with `is_fp4`, so the define means the same thing on both
    # vendors.
    comptime variant = get_defined_int["variant", 0]()

    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            var m = Bench(BenchConfig(num_repetitions=1))
            comptime if grouped:
                # Decode: batch 32 x top_k 6 routed pairs over 256 experts.
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 32](ctx, m, 192)
                bench_grouped_quantization[in_dtype, cols, 8](ctx, m, 192)
                # Smaller decode batch: 8 x top_k 6 routed pairs. This sits
                # just above the batching crossover, so it checks the
                # threshold picks the faster path there rather than only for
                # large batches.
                bench_grouped_quantization[in_dtype, cols, 48](ctx, m, 48)
                # Prefill-scale row counts, at both expert counts the shipping
                # models use.
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 768)
                bench_grouped_quantization[in_dtype, cols, 256](ctx, m, 3072)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 768)
                bench_grouped_quantization[in_dtype, cols, 128](ctx, m, 3072)
                # Single-expert contrast: every row on one expert, so the
                # occupied tile count collapses to the row tiles.
                bench_grouped_quantization[in_dtype, cols, 256, spread=False](
                    ctx, m, 192
                )
            else:
                bench_1d1d_quantization[in_dtype, cols, use_async, is_fp4](
                    ctx, m, "1d1d_quantization", rows
                )
            m.dump_report()
        elif ctx.default_device_info == MI355X:
            var m = Bench(BenchConfig(num_repetitions=1))
            # 0 times the ungated quantize, 1 the gated fusion, 2 the
            # two-kernel chain it replaces. All three take their output
            # format from `is_fp4`, as the SM100 arm above does, so
            # `-D is_fp4=False` times MXFP8 instead of timing MXFP4 under an
            # MXFP8 request. The fused and unfused arms report the same
            # BenchId, so two CSVs pair up bar for bar.
            comptime if variant == 0:
                bench_mx_amd_quantization[cols, is_fp4](ctx, m, rows)
            elif variant == 1:
                bench_mx_amd_gated[cols, is_fp4, True](ctx, m, rows)
            else:
                comptime assert variant == 2, (
                    "variant must be 0 (ungated), 1 (gated fused), or 2"
                    " (gated unfused)"
                )
                bench_mx_amd_gated[cols, is_fp4, False](ctx, m, rows)
            m.dump_report()
        else:
            print("this benchmark runs on NVIDIA SM100 and AMD MI355X only")
