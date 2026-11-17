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
"""Sparse MLA decode over a paged FP8 latent cache, as one sweep.

The sweep port of the deleted `gpu/nn/bench_mla_decode_sparse_fp8.mojo`. It
covers the speculative-decode verify: several query positions per sequence,
each attending its own top-k window. The measured region is the split-K decode and
the combine that reduces its partial outputs.

Runtime axes are `--batch_size`, `--seq_len`, `--cache_len` and `--top_k`,
each a comma list; `top_k` is only the index row stride to this kernel, so it
costs no compile. `q_dtype` is a comptime axis run innermost, because FP8 and
BF16 Q reach different kernels and their ratio is the comparison; `-D
q_dtype=<dtype>` pins it. AMD has no FP8 x BF16 MMA, so it runs FP8 Q only.
`num_heads` and `page_size` are `-D` defines, as before.

`--num_partitions` is one more axis: 0 keeps the dispatch's split, and a
positive count forces it into the grid and the kernel alike. `--logical=1`
also passes each pick's logical key position, which turns on the per-slot
position mask. `--pdl_pred_ns=<ns>` launches a one-CTA kernel ahead of each
decode that triggers its PDL successor at entry and then spins that long, so
the decode's entry wait has a predecessor still running; in the bench loop
the predecessor is otherwise the previous combine, which triggers only by
completing. With `-D MLA_STAMP=1` each config dumps the kernel's clock
stamps before it is timed.

`--index_stride=3` reproduces the old benchmark's gather pattern (it also
shifted every pick by one key); the default spreads the picks over the window,
as `bench_dsa_block` does. `--fill=host` and `--page_layout` are as in
`bench_dsa_block`; the measurement flags (`--target_time`, `--l2`,
`--throttle_check`, `--json`) are `SweepHarness`'s.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.sys import get_defined_int

from bench_sweep import (
    Config,
    Fields,
    Footprint,
    PageLayout,
    PagedBlocks,
    PageTable,
    Sweep,
    SweepHarness,
    device_budget,
    dtype_arm_enabled,
    fill_affine,
    fill_sparse_indices,
    fill_uniform,
    fill_uniform_host_rand,
    plan_groups,
)
from internal_utils import arg_parse
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from mla_sweep import dump_stamps, early_trigger_spin, force_split
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_utils import MHAConfig
from std.benchmark import BenchId
from nn.attention.gpu.mla import AMDMLA, MLAKernels, SM100MLA

comptime fp8 = DType.float8_e4m3fn
comptime KV_LATENT_DIM = 512
comptime ROPE_DEPTH = 64
comptime Q_DEPTH = KV_LATENT_DIM + ROPE_DEPTH
comptime V_DEPTH = KV_LATENT_DIM
comptime NUM_HEADS = get_defined_int["num_heads", 8]()
comptime PAGE_SIZE = get_defined_int["page_size", 128]()


struct Workspace(Movable):
    """Per-token buffers, sized for the largest config of a sweep group.

    Q contents do not affect timing, so both Q dtypes are filled once; index
    lists and row offsets depend on the shape and are rewritten per config.
    """

    var q_fp8: DeviceBuffer[fp8]
    var q_bf16: DeviceBuffer[DType.bfloat16]
    var out: DeviceBuffer[DType.bfloat16]
    var indices: DeviceBuffer[DType.int32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var logical: DeviceBuffer[DType.int32]
    var logical_lut: DeviceBuffer[DType.uint32]

    def __init__(
        out self, ctx: DeviceContext[], footprint: Footprint, on_host: Bool
    ) raises:
        var q_elems = footprint["tokens"] * NUM_HEADS * Q_DEPTH
        self.q_fp8 = ctx.enqueue_create_buffer[fp8](q_elems)
        self.q_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](q_elems)
        self.out = ctx.enqueue_create_buffer[DType.bfloat16](
            footprint["tokens"] * NUM_HEADS * V_DEPTH
        )
        self.indices = ctx.enqueue_create_buffer[DType.int32](
            footprint["index_slots"]
        )
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](
            footprint["batch"] + 1
        )
        self.logical = ctx.enqueue_create_buffer[DType.int32](
            max(1, footprint["logical_slots"])
        )
        self.logical_lut = ctx.enqueue_create_buffer[DType.uint32](
            max(1, footprint["logical_lut"])
        )
        if on_host:
            fill_uniform_host_rand(self.q_fp8)
            fill_uniform_host_rand(self.q_bf16)
        else:
            fill_uniform(ctx, self.q_fp8, 10)
            fill_uniform(ctx, self.q_bf16, 10)

    @staticmethod
    def require(
        cfg: Config, mut footprint: Footprint, copies: Int, logical: Bool
    ) raises:
        var tokens = cfg["batch_size"] * cfg["seq_len"]
        footprint.need("tokens", tokens)
        footprint.need("index_slots", copies * tokens * cfg["top_k"])
        if logical:
            footprint.need("logical_slots", tokens * cfg["top_k"])
            footprint.need(
                "logical_lut",
                cfg["batch_size"]
                * ceildiv(cfg["cache_len"] + cfg["seq_len"], PAGE_SIZE),
            )

    @staticmethod
    def bytes_for(footprint: Footprint) -> Int:
        return (
            footprint["tokens"] * NUM_HEADS * (3 * Q_DEPTH + 2 * V_DEPTH)
            + 4 * footprint["index_slots"]
            + 4 * (footprint["batch"] + 1)
            + 4 * footprint["logical_slots"]
            + 4 * footprint["logical_lut"]
        )


def _run_name[
    q_dtype: DType
](
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    top_k: Int,
    num_partitions: Int,
    logical: Bool,
) -> String:
    # fmt: off
    var name = String(
        "mla_decode_sparse_fp8 : ",
        "q_dtype=", q_dtype, ", ",
        "num_heads=", NUM_HEADS, ", ",
        "page_size=", PAGE_SIZE, ", ",
        "top_k=", top_k, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "cache_len=", cache_len,
    )
    # fmt: on
    if num_partitions > 0:
        name += String(", num_partitions=", num_partitions)
    if logical:
        name += ", logical"
    return name


@inline(.always)
def _decode_with[
    cache_t: KVCacheT,
    q_dtype: DType,
    //,
    kernels: MLAKernels,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    cache: cache_t,
    row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    scalars: TileTensor[.int64, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    top_k: Int,
    logical: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
    num_partitions: Int,
    ctx: DeviceContext,
) raises:
    var forced_np = Optional[Int]()
    if num_partitions > 0:
        forced_np = num_partitions
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_dtype](NUM_HEADS, Q_DEPTH),
        ragged=True,
        sparse=True,
        kernels=kernels,
    ](
        output,
        q,
        cache,
        CausalMask(),
        row_offsets,
        Float32(0.125),
        ctx,
        scalars,
        d_indices=indices,
        indices_stride=top_k,
        logical_indices=logical,
        num_partitions_in=forced_np,
    )


def _decode[
    cache_t: KVCacheT,
    q_dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[q_dtype, address_space=.GENERIC, ...],
    cache: cache_t,
    row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    scalars: TileTensor[.int64, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    top_k: Int,
    logical: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
    num_partitions: Int,
    ctx: DeviceContext,
) raises:
    # The sparse decode sweep runs on SM100 and AMD only.
    comptime if default_accelerator().is_amd_gpu():
        _decode_with[AMDMLA](
            output,
            q,
            cache,
            row_offsets,
            scalars,
            indices,
            top_k,
            logical,
            num_partitions,
            ctx,
        )
    else:
        _decode_with[SM100MLA](
            output,
            q,
            cache,
            row_offsets,
            scalars,
            indices,
            top_k,
            logical,
            num_partitions,
            ctx,
        )


def touched_bytes(
    batch_size: Int, seq_len: Int, cache_len: Int, top_k: Int
) -> Int:
    """Returns the latent bytes one launch gathers.

    A position's picks shift by one key from the previous position's, so a
    sequence touches about `top_k + seq_len` keys of its window.
    """
    var keys = min(cache_len + seq_len, top_k + seq_len)
    return batch_size * keys * Q_DEPTH


def run_config[
    q_dtype: DType
](
    ctx: DeviceContext[],
    mut h: SweepHarness,
    mut table: PageTable[PAGE_SIZE],
    mut latent: PagedBlocks[fp8],
    mut ws: Workspace,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
    top_k: Int,
    num_partitions: Int,
    index_stride: Int,
    logical: Bool,
    pdl_pred_ns: Int,
    on_host: Bool,
    copies: Int,
) raises:
    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    comptime Collection = PagedKVCacheCollection[
        fp8,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var tokens = batch_size * seq_len
    var pages = ceildiv(cache_len + seq_len, PAGE_SIZE)
    var slots = tokens * top_k

    h.timer.reset()
    fill_affine(ctx, ws.row_offsets, seq_len, 0, batch_size + 1)
    table.set_cache_len(ctx, cache_len)
    table.write_copies(ctx, max(copies, 1), batch_size, pages)
    # The index lists name physical rows, so each copy has its own, through
    # its own page table.
    for copy in range(max(copies, 1)):
        fill_sparse_indices(
            ctx,
            DeviceBuffer[DType.int32](
                ctx,
                ws.indices.unsafe_ptr().unsafe_offset(copy * slots),
                slots,
                owning=False,
            ),
            table.copy_lut(ctx, copy, batch_size, pages),
            pages,
            PAGE_SIZE,
            False,
            batch_size,
            seq_len,
            cache_len,
            top_k,
            on_host,
            index_stride,
        )
    if logical:
        # A page table whose entry for page `p` is `p` turns the generator's
        # physical rows into the picks' logical key positions.
        var lut_entries = batch_size * pages
        var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](lut_entries)
        ctx.synchronize()
        for i in range(lut_entries):
            lut_host[i] = UInt32(i % pages)
        var logical_lut = DeviceBuffer[DType.uint32](
            ctx, ws.logical_lut.unsafe_ptr(), lut_entries, owning=False
        )
        ctx.enqueue_copy(logical_lut, lut_host)
        fill_sparse_indices(
            ctx,
            DeviceBuffer[DType.int32](
                ctx, ws.logical.unsafe_ptr(), slots, owning=False
            ),
            logical_lut,
            pages,
            PAGE_SIZE,
            False,
            batch_size,
            seq_len,
            cache_len,
            top_k,
            on_host,
            index_stride,
        )

    comptime blocks_layout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*blocks_layout.shape_types]()
    blocks_shape[0] = Int64(table.num_blocks)
    blocks_shape[2] = Int64(1)
    var blocks_strides = Coord[*blocks_layout.stride_types]()
    blocks_strides[1] = blocks_shape[2] * Int64(blocks_strides[2].value())
    blocks_strides[0] = Int64(blocks_shape[1].value()) * blocks_strides[1]

    def cache_at(
        launch_ctx: DeviceContext[], copy: Int
    ) raises {mut latent, imm} -> Collection.CacheType:
        var lut = table.copy_lut(launch_ctx, copy, batch_size, pages)
        var kv_collection = Collection(
            TileTensor(
                latent.buf, blocks_layout(blocks_shape, blocks_strides)
            ).as_unsafe_any_origin(),
            TileTensor(table.cache_lengths, row_major(Int64(batch_size)))
            .as_imm()
            .as_unsafe_any_origin(),
            TileTensor(lut, row_major(Int64(batch_size), Int64(pages)))
            .as_imm()
            .as_unsafe_any_origin(),
            UInt32(seq_len),
            UInt32(cache_len),
        )
        return kv_collection.get_key_cache(0)

    var q_ptr: Pointer[Scalar[q_dtype], MutAnyOrigin]
    comptime if q_dtype == fp8:
        q_ptr = rebind[Pointer[Scalar[q_dtype], MutAnyOrigin]](
            ws.q_fp8.unsafe_ptr()
        )
    else:
        q_ptr = rebind[Pointer[Scalar[q_dtype], MutAnyOrigin]](
            ws.q_bf16.unsafe_ptr()
        )
    var q_tt = TileTensor(
        q_ptr, row_major((tokens, Idx[NUM_HEADS], Idx[Q_DEPTH]))
    )
    var out_tt = TileTensor(
        ws.out.unsafe_ptr(), row_major((tokens, Idx[NUM_HEADS], Idx[V_DEPTH]))
    )
    var row_offsets_tt = TileTensor(
        ws.row_offsets.unsafe_ptr(), row_major(batch_size + 1)
    )
    var d_indices = ws.indices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var mla_args = MLADispatchScalarArgs[num_heads=NUM_HEADS, is_fp8_kv=True](
        batch_size, cache_len, seq_len, ctx
    )
    if num_partitions > 0:
        force_split(ctx, mla_args.gpu_buf, batch_size, seq_len, num_partitions)
    var scalar_args = mla_args.gpu_tile_tensor()
    var logical_indices = OptionalReg[MutPointer[Int32, MutAnyOrigin]]()
    if logical:
        logical_indices = ws.logical.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
    ctx.synchronize()
    h.timer.lap("config_setup")

    @inline(.always)
    def kernel_launch(
        launch_ctx: DeviceContext[], copy: Int
    ) raises {mut out_tt, imm}:
        comptime if not default_accelerator().is_amd_gpu():
            if pdl_pred_ns > 0:
                launch_ctx.enqueue_function[early_trigger_spin](
                    UInt64(pdl_pred_ns), grid_dim=1, block_dim=32
                )
        _decode(
            out_tt,
            q_tt,
            cache_at(launch_ctx, copy),
            row_offsets_tt,
            scalar_args,
            d_indices.unsafe_offset(copy * slots),
            top_k,
            logical_indices,
            num_partitions,
            launch_ctx,
        )

    @inline(.always)
    def first_copy(launch_ctx: DeviceContext[]) raises {mut out_tt, imm}:
        kernel_launch(launch_ctx, 0)

    dump_stamps(
        ctx,
        first_copy,
        String(
            "b",
            batch_size,
            "_s",
            seq_len,
            "_c",
            cache_len,
            "_np",
            num_partitions,
            "_l" if logical else "_n",
        ),
    )
    h.first_launch(ctx, kernel_launch)
    var fields = Fields()
    fields.add("kernel", String("mla_decode_sparse_fp8"))
    fields.add("q_dtype", q_dtype)
    fields.add("num_heads", NUM_HEADS)
    fields.add("page_size", PAGE_SIZE)
    fields.add("top_k", top_k)
    fields.add("batch_size", batch_size)
    fields.add("seq_len", seq_len)
    fields.add("cache_len", cache_len)
    fields.add("index_stride", index_stride)
    fields.add("num_partitions", num_partitions)
    fields.add("logical", logical)
    fields.add("pdl_pred_ns", pdl_pred_ns)
    h.bench(
        ctx,
        kernel_launch,
        BenchId(
            _run_name[q_dtype](
                batch_size,
                seq_len,
                cache_len,
                top_k,
                num_partitions,
                logical,
            )
        ),
        fields^,
        copies,
    )
    _ = mla_args^


def main() raises:
    var h = SweepHarness("decode_sparse")

    var sweep = Sweep()
    sweep.axis("batch_size", "8")
    # 1 + num_speculative_tokens: the verify width a spec-decode step runs at.
    sweep.axis("seq_len", "6")
    # Crosses the point where the top-k window saturates and the work per
    # position stops growing.
    sweep.axis("cache_len", "512,1024,2048,4096,8192,16384")
    sweep.axis("top_k", "2048")
    sweep.axis("num_partitions", "0")
    var index_stride = arg_parse("index_stride", 0)
    var logical = arg_parse("logical", 0) != 0
    var pdl_pred_ns = arg_parse("pdl_pred_ns", 0)
    var fill = String(arg_parse("fill", "gpu"))
    var layout = PageLayout.parse(arg_parse("page_layout", "identity"))
    var on_host = fill == "host"

    h.timer.lap("main_prologue")
    with DeviceContext() as ctx:
        h.timer.lap("device_init")
        h.start(ctx)
        var rot = h.rotation()

        # One more copy allocates its pages whole, and its index list.
        def copies(cfg: Config) raises {imm} -> Int:
            var batch = cfg["batch_size"]
            var keys = cfg["cache_len"] + cfg["seq_len"]
            var copy_bytes = (
                batch * ceildiv(keys, PAGE_SIZE) * PAGE_SIZE * Q_DEPTH
                + 4 * batch * cfg["seq_len"] * cfg["top_k"]
            )
            return rot.copies(
                touched_bytes(
                    batch, cfg["seq_len"], cfg["cache_len"], cfg["top_k"]
                ),
                copy_bytes,
            )

        def require(cfg: Config, mut fp: Footprint) raises {imm}:
            PageTable[PAGE_SIZE].require(
                fp,
                cfg["batch_size"],
                cfg["cache_len"] + cfg["seq_len"],
                copies(cfg),
            )
            Workspace.require(cfg, fp, max(1, copies(cfg)), logical)

        def bytes(fp: Footprint) raises -> Int:
            return (
                PageTable[PAGE_SIZE].bytes_for(fp)
                + PagedBlocks[fp8].bytes_for[PAGE_SIZE](fp, Q_DEPTH)
                + Workspace.bytes_for(fp)
            )

        var groups = plan_groups(
            sweep.configs(), require, bytes, device_budget(ctx)
        )
        for ref group in groups:
            ref fp = group.footprint
            var table = PageTable[PAGE_SIZE].from_footprint(ctx, fp, layout)
            var latent = PagedBlocks[fp8](
                ctx, table, Q_DEPTH, seed=0, on_host=on_host
            )
            ctx.synchronize()
            h.timer.lap("fixture_fill")
            var ws = Workspace(ctx, fp, on_host)
            ctx.synchronize()
            h.timer.lap("workspace_fill")
            var info = Fields()
            info.add("fixture_bytes", table.bytes() + latent.bytes())
            info.add("planned_bytes", group.bytes)
            info.extend(fp)
            info.add("configs", len(group.configs))
            info.add("fill", fill)
            info.add("page_layout", String(layout))
            h.note(info^)
            for ref cfg in group.configs:
                comptime for q_dtype in [fp8, DType.bfloat16]:
                    comptime if dtype_arm_enabled["q_dtype", q_dtype]() and (
                        q_dtype == fp8 or not default_accelerator().is_amd_gpu()
                    ):
                        run_config[q_dtype](
                            ctx,
                            h,
                            table,
                            latent,
                            ws,
                            cfg["batch_size"],
                            cfg["seq_len"],
                            cfg["cache_len"],
                            cfg["top_k"],
                            cfg["num_partitions"],
                            index_stride,
                            logical,
                            pdl_pred_ns,
                            on_host,
                            copies(cfg),
                        )
            h.timer.reset()
        h.stop(ctx)
    h.timer.lap("device_teardown")
    h.finish()
