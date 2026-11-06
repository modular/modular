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
"""Decode context-parallel latent store of `mla_fused_rope_rmsnorm_quantization`.

Every rank of a `dcp`-way group gets the same lookup table and the same global
`cache_lengths` and runs the op with its own rank, on bf16, FP8 and NVFP4
latent caches. Each rank cache must equal, byte for byte, an image built from
one non-CP run of the same inputs: every token's row (and, for NVFP4, its
scales) on its owner at the owner's local slot, and the poison everywhere
else. The roped query uses the global position, so it must match the non-CP
query bit for bit on every rank.
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_float64, random_ui64, seed
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, RowMajorLayout, TileTensor, row_major
from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
from nn.attention.gpu.mla_graph import mla_fused_rope_rmsnorm_quantization
from std.testing import assert_equal
from std.utils import IndexList

comptime dtype = DType.bfloat16
comptime freq_dtype = DType.float32
comptime kv_norm_dim = 512
comptime rope_dim = 64
comptime head_size = kv_norm_dim + rope_dim
# NVFP4: the packed latent, then the FP8 rope bytes, plus one scale per 16.
comptime packed_size = kv_norm_dim // 2 + rope_dim
comptime sf_cols = kv_norm_dim // NVFP4_SF_VECTOR_SIZE
comptime num_q_heads = 16
comptime page_size = 128
comptime num_layers = 2
comptime layer_idx = 1
comptime max_seq_len = 4096
comptime epsilon = Float32(1e-6)
comptime poison = UInt8(0xAB)


def blocks_tt[
    dt: DType
](mut dev: DeviceBuffer[dt], num_blocks: Int, width: Int) -> TileTensor[
    dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], origin_of(dev)
]:
    return TileTensor(
        dev, row_major(num_blocks, 1, num_layers, page_size, 1, width)
    )


def row_index(block: Int, row: Int) -> Int:
    """Index of cache row `(block, layer_idx, row)` in rows."""
    return (block * num_layers + layer_idx) * page_size + row


struct Batch(Movable):
    """A ragged batch: prefill rows, decode rows and logical-page crossings."""

    var prompt_lens: List[Int]
    var cache_lens: List[Int]
    var lut_cols: Int
    var num_blocks: Int
    var lut: List[Int]

    def __init__(out self):
        seed(5)
        # Rows 1, 6 and 7 decode one token; 1 and 6 end a page. Row 5 crosses
        # global position 1024, a logical-page boundary for every group size
        # here, and row 3 crosses 256 and 384.
        self.prompt_lens = [17, 1, 31, 130, 3, 40, 1, 1]
        self.cache_lens = [0, 127, 13, 250, 6, 1000, 1023, 2047]
        var max_context = 0
        self.num_blocks = 4
        for i in range(len(self.prompt_lens)):
            var context = self.cache_lens[i] + self.prompt_lens[i]
            max_context = max(max_context, context)
            self.num_blocks += ceildiv(context, page_size)
        self.lut_cols = ceildiv(max_context, page_size)
        self.lut = List[Int](
            length=len(self.prompt_lens) * self.lut_cols, fill=self.num_blocks
        )
        var used = Set[Int]()
        for i in range(len(self.prompt_lens)):
            var pages = ceildiv(
                self.cache_lens[i] + self.prompt_lens[i], page_size
            )
            for c in range(pages):
                var b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                while b in used:
                    b = Int(random_ui64(0, UInt64(self.num_blocks - 1)))
                used.add(b)
                self.lut[i * self.lut_cols + c] = b

    def batch_size(self) -> Int:
        return len(self.prompt_lens)

    def total_length(self) -> Int:
        var total = 0
        for n in self.prompt_lens:
            total += n
        return total

    def max_prompt(self) -> Int:
        var m = 0
        for n in self.prompt_lens:
            m = max(m, n)
        return m

    def max_cache(self) -> Int:
        var m = 0
        for n in self.cache_lens:
            m = max(m, n)
        return m

    def block(self, bs: Int, slot: Int) -> Int:
        return self.lut[bs * self.lut_cols + slot // page_size]


struct Inputs(Movable):
    """Device inputs shared by every run of one case."""

    var row_offsets: DeviceBuffer[DType.uint32]
    var cache_lengths: DeviceBuffer[DType.uint32]
    var lut: DeviceBuffer[DType.uint32]
    var freqs: DeviceBuffer[freq_dtype]
    var gamma: DeviceBuffer[dtype]
    var kv: DeviceBuffer[dtype]
    var q: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext, batch: Batch) raises:
        var bsz = batch.batch_size()
        var total = batch.total_length()
        var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
            bsz + 1
        )
        var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
            bsz
        )
        var lut_host = ctx.enqueue_create_host_buffer[DType.uint32](
            bsz * batch.lut_cols
        )
        var freqs_host = ctx.enqueue_create_host_buffer[freq_dtype](
            max_seq_len * rope_dim
        )
        var gamma_host = ctx.enqueue_create_host_buffer[dtype](kv_norm_dim)
        var kv_host = ctx.enqueue_create_host_buffer[dtype](total * head_size)
        var q_host = ctx.enqueue_create_host_buffer[dtype](
            total * num_q_heads * rope_dim
        )
        ctx.synchronize()
        var offset = 0
        for i in range(bsz):
            row_offsets_host[i] = UInt32(offset)
            cache_lengths_host[i] = UInt32(batch.cache_lens[i])
            offset += batch.prompt_lens[i]
        row_offsets_host[bsz] = UInt32(offset)
        for i in range(bsz * batch.lut_cols):
            lut_host[i] = UInt32(batch.lut[i])
        for i in range(max_seq_len * rope_dim):
            freqs_host[i] = Float32(random_float64(-1.0, 1.0))
        for d in range(kv_norm_dim):
            gamma_host[d] = Float32(random_float64(0.5, 1.5)).cast[dtype]()
        for i in range(total * head_size):
            kv_host[i] = Float32(random_float64(-3.0, 3.0)).cast[dtype]()
        for i in range(total * num_q_heads * rope_dim):
            q_host[i] = Float32(random_float64(-2.0, 2.0)).cast[dtype]()

        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](bsz + 1)
        self.cache_lengths = ctx.enqueue_create_buffer[DType.uint32](bsz)
        self.lut = ctx.enqueue_create_buffer[DType.uint32](bsz * batch.lut_cols)
        self.freqs = ctx.enqueue_create_buffer[freq_dtype](
            max_seq_len * rope_dim
        )
        self.gamma = ctx.enqueue_create_buffer[dtype](kv_norm_dim)
        self.kv = ctx.enqueue_create_buffer[dtype](total * head_size)
        self.q = ctx.enqueue_create_buffer[dtype](
            total * num_q_heads * rope_dim
        )
        ctx.enqueue_copy(self.row_offsets, row_offsets_host)
        ctx.enqueue_copy(self.cache_lengths, cache_lengths_host)
        ctx.enqueue_copy(self.lut, lut_host)
        ctx.enqueue_copy(self.freqs, freqs_host)
        ctx.enqueue_copy(self.gamma, gamma_host)
        ctx.enqueue_copy(self.kv, kv_host)
        ctx.enqueue_copy(self.q, q_host)
        ctx.synchronize()


def launch[
    cache_dtype: DType, nvfp4: Bool, dcp_size: Int, dcp_interleave: Int
](
    ctx: DeviceContext,
    batch: Batch,
    inputs: Inputs,
    mut values: DeviceBuffer[cache_dtype],
    mut scales: DeviceBuffer[NVFP4_SF_DTYPE],
    mut q_out: DeviceBuffer[dtype],
    dcp_rank: Int,
) raises:
    """Runs the op for one rank into one latent cache."""
    var total = batch.total_length()
    var bsz = batch.batch_size()
    var row_offsets_tt = TileTensor[mut=False](
        inputs.row_offsets, row_major(bsz + 1)
    )
    var freqs_tt = TileTensor[mut=False](
        inputs.freqs, row_major[max_seq_len, rope_dim]()
    )
    var gamma_tt = TileTensor[mut=False](inputs.gamma, row_major[kv_norm_dim]())
    var kv_tt = TileTensor[mut=False](
        inputs.kv, row_major((total, Idx[head_size]))
    )
    var q_tt = TileTensor[mut=False](
        inputs.q, row_major((total, Idx[num_q_heads], Idx[rope_dim]))
    )
    var q_out_tt = TileTensor(
        q_out, row_major((total, Idx[num_q_heads], Idx[rope_dim]))
    )
    var cache_lengths_tt = TileTensor[mut=False](
        inputs.cache_lengths, row_major(bsz)
    )
    var lut_tt = TileTensor[mut=False](
        inputs.lut, row_major((bsz, batch.lut_cols))
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_tt} -> SIMD[DType.bfloat16, width]:
        return kv_tt.load[width=width]((coords[0], coords[1]))

    comptime if nvfp4:
        var coll = PagedKVCacheCollection[
            cache_dtype,
            KVCacheStaticParams(
                num_heads=1,
                head_size=packed_size,
                is_mla=True,
                fp8_tail_size=rope_dim,
            ),
            page_size,
            scale_dtype_=NVFP4_SF_DTYPE,
            quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
        ](
            blocks_tt(values, batch.num_blocks, packed_size),
            cache_lengths_tt,
            lut_tt,
            UInt32(batch.max_prompt()),
            UInt32(batch.max_cache()),
            scales=blocks_tt(scales, batch.num_blocks, sf_cols),
            scales_lookup_table=lut_tt,
        )
        mla_fused_rope_rmsnorm_quantization[
            dcp_size=dcp_size, dcp_interleave=dcp_interleave
        ](
            q_out_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            coll,
            UInt32(layer_idx),
            epsilon,
            kv_input_fn,
            ctx,
            dcp_rank,
        )
    else:
        var coll = PagedKVCacheCollection[
            cache_dtype,
            KVCacheStaticParams(num_heads=1, head_size=head_size, is_mla=True),
            page_size,
            scales_origin=MutUntrackedOrigin,
        ](
            blocks_tt(values, batch.num_blocks, head_size),
            cache_lengths_tt,
            lut_tt,
            UInt32(batch.max_prompt()),
            UInt32(batch.max_cache()),
        )
        mla_fused_rope_rmsnorm_quantization[
            dcp_size=dcp_size, dcp_interleave=dcp_interleave
        ](
            q_out_tt,
            q_tt,
            row_offsets_tt,
            freqs_tt,
            gamma_tt,
            coll,
            UInt32(layer_idx),
            epsilon,
            kv_input_fn,
            ctx,
            dcp_rank,
        )
    ctx.synchronize()


def poisoned[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.synchronize()
    var bytes = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(n * size_of[Scalar[dt]]()):
        bytes[i] = poison
    var dev = ctx.enqueue_create_buffer[dt](n)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    return dev


def to_bytes[
    dt: DType
](ctx: DeviceContext, dev: DeviceBuffer[dt], n: Int) raises -> List[UInt8]:
    var host = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    var bytes = host.unsafe_ptr().unsafe_bitcast[UInt8]()
    var out = List[UInt8](capacity=n * size_of[Scalar[dt]]())
    for i in range(n * size_of[Scalar[dt]]()):
        out.append(bytes[i])
    return out^


def expected_rank_image[
    dcp_size: Int, dcp_interleave: Int
](batch: Batch, ref_bytes: List[UInt8], row_bytes: Int, rank: Int) -> List[
    UInt8
]:
    """Poison, plus the reference row of every token `rank` owns at its slot."""
    var layout = ContextParallelLayout[dcp_size, dcp_interleave](rank)
    var image = List[UInt8](length=len(ref_bytes), fill=poison)
    for bs in range(batch.batch_size()):
        for t in range(batch.prompt_lens[bs]):
            var pos = batch.cache_lens[bs] + t
            if not layout.owns(pos):
                continue
            var slot = layout.local_slot(pos)
            var src = row_index(batch.block(bs, pos), pos % page_size)
            var dst = row_index(batch.block(bs, slot), slot % page_size)
            for i in range(row_bytes):
                image[dst * row_bytes + i] = ref_bytes[src * row_bytes + i]
    return image^


def count_mismatches(a: List[UInt8], b: List[UInt8]) -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            bad += 1
    return bad


def run_case[
    cache_dtype: DType, nvfp4: Bool, dcp_size: Int, dcp_interleave: Int
](ctx: DeviceContext) raises:
    print(
        "== mla_fused_rope_rmsnorm_cp",
        cache_dtype,
        "nvfp4" if nvfp4 else "",
        "dcp",
        dcp_size,
        "interleave",
        dcp_interleave,
    )
    var batch = Batch()
    var inputs = Inputs(ctx, batch)
    comptime row_elems = packed_size if nvfp4 else head_size
    comptime row_bytes = row_elems * size_of[Scalar[cache_dtype]]()
    comptime sf_row_bytes = sf_cols * size_of[Scalar[NVFP4_SF_DTYPE]]()
    var rows = batch.num_blocks * num_layers * page_size
    var values_n = rows * row_elems
    var scales_n = rows * sf_cols if nvfp4 else 1
    var q_n = batch.total_length() * num_q_heads * rope_dim

    var ref_values = poisoned[cache_dtype](ctx, values_n)
    var ref_scales = poisoned[NVFP4_SF_DTYPE](ctx, scales_n)
    var ref_q = ctx.enqueue_create_buffer[dtype](q_n)
    launch[cache_dtype, nvfp4, 1, 1](
        ctx, batch, inputs, ref_values, ref_scales, ref_q, 0
    )
    var ref_value_bytes = to_bytes(ctx, ref_values, values_n)
    var ref_scale_bytes = to_bytes(ctx, ref_scales, scales_n)
    var ref_q_bytes = to_bytes(ctx, ref_q, q_n)

    for rank in range(dcp_size):
        var values = poisoned[cache_dtype](ctx, values_n)
        var scales = poisoned[NVFP4_SF_DTYPE](ctx, scales_n)
        var q_out = ctx.enqueue_create_buffer[dtype](q_n)
        launch[cache_dtype, nvfp4, dcp_size, dcp_interleave](
            ctx, batch, inputs, values, scales, q_out, rank
        )
        var value_bad = count_mismatches(
            to_bytes(ctx, values, values_n),
            expected_rank_image[dcp_size, dcp_interleave](
                batch, ref_value_bytes, row_bytes, rank
            ),
        )
        assert_equal(value_bad, 0, msg="latent bytes, rank " + String(rank))
        comptime if nvfp4:
            var scale_bad = count_mismatches(
                to_bytes(ctx, scales, scales_n),
                expected_rank_image[dcp_size, dcp_interleave](
                    batch, ref_scale_bytes, sf_row_bytes, rank
                ),
            )
            assert_equal(scale_bad, 0, msg="scale bytes, rank " + String(rank))
        # RoPE uses the global position on every rank.
        assert_equal(
            count_mismatches(to_bytes(ctx, q_out, q_n), ref_q_bytes),
            0,
            msg="roped query, rank " + String(rank),
        )


def run_cache[
    cache_dtype: DType, nvfp4: Bool = False
](ctx: DeviceContext) raises:
    # A group of one is the plain op.
    run_case[cache_dtype, nvfp4, 1, 1](ctx)
    run_case[cache_dtype, nvfp4, 2, 1](ctx)
    run_case[cache_dtype, nvfp4, 2, page_size](ctx)
    run_case[cache_dtype, nvfp4, 4, 1](ctx)
    run_case[cache_dtype, nvfp4, 4, page_size](ctx)
    run_case[cache_dtype, nvfp4, 8, 1](ctx)
    run_case[cache_dtype, nvfp4, 8, page_size](ctx)


def main() raises:
    with DeviceContext() as ctx:
        run_cache[DType.bfloat16](ctx)
        run_cache[DType.float8_e4m3fn](ctx)
        run_cache[DType.uint8, nvfp4=True](ctx)
