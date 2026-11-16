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

"""Runs prepared sparse attention with the SM100 decode attention core."""

from max.gpu.host import DeviceContext
from layout import (
    ComptimeInt,
    DefaultEngine,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from kv_cache.types import KVCacheT
from layout.tile_tensor import NullableTileTensor
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import CausalMask, NullMask, MHAMask
from nn.attention.mha_utils import (
    MHAConfig,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    mla_decode_sm100_dispatch,
)


def mla_sparse_attention[
    cache_t: KVCacheT,
    SinkType: OptionalPointer,
    //,
    num_heads: Int,
    depth: Int,
    causal: Bool = False,
    ragged: Bool = True,
    fp8_scale_bf16: Bool = False,
    scale_block_size_override: Int = -1,
](
    output: TileTensor[address_space=.GENERIC, ...],
    q: TileTensor[address_space=.GENERIC, ...],
    cache: cache_t,
    indices: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: SinkType,
    scale: Float32,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
    ctx: DeviceContext,
    num_partitions_in: Optional[Int] = None,
    scales_ptr_override: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
) raises:
    """Runs attention over packed physical index rows and per-query lengths.

    Query tokens follow `input_row_offsets`; cache lengths exclude this query
    chunk. `causal` requires each sparse row to contain only the query's
    causal prefix, with its valid entries packed first, exactly
    `min(top_k, causal count)` of them. Otherwise the whole selected row is
    visible and each slot's index decides it, so a `-1` anywhere is masked.
    Both forms read one length per query.

    Capture requires `num_partitions_in=1`; split accumulator storage is
    currently local to dispatch. Eager calls also support multiple partitions.
    """
    # Legacy prefill writes through a borrowed tensor handle; retain its
    # engine and geometry when restoring the output's writable view.
    var writable: type_of(output).OriginCastType[MutAnyOrigin] = {
        output._unsafe_storage_cast[to_mut=True, to_origin=MutAnyOrigin](),
        output.layout,
    }
    var k = KVCacheMHAOperand(cache)
    comptime k_t = type_of(k)
    var batch_size = Int(input_row_offsets.dim(0)) - 1 if ragged else 1
    if batch_size == 0 or Int(q.dim(0)) == 0:
        return
    var args = NullableTileTensor[
        DType.int64,
        RowMajorLayout[ComptimeInt[3]],
        MutAnyOrigin,
        Engine=DefaultEngine[element_width=1],
    ](None, row_major[3]())
    var lengths = NonNullPointer[DType.int32](
        topk_lengths.ptr.unsafe_bitcast[Int32]().as_imm().as_unsafe_any_origin()
    )

    @inline(.always)
    def launch_decode[MaskType: MHAMask](mask: MaskType) raises {imm}:
        mla_decode_sm100_dispatch[
            q.dtype,
            k_t,
            output.dtype,
            MaskType,
            config=MHAConfig[q.dtype](num_heads, depth),
            depth=depth,
            num_heads=num_heads,
            group=num_heads,
            ragged=ragged,
            sparse=True,
            physical_indices=True,
            TopkLengthsPtrType=type_of(lengths),
            AttnSinkPtrType=SinkType,
            skip_masked_tiles=True,
            prepared_bounds=True,
            fp8_scale_bf16=fp8_scale_bf16,
            scale_block_size_override=scale_block_size_override,
        ](
            q,
            k,
            writable,
            scale,
            input_row_offsets,
            mask,
            args,
            batch_size,
            q_max_seq_len,
            max_cache_valid_length,
            ctx,
            d_indices=indices,
            indices_stride=indices_stride,
            topk_lengths=lengths,
            attn_sink_ptr=attn_sink_ptr,
            num_partitions_in=num_partitions_in,
            scales_ptr_override=scales_ptr_override,
        )

    comptime if causal:
        launch_decode(CausalMask())
    else:
        launch_decode(NullMask())


def mla_sparse_attention_flat[
    cache_t: KVCacheT,
    //,
    num_heads: Int,
    depth: Int,
    fp8_scale_bf16: Bool = False,
    scale_block_size_override: Int = 0,
](
    output: TileTensor[address_space=.GENERIC, ...],
    q: TileTensor[address_space=.GENERIC, ...],
    cache: cache_t,
    indices: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[ImmPointer[Float32, ImmutAnyOrigin]],
    scale: Float32,
    ctx: DeviceContext,
    scales_ptr_override: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
) raises:
    """Runs independent physical sparse rows with the shared attention core.

    All queries form one scheduling segment. Their per-query selected lengths
    are authoritative; cache batch lengths do not restrict these physical rows.
    The form is null-masked, so a row's `-1` slots are excluded by their index
    wherever they sit rather than by position.
    """
    var query_tokens = q.num_elements() // (num_heads * depth)
    if query_tokens == 0:
        return
    var lengths = topk_lengths.as_imm()

    @inline(.always)
    def launch[SinkType: OptionalPointer](sink: SinkType) raises {imm}:
        mla_sparse_attention[
            num_heads=num_heads,
            depth=depth,
            ragged=False,
            fp8_scale_bf16=fp8_scale_bf16,
            scale_block_size_override=scale_block_size_override,
        ](
            output,
            q,
            cache,
            indices,
            indices_stride,
            lengths,
            lengths,
            sink,
            scale,
            query_tokens,
            indices_stride,
            ctx,
            num_partitions_in=Optional[Int](1),
            scales_ptr_override=scales_ptr_override,
        )

    if case .Some(sink) = attn_sink_ptr:
        launch(NonNullPointer[DType.float32](sink))
    else:
        launch(NullPointer[DType.float32]())
