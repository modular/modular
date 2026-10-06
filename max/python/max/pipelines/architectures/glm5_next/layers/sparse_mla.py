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
"""NoPE sparse MLA for GLM-5.3-Flash, as the decoder layer calls it.

The 11 sparse-MLA layers (plus the MTP draft layer) are GLM-5.2's block with
two changes, both of which land here rather than in
:mod:`max.pipelines.architectures.deepseekV3_2`:

**NoPE.** ``qk_rope_head_dim`` is 0, so the absorbed decode latent is
``kv_lora_rank`` = 512 rather than 576. Every MLA kernel in the tree is
compiled against 576 --- the three sparse prefill kernels comptime-assert
``config.qk_depth == 576``
(``max/kernels/src/nn/attention/gpu/nvidia/sm100/mla_prefill_sparse.mojo:1418``
and its two FP8 siblings), the SM100 decode config's ``supported()`` requires
``q_depth == 576 and depth == 512``
(``nvidia/sm100/mla_decode_utils.mojo:800``), and ``flareMLA_decoding`` asserts
576 (``gpu/mla.mojo:771``). So this module zero-pads the query and the latent
to the 576 geometry on the way into the kernel, which is exact: the padded
columns are zero in *both* operands, adding ``0.0`` to a float32 accumulator
does not round, and ``BN_QK`` = 64 divides 512 and 576 alike, so the first
eight tiles decompose identically and the ninth contributes exactly zero. The
padding is deliberately not in the weights: ``q_b_proj`` and
``kv_a_proj_with_mqa`` stay at their checkpoint widths, so nothing pays the
25% wider ``q_b_proj`` matmul that a padded weight would cost forever.

**k-pooled indexing.** The selection comes from :class:`~max.pipelines.architectures.deepseekV3_2.layers.indexer.Indexer`, which
scores 4-token pools and emits ``index_topk + index_kpool - 1`` = 2051 token
positions per query. The shared sparse-MLA block is therefore built with
``skip_topk=True`` --- it declares no indexer weights of its own and its rope is
never reached --- and ``index_topk`` set to that 2051 *selection width*, which
is what it uses for both ``sparse_topk_lengths`` and ``sparse_indices_stride``.

Nothing here subclasses the shared block. It is composed: the per-device module
is `SparseLatentAttentionWithRopeFp8` unmodified, and this file supplies the
projections' call order around its ``_mla_impl`` so the padding has somewhere to
live without forking a file five shipping architectures ride on.
"""

from __future__ import annotations

from dataclasses import field

from max import tree
from max.dtype import DType
from max.graph import (
    BufferValue,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.comm import Allreduce
from max.nn.kernels import mla_decode_graph
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.quant_ops import quantized_matmul
from max.nn.rotary_embedding import RotaryEmbedding
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    Indexer,
)
from max.pipelines.architectures.deepseekV3_2.layers.sparse_mla import (
    SparseLatentAttentionWithRopeFp8,
)

from ..model_config import Glm5NextConfig
from ..quantization import UNQUANTIZED_ATTN_PROJECTIONS

__all__ = [
    "Glm5NextSparseMLASublayer",
    "SparseMLASublayerInputs",
    "nope_pad_rotary",
]

# Attention sink the sparse decode kernel adds for slots a query does not
# select. Matches DeepSeek-V3.2's value; -1e38 is float32's most negative
# normal power of ten, so `exp` of it is exactly zero.
_SPARSE_ATTN_SINK = -1.0e38


class _FlatSparseMLA(SparseLatentAttentionWithRopeFp8):
    """The shared sparse-MLA block, holding its checkpoint weight names.

    The only thing this changes is the FQN prefix. The checkpoint puts the MLA
    projections directly under ``self_attn`` --- ``self_attn.q_a_proj.weight``
    --- and the decoder layer binds this sublayer as ``self_attn``, so the
    attribute name it is held under here must not appear in between. The KDA
    lane's :class:`KimiDeltaAttention` carries the same property for the same
    reason. No behaviour is overridden.
    """

    @property
    def _omit_module_attr_name(self) -> bool:
        return True


@tree.dataclass(frozen=True, kw_only=True)
class SparseMLASublayerInputs:
    """One sparse-MLA layer's per-step inputs, one entry per device.

    The decoder layer is generic over each sublayer's bundle
    (``tasks/glm53-scope/contracts/mhc-decoder-layer.md``), so this lives here
    rather than in a shared dataclass nobody owns.
    """

    signal_buffers: list[BufferValue]
    """Allreduce signal buffers for ``o_proj``'s partial sums."""

    input_row_offsets: list[TensorValue]
    """``[batch_size + 1]`` uint32 exclusive prefix offsets over the packed
    tokens."""

    mla_kv_collections: list[PagedCacheValues]
    """The ``"mla"`` leaf of the multi-cache: the 576-wide padded latent."""

    indexer_kv_collections: list[PagedCacheValues]
    """The ``"indexer"`` leaf: the pooled-key cache, declared
    ``index_head_dim // index_kpool`` = 32 wide. Only the fused pooled scorer
    reads it; the dense bring-up scorer projects its keys from ``xs``."""

    layer_idx: TensorValue
    """uint32 scalar on CPU. This is the layer's index *within the cached
    subset*, not its index in the model: only the sparse-MLA layers hold a KV
    cache, so layer 3 of the model is cache layer 0."""

    tail_pools: list[BufferValue]
    """``[num_rows, 2, index_kpool, index_head_dim]`` k-pool tail ring, one per
    device. Every sparse layer shares the pool and differs only in the row it
    runs in, so this is the whole pool rather than one layer's slice."""

    tail_row_ids: list[TensorValue]
    """``[batch]`` uint32 ring row this layer runs each sequence in, one per
    device. Drawn from the ring leaf at the layer's index within the *ring's*
    layer set -- the indexer cache's -- so a model index would bind one
    layer's ring to another and pass every shape check."""

    prev_selection: TensorValue | None = None
    """``[total_tokens, 2051]`` int32 selection to reuse instead of running the
    indexer, for ``index_share_for_mtp_iteration``. The shared value is the
    *expanded token* selection, not the 512 pool ids."""

    selection: list[TensorValue] = field(default_factory=list)
    """Per-device sink for this layer's selection. Appended to the model's
    graph outputs so the draft graph can take it as ``prev_topk_indices``, the
    pattern ``deepseekV3_2_nextn.py:342`` already uses. Mutated in place, which
    is why this dataclass is frozen but this field is a list."""


def nope_pad_rotary(config: Glm5NextConfig) -> RotaryEmbedding:
    """Builds the rotary table the 576 padding needs, and nothing else uses.

    GLM-5.3-Flash has no rotary embedding anywhere in its decoder. The padded
    kernel geometry nevertheless carries 64 rotary columns, and the fused MLA
    ops derive that width from ``freqs_cis.shape[1]``, so a table of that shape
    has to exist. Its *values* are irrelevant --- every query column it rotates
    is zero, and rotating zero yields zero --- but its first dimension has to
    cover the longest position the model will see.

    Build it **once per model** and pass the same instance to every sparse
    layer: at ``max_seq_len`` = 1M the table is 134 MB, and eleven copies of it
    would not be. It disappears entirely when the MLA kernels grow a native
    512-wide configuration.

    Args:
        config: The model config, read for ``max_seq_len``.

    Returns:
        A rotary embedding whose ``freqs_cis`` is ``[max_seq_len, 64]``.
    """
    rope_width = config.mla_head_dim - config.kv_lora_rank
    if rope_width <= 0:
        raise ValueError(
            "nope_pad_rotary is only needed when the latent is padded: "
            f"mla_head_dim ({config.mla_head_dim}) must exceed kv_lora_rank "
            f"({config.kv_lora_rank}). Set Glm5NextConfig.mla_latent_pad_to."
        )
    return RotaryEmbedding(
        dim=rope_width,
        n_heads=config.num_attention_heads,
        theta=10000.0,
        max_seq_len=config.max_seq_len,
        head_dim=rope_width,
        interleaved=False,
    )


def _zero_pad_last(x: TensorValue, width: int) -> TensorValue:
    """Appends ``width`` zero columns to ``x``'s last axis."""
    if width == 0:
        return x
    zeros = ops.broadcast_to(
        ops.constant(0, x.dtype, device=x.device).reshape(
            (1,) * (x.rank - 1) + (1,)
        ),
        tuple(x.shape[:-1]) + (width,),
    )
    return ops.concat([x, zeros], axis=-1)


class Glm5NextSparseMLASublayer(Module):
    """Sparse MLA plus the k-pooled indexer, across all devices.

    Satisfies ``Glm5NextSublayer[SparseMLASublayerInputs]`` structurally: it
    takes one already normalized ``[total_tokens, hidden_size]`` per device and
    returns one output per device with ``o_proj``'s partial sums all-reduced.
    It adds no residual and declares no input layernorm --- the decoder layer
    owns both, because the mHC write-back is not an add.

    The indexer is **replicated**, not head-sharded, and that is a correctness
    requirement rather than a tuning choice: its 32 head scores are summed
    before the top-k, so a sharded head axis would leave each rank holding a
    partial sum and ranks would select different tokens. The all-reduce that
    would repair it is O(context) --- about 1 MB per sequence per layer at 1M
    --- against a fixed 164 MB of replicated weights across the 11 layers.

    Args:
        config: The model config.
        layer_idx: This layer's index in the model, for error messages only.
            The cache-facing index travels in the bundle.
        rope: The padding rotary table from :func:`nope_pad_rotary`, shared
            across every sparse layer.
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        layer_idx: int,
        *,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        devices = config.devices
        self.devices = list(devices)

        kv_params = _mla_leaf(config)
        if kv_params.head_dim != config.mla_head_dim:
            raise ValueError(
                f"The 'mla' cache leaf is {kv_params.head_dim} wide but "
                f"Glm5NextConfig.mla_head_dim is {config.mla_head_dim}. The "
                "kernel reads the cache row and the padded query as one "
                "geometry, so a disagreement is a silent wrong answer rather "
                "than a shape error. Both must come from mla_latent_pad_to."
            )
        self.rope = rope
        self.latent_pad = config.mla_head_dim - config.kv_lora_rank

        quant_scheme = config.quant_scheme
        if quant_scheme is None:
            raise ValueError(
                "Glm5NextSparseMLASublayer needs a resolved quant scheme: the "
                "sparse-MLA block is FP8 except kv_b_proj and the indexer, and "
                "that split is read off the checkpoint's weight scales."
            )
        assert "kv_b_proj" in UNQUANTIZED_ATTN_PROJECTIONS

        # The shared DSA indexer, not a GLM-specific one: k-pool compression
        # and the zero-width RoPE GLM-5.3-Flash needs both live there now, so
        # forking would mean two implementations of one thing.
        self.indexer = Indexer(
            dim=config.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            qk_rope_head_dim=0,  # NoPE throughout GLM-5.3-Flash's decoder
            index_topk=config.index_topk,
            q_lora_rank=config.q_lora_rank,
            devices=self.devices,
            quant_config=quant_scheme.config,
            # GLM norms the indexer key in the compute dtype, not float32.
            k_norm_dtype=config.compute_dtype,
            index_kpool=config.index_kpool,
            index_kpool_compress=config.index_kpool_compress,
            index_kpool_always_select_tail=config.index_kpool_always_select_tail,
            # GLM-5.3-Flash leaves the indexer out of its FP8 map.
            indexer_weights_fp8=False,
        )
        self.indexer.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.indexer_shards = self.indexer.shard(self.devices)

        self.attention = _FlatSparseMLA(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            quant_config=quant_scheme.config,
            devices=self.devices,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            # The block's own geometry stays NoPE so its weights match the
            # checkpoint; the padding is applied to activations below.
            qk_rope_head_dim=0,
            v_head_dim=config.v_head_dim,
            norm_dtype=config.compute_dtype,
            # TODO(GLM53-PREFILLSCALE): "auto" would route prefill through
            # `mla_prefill_decode_graph`, which asserts the absorbed scales are
            # present (`max/python/max/nn/kernels.py:4580`) and so cannot serve
            # a checkpoint that ships `kv_b_proj` unquantized --- the very
            # split this model needs. `mla_decode_graph` already takes them as
            # `TensorValue | None`, so the decode route is correct today and
            # the prefill route needs the same three asserts relaxed. Until
            # then prefill runs the decode kernel, which is the same fallback
            # the shared block already takes for unsupported head counts.
            graph_mode="decode",
            # No indexer of its own: `skip_topk` declares none, and the
            # selection arrives from `Indexer` instead.
            skip_topk=True,
            # Both `sparse_topk_lengths` and `sparse_indices_stride` come from
            # this, so it is the 2051 selection width and not `index_topk`.
            index_topk=self.indexer.selection_width,
            kv_b_proj_dtype=config.kv_b_proj_dtype,
        )
        self.attention.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(self.devices)
        )
        self.shards = self.attention.shard(self.devices)
        if self.attention.kv_b_proj_scale is None:
            for shard in self.shards:
                # TODO(GLM53-KVBSCALE): `SparseLatentAttentionWithRopeFp8.shard`
                # builds each replica without forwarding `kv_b_proj_dtype`, so
                # a replica re-declares `kv_b_proj.weight_scale` even when the
                # parent has none -- an unmapped weight at load, and a scale
                # handed to the kernel for a weight that was never quantized.
                # Every absorb path keys off this attribute being None
                # (`multi_latent_attention_fp8.py:_gather_per_head_scale`), so
                # clearing it here is the whole repair. The one-line fix is to
                # pass `kv_b_proj_dtype=self.kv_b_proj.dtype` in that
                # constructor call; this loop is inert once it lands.
                shard.kv_b_proj_scale = None
                shard._kv_b_proj_quantized = False
        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def __call__(
        self, xs: list[TensorValue], inputs: SparseMLASublayerInputs
    ) -> list[TensorValue]:
        """Runs the layer on every device and all-reduces the result.

        Args:
            xs: ``[total_tokens, hidden_size]`` per device, normalized.
            inputs: This layer's per-step caches, offsets and selection.

        Returns:
            ``[total_tokens, hidden_size]`` per device.
        """
        outputs: list[TensorValue] = []
        for i, shard in enumerate(self.shards):
            # `q_a_proj` and `kv_a_proj_with_mqa` are one fused matmul, and the
            # indexer consumes the normalized query residual it produces, so
            # this runs once and feeds both.
            q_a_normed, kv = self._project(shard, xs[i])
            selection = self._select(i, xs[i], q_a_normed, inputs)
            inputs.selection.append(selection)
            outputs.append(
                self._attend(
                    shard,
                    q_a_normed,
                    kv,
                    selection,
                    kv_collection=inputs.mla_kv_collections[i],
                    layer_idx=inputs.layer_idx,
                    input_row_offsets=inputs.input_row_offsets[i],
                )
            )
        if len(self.shards) == 1:
            # `o_proj` produced the whole sum, and `Allreduce` still demands a
            # matching signal buffer per device, so a single-device graph would
            # have to fabricate one for a no-op.
            return outputs
        return self.allreduce(outputs, inputs.signal_buffers)

    def _project(
        self, shard: SparseLatentAttentionWithRopeFp8, x: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Returns ``(q_a_layernorm(q_a_proj(x)), kv_a_proj_with_mqa(x))``."""
        wqkv, wqkv_scale = shard.wqkv
        qkv = quantized_matmul(
            x=x,
            weight=wqkv,
            weight_scale=wqkv_scale,
            input_scale=None,
            quant_config=shard.quant_config,
        )
        q_a_out, kv = ops.split(
            qkv, [shard.q_lora_rank, shard.cache_head_dim], axis=1
        )
        return shard.q_a_layernorm(q_a_out), kv

    def _select(
        self,
        device_idx: int,
        x: TensorValue,
        q_a_normed: TensorValue,
        inputs: SparseMLASublayerInputs,
    ) -> TensorValue:
        """Returns this device's ``[total_tokens, 2051]`` token selection.

        Reuses ``prev_selection`` when the caller already has one, and
        otherwise runs the pooled indexer: compress, cache, score pools, expand
        back to token positions.
        """
        if inputs.prev_selection is not None:
            return inputs.prev_selection.to(x.device)
        return self.indexer_shards[device_idx](
            x,
            q_a_normed,
            # NoPE: `qk_rope_head_dim` is 0, so the indexer skips the rotary
            # entirely and never reads this table.
            self.rope.freqs_cis,
            inputs.input_row_offsets[device_idx],
            inputs.indexer_kv_collections[device_idx],
            inputs.layer_idx,
            tail=inputs.tail_pools[device_idx],
            slot_idx=inputs.tail_row_ids[device_idx],
        )

    def _attend(
        self,
        shard: SparseLatentAttentionWithRopeFp8,
        q_a_normed: TensorValue,
        kv: TensorValue,
        selection: TensorValue,
        *,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Projects the query, pads to 576, and runs sparse MLA.

        The shared block's own ``__call__`` is bypassed rather than subclassed:
        it would run its rope, and it offers no seam between ``q_b_proj`` and
        the kernel, which is exactly where the NoPE padding has to go. Every
        weight, the absorbed ``w_uk`` / ``w_uv`` / ``w_k`` and ``_mla_impl``
        itself are the shared block's, unmodified.
        """
        xq = quantized_matmul(
            x=q_a_normed,
            weight=shard.q_b_proj,
            weight_scale=shard.q_b_proj_scale,
            input_scale=None,
            quant_config=shard.quant_config,
        ).reshape((-1, shard.n_heads, shard.qk_head_dim))

        # Zero-pad both operands to the geometry every MLA kernel compiles.
        xq = _zero_pad_last(xq, self.latent_pad)
        kv = _zero_pad_last(kv, self.latent_pad)
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)

        w_uk, w_uk_scale = shard.w_uk
        w_uv, w_uv_scale = shard.w_uv
        if w_uk_scale is not None or w_uv_scale is not None:
            raise ValueError(
                "GLM-5.3-Flash absorbs a BF16 kv_b_proj, so the MLA kernel "
                "takes no absorbed scales. A scale here means the block was "
                "built without kv_b_proj_dtype."
            )
        if kv_collection.kv_scales is not None:
            raise ValueError(
                "An FP8 latent cache with a BF16 kv_b_proj is not reachable: "
                "the FP8 MLA op requires absorbed weight scales this "
                "checkpoint does not ship (max/python/max/nn/kernels.py:4410). "
                "Run the BF16 latent cache, which is what the accuracy gates "
                "use, until that pairing is supported."
            )

        # `mla_decode_graph` directly rather than the block's `_mla_impl`: that
        # helper always forwards its FP8 `quant_config`, and every operand this
        # kernel sees is BF16. The block's FP8-ness lives in `q_a_proj`,
        # `q_b_proj`, `kv_a_proj_with_mqa` and `o_proj`, which are separate
        # `quantized_matmul` calls; `kv_b_proj` --- the only weight the MLA
        # kernel itself consumes, through the absorb --- is BF16 in this
        # checkpoint. Passing a quant config here would assert on the absorbed
        # scales that do not exist.
        assert kv_collection.attention_dispatch_metadata is not None
        assert kv_collection.mla_num_partitions is not None
        attn_out = mla_decode_graph(
            q=xq,
            kv=kv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_norm_gamma=shard.kv_a_proj_layernorm,
            w_uk=w_uk,
            w_uv=w_uv,
            kv_params=shard.kv_params,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=shard.scale,
            epsilon=1e-6,
            v_head_dim=shard.v_head_dim,
            scalar_args=kv_collection.attention_dispatch_metadata,
            num_partitions_scalar=kv_collection.mla_num_partitions,
            sparse_indices=selection,
            sparse_topk_lengths=ops.broadcast_to(
                ops.constant(
                    self.indexer.selection_width,
                    dtype=DType.int32,
                    device=xq.device,
                ),
                (xq.shape[0],),
            ),
            sparse_attn_sink=ops.broadcast_to(
                ops.constant(
                    _SPARSE_ATTN_SINK, dtype=DType.float32, device=xq.device
                ),
                (shard.n_heads,),
            ),
            sparse_indices_stride=self.indexer.selection_width,
        ).reshape((-1, shard.n_heads * shard.v_head_dim))
        return shard.o_proj(attn_out)


def _mla_leaf(config: Glm5NextConfig) -> KVCacheParams:
    """Returns the ``"mla"`` leaf of the model's multi-cache."""
    children = getattr(config.kv_params, "children", None)
    if children is None or "mla" not in children:
        raise ValueError(
            "Glm5NextSparseMLASublayer expects a MultiKVCacheParams with an "
            "'mla' leaf alongside 'indexer', as Glm5NextConfig."
            "construct_kv_params builds."
        )
    leaf = children["mla"]
    assert isinstance(leaf, KVCacheParams)
    return leaf
