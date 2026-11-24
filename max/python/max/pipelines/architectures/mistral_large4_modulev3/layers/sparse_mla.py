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
"""Mistral Large 4 sparse MLA: no query LoRA, native NoPE rows, KV readers.

Three structural differences from the DeepSeek-V3.2 layer this forks:

* **No query LoRA.** ``q_lora_rank`` is null, so the query is one FP8
  block-scaled ``wq`` over the hidden state. There is no ``q_a_proj``, no
  ``q_norm``, and the indexer takes its query from the hidden state directly.
* **Native 512-wide NoPE rows.** On every fourth layer the rope slice is
  absent: ``wkv_a_with_mqa`` is 512 wide, ``wkv_b`` is
  ``heads * (128 + 128)``, the query keeps all 128 content dims, and the
  softmax scale is still ``128 ** -0.5`` because it uses the *config*'s
  ``qk_head_dim``, not the layer's absorbed width.
* **KV readers.** The nine layers that read another layer's latent own no
  ``wkv_a_with_mqa`` and no ``kv_a_norm``. They attend with their own ``wq``,
  SSMax, ``wkv_b``, ``wo`` and indexer over the source layer's rows, which
  live in the source's cache slot, so the reader passes the source's
  projection and normalization and maps to the source's slot.

SSMax is applied to the query here, outside the fused op's rotation; see
:mod:`.ssmax` for why that is the same linear map and which rounding differs.
"""

from __future__ import annotations

from typing import Any

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Linear, Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.common_layers.quant_tensor import FP8BlockActivation
from max.nn.kernels import (
    mla_decode_graph,
    mla_prefill_decode_graph,
    row_invariant_matmul,
)
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.norm import RMSNorm
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, NamedMapping

from ...deepseekV3_2.layers.sparse_mla import sparse_selection_width
from ..model_config import ML4_DSPARK_VERIFY_ROWS, pins_verify_split
from .indexer import MistralLarge4Indexer
from .ssmax import SSMaxScaling


def _zero_prefill_plan(
    input_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Returns zeroed outputs shaped like a one-chunk dense prefill plan.

    Sparse kernels gather physical cache rows in both prefill and decode, so
    the dense planner, whose budget a valid batch's page-aligned aggregate
    context can exceed, never runs. The op still takes the three fields.
    """
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError("sparse metadata requires uint32 input row offsets")
    offsets = ops.broadcast_to(
        ops.constant(0, DType.uint32, device=input_row_offsets.device),
        (1, input_row_offsets.shape[0]),
    )
    lengths = ops.broadcast_to(
        ops.constant(0, DType.int32, device=input_row_offsets.device), (1,)
    )
    return offsets, offsets, lengths


_sparse_prefill_plan = F.functional(_zero_prefill_plan)


def _zero_prefill_budget(input_row_offsets: Tensor) -> Tensor:
    """Returns a host-resident zero for the dense arm's staging budget.

    A graph whose signature declares no batch context length -- the fused
    speculative graph, which declares one only under data parallelism -- still
    has to hand the combined prefill/decode op a dense planner budget. The
    sparse arm reads it nowhere: it selects its rows through ``sparse_indices``
    and never sizes a dense staging buffer from it. A host constant keeps the
    value off the device, so no layer performs a device-to-host copy of it,
    which device graph capture cannot record.

    Args:
        input_row_offsets: The batch's ragged offsets, read for their device.

    Returns:
        ``[1]`` int32 zeros, on the host.
    """
    del input_row_offsets
    return F.constant(0, DType.int32, device=CPU()).reshape([1])


def tensor_parallel_sparse_mla(
    layer: MistralLarge4SparseMLA,
) -> MistralLarge4SparseMLA:
    """Shards the attention projections along the tensor-parallel axis.

    The head axis is what splits: ``wq`` produces ``heads * qk_head_dim`` rows
    and ``wkv_b`` ``heads * (qk_nope + v)``, so both shard on their output axis
    and ``wo`` consumes the split heads, so it shards on its input axis. The
    latent projection, the two norms and the whole indexer stay replicated:
    the 576-wide latent is not a head axis, and every device must derive the
    same top-k selection from the same keys, as the DeepSeek-V3.2 spine does.

    SSMax's learned ``s``/``b`` follow the heads, so they shard with ``wq``'s
    output axis; at TP8 each rank holds its own 8 heads' pair.

    Args:
        layer: The attention layer to annotate with TP placements.

    Returns:
        The same layer, with its weights placed.
    """
    mesh = layer.wq.weight.mesh
    if mesh.num_devices == 1:
        return layer
    layer.wq.weight = layer.wq.weight.to(NamedMapping(mesh, (TP, None)))
    layer.wkv_b.weight = layer.wkv_b.weight.to(NamedMapping(mesh, (TP, None)))
    layer.wo.weight = layer.wo.weight.to(NamedMapping(mesh, (None, TP)))
    layer.ssmax_scaling.s = layer.ssmax_scaling.s.to(NamedMapping(mesh, (TP,)))
    layer.ssmax_scaling.b = layer.ssmax_scaling.b.to(NamedMapping(mesh, (TP,)))
    return layer


class MistralLarge4SparseMLA(
    Module[..., tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]]
):
    """Sparse latent attention with SSMax and optional cross-layer KV reading."""

    wq: QuantizedLinear
    """The query projection, ``hidden -> heads * qk_head_dim``; FP8 block."""
    wkv_a_with_mqa: Linear | None
    """The latent projection; absent on a KV reader."""
    kv_a_norm: RMSNorm | None
    """The 512-wide latent norm; absent on a KV reader."""
    wkv_b: Linear
    """The absorbed K/V projection; BF16, and present on every layer."""
    wo: QuantizedLinear
    """The output projection; FP8 block."""
    ssmax_scaling: SSMaxScaling
    """Per-(token, head) query scale."""
    indexer: MistralLarge4Indexer | None
    """The layer's own indexer, or ``None`` when it carries the previous full
    layer's top-k selection forward."""

    def __init__(
        self,
        *,
        num_attention_heads: int,
        hidden_size: int,
        layer_idx: int,
        kv_params: KVCacheParams | None,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        scale: float,
        graph_mode: str,
        buffer_size: int,
        quant_config: Any,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        owns_indexer: bool = True,
        owns_kv: bool = True,
        rope_interleaved: bool = True,
        norm_eps: float = 1e-6,
        index_norm_eps: float = 1e-6,
        row_invariant_decode: bool = False,
    ) -> None:
        """Builds the layer's projections for its place in the schedule.

        Args:
            num_attention_heads: Query heads (64).
            hidden_size: Width of the hidden state.
            layer_idx: This layer's index, for error messages.
            kv_params: The MLA cache leaf this layer's group uses.
            qk_nope_head_dim: Content dims per head; 128 on a NoPE layer.
            qk_rope_head_dim: Rope dims per head; 0 on a NoPE layer.
            kv_lora_rank: Latent width (512).
            v_head_dim: Value dims per head (128).
            scale: The explicit softmax scale, ``128 ** -0.5`` everywhere.
            graph_mode: ``"prefill"``, ``"decode"`` or ``"auto"``.
            buffer_size: Dense prefill buffer size the sparse plan ignores.
            quant_config: The FP8 block config ``wq`` and ``wo`` load with.
            index_n_heads: Index heads of an owned indexer.
            index_head_dim: Width of an index head.
            index_topk: Keys an owned indexer keeps.
            owns_indexer: Whether this layer runs its own indexer.
            owns_kv: Whether this layer owns its latent cache rows.
            rope_interleaved: Whether the MLA rope is GPT-J interleaved.
            norm_eps: Epsilon of the fused op's latent RMSNorm.
            index_norm_eps: Epsilon of the indexer's two LayerNorms.
            row_invariant_decode: Whether a decode step takes the DSpark
                verify's split-K count and row-invariant BF16 projections,
                so it computes each row bit for bit as the verify does.
        """
        super().__init__()
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        assert kv_params is not None, (
            "the model assigns this layer's MLA cache leaf before compile"
        )
        self.kv_params = kv_params
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim
        self.scale = scale
        self.graph_mode = graph_mode
        self.BUFFER_TOK_SIZE = buffer_size
        self.owns_kv = owns_kv
        self.owns_indexer = owns_indexer
        self.index_topk = index_topk
        self.row_invariant_decode = row_invariant_decode
        self.pins_verify_split = pins_verify_split(
            row_invariant_decode, graph_mode
        )
        self.norm_eps = norm_eps
        self.wq = QuantizedLinear(
            in_dim=hidden_size,
            out_dim=num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        if owns_kv:
            self.wkv_a_with_mqa = Linear(
                in_dim=hidden_size, out_dim=self.cache_head_dim, bias=False
            )
            self.kv_a_norm = RMSNorm(kv_lora_rank, eps=self.norm_eps)
        else:
            self.wkv_a_with_mqa = None
            self.kv_a_norm = None
        self.wkv_b = Linear(
            in_dim=kv_lora_rank,
            out_dim=num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.wo = QuantizedLinear(
            in_dim=num_attention_heads * v_head_dim,
            out_dim=hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.ssmax_scaling = SSMaxScaling(num_attention_heads)
        self.indexer = (
            MistralLarge4Indexer(
                hidden_size=hidden_size,
                n_heads=index_n_heads,
                head_dim=index_head_dim,
                rope_head_dim=qk_rope_head_dim,
                index_topk=index_topk,
                quant_config=quant_config,
                rope_interleaved=rope_interleaved,
                norm_eps=index_norm_eps,
                row_invariant_decode=row_invariant_decode,
            )
            if owns_indexer
            else None
        )

    def to(self, target: Any) -> MistralLarge4SparseMLA:
        """Moves the layer and its quantized projections to ``target``."""
        super().to(target)
        if self.wkv_a_with_mqa is not None:
            self.wkv_a_with_mqa = self.wkv_a_with_mqa.to(target)
        self.wkv_b = self.wkv_b.to(target)
        return self

    @property
    def _kv_b_proj_weight(self) -> Tensor:
        """``wkv_b`` reshaped to ``[kv_rank, heads, qk_nope + v]``."""
        return self.wkv_b.weight.T.reshape(
            (self.kv_lora_rank, self.n_heads, -1)
        )

    @property
    def w_uk(self) -> Tensor:
        """Decode K-projection, ``[heads, kv_rank, qk_nope]``."""
        return self._kv_b_proj_weight[..., : self.qk_nope_head_dim].transpose(
            0, 1
        )

    @property
    def w_uv(self) -> Tensor:
        """Decode V-projection, ``[heads, v_head, kv_rank]``."""
        return self._kv_b_proj_weight[..., self.qk_nope_head_dim :].permute(
            [1, 2, 0]
        )

    @property
    def w_k(self) -> Tensor:
        """Prefill K-projection, flattened to ``[heads * qk_nope, kv_rank]``."""
        return (
            self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
            .permute([1, 2, 0])
            .reshape((-1, self.kv_lora_rank))
        )

    def create_mla_prefill_metadata(
        self,
        input_row_offsets: Tensor,
        batch_context_length: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the prefill plan fields the dense prefill arm of the op takes.

        The plan depends only on the batch's layout, so the model builds it once
        and threads it into every layer rather than rebuilding it per layer.

        Args:
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            batch_context_length: Host-resident page-aligned context length,
                substituted for the plan's device-resident buffer length so no
                layer performs a device-to-host copy of it. ``None`` for a
                graph whose signature has no slot to carry one -- the fused
                speculative graph, whose canonical inputs declare a batch
                context length only under data parallelism -- which then stages
                a zero-length budget instead (see
                :func:`_zero_prefill_budget`).

        Returns:
            The plan's buffer row offsets, cache offsets and host buffer length.
        """
        replicated = DeviceMapping.replicated(input_row_offsets.mesh)
        buffer_row_offsets, cache_offsets, _ = _sparse_prefill_plan(
            input_row_offsets
        )
        return (
            buffer_row_offsets.rebind_mapping(replicated),
            cache_offsets.rebind_mapping(replicated),
            (
                _zero_prefill_budget(input_row_offsets)
                if batch_context_length is None
                else batch_context_length
            ),
        )

    def _mla_impl(
        self,
        xq: Tensor,
        kv: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        kv_norm_gamma: Tensor,
        sparse_indices: Tensor,
        mla_prefill_metadata: tuple[Tensor, Tensor, Tensor] | None,
    ) -> Tensor:
        """Runs the fused sparse MLA op and returns the flattened output."""
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_norm_gamma,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": self.norm_eps,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "w_uk": self.w_uk,
            "w_uv": self.w_uv,
        }
        # Only the combined prefill/decode op takes the dense planner's plan
        # and the prefill K projection; mla_decode_graph's signature has
        # neither, and a decode-only pipeline would otherwise fail to build.
        if self.graph_mode != "decode":
            assert mla_prefill_metadata is not None
            buffer_row_offsets, cache_offsets, buffer_length = (
                mla_prefill_metadata
            )
            attn_kwargs["buffer_row_offsets"] = buffer_row_offsets
            attn_kwargs["cache_offsets"] = cache_offsets
            # The dense arm sizes its paged write from `buffer_length`, so
            # handing it the batch's context length instead of the plan's own
            # buffer length is only safe on the sparse arm, which selects its
            # rows through `sparse_indices` and never reads this value.
            assert sparse_indices is not None
            attn_kwargs["buffer_length"] = buffer_length.to(CPU())
            attn_kwargs["w_k"] = self.w_k
        assert kv_collection.attention_dispatch_metadata is not None
        attn_kwargs["scalar_args"] = kv_collection.attention_dispatch_metadata
        assert kv_collection.mla_num_partitions is not None
        attn_kwargs["num_partitions_scalar"] = kv_collection.mla_num_partitions
        sparse_kw: dict[str, Any] = {
            "sparse_indices": sparse_indices,
            "physical_indices": True,
            "sparse_indices_stride": sparse_selection_width(
                self.kv_params, self.index_topk
            ),
            # The op always passes a no-op sink; the AMD sparse arms reject one
            # unless it is declared inert.
            "sparse_attn_sink_inert": True,
        }
        if self.pins_verify_split:
            # The pinned split is sized for a causal fixed-top-k selection;
            # the op rejects anything else only when Mojo lowers it.
            assert attn_kwargs["mask_variant"] is MHAMaskVariant.CAUSAL_MASK
            assert "sparse_topk_lengths" not in sparse_kw
            attn_kwargs["sparse_split_rows_per_request"] = (
                ML4_DSPARK_VERIFY_ROWS
            )
        if self.graph_mode == "decode":
            # TODO(DISTINF-832): the decode-only op takes no split pin, so a
            # disaggregated decode worker's DSpark verify and decode steps
            # still split differently. Thread the pin before that path ships.
            result = mla_decode_graph(**attn_kwargs, **sparse_kw)
        else:
            result = mla_prefill_decode_graph(**attn_kwargs, **sparse_kw)
        result = result.rebind_mapping(xq.mapping)
        return result.reshape((result.shape[0], self.n_heads * self.v_head_dim))

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_k_collection: PagedCacheValues | None,
        freqs_cis: Tensor,
        positions: Tensor,
        layer_idx: Tensor,
        indexer_layer_idx: Tensor,
        input_row_offsets: Tensor,
        quant_config: Any,
        prev_topk_indices: Tensor | None = None,
        reuse_prev_topk: bool = False,
        shared_kv: tuple[Tensor, Tensor] | None = None,
        *,
        query_activation: FP8BlockActivation | None = None,
        mla_prefill_metadata: tuple[Tensor, Tensor, Tensor] | None = None,
        ssmax_log_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Returns the attention output, its top-k selection and its latent.

        Args:
            x: ``[tokens, hidden]`` input-normed activations.
            kv_collection: This layer's MLA cache leaf.
            indexer_k_collection: The indexer key cache; ``None`` for a layer
                that reuses the previous full layer's selection.
            freqs_cis: The rotary table.
            positions: ``[tokens]`` absolute 0-based positions, for SSMax.
            layer_idx: This layer's slot in its MLA leaf.
            indexer_layer_idx: This layer's slot in the indexer cache.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            quant_config: The FP8 block config the indexer scores with.
            prev_topk_indices: The most recent full layer's selection.
            reuse_prev_topk: Whether to carry ``prev_topk_indices`` forward.
            shared_kv: ``(projection, norm_gamma)`` of the source layer whose cache
                rows this reader attends; required when the layer owns no
                ``wkv_a_with_mqa``.
            query_activation: ``x``'s FP8 block-quantized form, when the caller
                already launched the quantize as part of the input norm
                (:func:`~max.nn.common_layers.quant_ops.rms_norm_quantize_fp8_block`).
                ``None`` quantizes ``x`` here.
            mla_prefill_metadata: The plan the combined prefill/decode arm
                takes, built once for the batch; unused by a decode-only
                pipeline.
            ssmax_log_positions: ``positions``' SSMax logs, computed once for
                every layer; ``None`` takes them from ``positions`` here.

        Returns:
            The attention output, the top-k selection it used, and the
            ``(projection, gamma)`` pair a later reader may reuse (``None`` on
            a layer that produced none). The projection's leading columns are
            the latent; on a layer that owns its indexer and takes the
            row-invariant matmul, the indexer key's columns follow.

        Raises:
            ValueError: If a layer that neither owns a latent nor reads one
                calls this, or a reused selection arrives without its source.
        """
        xq = quant_ops.matmul(
            x if query_activation is None else query_activation,
            self.wq.weight,
        )
        xq = xq.reshape((xq.shape[0], self.n_heads, self.qk_head_dim))
        # SSMax on the whole query, after the rope the fused op will apply.
        if ssmax_log_positions is None:
            xq = self.ssmax_scaling(xq, positions)
        else:
            # One log per step is rounded once, then scaled and rounded to the
            # query's dtype: a log rounded to another dtype would round twice.
            if ssmax_log_positions.dtype != xq.dtype:
                raise ValueError(
                    "the batch's SSMax logs are"
                    f" {ssmax_log_positions.dtype} where the query path rounds"
                    f" {xq.dtype}"
                )
            xq = self.ssmax_scaling.scale_by_log(xq, ssmax_log_positions)

        runs_indexer = self.indexer is not None and not reuse_prev_topk
        carried: tuple[Tensor, Tensor] | None = None
        merged_projection: Tensor | None = None
        kv: Tensor
        kv_norm_gamma: Tensor
        if self.owns_kv:
            assert self.wkv_a_with_mqa is not None
            assert self.kv_a_norm is not None
            latent_weight = self.wkv_a_with_mqa.weight
            key_width = 0
            # Only the row-invariant matmul keeps its dispatch across the
            # width: each output column reduces K exactly as it does alone, so
            # the two halves equal the separate matmuls. A
            # `quant_ops.matmul` picks its split from the output width, so
            # merging would change what a fallback configuration computes and
            # the two projections stay separate there.
            if runs_indexer and self.row_invariant_decode:
                # The latent and the indexer key read the same rows, so one
                # launch projects both.
                assert self.indexer is not None
                key_weight = self.indexer.wk.weight
                assert isinstance(key_weight, Tensor)
                key_width = int(key_weight.shape[0])
                latent_weight = F.concat([latent_weight, key_weight], axis=0)
            projection = (
                row_invariant_matmul(x, latent_weight)
                if self.row_invariant_decode
                else quant_ops.matmul(x, latent_weight)
            )
            if key_width:
                # The whole projection is handed over, not its key columns:
                # slicing them here would materialise them in a copy launch of
                # their own, which is the launch this fusion is meant to
                # remove. The indexer is told, separately, the width of the
                # latent columns this layer put before the key's.
                merged_projection = projection
            kv_norm_gamma = self.kv_a_norm.weight
            # The whole projection is handed on and every layer slices its
            # latent off it: a sliced latent leaving the subgraph would be
            # materialized in a launch of its own, where views fuse.
            carried = (projection, kv_norm_gamma)
        else:
            if shared_kv is None:
                raise ValueError(
                    f"layer {self.layer_idx} reads another layer's MLA latent"
                    " and was given none"
                )
            # A reader has no projection of its own, so it hands nothing on.
            projection, kv_norm_gamma = shared_kv
        kv = projection[:, : self.cache_head_dim]

        if runs_indexer:
            assert self.indexer is not None
            assert indexer_k_collection is not None
            topk_indices = self.indexer(
                x,
                positions,
                freqs_cis,
                input_row_offsets,
                indexer_k_collection,
                indexer_layer_idx,
                kv_collection,
                quant_config,
                MHAMaskVariant.CAUSAL_MASK,
                key_projection=merged_projection,
                # The key's columns start where the latent's end, at the width
                # this layer concatenated before it, so the indexer is told
                # that width rather than recovering it from the operand.
                key_col_offset=(
                    self.cache_head_dim if merged_projection is not None else 0
                ),
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "A layer that carries the previous full layer's top-k "
                    "selection received none."
                )
            topk_indices = prev_topk_indices

        attn_out = self._mla_impl(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            sparse_indices=topk_indices,
            mla_prefill_metadata=mla_prefill_metadata,
        )
        return self.wo(attn_out), topk_indices, carried
