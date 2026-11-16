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
"""Mistral Large 4 transformer block (ModuleV3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
)
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.legacy.quant_config import QuantConfig
from max.nn.norm import RMSNorm
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from ...deepseekV3_modulev3.layers.transformer_block import (
    ParallelismMode,
    post_attention,
    post_mlp,
)
from ..config import MistralLarge4Config, mla_group
from ..model_config import ML4_DSPARK_VERIFY_ROWS
from ._dispatch import SM100_ARCHS
from .mlp import MistralLarge4MLP
from .moe import (
    MistralLarge4ExpertParallelMoE,
    MistralLarge4MoE,
    MistralLarge4TensorParallelMoE,
)
from .sparse_mla import (
    MistralLarge4SparseMLA,
    tensor_parallel_sparse_mla,
)


def _normed_query_input(
    x: Tensor, norm: RMSNorm, weight: object
) -> tuple[Tensor, FP8BlockActivation | None]:
    """Returns the attention input norm, and its FP8 quantized form if fused.

    The norm's only other readers are BF16, so the FP8 quantize ``wq`` needs
    can ride the norm in one launch
    (:func:`~max.nn.common_layers.quant_ops.rms_norm_quantize_fp8_block`);
    a BF16 ``wq`` needs no quantize and keeps the norm on its own.

    Only a plain :class:`~max.nn.norm.RMSNorm` fuses: the fused op hardcodes
    ``weight_offset=0`` and ``multiply_before_cast=False``, while a
    ``GemmaRMSNorm`` sets those in an overridden ``forward``, so fusing one
    would compute a different normalization with no error.

    Args:
        x: ``[tokens, hidden]`` the previous layer produced.
        norm: The block's input norm.
        weight: ``wq``'s weight, which sets the quantize's block size.

    Returns:
        The normalized row, and the quantized activation when it was fused.
    """
    if isinstance(weight, FP8BlockTensor) and type(norm) is RMSNorm:
        return quant_ops.rms_norm_quantize_fp8_block(
            x, weight, norm.weight, norm.eps
        )
    return norm(x), None


def _parallelism_mode(config: MistralLarge4Config) -> ParallelismMode:
    """Resolves the layer's parallelism strategy from the config.

    Mistral Large 4's 128 NVFP4 experts fit on one device or split over the
    ranks two ways: the FFN width of every expert (``TP_TP``), or whole experts
    per rank beside TP attention (``TP_EP``, or ``DP_EP`` under data-parallel
    attention). Only an ``ep_config`` with more than one device selects the
    latter; a multi-device export without an ``ep_size`` lands on ``TP_TP``.
    """
    num_devices = len(config.devices)
    if num_devices <= 1:
        return ParallelismMode.TP_TP
    if config.ep_config is not None:
        if config.data_parallel_degree == 1:
            return ParallelismMode.TP_EP
        return ParallelismMode.DP_EP
    if config.data_parallel_degree == num_devices:
        return ParallelismMode.DP_EP
    return ParallelismMode.TP_TP


@dataclass(frozen=True)
class RowInvariantDecodeBlocker:
    """Why the decode step cannot match the DSpark verify bit for bit."""

    reason: str
    warn: bool
    """Whether losing the identity deserves a warning: the target could not
    be read, or the cache is sized for a speculative verify."""


def row_invariant_decode_blocker(
    config: MistralLarge4Config, kv_params: KVCacheParamInterface
) -> RowInvariantDecodeBlocker | None:
    """Returns why the decode step cannot match the DSpark verify bit for bit.

    The decode pins (the verify's sparse MLA split-K count and the
    row-invariant BF16 matmul) are kernel opt-ins whose compile-time checks
    hold only on SM100, with an FP8 latent cache and at most 64 query rows per
    head group across a verify. Outside that regime the model keeps the
    default kernels, so it still builds; ``None`` means the pins apply. The
    base graph takes them too: a solo request's greedy output with DSpark must
    match the base deployment's.

    Args:
        config: The model config the decoder is built from.
        kv_params: The cache tree the graph's inputs are flattened from. Its
            latent leaves are the cache the sparse MLA ops receive, which can
            differ from the config's own tree.
    """
    assert isinstance(kv_params, MultiKVCacheParams)
    leaves = []
    for rope in sorted(set(config.rope_layers)):
        leaf = kv_params.children[mla_group(rope)]
        assert isinstance(leaf, KVCacheParams)
        leaves.append(leaf)
    speculative = any(leaf.speculative_method is not None for leaf in leaves)
    try:
        arch = accelerator_architecture_name()
    except Exception:
        return RowInvariantDecodeBlocker(
            "the target architecture could not be read", warn=True
        )
    if arch not in SM100_ARCHS:
        return RowInvariantDecodeBlocker(
            f"the target is {arch}, not SM100", warn=speculative
        )
    for leaf in leaves:
        if leaf.dtype != DType.float8_e4m3fn:
            return RowInvariantDecodeBlocker(
                f"the MLA cache is {leaf.dtype}, not float8_e4m3fn",
                warn=speculative,
            )
    tp = (
        1
        if _parallelism_mode(config) is ParallelismMode.DP_EP
        else len(config.devices)
    )
    heads = config.num_attention_heads // tp
    if heads * ML4_DSPARK_VERIFY_ROWS > 64:
        return RowInvariantDecodeBlocker(
            f"{heads} heads per rank times a {ML4_DSPARK_VERIFY_ROWS}-row"
            " verify exceeds one 64-row head group",
            warn=speculative,
        )
    return None


def _shards_dense_ffn(
    mode: ParallelismMode, use_allreduce: bool | None
) -> bool:
    """Whether the dense layer-0 FFN may be tensor-parallel sharded.

    ``w2`` is row parallel, so its output is a partial sum: the split is legal
    only where every rank holds every token -- under ``TP_TP``, or under TP
    attention beside EP's allreduce backend. A data-parallel replica holds its
    own rows, and under EP's reduce-scatter backend each rank holds a disjoint
    row subset, so nothing later completes the sum in either.

    Args:
        mode: The layer's parallelism strategy.
        use_allreduce: The EP backend's allreduce setting, when EP is on.

    Returns:
        Whether the dense FFN's projections split along the tensor-parallel
        axis.
    """
    if mode is ParallelismMode.TP_TP:
        return True
    # DP replicas hold disjoint rows, so even EP's allreduce backend leaves the
    # dense FFN replicated rather than completing a row-parallel sum.
    return mode is ParallelismMode.TP_EP and use_allreduce is True


class MistralLarge4TransformerBlock(
    Module[..., tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]]
):
    """Sparse MLA, MoE or dense MLP, and the two RMSNorms around them.

    Differs from the DeepSeek-V3.2 block in that attention takes the row
    positions (SSMax) and can read another layer's latent, so ``forward``
    threads those through and returns the latent a later reader may reuse.
    """

    self_attn: MistralLarge4SparseMLA
    mlp: (
        MistralLarge4MoE
        | MistralLarge4TensorParallelMoE
        | MistralLarge4ExpertParallelMoE
        | MistralLarge4MLP
    )

    def __init__(
        self,
        *,
        config: MistralLarge4Config,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        attention_scale: float,
        rms_norm_eps: float,
        kv_params: KVCacheParams,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        owns_indexer: bool,
        owns_kv: bool,
        is_moe: bool,
        moe_intermediate_size: int,
        dense_intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        attention_quant_config: QuantConfig,
        expert_quant_config: QuantConfig,
        graph_mode: str,
        buffer_size: int,
        rope_interleaved: bool = True,
        ep_batch_manager: EPBatchManager | None = None,
        index_norm_eps: float = 1e-6,
        row_invariant_decode: bool = False,
    ) -> None:
        """Builds one layer from the resolved schedule.

        Args:
            config: The model config, read for the parallelism strategy the
                residual collectives follow.
            layer_idx: This layer's index.
            hidden_size: Width of the hidden state.
            num_attention_heads: Query heads.
            qk_nope_head_dim: Content dims per head; this layer's own width,
                128 on a NoPE layer.
            qk_rope_head_dim: Rope dims per head; 0 on a NoPE layer.
            kv_lora_rank: Latent width.
            v_head_dim: Value dims per head.
            attention_scale: The explicit softmax scale.
            rms_norm_eps: Epsilon of the block's two norms.
            kv_params: The MLA cache leaf this layer's group uses.
            index_n_heads: Index heads.
            index_head_dim: Width of an index head.
            index_topk: Keys the indexer keeps.
            owns_indexer: Whether this layer runs its own indexer.
            owns_kv: Whether this layer owns its latent cache rows.
            is_moe: Whether the layer is sparse (MoE) rather than dense.
            moe_intermediate_size: Expert hidden width.
            dense_intermediate_size: Dense FFN hidden width.
            num_experts: Routed experts.
            num_experts_per_token: Experts a token takes.
            attention_quant_config: FP8 block config for ``wq``/``wo``/``wq_b``.
            expert_quant_config: NVFP4 config for the routed experts.
            graph_mode: ``"prefill"``, ``"decode"`` or ``"auto"``.
            buffer_size: Dense prefill buffer size the sparse plan ignores.
            rope_interleaved: Whether both ropes are GPT-J interleaved.
            ep_batch_manager: The expert-parallel runtime, when EP is on.
            index_norm_eps: Epsilon of the indexer's two LayerNorms.
            row_invariant_decode: Whether the decode step takes the DSpark
                verify's kernels; see :func:`row_invariant_decode_blocker`.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mode = _parallelism_mode(config)
        num_devices = len(config.devices)
        self.self_attn = MistralLarge4SparseMLA(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            kv_params=kv_params,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            scale=attention_scale,
            graph_mode=graph_mode,
            buffer_size=buffer_size,
            quant_config=attention_quant_config,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            owns_indexer=owns_indexer,
            owns_kv=owns_kv,
            rope_interleaved=rope_interleaved,
            norm_eps=rms_norm_eps,
            index_norm_eps=index_norm_eps,
            row_invariant_decode=row_invariant_decode,
        )
        if self.mode is not ParallelismMode.DP_EP:
            tensor_parallel_sparse_mla(self.self_attn)
        self.mlp: (
            MistralLarge4MoE
            | MistralLarge4TensorParallelMoE
            | MistralLarge4ExpertParallelMoE
            | MistralLarge4MLP
        )
        if is_moe:
            if ep_batch_manager is not None:
                self.mlp = MistralLarge4ExpertParallelMoE(
                    hidden_dim=hidden_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    moe_dim=moe_intermediate_size,
                    expert_quant_config=expert_quant_config,
                    shared_expert_quant_config=attention_quant_config,
                    ep_batch_manager=ep_batch_manager,
                )
            elif self.mode is ParallelismMode.DP_EP or num_devices == 1:
                self.mlp = MistralLarge4MoE(
                    hidden_dim=hidden_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    moe_dim=moe_intermediate_size,
                    expert_quant_config=expert_quant_config,
                    shared_expert_quant_config=attention_quant_config,
                )
            else:
                self.mlp = MistralLarge4TensorParallelMoE(
                    hidden_dim=hidden_size,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    moe_dim=moe_intermediate_size,
                    expert_quant_config=expert_quant_config,
                    shared_expert_quant_config=attention_quant_config,
                    graph_mode=graph_mode,
                )
        else:
            dense = MistralLarge4MLP(
                hidden_size,
                dense_intermediate_size,
                quant_config=attention_quant_config,
            )
            # Layer 0's FFN is the only dense one, so it is the only place a
            # TP split shows up outside attention.
            if _shards_dense_ffn(
                self.mode,
                None
                if self.config.ep_config is None
                else self.config.ep_config.use_allreduce,
            ):
                # Gate and up shard on their output axis, down on its input
                # axis, which is the head-in/out mirror of the attention
                # projections' TP split.
                for gate_or_up in (dense.w1, dense.w3):
                    gate_or_up.weight = gate_or_up.weight.to(
                        NamedMapping(gate_or_up.weight.mesh, (TP, None))
                    )
                dense.w2.weight = dense.w2.weight.to(
                    NamedMapping(dense.w2.weight.mesh, (None, TP))
                )
            self.mlp = dense
        self.input_layernorm = RMSNorm(dim=hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            dim=hidden_size, eps=rms_norm_eps
        )
        self.graph_mode = graph_mode
        self.buffer_size = buffer_size

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        indexer_k_collection: PagedCacheValues | None,
        freqs_cis: Tensor,
        positions: Tensor,
        input_row_offsets: Tensor,
        layer_idx: Tensor,
        indexer_layer_idx: Tensor,
        quant_config: Any,
        prev_topk_indices: Tensor | None = None,
        reuse_prev_topk: bool = False,
        shared_kv: tuple[Tensor, Tensor] | None = None,
        comm_buffers: EPCommBuffers | None = None,
        *,
        mla_prefill_metadata: tuple[Tensor, Tensor, Tensor] | None = None,
        ssmax_log_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Returns the block output, its top-k selection and its latent.

        Args:
            x: ``[tokens, hidden]`` input activations.
            kv_collection: This layer's MLA cache leaf.
            indexer_k_collection: The indexer key cache, or ``None`` when the
                layer carries the previous full layer's selection forward.
            freqs_cis: The rotary table.
            positions: ``[tokens]`` absolute 0-based positions.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            layer_idx: This layer's slot in its MLA leaf.
            indexer_layer_idx: This layer's slot in the indexer cache.
            quant_config: FP8 block config the indexer scores with.
            prev_topk_indices: The most recent full layer's selection.
            reuse_prev_topk: Whether to carry ``prev_topk_indices`` forward.
            shared_kv: The source layer's ``(projection, norm_gamma)`` when this
                layer reads another layer's cache rows.
            comm_buffers: The expert-parallel communication buffers, when EP
                is on.
            mla_prefill_metadata: The prefill plan, built once for the batch.
            ssmax_log_positions: The rows' SSMax logs, built once for the
                batch; ``None`` derives them from ``positions`` in the layer.
        """
        residual = x
        norm_x, query_activation = _normed_query_input(
            x, self.input_layernorm, self.self_attn.wq.weight
        )
        attn_out, topk_indices, carried = self.self_attn(
            norm_x,
            kv_collection,
            indexer_k_collection,
            freqs_cis,
            positions,
            layer_idx,
            indexer_layer_idx,
            input_row_offsets,
            quant_config,
            prev_topk_indices,
            reuse_prev_topk,
            shared_kv,
            query_activation=query_activation,
            mla_prefill_metadata=mla_prefill_metadata,
            ssmax_log_positions=ssmax_log_positions,
        )
        hidden_states = post_attention(
            self.mode, self.config, residual, attn_out
        )
        norm_h = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MistralLarge4ExpertParallelMoE):
            mlp_out = self.mlp(norm_h, comm_buffers)
        else:
            mlp_out = self.mlp(norm_h)
        return (
            post_mlp(self.mode, self.config, hidden_states, mlp_out),
            topk_indices,
            carried,
        )
