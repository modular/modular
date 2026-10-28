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
"""One Kimi K3 decoder layer (ModuleV3), batch-size-1 scope.

Direct V3 port of ``kimi_k3/kimi_k3.py``'s ``KimiK3DecoderLayer.__call__``.
This file's job is ONLY the assembly order, which this model gets uniquely
wrong if copied from a plain transformer: there is no additive residual (see
``attn_residual.py``), and each layer's sublayer type (KDA vs MLA, dense MLP
vs MoE) is a config-time choice, not a runtime branch.

MLA runs one of three forms, selected by what the caller passes: padded
(no ``kv_collection``, ``[batch, seq_len, hidden]``, reshaped at the call
site to share KDA's ragged ``[total_seq_len, hidden]``), ragged over a
decompressed cache, or ragged over a compressed latent cache -- the latter
two take a ``kv_collection`` and ``input_row_offsets``, and which runs is
read off ``self.attention.kv_params``'s type (an ``MLAKVCacheParams`` means
latent) rather than threaded as a separate flag. The latent form additionally
needs ``mla_prefill_plan``/``mla_freqs_cis``, shared across every MLA layer
this step -- see ``mla_attention.py``'s ``kimi_k3_mla_prefill_plan`` and
``kimi_k3_nope_freqs_cis``.

Under tensor parallelism both attention forms and the dense MLP return
partial sums, which ``forward`` reduces. ``KimiK3LatentMoE`` reduces
internally, so its output is used as is. Under expert parallelism,
``forward`` takes an ``ep_config``/``ep_buffers`` pair and passes them to
the FFN; nothing else in the layer changes.
"""

from __future__ import annotations

from max import functional as F
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.kv_cache import MLAKVCacheParams
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from . import ep
from .attn_residual import (
    kimi_k3_attn_res_post_attention,
    kimi_k3_attn_res_pre_attention,
)
from .kda_attention import KimiK3KDAAttention, tensor_parallel_kda_attention
from .mla_attention import KimiK3MLAAttention, tensor_parallel_mla_attention
from .mlp import KimiK3DenseMLP, tensor_parallel_dense_mlp
from .moe_latent import (
    KimiK3LatentMoE,
    expert_parallel_latent_moe,
    tensor_parallel_latent_moe,
)
from .norm import KIMI_RMS_NORM_EPS, kimi_rms_norm


def _tp_active(mesh: DeviceMesh) -> bool:
    """Whether ``mesh`` actually shards along ``tp``.

    A mesh can span several devices for another reason, such as expert
    parallelism, and a pure-EP mesh has no ``tp`` axis at all.
    """
    return TP in mesh.axis_names and mesh.axis_size(TP) > 1


class KimiK3DecoderLayer(Module[..., tuple[Tensor, list[Tensor]]]):
    """One decoder layer: residual mixture, attention, residual mixture, FFN."""

    def __init__(
        self,
        *,
        layer_idx: int,
        attn_res_block_size: int,
        is_kda: bool,
        is_dense: bool,
        hidden_size: int,
        attention: KimiK3KDAAttention | KimiK3MLAAttention,
        ffn: KimiK3DenseMLP | KimiK3LatentMoE,
        rms_norm_eps: float = KIMI_RMS_NORM_EPS,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_res_block_size = attn_res_block_size
        self.is_kda = is_kda
        self.is_dense = is_dense
        self.rms_norm_eps = rms_norm_eps
        self.attention = attention
        self.ffn = ffn

        self.input_layernorm_weight = Tensor.ones([hidden_size])
        self.post_attention_layernorm_weight = Tensor.ones([hidden_size])
        self.self_attention_res_proj_weight = Tensor.zeros([1, hidden_size])
        self.self_attention_res_norm_weight = Tensor.ones([hidden_size])
        self.mlp_res_proj_weight = Tensor.zeros([1, hidden_size])
        self.mlp_res_norm_weight = Tensor.ones([hidden_size])

    def forward(
        self,
        hidden_states: Tensor,
        block_residual: list[Tensor],
        *,
        conv_pool: Tensor | None = None,
        conv_row_id: Tensor | None = None,
        state_pool: Tensor | None = None,
        recurrent_row_id: Tensor | None = None,
        input_row_offsets: Tensor | None = None,
        ep_config: EPConfig | None = None,
        ep_buffers: ep.EPBuffers | None = None,
        kv_collection: PagedCacheValues | None = None,
        mla_prefill_plan: MLAPrefillMetadata | None = None,
        mla_freqs_cis: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Runs the layer over the batch's tokens.

        Args:
            hidden_states: ``[total_seq_len, hidden]`` ragged (batch size 1
            at this scope, so ``total_seq_len == seq_len``).
            block_residual: The candidate stack entering this layer.
            conv_pool: KDA's conv-history pool, mutated in place at
            ``conv_row_id``; required when ``is_kda``.
            conv_row_id: The conv-pool row of each sequence.
            state_pool: KDA's recurrent-state pool, mutated in place at
            ``recurrent_row_id``; required when ``is_kda``.
            recurrent_row_id: The recurrent-state row of each sequence.
            input_row_offsets: The batch's ragged metadata. An MLA layer
            reads only this, and only when serving over ``kv_collection``.
            ep_config: The FFN's expert-parallel exchange setup; only
            meaningful when the FFN is an expert-parallel
            ``KimiK3LatentMoE``, unused otherwise.
            ep_buffers: The exchange's graph buffers, passed whenever
            ``ep_config`` is.
            kv_collection: An MLA layer's paged cache handle for this batch.
            ``None`` runs the padded/parity form instead -- unused for KDA.
            mla_prefill_plan: The compressed latent cache's prefill plan,
            required with ``kv_collection`` when
            ``self.attention.kv_params`` is an ``MLAKVCacheParams``.
            mla_freqs_cis: The rotary frequencies that go with
            ``mla_prefill_plan``.
        """
        x, prefix_sum, stack = kimi_k3_attn_res_pre_attention(
            hidden_states,
            block_residual,
            self.layer_idx,
            self.attn_res_block_size,
            self.self_attention_res_proj_weight,
            self.self_attention_res_norm_weight,
            self.rms_norm_eps,
        )
        normed = kimi_rms_norm(
            x, self.input_layernorm_weight, self.rms_norm_eps
        )

        if isinstance(self.attention, KimiK3KDAAttention):
            assert (
                conv_pool is not None
                and conv_row_id is not None
                and state_pool is not None
                and recurrent_row_id is not None
                and input_row_offsets is not None
            )
            attn_out = self.attention(
                normed,
                conv_pool,
                conv_row_id,
                state_pool,
                recurrent_row_id,
                input_row_offsets,
            )
        elif kv_collection is not None:
            assert isinstance(self.attention, KimiK3MLAAttention)
            assert input_row_offsets is not None
            if isinstance(self.attention.kv_params, MLAKVCacheParams):
                assert (
                    mla_prefill_plan is not None and mla_freqs_cis is not None
                )
                attn_out = self.attention.forward_latent_ragged(
                    normed,
                    kv_collection,
                    input_row_offsets,
                    mla_prefill_plan,
                    mla_freqs_cis,
                )
            else:
                attn_out = self.attention.forward_ragged(
                    normed, kv_collection, input_row_offsets
                )
        else:
            total_seq_len = normed.shape[0]
            hidden = normed.shape[1]
            padded = F.reshape(normed, [1, total_seq_len, hidden])
            attn_out = F.reshape(
                self.attention(padded), [total_seq_len, hidden]
            )
        if _tp_active(self.attention.o_proj.weight.mesh):
            attn_out = F.allreduce_sum(attn_out, mesh_axis=TP)

        mixed, prefix = kimi_k3_attn_res_post_attention(
            prefix_sum,
            attn_out,
            stack,
            self.mlp_res_proj_weight,
            self.mlp_res_norm_weight,
            self.rms_norm_eps,
        )
        normed2 = kimi_rms_norm(
            mixed, self.post_attention_layernorm_weight, self.rms_norm_eps
        )
        if isinstance(self.ffn, KimiK3LatentMoE):
            ffn_out = self.ffn(
                normed2, ep_config=ep_config, ep_buffers=ep_buffers
            )
        else:
            ffn_out = self.ffn(normed2)
        if isinstance(self.ffn, KimiK3DenseMLP) and _tp_active(
            self.ffn.down_proj.weight.mesh
        ):
            ffn_out = F.allreduce_sum(ffn_out, mesh_axis=TP)
        return prefix + ffn_out, stack


def tensor_parallel_decoder_layer(
    layer: KimiK3DecoderLayer, mesh: DeviceMesh
) -> KimiK3DecoderLayer:
    """Shards a decoder layer's sublayers along the ``tp`` mesh axis.

    The layer's weights must already be on ``mesh``. The norm and
    residual-mixture weights work at the full hidden width and stay
    replicated.
    """
    if isinstance(layer.attention, KimiK3KDAAttention):
        tensor_parallel_kda_attention(layer.attention, mesh)
    else:
        tensor_parallel_mla_attention(layer.attention, mesh)
    if isinstance(layer.ffn, KimiK3DenseMLP):
        tensor_parallel_dense_mlp(layer.ffn, mesh)
    else:
        tensor_parallel_latent_moe(layer.ffn, mesh)
    return layer


def expert_parallel_decoder_layer(
    layer: KimiK3DecoderLayer, mesh: DeviceMesh
) -> KimiK3DecoderLayer:
    """Shards a decoder layer's MoE FFN across ``mesh``, expert by expert.

    A dense layer's MLP stays replicated. Call
    :meth:`KimiK3DecoderLayer.forward` with this layer's
    ``ep_config``/``ep_buffers`` to use the sharded experts.
    """
    if isinstance(layer.ffn, KimiK3LatentMoE):
        expert_parallel_latent_moe(layer.ffn, mesh)
    return layer
