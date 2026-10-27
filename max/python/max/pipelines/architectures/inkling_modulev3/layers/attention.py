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
"""Inkling attention: qkvr projection, K/V short convolutions, relative bias.

Reference: ``nvidia/ops/qkvr_prep.py`` and ``nvidia/attention.py`` at vLLM
v0.26.0."""

from __future__ import annotations

from dataclasses import dataclass

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import (
    RowParallelLinear,
    col_parallel,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.kv_cache import PagedCacheValues as PagedCacheGraphValues

from ...inkling.layers.attention import _scale_rows
from ...inkling.layers.short_convolution import (
    fused_qk_rms_norm_short_conv_ragged,
)
from ...inkling.model_config import InklingTextConfig
from .placement import as_sharded, shard_axis
from .short_convolution import InklingRMSNorm, ShortConvolution


@dataclass(frozen=True)
class _DeviceAttention:
    """One device's share of an attention block's static shape."""

    kv_params: MHAKVCacheParams
    out_dims: tuple[int, int, int, int]
    num_heads: int
    d_rel: int
    rel_extent: int
    scale: float
    local_window_size: int | None
    applies_log_scaling: bool
    epsilon: float
    commit: bool


def _attention_on_device(
    qkvr: TensorValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    log_scaling: TensorValue,
    kv_collection: PagedCacheGraphValues,
    q_gamma: TensorValue,
    k_gamma: TensorValue,
    k_taps: TensorValue,
    v_taps: TensorValue,
    rel_logits_proj: TensorValue,
    k_conv_ring: BufferValue,
    v_conv_ring: BufferValue,
    k_conv_rows: TensorValue,
    v_conv_rows: TensorValue,
    k_layer_row: TensorValue,
    v_layer_row: TensorValue,
    cache_layer_idx: TensorValue,
    *,
    block: _DeviceAttention,
) -> TensorValue:
    """Runs one device's heads from the qkvr projection to the attention
    output, ``[total_tokens, q_dim]``."""
    q_dim, k_dim, v_dim, _ = block.out_dims
    total_tokens = qkvr.shape[0]
    r = qkvr[:, q_dim + k_dim + v_dim :]
    q = fused_qk_rms_norm_short_conv_ragged(
        block.kv_params,
        qkvr,
        input_row_offsets,
        positions,
        kv_collection,
        q_gamma,
        k_gamma,
        k_taps,
        v_taps,
        k_conv_ring,
        v_conv_ring,
        k_conv_rows,
        v_conv_rows,
        k_layer_row,
        v_layer_row,
        log_scaling,
        block.epsilon,
        cache_layer_idx,
        q_num_heads=block.num_heads,
        apply_log_scaling=block.applies_log_scaling,
        commit=block.commit,
    )
    rows = r.reshape([total_tokens * block.num_heads, block.d_rel])
    bias = (rows @ rel_logits_proj).reshape(
        [total_tokens, block.num_heads, block.rel_extent]
    )
    if block.applies_log_scaling:
        bias = _scale_rows(bias, log_scaling)

    attn_out = flash_attention_ragged(
        block.kv_params,
        input=q,
        input_row_offsets=input_row_offsets,
        kv_collection=kv_collection,
        layer_idx=cache_layer_idx,
        mask_variant=(
            MHAMaskVariant.CAUSAL_MASK
            if block.local_window_size is None
            else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
        ),
        scale=block.scale,
        local_window_size=(
            -1 if block.local_window_size is None else block.local_window_size
        ),
        rel_logits=bias,
    )
    return attn_out.reshape([total_tokens, q_dim])


_attention = F.functional(_attention_on_device)


class _RelLogitsProj(Module[[], None]):
    """Holds the relative-logits projection under the checkpoint's name."""

    def __init__(self, d_rel: int, rel_extent: int) -> None:
        self.proj = Tensor.zeros([d_rel, rel_extent])


class InklingAttention(Module[..., Tensor]):
    """One decoder layer's attention block.

    Splits the heads across devices: the qkvr projection column-parallel,
    ``wo_ud`` row-parallel, norms and the relative-logits projection
    replicated. The prologue (K/V short conv with its ring commit, Q/K norms,
    KV-cache stores) is one fused launch for any chunk length.
    """

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        num_devices: int,
        is_local: bool | None = None,
        commit_conv_state: bool = True,
    ) -> None:
        if is_local is None:
            is_local = text_config.is_local_attention(layer_idx)
        num_heads = text_config.num_heads(is_local)
        num_kv_heads = text_config.num_kv_heads(is_local)
        if num_heads % num_devices or num_kv_heads % num_devices:
            raise ValueError(
                f"layer {layer_idx} has {num_heads} query heads and "
                f"{num_kv_heads} KV heads, which do not both divide over "
                f"{num_devices} devices"
            )
        if kv_params.n_kv_heads_per_device != num_kv_heads // num_devices:
            raise ValueError(
                f"layer {layer_idx} is sharded {num_devices} ways into "
                f"{num_kv_heads // num_devices} KV heads per rank, but its "
                f"cache holds {kv_params.n_kv_heads_per_device} per device; "
                "the cache and the projection must be split the same way"
            )
        self.text_config = text_config
        self.layer_idx = layer_idx
        head_dim = text_config.head_dim_for(is_local)
        out_dims = text_config.qkvr_out_dims(is_local)
        q_dim, k_dim, v_dim, r_dim = out_dims
        local_window_size = text_config.attention_window(is_local)
        self.block = _DeviceAttention(
            kv_params=kv_params,
            out_dims=(
                q_dim // num_devices,
                k_dim // num_devices,
                v_dim // num_devices,
                r_dim // num_devices,
            ),
            num_heads=num_heads // num_devices,
            d_rel=text_config.d_rel,
            rel_extent=text_config.rel_extent_for(is_local),
            # muP scales attention logits by 1/d, not the usual 1/sqrt(d).
            scale=1.0 / head_dim,
            local_window_size=local_window_size,
            applies_log_scaling=text_config.applies_log_scaling(is_local),
            epsilon=text_config.rms_norm_eps,
            commit=commit_conv_state,
        )

        hidden = text_config.hidden_size
        self.wq_du = col_parallel(
            Linear(hidden, q_dim, bias=text_config.q_bias)
        )
        self.wk_dv = col_parallel(
            Linear(hidden, k_dim, bias=text_config.q_bias)
        )
        self.wv_dv = col_parallel(
            Linear(hidden, v_dim, bias=text_config.q_bias)
        )
        self.wr_du = col_parallel(
            Linear(hidden, r_dim, bias=text_config.q_bias)
        )

        kv_conv_dim = text_config.kv_conv_dim(is_local)
        self.k_sconv = ShortConvolution(
            channels=kv_conv_dim, kernel_size=text_config.sconv_kernel_size
        )
        self.v_sconv = ShortConvolution(
            channels=kv_conv_dim, kernel_size=text_config.sconv_kernel_size
        )
        self.k_sconv.weight = shard_axis(self.k_sconv.weight, 0)
        self.v_sconv.weight = shard_axis(self.v_sconv.weight, 0)
        self.q_norm = InklingRMSNorm(head_dim, eps=text_config.rms_norm_eps)
        self.k_norm = InklingRMSNorm(head_dim, eps=text_config.rms_norm_eps)
        self.rel_logits_proj = _RelLogitsProj(
            text_config.d_rel, self.block.rel_extent
        )
        self.wo_ud = RowParallelLinear(q_dim, hidden, bias=text_config.o_bias)

    def _qkvr(self, x: Tensor) -> Tensor:
        """Projects ``x`` to each device's ``[q | k | v | r]`` columns.

        Concatenating the column-parallel parts on their sharded axis joins
        each device's own pieces, so every device holds its heads of all
        four.
        """
        projections = (self.wq_du, self.wk_dv, self.wv_dv, self.wr_du)
        weight = F.concat([proj.weight for proj in projections], axis=0)
        out = x @ weight.T
        biases = [proj.bias for proj in projections]
        if all(isinstance(bias, Tensor) for bias in biases):
            out = out + F.concat(biases, axis=0)
        return out

    def forward(
        self,
        x: Tensor,
        *,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        k_conv_ring: Tensor,
        k_conv_rows: Tensor,
        k_layer_row: Tensor,
        v_conv_ring: Tensor,
        v_conv_rows: Tensor,
        v_layer_row: Tensor,
        cache_layer_idx: Tensor,
    ) -> Tensor:
        """Runs the block over one ragged batch. Returns each device's
        partial sum of the output.

        ``cache_layer_idx`` is an operand, not a folded constant, so every
        layer of one attention flavor can share a compiled subgraph.
        """
        attn_out = _attention(
            self._qkvr(x),
            input_row_offsets,
            positions,
            log_scaling,
            kv_collection,
            self.q_norm.weight,
            self.k_norm.weight,
            self.k_sconv.taps,
            self.v_sconv.taps,
            self.rel_logits_proj.proj,
            k_conv_ring,
            v_conv_ring,
            k_conv_rows,
            v_conv_rows,
            k_layer_row,
            v_layer_row,
            cache_layer_idx,
            block=self.block,
        )
        return self.wo_ud(as_sharded(attn_out, x.mesh, axis=1))
