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
"""Inkling 2 attention: projections, K/V short convolutions, relative bias."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module, TransparentModule
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import RowParallelLinear, col_parallel
from max.nn.kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.linear import Linear
from max.support.math import ceildiv
from max.tensor import Tensor

from ..model_config import Inkling2TextConfig
from .norm import unweighted_rms_norm
from .placement import as_sharded, shard_axis
from .short_convolution import (
    DynamicShortConvolution,
    LayerConvState,
    ShortConvolution,
)

_CLIP_EPS = 1e-6


class _ScaleParam(Module[[], None]):
    """A learned scale under the checkpoint's name."""

    def __init__(self, shape: list[int]) -> None:
        self.scale = shard_axis(Tensor.zeros(shape), 0)


class _RelLogitsProj(Module[[], None]):
    """The bank of bias-versus-distance profiles, ``[d_rel, extent]``."""

    def __init__(self, d_rel: int, extent: int) -> None:
        self.proj = Tensor.zeros([d_rel, extent])


_ROW_ALIGN = 8
"""Rows per head in the fused weight round up to this, for 16-byte bf16 rows."""


class _QuerySideProjection(
    TransparentModule[[Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]
):
    """The q, r, gate and sink projections behind one GEMM.

    At decode each projection is a weight-bound GEMV, so one launch over their
    row interleaved weight replaces four. The rows are head-major, placing
    each head's q block, r block, gate block and sink row together, so the
    output reshapes to ``[tokens, heads, q | r | gate | sink | pad]`` and splits on
    the last axis. That keeps the split off the head axis, where the
    column-parallel weights shard: each rank's rows are its own heads' four
    blocks, where a plain ``[q; r; gate; sink]`` row concat would hand a rank
    a slice of every projection instead.

    Each head's rows are padded with zeros to a multiple of 8, so a rank's
    output row stays 16-byte aligned for the SM100 matmul. Unpadded, 12 heads
    of 273 rows is 3276 columns, which the dispatcher hands to cuBLAS's sm80
    kernel at several times the cost.

    Name-transparent, so the checkpoint's four names still load. The weights
    are dense bf16; a quantized projection would have to interleave its scale
    blocks in the same head-major order.
    """

    def __init__(
        self,
        *,
        hidden: int,
        num_heads: int,
        head_dim: int,
        d_rel: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.hidden = hidden
        self.width = 2 * head_dim + d_rel + 1
        self.padded_width = ceildiv(self.width, _ROW_ALIGN) * _ROW_ALIGN
        self.wq_du = col_parallel(
            Linear(hidden, num_heads * head_dim, bias=False)
        )
        self.wr_du = col_parallel(Linear(hidden, num_heads * d_rel, bias=False))
        self.gate_attn = col_parallel(
            Linear(hidden, num_heads * head_dim, bias=False)
        )
        self.attn_sink_proj = col_parallel(
            Linear(hidden, num_heads, bias=False)
        )

    @property
    def fused_weight(self) -> Tensor:
        """The interleaved ``[heads * padded_width, hidden]`` weight, one GEMM."""
        parts = [
            self.wq_du.weight.reshape(
                [self.num_heads, self.head_dim, self.hidden]
            ),
            self.wr_du.weight.reshape(
                [self.num_heads, self.d_rel, self.hidden]
            ),
            self.gate_attn.weight.reshape(
                [self.num_heads, self.head_dim, self.hidden]
            ),
            self.attn_sink_proj.weight.reshape(
                [self.num_heads, 1, self.hidden]
            ),
        ]
        pad = self.padded_width - self.width
        if pad:
            parts.append(
                F.zeros_like(
                    F.broadcast_to(parts[3], [self.num_heads, pad, self.hidden])
                )
            )
        return F.concat(parts, axis=1).reshape(
            [self.num_heads * self.padded_width, self.hidden]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Projects ``x`` to per-head q, r, gate and sink with one matmul."""
        per_head = (x @ self.fused_weight.T).reshape(
            [-1, self.num_heads, self.padded_width]
        )
        sizes = [self.head_dim, self.d_rel, self.head_dim, 1]
        pad = self.padded_width - self.width
        if pad:
            sizes.append(pad)
        q, r, gate, sink = per_head.split(sizes, axis=-1)[:4]
        return q, r, gate, sink.reshape([-1, self.num_heads])


def l2_clip(x: Tensor, threshold: float, scale: Tensor | None = None) -> Tensor:
    """Scales each head vector down to ``threshold`` when its norm is larger.

    A learned ``scale`` of shape ``[heads, dim]`` applies first. The math runs
    in float32 and the result rounds once to ``x``'s dtype.
    """
    x32 = x.cast(DType.float32)
    if scale is not None:
        x32 = x32 * scale.cast(DType.float32)
    norm = F.sqrt((x32 * x32).sum(axis=-1))
    factor = F.elementwise_min(threshold / (norm + _CLIP_EPS), 1.0)
    return (x32 * factor).cast(x.dtype)


def scale_rows(x: Tensor, factor: Tensor) -> Tensor:
    """Scales each token's ``[tokens, ...]`` rows by a float32 ``[tokens]``."""
    rows = factor.reshape([-1] + [1] * (x.rank - 1))
    return (x.cast(DType.float32) * rows).cast(x.dtype)


class Inkling2Attention(Module[..., Tensor]):
    """One decoder layer's attention block.

    The heads split across devices: the projections column-parallel,
    ``wo_ud`` row-parallel. A local layer's K and V pass through a static
    short convolution and a global layer's through a dynamic one. Queries and
    relative states are clipped, and on a global layer also scaled by the
    position-dependent factor ``tau``.
    """

    def __init__(
        self,
        *,
        text_config: Inkling2TextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        num_devices: int,
    ) -> None:
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
                f"cache holds {kv_params.n_kv_heads_per_device} per device"
            )
        self.text_config = text_config
        self.is_local = is_local
        self.kv_params = kv_params
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = text_config.head_dim_for(is_local)
        self.window = text_config.attention_window(is_local)

        hidden = text_config.hidden_size
        q_dim = num_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        d_rel = text_config.d_rel
        self.q_side = _QuerySideProjection(
            hidden=hidden,
            num_heads=num_heads,
            head_dim=self.head_dim,
            d_rel=d_rel,
        )
        self.wk_dv = col_parallel(Linear(hidden, kv_dim, bias=False))
        self.wv_dv = col_parallel(Linear(hidden, kv_dim, bias=False))
        self.wo_ud = RowParallelLinear(q_dim, hidden, bias=False)
        self.q_l2_clip = _ScaleParam([num_heads, self.head_dim])
        self.rel_l2_clip = _ScaleParam([num_heads, d_rel])
        self.rel_logits_proj = _RelLogitsProj(
            d_rel, text_config.rel_extent_for(is_local)
        )
        kernel = text_config.sconv_kernel_size
        self.k_sconv: ShortConvolution | DynamicShortConvolution
        self.v_sconv: ShortConvolution | DynamicShortConvolution
        if is_local:
            k_conv = ShortConvolution(channels=kv_dim, kernel_size=kernel)
            v_conv = ShortConvolution(channels=kv_dim, kernel_size=kernel)
            k_conv.weight = shard_axis(k_conv.weight, 0)
            v_conv.weight = shard_axis(v_conv.weight, 0)
            self.k_sconv, self.v_sconv = k_conv, v_conv
        else:
            k_dyn, v_dyn = (
                DynamicShortConvolution(
                    channels=kv_dim,
                    input_size=hidden,
                    num_heads=num_kv_heads,
                    kernel_size=kernel,
                )
                for _ in range(2)
            )
            for conv in (k_dyn, v_dyn):
                conv.static_w = shard_axis(conv.static_w, 1)
                conv.weight_proj = shard_axis(conv.weight_proj, 0)
            self.k_sconv, self.v_sconv = k_dyn, v_dyn

    def _convolve_kv(
        self,
        k: Tensor,
        v: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        conv: LayerConvState,
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the K and V short convolutions, each on its own channels."""
        if self.is_local:
            assert isinstance(self.k_sconv, ShortConvolution)
            assert isinstance(self.v_sconv, ShortConvolution)
            k = self.k_sconv(k, conv.k, input_row_offsets, positions)
            v = self.v_sconv(v, conv.v, input_row_offsets, positions)
        else:
            assert isinstance(self.k_sconv, DynamicShortConvolution)
            assert isinstance(self.v_sconv, DynamicShortConvolution)
            k = self.k_sconv(k, x_k, conv.k, input_row_offsets, positions)
            v = self.v_sconv(v, x_v, conv.v, input_row_offsets, positions)
        mesh = k.mesh
        return as_sharded(k, mesh, axis=1), as_sharded(v, mesh, axis=1)

    def forward(
        self,
        x_q: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        *,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        tau: Tensor,
        conv: LayerConvState,
        cache_layer_idx: Tensor,
    ) -> Tensor:
        """Runs the block over one ragged batch.

        Args:
            x_q: The normalized input of the q, r, gate and sink projections.
            x_k: The normalized input of the K projection and, on a global
                layer, of its dynamic taps. ``x_q`` itself on a local layer.
            x_v: The same for V.
            kv_collection: The KV cache group of this layer's kind.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            positions: Absolute position of each token.
            tau: ``[tokens]`` float32 log-scaling factor.
            conv: This layer's conv rings.
            cache_layer_idx: The layer's index within its cache group. An
                operand, not a constant, so layers can share a subgraph.

        Returns:
            Each device's partial sum of the block's output.
        """
        heads, kv_heads, dim = self.num_heads, self.num_kv_heads, self.head_dim
        text_config = self.text_config
        k, v = self._convolve_kv(
            self.wk_dv(x_k),
            self.wv_dv(x_v),
            x_k,
            x_v,
            conv,
            input_row_offsets,
            positions,
        )
        k = l2_clip(
            k.reshape([-1, kv_heads, dim]), text_config.qk_l2_clip_threshold
        )
        v = unweighted_rms_norm(
            v.cast(DType.float32).reshape([-1, kv_heads, dim]),
            text_config.rms_norm_eps,
        ).cast(v.dtype)
        q, r, gate, sink = self.q_side(x_q)
        q = l2_clip(q, text_config.qk_l2_clip_threshold, self.q_l2_clip.scale)
        r = l2_clip(
            r, text_config.rel_l2_clip_threshold, self.rel_l2_clip.scale
        )
        if not self.is_local:
            q = scale_rows(q, tau)
            r = scale_rows(r, tau)
        bias = r @ self.rel_logits_proj.proj

        store_k_cache_ragged(
            kv_collection, k, input_row_offsets, cache_layer_idx
        )
        store_v_cache_ragged(
            kv_collection, v, input_row_offsets, cache_layer_idx
        )
        attn = flash_attention_ragged(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=cache_layer_idx,
            mask_variant=(
                MHAMaskVariant.CAUSAL_MASK
                if self.window is None
                else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            ),
            scale=dim**-0.5,
            local_window_size=-1 if self.window is None else self.window,
            rel_logits=bias,
            token_sink=sink,
        )
        attn = as_sharded(attn, x_q.mesh, axis=1)
        gate = F.sigmoid(gate.cast(DType.float32))
        gated = attn.cast(DType.float32) * gate
        return self.wo_ud(gated.reshape([-1, heads * dim]).cast(x_q.dtype))
