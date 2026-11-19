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
"""One Inkling 2 decoder layer, on the block attention residual stream."""

from __future__ import annotations

import os
from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Module, TransparentModule, as_subgraph
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import MHAKVCacheParams
from max.tensor import Tensor

from ..model_config import ExpertFormat, Inkling2TextConfig
from .attention import Inkling2Attention
from .attn_res import attn_res_mix, attn_res_mix_norm, can_fuse_mix_norm
from .moe import Inkling2MoE
from .norm import Inkling2RMSNorm
from .placement import as_partial, reduce_partial
from .short_convolution import LayerConvState, ShortConvolution

# The attn_res mixture and the sublayer norm run as one `inkling2_mix_norm`
# launch per query row; set INKLING2_UNFUSED_MIX_NORM=1 for the unfused ops.
_FUSED_MIX_NORM = os.environ.get("INKLING2_UNFUSED_MIX_NORM", "") != "1"


class _AttentionSite(TransparentModule[..., Tensor]):
    """The attention and its output convolution, reduced across devices.

    Name-transparent, so the checkpoint names stay under the layer. Every
    layer of one kind shares one compiled subgraph for it.
    """

    def __init__(self, attn: Inkling2Attention, attn_sconv: ShortConvolution):
        self.attn = attn
        self.attn_sconv = attn_sconv

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
        """Returns the block's output, summed over the devices."""
        attn = self.attn(
            x_q,
            x_k,
            x_v,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            positions=positions,
            tau=tau,
            conv=conv,
            cache_layer_idx=cache_layer_idx,
        )
        # Each rank convolves its full-width partial sum of the branch
        # output. The convolution is linear, so reducing afterwards matches
        # reducing first.
        return reduce_partial(
            as_partial(
                self.attn_sconv(
                    attn, conv.attn_out, input_row_offsets, positions
                ),
                x_q.mesh,
            )
        )


class Inkling2DecoderLayer(Module[..., tuple[Tensor, list[Tensor]]]):
    """Attention then a mixture of experts, each read through attn_res.

    A local layer normalizes one mixture for attention. A global layer has no
    ``attn_norm``: its q, k and v each take their own mixture and norm.
    """

    def __init__(
        self,
        *,
        text_config: Inkling2TextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        num_devices: int,
        device: DeviceRef,
        expert_format: ExpertFormat,
        use_subgraphs: bool,
    ) -> None:
        self.is_local = text_config.is_local_attention(layer_idx)
        self.is_block_boundary = text_config.is_block_boundary(layer_idx)
        self.normalize_values = text_config.attn_res_normalize_values
        # Layers of one kind share a subgraph; the mixtures and the output
        # convolutions stay out because their signatures differ by depth and
        # by layer kind.
        self.attn_subgraph = (
            f"inkling2_{'local' if self.is_local else 'global'}_attention"
            if use_subgraphs
            else None
        )
        self.mlp_subgraph = (
            f"inkling2_{expert_format.value}_moe" if use_subgraphs else None
        )
        hidden = text_config.hidden_size
        eps = text_config.rms_norm_eps
        if self.is_local:
            self.attn_norm = Inkling2RMSNorm(hidden, eps=eps)
        else:
            self.q_input_norm = Inkling2RMSNorm(hidden, eps=eps)
            self.k_input_norm = Inkling2RMSNorm(hidden, eps=eps)
            self.v_input_norm = Inkling2RMSNorm(hidden, eps=eps)
        self.attn_site = _AttentionSite(
            Inkling2Attention(
                text_config=text_config,
                layer_idx=layer_idx,
                kv_params=kv_params,
                num_devices=num_devices,
            ),
            ShortConvolution(
                channels=hidden, kernel_size=text_config.sconv_kernel_size
            ),
        )
        self.mlp_norm = Inkling2RMSNorm(hidden, eps=eps)
        self.mlp = Inkling2MoE(
            text_config=text_config,
            expert_format=expert_format,
            device=device,
            num_devices=num_devices,
        )
        self.mlp_sconv = ShortConvolution(
            channels=hidden, kernel_size=text_config.sconv_kernel_size
        )

    def _attention_inputs(
        self, h: Tensor, blocks: Sequence[Tensor], attn_query: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the normalized q, k and v inputs of the attention."""
        if attn_query is None:
            # Layer 0 reads the embeddings, with no mixture.
            mixes = [h] * (1 if self.is_local else 3)
        elif _FUSED_MIX_NORM and can_fuse_mix_norm(blocks, h):
            norms = (
                [self.attn_norm]
                if self.is_local
                else [self.q_input_norm, self.k_input_norm, self.v_input_norm]
            )
            xs = [
                attn_res_mix_norm(
                    blocks,
                    h,
                    attn_query,
                    norm.weight,
                    query_row=row,
                    norm_eps=norm.eps,
                    normalize_values=self.normalize_values,
                )
                for row, norm in enumerate(norms)
            ]
            if self.is_local:
                return xs[0], xs[0], xs[0]
            return xs[0], xs[1], xs[2]
        else:
            mixes = attn_res_mix(
                [*blocks, h],
                attn_query,
                normalize_values=self.normalize_values,
            )
        if self.is_local:
            x = self.attn_norm(mixes[0])
            return x, x, x
        return (
            self.q_input_norm(mixes[0]),
            self.k_input_norm(mixes[1]),
            self.v_input_norm(mixes[2]),
        )

    def forward(
        self,
        h: Tensor,
        blocks: Sequence[Tensor],
        attn_query: Tensor | None,
        mlp_query: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        tau: Tensor,
        conv: LayerConvState,
        cache_layer_idx: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Runs the layer over one ragged batch.

        Args:
            h: The partial sum of the current block, ``[tokens, hidden]``.
            blocks: The stored sums of the completed blocks.
            attn_query: The attention mixture's queries, ``None`` on layer 0.
            mlp_query: The MLP mixture's query.
            kv_collection: The KV cache group of this layer's kind.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            positions: Absolute position of each token.
            tau: ``[tokens]`` float32 log-scaling factor.
            conv: This layer's conv rings.
            cache_layer_idx: The layer's index within its cache group.

        Returns:
            The new partial sum and the stored blocks. A block boundary adds
            this layer's input to the blocks and restarts the partial sum at
            the attention output.
        """
        x_q, x_k, x_v = self._attention_inputs(h, blocks, attn_query)
        attend = (
            self.attn_site
            if self.attn_subgraph is None
            else as_subgraph(self.attn_site, name=self.attn_subgraph)
        )
        attn = attend(
            x_q,
            x_k,
            x_v,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            positions=positions,
            tau=tau,
            conv=conv,
            cache_layer_idx=cache_layer_idx,
        )
        # The sum stays float32 until the layer output rounds it once.
        residual = attn.cast(DType.float32)
        if self.is_block_boundary:
            blocks = [*blocks, h]
        else:
            residual = h.cast(DType.float32) + residual
        if _FUSED_MIX_NORM and can_fuse_mix_norm(blocks, residual):
            mlp_input = attn_res_mix_norm(
                blocks,
                residual,
                mlp_query,
                self.mlp_norm.weight,
                query_row=0,
                norm_eps=self.mlp_norm.eps,
                normalize_values=self.normalize_values,
            )
        else:
            (mixed,) = attn_res_mix(
                [*blocks, residual],
                mlp_query,
                normalize_values=self.normalize_values,
            )
            mlp_input = self.mlp_norm(mixed)
        mlp = (
            self.mlp
            if self.mlp_subgraph is None
            else as_subgraph(self.mlp, name=self.mlp_subgraph)
        )
        moe = reduce_partial(
            as_partial(
                self.mlp_sconv(
                    mlp(mlp_input),
                    conv.mlp_out,
                    input_row_offsets,
                    positions,
                ),
                h.mesh,
            )
        )
        return (residual + moe.cast(DType.float32)).cast(h.dtype), list(blocks)
