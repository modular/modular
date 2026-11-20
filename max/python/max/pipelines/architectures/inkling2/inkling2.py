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
"""The Inkling 2 text decoder, assembled."""

from __future__ import annotations

from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import DimLike, TensorValue
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues, kv_input_values
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import TP
from max.nn.kv_cache import (
    CacheLeafKind,
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.outputs import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib.role_contracts import DecoderOutputs, VisionInputs
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)

from .layers.attn_res import AttnResQueries, attn_res_mix
from .layers.decoder_layer import Inkling2DecoderLayer
from .layers.norm import Inkling2RMSNorm
from .layers.placement import as_replicated
from .layers.short_convolution import LayerConvState, SiteState
from .model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    STATE_CACHE_KEY,
    Inkling2Config,
)
from .state_cache import ConvSite, ConvStateLayout

_merge_multimodal_embeddings = F.functional(merge_multimodal_embeddings)


def token_positions(
    input_row_offsets: Tensor, cache_lengths: Tensor, total_tokens: DimLike
) -> Tensor:
    """Returns each ragged token's absolute position as uint32.

    That is its index in its sequence plus the tokens the sequence already
    has cached.
    """
    offsets = input_row_offsets.cast(DType.int64)
    starts = offsets[:-1]
    shift = F.rebind(cache_lengths.cast(DType.int64), starts.shape) - starts
    positions = F.range(
        0,
        F.shape_to_tensor([total_tokens])[0],
        1,
        out_dim=total_tokens,
        dtype=DType.int64,
        device=offsets.mapping,
    ) + F.repeat_interleave(
        shift, offsets[1:] - starts, axis=0, out_dim=total_tokens
    )
    return positions.cast(DType.uint32)


def log_scaling_tau(positions: Tensor, *, alpha: float, n_floor: int) -> Tensor:
    """``tau = 1 + alpha * log(max(1, (position + 1) / n_floor))``, float32."""
    ratio = (positions.cast(DType.float32) + 1.0) / float(n_floor)
    return 1.0 + alpha * F.log(F.elementwise_max(ratio, 1.0))


class Inkling2(Module[..., DecoderOutputs]):
    """The Inkling 2 text model: embedding, decoder layers, LM head."""

    def __init__(self, config: Inkling2Config) -> None:
        text_config = config.text_config
        self.text_config = text_config
        self.num_devices = len(config.devices)
        self.return_logits = config.outputs.return_logits
        # Only used to address leaves and rows; the kernels take the ring
        # length from the pool's shape.
        self.conv_layout = ConvStateLayout.from_config(
            text_config, tp_size=self.num_devices
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params: dict[str, MHAKVCacheParams] = {}
        for key, params in config.kv_params.children.items():
            if params.leaf_kind is not CacheLeafKind.ATTENTION:
                continue
            assert isinstance(params, MHAKVCacheParams)
            kv_params[key] = params
        self.kv_params = config.kv_params

        # The embedding and the LM head are 2.4 GB each, so TP splits them.
        self.embed = VocabParallelEmbedding(
            text_config.vocab_size, dim=text_config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.attn_res = AttnResQueries(
            hidden_size=text_config.hidden_size,
            num_layers=text_config.num_hidden_layers,
            is_local=self.conv_layout.is_local,
        )

        layers = []
        self.layer_kv_keys: list[str] = []
        self.layer_cache_indices: list[int] = []
        layers_in_cache = {GLOBAL_ATTENTION: 0, LOCAL_ATTENTION: 0}
        for layer_idx in range(text_config.num_hidden_layers):
            kv_key = (
                LOCAL_ATTENTION
                if text_config.is_local_attention(layer_idx)
                else GLOBAL_ATTENTION
            )
            layers.append(
                Inkling2DecoderLayer(
                    text_config=text_config,
                    layer_idx=layer_idx,
                    kv_params=kv_params[kv_key],
                    num_devices=self.num_devices,
                    device=config.devices[0],
                    expert_format=config.expert_formats[layer_idx],
                    use_subgraphs=config.use_subgraphs,
                )
            )
            self.layer_kv_keys.append(kv_key)
            self.layer_cache_indices.append(layers_in_cache[kv_key])
            layers_in_cache[kv_key] += 1
        self.layers = ModuleList(layers)

        self.norm = Inkling2RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.unembed = ColumnParallelLinear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self._padded_tail_mask: np.ndarray | None = None
        if text_config.unpadded_vocab_size < text_config.vocab_size:
            self._padded_tail_mask = np.zeros(
                text_config.vocab_size, dtype=np.float32
            )
            self._padded_tail_mask[text_config.unpadded_vocab_size :] = -np.inf

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        vision: VisionInputs[Tensor] | None,
        kv: Any,
    ) -> DecoderOutputs:
        """Runs the whole text model over one ragged batch.

        ``vision`` carries the image embeddings and the token rows they
        replace, when a vision tower feeds the decoder.
        """
        text_config = self.text_config
        mesh = self.norm.weight.mesh
        first_row_offsets = input_row_offsets
        tokens, input_row_offsets = (
            self._broadcast(t, mesh) for t in (tokens, input_row_offsets)
        )
        replicated = tokens.mapping

        kv_tree = self.kv_params.unflatten_kv_inputs(iter(kv_input_values(kv)))
        kv_collections = {
            key: PagedCacheValues.from_upstream(
                tree.leaves(child, leaf=KVCacheInputsPerDevice), replicated
            )
            for key, child in kv_tree.items()
            if key != STATE_CACHE_KEY
        }
        positions = token_positions(
            input_row_offsets,
            next(iter(kv_collections.values())).cache_lengths,
            tokens.shape[0],
        )
        pools, slot_rows = self._conv_state(
            tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            ),
            mesh,
            replicated,
        )
        tau = as_replicated(
            log_scaling_tau(
                positions,
                alpha=text_config.log_scaling_alpha,
                n_floor=text_config.log_scaling_n_floor,
            ),
            mesh,
        )

        h = self.embed_norm(self.embed(tokens))
        if vision is not None:
            # Merged after the embedding norm, so the tower's rows skip it.
            h = as_replicated(
                _merge_multimodal_embeddings(
                    h,
                    self._broadcast(vision.embeddings, mesh),
                    self._broadcast(vision.scatter_indices, mesh),
                ),
                mesh,
            )
        blocks: list[Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            sites = [
                self.conv_layout.row_for(layer_idx, site) for site in ConvSite
            ]
            conv = LayerConvState(
                *(
                    SiteState(
                        ring=pools[leaf],
                        rows=slot_rows[leaf],
                        layer_row=F.constant(row, DType.uint32, device=CPU()),
                    )
                    for leaf, row in sites
                )
            )
            h, blocks = layer(
                h,
                blocks,
                self.attn_res.attn_queries[layer_idx - 1]
                if layer_idx > 0
                else None,
                self.attn_res.mlp_queries[layer_idx],
                kv_collections[self.layer_kv_keys[layer_idx]],
                input_row_offsets,
                positions,
                tau,
                conv,
                F.constant(
                    self.layer_cache_indices[layer_idx],
                    DType.uint32,
                    device=CPU(),
                ),
            )
        return self._logits(
            blocks, h, input_row_offsets, first_row_offsets, return_n_logits
        )

    @staticmethod
    def _broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
        """Replicates a single-device input across the mesh."""
        if mesh.num_devices > 1:
            # Device graph capture records each device's stream on its own,
            # and a peer copy makes one stream wait on another, which
            # invalidates the capture. The collective syncs on the devices.
            return F.distributed_broadcast(t, mesh)
        return t.to(mesh)

    def _conv_state(
        self,
        per_device: list[RecurrentStateInputsPerDevice[TensorValue, Any]],
        mesh: DeviceMesh,
        replicated: DeviceMapping,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Builds one pool tensor and one slot table per conv-state leaf.

        The K and V pools hold each device's channels. The residual pools are
        full width on every device and hold that device's partial sums. The
        slot tables are the same on every device.
        """
        k_and_v = {ConvSite.K, ConvSite.V}
        pools = []
        rows = []
        for index in range(len(per_device[0].leaves)):
            site = ConvSite(index % len(ConvSite))
            mapping = (
                NamedMapping(mesh, (None, None, TP))
                if site in k_and_v
                else DeviceMapping(mesh, (Partial(),))
            )
            if mesh.num_devices == 1:
                mapping = replicated
            pools.append(
                Tensor.from_shard_values(
                    [device.leaves[index].pool for device in per_device],
                    mapping,
                )
            )
            rows.append(
                Tensor.from_shard_values(
                    [
                        device.leaves[index].live_row_ids
                        for device in per_device
                    ],
                    replicated,
                )
            )
        return pools, rows

    def _lm_head(self, blocks: list[Tensor], h: Tensor) -> Tensor:
        """Mixes the blocks, then returns softcapped float32 logits."""
        (mixed,) = attn_res_mix(
            [*blocks, h],
            self.attn_res.output_query,
            normalize_values=self.text_config.attn_res_normalize_values,
        )
        logits = self.unembed(self.norm(mixed))
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        cap = self.text_config.final_logit_softcapping
        logits = cap * F.tanh(logits.cast(DType.float32) / cap)
        if self._padded_tail_mask is not None:
            # Sends the untrained tail of the vocabulary to negative infinity
            # after the softcap, which would pull it to -cap. A materialized
            # row: concatenated stride-zero broadcasts fault on B200.
            logits = logits + F.constant(
                self._padded_tail_mask, DType.float32, device=logits.device
            )
        return logits

    def _logits(
        self,
        blocks: list[Tensor],
        h: Tensor,
        input_row_offsets: Tensor,
        first_row_offsets: Tensor,
        return_n_logits: Tensor,
    ) -> DecoderOutputs:
        """Computes the logits the pipeline asked for.

        The mixture is local to each token, so it runs only on the rows that
        get logits.
        """

        def rows_of(indices: Tensor) -> tuple[list[Tensor], Tensor]:
            return [F.gather(b, indices, axis=0) for b in blocks], F.gather(
                h, indices, axis=0
            )

        last = input_row_offsets[1:] - 1
        last_logits = self._lm_head(*rows_of(last))
        logits: Tensor | None = None
        offsets: Tensor | None = None
        if self.return_logits == ReturnLogits.VARIABLE:
            count = return_n_logits[0]
            distance = F.range(
                count,
                0,
                -1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=h.mesh,
            )
            indices = (input_row_offsets[1:].unsqueeze(-1) - distance).reshape(
                (-1,)
            )
            logits = self._lm_head(*rows_of(indices))
            offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=h.mesh,
            ).local_shards[0]
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._lm_head(blocks, h)
            offsets = first_row_offsets
        return DecoderOutputs(
            next_token_logits=last_logits,
            logits=logits,
            logit_offsets=offsets,
        )
