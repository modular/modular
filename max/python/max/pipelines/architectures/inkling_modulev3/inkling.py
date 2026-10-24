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
"""The Inkling text decoder, assembled."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module, as_subgraph
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)
from max.experimental.tensor import Tensor
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn.kv_cache import (
    CacheLeafKind,
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from ..inkling.inkling import _subgraph_layer_groups
from ..inkling.layers.attention import log_scaling_tau
from ..inkling.model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    STATE_CACHE_KEY,
    InklingConfig,
    InklingTextConfig,
)
from ..inkling.state_cache import ConvSite, InklingConvStateLayout
from .layers.attention import InklingAttention
from .layers.moe import InklingDenseMLP, InklingMoE
from .layers.placement import as_partial, as_replicated, reduce_partial
from .layers.short_convolution import InklingRMSNorm, ShortConvolution

_log_scaling_tau = F.functional(log_scaling_tau)
_merge_multimodal_embeddings = F.functional(merge_multimodal_embeddings)


class _ConvSites:
    """One layer's conv pools, slot tables and rows, in :class:`ConvSite`
    order.

    The kernels index the whole ``[num_layers, batch]`` slot table, so no
    per-layer slice is materialized.
    """

    def __init__(
        self, pools: list[Tensor], rows: list[Tensor], layer_rows: list[Tensor]
    ) -> None:
        self.pools = pools
        self.rows = rows
        self.layer_rows = layer_rows

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {
            "pools": self.pools,
            "rows": self.rows,
            "layer_rows": self.layer_rows,
        }, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: dict[str, Any]
    ) -> _ConvSites:
        return cls(children["pools"], children["rows"], children["layer_rows"])


class InklingDecoderLayer(Module[..., tuple[Tensor, Tensor]]):
    """One decoder layer: attention and feed-forward, each with an output
    conv."""

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        num_devices: int,
        quant_config: QuantConfig | None = None,
    ) -> None:
        hidden = text_config.hidden_size
        self.attn_norm = InklingRMSNorm(hidden, eps=text_config.rms_norm_eps)
        self.attn = InklingAttention(
            text_config=text_config,
            layer_idx=layer_idx,
            kv_params=kv_params,
            num_devices=num_devices,
        )
        # Replicated: each rank convolves its full-width partial sum of the
        # branch output. The convolution's linearity makes reducing
        # afterwards equivalent to reducing first.
        self.attn_sconv = ShortConvolution(
            channels=hidden, kernel_size=text_config.sconv_kernel_size
        )
        self.mlp_norm = InklingRMSNorm(hidden, eps=text_config.rms_norm_eps)
        self.mlp: InklingMoE | InklingDenseMLP
        if text_config.is_moe_layer(layer_idx):
            # Only routed experts quantize; sinks and gate stay at dtype.
            routed_quant = (
                quant_config
                if quant_config is not None
                and layer_idx in quant_config.mlp_quantized_layers
                else None
            )
            self.mlp = InklingMoE(
                hidden_dim=hidden,
                num_experts=text_config.n_routed_experts,
                num_experts_per_token=text_config.num_experts_per_tok,
                moe_dim=text_config.intermediate_size,
                n_shared_experts=text_config.n_shared_experts,
                route_scale=text_config.route_scale,
                num_devices=num_devices,
                quant_config=routed_quant,
            )
        else:
            self.mlp = InklingDenseMLP(
                hidden, text_config.dense_intermediate_size
            )
        self.mlp_sconv = ShortConvolution(
            channels=hidden, kernel_size=text_config.sconv_kernel_size
        )

    def forward(
        self,
        h: Tensor,
        delta_in: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        conv: _ConvSites,
        cache_layer_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the layer over one ragged batch.

        Takes the residual stream and the previous layer's unreduced
        mlp-branch delta, and returns the same pair. Reducing the
        predecessor's delta here, rather than at the end of the layer that
        produced it, keeps the collective, the residual add and the norm in
        one subgraph, which is where the graph compiler can fuse them. Layer
        0 gets a zero delta so every layer shares one signature; the stack
        reduces the last layer's delta.
        """
        mesh = h.mesh
        h = h + reduce_partial(delta_in)
        attention = self.attn(
            self.attn_norm(h),
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            positions=positions,
            log_scaling=log_scaling,
            k_conv_ring=conv.pools[ConvSite.K],
            k_conv_rows=conv.rows[ConvSite.K],
            k_layer_row=conv.layer_rows[ConvSite.K],
            v_conv_ring=conv.pools[ConvSite.V],
            v_conv_rows=conv.rows[ConvSite.V],
            v_layer_row=conv.layer_rows[ConvSite.V],
            cache_layer_idx=cache_layer_idx,
        )
        attn_delta = self._convolve(
            self.attn_sconv,
            attention,
            conv,
            ConvSite.ATTN_OUT,
            input_row_offsets,
            positions,
            mesh,
        )
        h = h + reduce_partial(attn_delta)
        mlp_delta = self._convolve(
            self.mlp_sconv,
            self.mlp(self.mlp_norm(h)),
            conv,
            ConvSite.MLP_OUT,
            input_row_offsets,
            positions,
            mesh,
        )
        return h, mlp_delta

    @staticmethod
    def _convolve(
        sconv: ShortConvolution,
        partial: Tensor,
        conv: _ConvSites,
        site: ConvSite,
        input_row_offsets: Tensor,
        positions: Tensor,
        mesh: DeviceMesh,
    ) -> Tensor:
        """Convolves each rank's full-width partial sum of a branch's output."""
        return as_partial(
            sconv(
                partial,
                conv.pools[site],
                conv.rows[site],
                conv.layer_rows[site],
                input_row_offsets,
                positions,
            ),
            mesh,
        )


class Inkling(Module[..., ModuleV3Outputs]):
    """The Inkling text model: embedding, decoder layers, LM head."""

    def __init__(
        self,
        config: InklingConfig,
        *,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        if return_hidden_states != ReturnHiddenStates.NONE:
            raise NotImplementedError(
                "Inkling ModuleV3 does not return hidden states"
            )
        text_config = config.text_config
        self.text_config = text_config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.dtype = config.dtype
        self.return_logits = return_logits

        # Only used to address leaves and rows; the kernels take the ring
        # length from the pool's shape.
        self.conv_layout = InklingConvStateLayout.from_config(
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
        self.embed_norm = (
            InklingRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
            if text_config.use_embed_norm
            else None
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
                InklingDecoderLayer(
                    text_config=text_config,
                    layer_idx=layer_idx,
                    kv_params=kv_params[kv_key],
                    num_devices=self.num_devices,
                    quant_config=config.quant_config,
                )
            )
            self.layer_kv_keys.append(kv_key)
            self.layer_cache_indices.append(layers_in_cache[kv_key])
            layers_in_cache[kv_key] += 1
        self.layers = ModuleList(layers)

        self.subgraph_names: dict[int, str] = {}
        if config.use_subgraphs:
            names, groups = _subgraph_layer_groups(
                text_config, config.quant_config
            )
            for name, group in zip(names, groups, strict=True):
                for layer_idx in group:
                    self.subgraph_names[layer_idx] = name

        self._padded_tail_mask: np.ndarray | None = None
        if text_config.unpadded_vocab_size < text_config.vocab_size:
            self._padded_tail_mask = np.zeros(
                text_config.vocab_size, dtype=np.float32
            )
            self._padded_tail_mask[text_config.unpadded_vocab_size :] = -np.inf

        self.norm = InklingRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.unembed = ColumnParallelLinear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )

    def forward(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        positions: Tensor,
        return_n_logits: Tensor,
        image_embeddings: Tensor,
        image_indices: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the whole text model over one ragged batch."""
        mesh = self.norm.weight.mesh
        first_row_offsets = input_row_offsets
        (
            tokens,
            input_row_offsets,
            positions,
            image_embeddings,
            image_indices,
        ) = (
            self._broadcast(t, mesh)
            for t in (
                tokens,
                input_row_offsets,
                positions,
                image_embeddings,
                image_indices,
            )
        )
        replicated = tokens.mapping

        kv_tree = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in kv_inputs)
        )
        kv_collections = {
            key: PagedCacheValues.from_upstream(
                tree.leaves(child, leaf=KVCacheInputsPerDevice), replicated
            )
            for key, child in kv_tree.items()
            if key != STATE_CACHE_KEY
        }
        assert STATE_CACHE_KEY in kv_tree, (
            "Inkling always convolves; the cache declared no conv-state child"
        )
        pools, slot_rows = self._conv_state(
            tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            ),
            mesh,
            replicated,
        )

        h = self.embed(tokens)
        # Every rank needs the positions for its conv rings.
        log_scaling = as_replicated(
            _log_scaling_tau(
                positions,
                alpha=self.text_config.log_scaling_alpha,
                n_floor=self.text_config.log_scaling_n_floor,
            ),
            mesh,
        )
        if self.embed_norm is not None:
            h = self.embed_norm(h)
        # Merged after the embedding norm, so the tower's rows skip it.
        h = as_replicated(
            _merge_multimodal_embeddings(h, image_embeddings, image_indices),
            mesh,
        )

        delta = as_partial(F.zeros_like(h), mesh)
        for layer_idx, layer in enumerate(self.layers):
            sites = [
                self.conv_layout.row_for(layer_idx, site) for site in ConvSite
            ]
            conv = _ConvSites(
                pools=[pools[leaf] for leaf, _ in sites],
                rows=[slot_rows[leaf] for leaf, _ in sites],
                layer_rows=[
                    F.constant(row, DType.uint32, device=CPU())
                    for _, row in sites
                ],
            )
            call: Callable[..., tuple[Tensor, Tensor]] = layer
            if layer_idx in self.subgraph_names:
                call = as_subgraph(layer, name=self.subgraph_names[layer_idx])
            h, delta = call(
                h,
                delta,
                kv_collections[self.layer_kv_keys[layer_idx]],
                input_row_offsets,
                positions,
                log_scaling,
                conv,
                F.constant(
                    self.layer_cache_indices[layer_idx],
                    DType.uint32,
                    device=CPU(),
                ),
            )
        # Only the last layer's delta is still unreduced.
        h = h + reduce_partial(delta)
        return self._logits(
            h, input_row_offsets, first_row_offsets, return_n_logits
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

    def _lm_head(self, h: Tensor) -> Tensor:
        """Projects normed rows to float32 logits on the first device."""
        logits = self.unembed(self.norm(h))
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        if self._padded_tail_mask is not None:
            # Sends the untrained tail of the vocabulary to negative
            # infinity. Materialized row: concatenated stride-zero broadcasts
            # fault on B200.
            logits = logits + F.constant(
                self._padded_tail_mask, DType.float32, device=logits.device
            ).cast(logits.dtype)
        logits = logits.cast(DType.float32)
        scaling = self.text_config.logits_mup_width_multiplier
        if scaling != 1.0:
            logits = logits / scaling
        return logits

    def _logits(
        self,
        h: Tensor,
        input_row_offsets: Tensor,
        first_row_offsets: Tensor,
        return_n_logits: Tensor,
    ) -> ModuleV3Outputs:
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._lm_head(last_h)
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
            logits = self._lm_head(F.gather(h, indices, axis=0))
            offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=h.mesh,
            ).local_shards[0]
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._lm_head(h)
            offsets = first_row_offsets
        return ModuleV3Outputs(
            next_token_logits=last_logits,
            logits=logits,
            logit_offsets=offsets,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Graph input types, in the order :meth:`forward` consumes them."""
        device = self.devices[0]
        return (
            TensorType(DType.int64, shape=["total_seq_len"], device=device),
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            ),
            TensorType(DType.uint32, shape=["total_seq_len"], device=device),
            TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
            TensorType(
                self.dtype,
                shape=["total_image_tokens", self.text_config.hidden_size],
                device=device,
            ),
            TensorType(
                DType.int32, shape=["total_image_tokens"], device=device
            ),
            # The conv state is a child of the cache tree, so one flatten
            # covers the attention pages and the state leaves both.
            *self.kv_params.flattened_kv_inputs(),
        )
