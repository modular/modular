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
"""The Inkling2 text decoder: embedding, residual banks, layers, LM head."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn import Module, as_subgraph
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import ColumnParallelLinear
from max.nn.common_layers.mesh_axis import TP
from max.nn.kv_cache import (
    CacheLeafKind,
    KVCacheInputsPerDevice,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)

from .layers.attention import InklingAttention
from .layers.moe import Inkling2MoE
from .layers.placement import as_partial, as_replicated, reduce_partial
from .layers.short_convolution import InklingRMSNorm, ShortConvolution
from .model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    STATE_CACHE_KEY,
    ExpertFormat,
    InklingConfig,
    InklingTextConfig,
)
from .state_cache import ConvSite, InklingConvStateLayout

_FP32 = DType.float32

_ATTN_RES_EPS = 1e-6
"""Variance epsilon of the block residual's parameter-free RMSNorm."""


def log_scaling_tau(
    positions: TensorValue, *, alpha: float, n_floor: int
) -> TensorValue:
    """Per-token log-scaling factor ``tau`` for queries and relative states."""
    ratio = (ops.cast(positions, _FP32) + 1.0) / float(n_floor)
    return 1.0 + alpha * ops.log(ops.max(ratio, 1.0))


_log_scaling_tau = F.functional(log_scaling_tau)
_merge_multimodal_embeddings = F.functional(merge_multimodal_embeddings)


def _tanh(x: Tensor) -> Tensor:
    return F.sigmoid(x + x) * 2.0 - 1.0


def _query_shape(num_queries: int, hidden: int) -> list[int]:
    return [hidden] if num_queries == 1 else [num_queries, hidden]


def _attn_res_read(
    query: Tensor, stack: Tensor, residual: Tensor, hidden: int
) -> tuple[Tensor, ...]:
    num_queries = int(query.shape[0]) if query.rank == 2 else 1
    stacked = F.concat([stack, F.unsqueeze(residual, -2)], axis=-2)
    keys = F.cast(stacked, _FP32)
    keys = keys * F.rsqrt(F.mean(keys * keys, axis=-1) + _ATTN_RES_EPS)
    queries = F.reshape(F.cast(query, _FP32), [1, num_queries, hidden])
    logits = F.matmul(queries, F.transpose(keys, -1, -2))
    probs = F.softmax(logits * hidden**-0.5, axis=-1)
    out = F.cast(F.matmul(probs, keys), residual.dtype)
    return tuple(
        F.squeeze(part, axis=1)
        for part in F.split(out, [1] * num_queries, axis=1)
    )


def _bank_length(text_config: InklingTextConfig, layer_idx: int) -> int:
    return sum(
        1
        for index in range(layer_idx)
        if text_config.is_attn_res_block_boundary(index)
    )


def _subgraph_layer_groups(
    text_config: InklingTextConfig, quant_config: QuantConfig | None
) -> tuple[list[str], list[list[int]]]:
    groups: dict[tuple[bool, bool, int, bool, int], list[int]] = {}
    for layer_idx in range(text_config.num_hidden_layers):
        key = (
            text_config.is_local_attention(layer_idx),
            quant_config is not None
            and layer_idx in quant_config.mlp_quantized_layers,
            text_config.attention_input_queries(layer_idx),
            text_config.is_attn_res_block_boundary(layer_idx),
            _bank_length(text_config, layer_idx),
        )
        groups.setdefault(key, []).append(layer_idx)

    names = []
    shared = []
    for geometry, layer_indices in groups.items():
        local, quantized, queries, boundary, states = geometry
        if len(layer_indices) == 1:
            continue
        flavor = "local" if local else "global"
        mlp = "nvfp4_moe" if quantized else "moe"
        names.append(
            f"inkling_{flavor}_{mlp}_q{queries}_n{states}"
            f"_b{int(boundary)}_block"
        )
        shared.append(layer_indices)
    return names, shared


class _AttentionResidualBanks(Module[[], None]):
    def __init__(self, *, text_config: InklingTextConfig) -> None:
        hidden = text_config.hidden_size
        num_layers = text_config.num_hidden_layers
        # Layer 0 has no attention input (:760, :816): the attention banks are
        # offset by one, the feed-forward ones are not. Entries are tensors.
        self.attn_queries: ModuleList[Any] = ModuleList[Any](
            [
                Tensor.zeros(
                    _query_shape(
                        text_config.attention_input_queries(layer_idx + 1),
                        hidden,
                    ),
                    dtype=_FP32,
                )
                for layer_idx in range(num_layers - 1)
            ]
        )
        self.mlp_queries: ModuleList[Any] = ModuleList[Any](
            [
                Tensor.zeros(_query_shape(1, hidden), dtype=_FP32)
                for _ in range(num_layers)
            ]
        )
        self.output_query = Tensor.zeros([hidden], dtype=_FP32)


class _ConvSites:
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


class InklingDecoderLayer(Module[..., tuple[Tensor, Tensor, Tensor]]):
    """One decoder layer: attention, feed-forward, output convs, residual bank."""

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: KVCacheParams,
        expert_format: ExpertFormat,
        device: DeviceRef,
        num_devices: int,
        max_seq_len: int,
        max_query_rows: int,
    ) -> None:
        hidden = text_config.hidden_size
        eps = text_config.rms_norm_eps
        self.hidden = hidden
        self.layer_idx = layer_idx
        self.is_local = text_config.is_local_attention(layer_idx)
        self.attention_queries = text_config.attention_input_queries(layer_idx)
        self.is_block_boundary = text_config.is_attn_res_block_boundary(
            layer_idx
        )

        if self.is_local:
            self.attn_norm = InklingRMSNorm(hidden, eps=eps)
        else:
            self.q_input_norm = InklingRMSNorm(hidden, eps=eps)
            self.k_input_norm = InklingRMSNorm(hidden, eps=eps)
            self.v_input_norm = InklingRMSNorm(hidden, eps=eps)

        self.attn = InklingAttention(
            text_config=text_config,
            layer_idx=layer_idx,
            kv_params=kv_params,
            num_devices=num_devices,
            max_seq_len=max_seq_len,
            max_query_rows=max_query_rows,
        )
        # Replicated: each rank convolves its full-width partial sum of the
        # branch output. The convolution's linearity makes reducing
        # afterwards equivalent to reducing first.
        self.attn_sconv = ShortConvolution(
            channels=hidden, kernel_size=text_config.sconv_kernel_size
        )
        self.mlp_norm = InklingRMSNorm(hidden, eps=eps)
        # Every Inkling2 layer routes to experts; there is no dense prefix
        # (:719-730). The experts' storage format is per layer, detected
        # from the checkpoint (config.expert_formats).
        self.mlp = Inkling2MoE(
            text_config=text_config,
            expert_format=expert_format,
            device=device,
            num_devices=num_devices,
        )
        self.mlp_sconv = ShortConvolution(
            channels=hidden, kernel_size=text_config.sconv_kernel_size
        )

    def forward(
        self,
        h: Tensor,
        delta_in: Tensor,
        block_states: Tensor | None,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        conv: _ConvSites,
        cache_layer_idx: Tensor,
        cache_row: int,
        attn_query: Tensor | None,
        mlp_query: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the layer over one ragged batch."""
        mesh = h.mesh
        hidden = self.hidden
        h = h + reduce_partial(delta_in)

        attention = self._attention(
            h,
            block_states,
            attn_query,
            kv_collection,
            input_row_offsets,
            positions,
            log_scaling,
            conv,
            cache_layer_idx,
            cache_row,
            hidden,
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
        residual = reduce_partial(attn_delta)
        if self.is_block_boundary:
            # The layer's input goes into the bank instead of into the stream
            # (:799-800, :863-864), and the bank grows before the
            # feed-forward read.
            entry = F.unsqueeze(h, -2)
            stack = (
                entry
                if block_states is None
                else F.concat([block_states, entry], axis=-2)
            )
        else:
            if block_states is None:
                raise ValueError(
                    f"layer {self.layer_idx} reads the block residual but no "
                    "bank was given; only layer 0 may run without one"
                )
            stack = block_states
            residual = h + residual

        mlp_in = _attn_res_read(mlp_query, stack, residual, hidden)[0]
        mlp_delta = self._convolve(
            self.mlp_sconv,
            self.mlp(self.mlp_norm(mlp_in)),
            conv,
            ConvSite.MLP_OUT,
            input_row_offsets,
            positions,
            mesh,
        )
        return residual, mlp_delta, stack

    def _attention(
        self,
        h: Tensor,
        block_states: Tensor | None,
        attn_query: Tensor | None,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        conv: _ConvSites,
        cache_layer_idx: Tensor,
        cache_row: int,
        hidden: int,
    ) -> Tensor:
        if self.attention_queries == 0:
            # Layer 0 has no bank, so it projects every side from its own
            # state (:760, :816, and :843 for a full first layer).
            states: tuple[Tensor, ...] = (h,) if self.is_local else (h, h, h)
        else:
            if attn_query is None or block_states is None:
                raise ValueError(
                    f"layer {self.layer_idx} reads {self.attention_queries} "
                    "block-residual queries but was given no bank"
                )
            states = _attn_res_read(attn_query, block_states, h, hidden)

        if self.is_local:
            q_in = k_in = v_in = states[0]
            x = self.attn_norm(q_in)
        else:
            q_in, k_in, v_in = states
            x = self.q_input_norm(q_in)
            k_in = self.k_input_norm(k_in)
            v_in = self.v_input_norm(v_in)
        return self.attn(
            x,
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
            cache_row=cache_row,
            key_hidden_states=k_in,
            value_hidden_states=v_in,
        )

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

        # Addresses leaves and rows only; the kernels take the ring
        # length from the pool's shape.
        self.conv_layout = InklingConvStateLayout.from_config(
            text_config, tp_size=self.num_devices
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params: dict[str, KVCacheParams] = {}
        for key, params in config.kv_params.children.items():
            # Select on ``leaf_kind``: the base ``KVCacheParams`` reaches here and
            # carries everything the layer reads.
            if params.leaf_kind is not CacheLeafKind.ATTENTION:
                continue
            assert isinstance(params, KVCacheParams)
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
        self.attn_res = _AttentionResidualBanks(text_config=text_config)

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
                    expert_format=config.expert_formats[layer_idx],
                    device=self.devices[0],
                    num_devices=self.num_devices,
                    max_seq_len=config.max_seq_len,
                    max_query_rows=getattr(config, "max_batch_tokens", 512),
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
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        positions: Tensor,
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
        block_states: Tensor | None = None
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
            bank = self.text_config.attention_input_bank(layer_idx)
            call: Callable[..., tuple[Tensor, Tensor, Tensor]] = layer
            if layer_idx in self.subgraph_names:
                call = as_subgraph(layer, name=self.subgraph_names[layer_idx])
            h, delta, block_states = call(
                h,
                delta,
                block_states,
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
                self.layer_cache_indices[layer_idx],
                None if bank is None else self.attn_res.attn_queries[bank],
                self.attn_res.mlp_queries[layer_idx],
            )
        if block_states is None:
            raise ValueError(
                "the decoder produced no block residual bank; every stack "
                "starts with a boundary layer (layer 0 is one by construction)"
            )
        # Only the last layer's delta is still unreduced.
        h = h + reduce_partial(delta)
        # The final read the reference calls `final_input` (:1249, :1311).
        h = _attn_res_read(
            self.attn_res.output_query,
            block_states,
            h,
            self.text_config.hidden_size,
        )[0]
        return self._logits(
            h, input_row_offsets, first_row_offsets, return_n_logits
        )

    @staticmethod
    def _broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
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
        logits = self.unembed(self.norm(h))
        if logits.is_distributed:
            logits = F.allgather(logits, tensor_axis=-1).local_shards[0]
        logits = F.cast(logits, _FP32)
        cap = self.text_config.final_logit_softcapping
        logits = _tanh(logits / cap) * cap
        if self._padded_tail_mask is not None:
            # Sends the untrained tail of the vocabulary to negative
            # infinity. Materialized row: concatenated stride-zero broadcasts
            # fault on B200.
            logits = logits + F.constant(
                self._padded_tail_mask, _FP32, device=logits.device
            )
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
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            ),
            TensorType(DType.uint32, shape=["total_seq_len"], device=device),
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
