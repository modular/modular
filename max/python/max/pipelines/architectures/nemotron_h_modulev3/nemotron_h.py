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
"""The Nemotron-H hybrid decoder.

The module tree mirrors the checkpoint's (``backbone.layers.{i}.mixer``), so
weights load under their own names.
"""

from __future__ import annotations

from collections.abc import Callable

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module, as_subgraph
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import DeviceMapping, NamedMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import ModuleV3Outputs

from .layers.attention import NemotronHAttention
from .layers.mamba2 import MambaStateAccess, NemotronHMamba2Mixer
from .layers.moe import NemotronHMLP, NemotronHMoE
from .layers.quantized import quantized_linear
from .model_config import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    LayerKind,
    NemotronHConfig,
)


def _distributed_state(
    per_device: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    mapping: DeviceMapping,
) -> RecurrentStateInputsPerDevice[Tensor, Tensor]:
    """Builds one state tensor per leaf from each device's pool.

    Each device's pool holds the state of its own Mamba heads, so the pools
    are sharded on their channel axis. The pool rows of each request are the
    same on every device, so they take the replicated token placement.
    """
    leaves: list[RecurrentLeafInputs[Tensor, Tensor]] = []
    for index in range(len(per_device[0].leaves)):
        pools = [device.leaves[index].pool for device in per_device]
        spec = (None, TP) + (None,) * (pools[0].rank - 2)
        leaves.append(
            RecurrentLeafInputs(
                pool=Tensor.from_shard_values(
                    pools, NamedMapping(mapping.mesh, spec)
                ),
                live_row_ids=Tensor.from_shard_values(
                    [
                        device.leaves[index].live_row_ids
                        for device in per_device
                    ],
                    mapping,
                ),
            )
        )
    return RecurrentStateInputsPerDevice(leaves=tuple(leaves))


class NemotronHBlock(Module[..., Tensor]):
    """A pre-norm residual block around one mixer."""

    def __init__(self, mixer: Module[..., Tensor], config: NemotronHConfig):
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = mixer

    def forward(self, h: Tensor, *mixer_args: object) -> Tensor:
        # Row-parallel mixers leave partial sums; reduce before the residual add.
        mixed = self.mixer(self.norm(h), *mixer_args).to(h.mapping)
        return h + mixed


class NemotronHBackbone(Module[..., Tensor]):
    """Embedding, then the hybrid layer stack. Returns pre-norm hidden states."""

    def __init__(self, config: NemotronHConfig, attn_params: KVCacheParams):
        self.embeddings = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layer_kinds = tuple(config.layer_kinds)
        layers: list[NemotronHBlock] = []
        w4a4_mixers = config.w4a4_mixers()
        for i, kind in enumerate(self.layer_kinds):
            name = f"backbone.layers.{i}.mixer"
            mixer: Module[..., Tensor]
            match kind:
                case LayerKind.MAMBA:
                    mixer = NemotronHMamba2Mixer(config, name)
                case LayerKind.ATTENTION:
                    attn_idx = self.layer_kinds[:i].count(LayerKind.ATTENTION)
                    mixer = NemotronHAttention(config, attn_params, attn_idx)
                case LayerKind.MOE:
                    mixer = NemotronHMoE(
                        config, name, w4a4_experts=name in w4a4_mixers
                    )
                case LayerKind.MLP:
                    mixer = NemotronHMLP(config, name)
            layers.append(NemotronHBlock(mixer, config))
        self.layers = ModuleList(layers)
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor],
        input_row_offsets: Tensor,
    ) -> Tensor:
        h = self.embeddings(tokens)
        query_start_loc = F.cast(input_row_offsets, DType.int32)
        # In the order NemotronHConfig.construct_kv_params declares them.
        conv, ssm = state.leaves
        # Jenga zeroes a request's state rows before its first forward, so
        # every request can resume from its rows.
        batch_size = conv.live_row_ids.shape[1]
        has_initial_state = F.full(
            [batch_size], True, dtype=DType.bool, device=h.mesh
        )
        mamba_idx = 0
        for kind, layer in zip(self.layer_kinds, self.layers, strict=True):
            if kind is LayerKind.ATTENTION:
                h = layer(h, kv_collection, input_row_offsets)
                continue
            # The blocks of each other kind share one subgraph, and each call
            # resolves its own layer's weights. Attention layers bake their
            # KV-cache layer index in as a constant, so they can't share a
            # subgraph.
            call: Callable[..., Tensor] = as_subgraph(layer, name=kind.value)
            if kind is LayerKind.MAMBA:
                # Every layer's rows go into the shared subgraph with the
                # layer index, and the layer slices its own rows there.
                access = MambaStateAccess(
                    conv_pool=conv.pool,
                    conv_rows=conv.live_row_ids,
                    ssm_pool=ssm.pool,
                    ssm_rows=ssm.live_row_ids,
                    layer=F.constant(mamba_idx, DType.int64, device=CPU()),
                )
                h = call(h, access, query_start_loc, has_initial_state)
                mamba_idx += 1
            else:
                h = call(h)
        return h


class NemotronH(Module[..., ModuleV3Outputs]):
    """Nemotron-H for causal language modeling."""

    def __init__(self, config: NemotronHConfig) -> None:
        if config.return_logits == ReturnLogits.VARIABLE:
            raise NotImplementedError(
                "Nemotron-H does not return a variable number of logits, "
                "which speculative decoding needs"
            )
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        attn_params = config.kv_params.child(ATTN_CACHE_KEY, MHAKVCacheParams)
        self.backbone = NemotronHBackbone(config, attn_params)
        self.lm_head = quantized_linear(config, "lm_head")

    def _logits(self, h: Tensor) -> Tensor:
        return F.cast(self.lm_head(self.backbone.norm_f(h)), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        del return_n_logits
        mesh = self.backbone.norm_f.weight.mesh
        row_offsets = input_row_offsets
        if mesh.num_devices > 1:
            # Device graph capture records each device's stream on its own,
            # and a peer copy makes one stream wait on another, which
            # invalidates the capture. The collective syncs on the devices.
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
        else:
            tokens = tokens.to(mesh)
            input_row_offsets = input_row_offsets.to(mesh)
        kv_tree = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in kv_inputs)
        )
        kv_collection = PagedCacheValues.from_upstream(
            tree.leaves(kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice),
            tokens.mapping,
        )
        state = _distributed_state(
            tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            ),
            tokens.mapping,
        )
        h = self.backbone(tokens, kv_collection, state, input_row_offsets)

        last = self._logits(F.gather(h, input_row_offsets[1:] - 1, axis=0))
        # The head is replicated, so device 0 holds the full vocabulary.
        # The sampler reads that one buffer.
        if self.return_logits == ReturnLogits.ALL:
            return ModuleV3Outputs(
                next_token_logits=last.local_shards[0],
                logits=self._logits(h).local_shards[0],
                logit_offsets=row_offsets,
            )
        return ModuleV3Outputs(next_token_logits=last.local_shards[0])
