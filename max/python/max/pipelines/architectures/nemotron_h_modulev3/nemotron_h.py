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
from typing import Any, cast

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
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import ModuleV3Outputs

from .layers.attention import NemotronHAttention
from .layers.mamba2 import (
    MambaReplayCapture,
    MambaStateAccess,
    NemotronHMamba2Mixer,
)
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

    def __init__(
        self,
        mixer: Module[..., Tensor],
        config: NemotronHConfig,
        capture_replay: bool = False,
    ):
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = mixer
        self.capture_replay = capture_replay

    def forward(self, h: Tensor, *mixer_args: object) -> Any:
        # Row-parallel mixers leave partial sums; reduce before the residual add.
        if self.capture_replay:
            # A Mamba block that hands back the inputs of its state updates,
            # after its output.
            # Only Mamba mixers capture replay inputs, with their own arguments.
            mixed, replay = cast(Any, self.mixer).forward_with_replay(
                self.norm(h), *mixer_args
            )
            return h + mixed.to(h.mapping), replay
        mixed = self.mixer(self.norm(h), *mixer_args).to(h.mapping)
        return h + mixed


class NemotronHBackbone(
    Module[..., tuple[Tensor, tuple[Tensor, ...], tuple[Any, ...]]]
):
    """Embedding, then the hybrid layer stack. Returns pre-norm hidden states."""

    def __init__(self, config: NemotronHConfig, attn_params: KVCacheParams):
        self.embeddings = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layer_kinds = tuple(config.layer_kinds)
        self.target_layer_ids = config.target_layer_ids
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
            layers.append(
                NemotronHBlock(
                    mixer,
                    config,
                    capture_replay=(
                        config.capture_mamba_replay and kind is LayerKind.MAMBA
                    ),
                )
            )
        self.layers = ModuleList(layers)
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor],
        input_row_offsets: Tensor,
        capture_rows: Tensor | None = None,
        update_state: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[MambaReplayCapture, ...]]:
        """Runs the layer stack.

        Args:
            tokens: The ragged token ids.
            kv_collection: The attention layers' cache.
            state: The Mamba state to run on. Only its first two leaves, the
                convolution and SSM pools, are read.
            input_row_offsets: The ragged row offsets of the sequences.
            capture_rows: The token rows to capture each Mamba layer's replay
                inputs of. Required when the config sets
                ``capture_mamba_replay``.
            update_state: The ``[batch_size]`` bool flags selecting which
                requests write their new Mamba state back to their rows. A
                request with the flag false only reads its state. ``None``
                updates every request.

        Returns:
            The final pre-norm hidden states, the hidden states selected by
            ``target_layer_ids`` and each Mamba layer's captured replay inputs.
        """
        h = self.embeddings(tokens)
        replays: list[MambaReplayCapture] = []
        captures: dict[int, Tensor] = {}
        query_start_loc = F.cast(input_row_offsets, DType.int32)
        # In the order NemotronHConfig.construct_kv_params declares them.
        conv, ssm = state.leaves[:2]
        # Jenga zeroes a request's state rows before its first forward, so
        # every request can resume from its rows.
        batch_size = conv.live_row_ids.shape[1]
        has_initial_state = F.full(
            [batch_size], True, dtype=DType.bool, device=h.mesh
        )
        if update_state is None:
            update_state = has_initial_state
        mamba_idx = 0
        for index, (kind, layer) in enumerate(
            zip(self.layer_kinds, self.layers, strict=True)
        ):
            if index in self.target_layer_ids:
                captures[index] = h
            if kind is LayerKind.ATTENTION:
                h = layer(h, kv_collection, input_row_offsets)
                continue
            # The blocks of each other kind share one subgraph, and each call
            # resolves its own layer's weights. Attention layers bake their
            # KV-cache layer index in as a constant, so they can't share a
            # subgraph.
            call: Callable[..., Any] = as_subgraph(layer, name=kind.value)
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
                if layer.capture_replay:
                    assert capture_rows is not None, (
                        "capture_mamba_replay needs the rows to capture"
                    )
                    h, replay = call(
                        h,
                        access,
                        query_start_loc,
                        has_initial_state,
                        update_state,
                        capture_rows,
                    )
                    replays.append(replay)
                else:
                    h = call(
                        h,
                        access,
                        query_start_loc,
                        has_initial_state,
                        update_state,
                    )
                mamba_idx += 1
            else:
                h = call(h)
        if len(self.layers) in self.target_layer_ids:
            captures[len(self.layers)] = h
        return (
            h,
            tuple(captures[index] for index in self.target_layer_ids),
            tuple(replays),
        )


class NemotronH(Module[..., ModuleV3Outputs]):
    """Nemotron-H for causal language modeling."""

    def __init__(self, config: NemotronHConfig) -> None:
        if any(
            i < 0 or i > len(config.layer_kinds)
            for i in config.target_layer_ids
        ):
            raise ValueError(
                "Nemotron-H hidden-state layer ids must be between 0 and the layer count"
            )
        if (
            config.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
            and not config.target_layer_ids
        ):
            raise ValueError("Selected hidden states require target_layer_ids")
        if config.return_hidden_states == ReturnHiddenStates.LAST_PER_DEVICE:
            raise ValueError(
                "Nemotron-H cannot return per-device hidden states: its"
                " ModuleV3 outputs hold one copy, so LAST_PER_DEVICE would"
                " silently drop every device but the first. Use LAST."
            )
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        attn_params = config.kv_params.child(ATTN_CACHE_KEY, MHAKVCacheParams)
        self.backbone = NemotronHBackbone(config, attn_params)
        self.lm_head = quantized_linear(config, "lm_head")

    def _logits(self, h: Tensor) -> Tensor:
        logits = self.lm_head(self.backbone.norm_f(h))
        # Gather the vocabulary before selecting the sampler's device.
        logits = F.allgather(logits, tensor_axis=-1)
        return logits.cast(DType.float32)

    def _place(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replicates the single-device tokens and offsets across the mesh."""
        mesh = self.backbone.norm_f.weight.mesh
        if mesh.num_devices > 1:
            # Device graph capture records each device's stream on its own,
            # and a peer copy makes one stream wait on another, which
            # invalidates the capture. The collective syncs on the devices.
            return (
                F.distributed_broadcast(tokens, mesh),
                F.distributed_broadcast(input_row_offsets, mesh),
            )
        return tokens.to(mesh), input_row_offsets.to(mesh)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        if self.return_logits == ReturnLogits.VARIABLE:
            # Only `verify` builds the variable-logits range, from a host
            # count. This graph's count would land on the device, which
            # `F.range` rejects.
            raise NotImplementedError(
                "Nemotron-H returns a variable number of logits only from"
                " its speculative verify pass"
            )
        del return_n_logits
        row_offsets = input_row_offsets
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
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
        outputs, _, _ = self._run(
            tokens,
            None,
            input_row_offsets,
            row_offsets,
            kv_collection,
            state,
            None,
        )
        return outputs

    def verify(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor],
        capture_rows: Tensor,
        update_state: Tensor,
    ) -> tuple[ModuleV3Outputs, Tensor, tuple[MambaReplayCapture, ...]]:
        """Runs a speculative verify pass on the given Mamba state.

        Args:
            tokens: The single-device merged tokens, ``[tokens]``.
            return_n_logits: How many tokens of logits each request returns.
            input_row_offsets: The single-device ragged offsets.
            kv_collection: The attention layers' cache, on the mesh.
            state: The Mamba state the verify runs on.
            capture_rows: The token rows to capture each Mamba layer's replay
                inputs of.
            update_state: The ``[batch_size]`` bool flags selecting which
                requests advance their state. A request with drafts must not,
                since its accepted prefix is not known until the replay.

        Returns:
            The model outputs, the selected hidden states as one tensor
            placed on the mesh, and each Mamba layer's replay inputs.
        """
        row_offsets = input_row_offsets
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        outputs, hidden, replays = self._run(
            tokens,
            # A host tensor, as the logit range below needs.
            return_n_logits[0],
            input_row_offsets,
            row_offsets,
            kv_collection,
            state,
            capture_rows,
            update_state,
        )
        assert hidden is not None, "a verify pass returns hidden states"
        return outputs, hidden, replays

    def _run(
        self,
        tokens: Tensor,
        count: Tensor | None,
        input_row_offsets: Tensor,
        row_offsets: Tensor,
        kv_collection: PagedCacheValues,
        state: RecurrentStateInputsPerDevice[Tensor, Tensor],
        capture_rows: Tensor | None,
        update_state: Tensor | None = None,
    ) -> tuple[
        ModuleV3Outputs,
        Tensor | None,
        tuple[MambaReplayCapture, ...],
    ]:
        mesh = self.backbone.norm_f.weight.mesh
        h, captures, replays = self.backbone(
            tokens,
            kv_collection,
            state,
            input_row_offsets,
            capture_rows,
            update_state,
        )

        # One projection and one gather serve both outputs when ALL logits
        # are requested: the head runs once on the full hidden states and the
        # last rows are gathered for the sampler. For other modes the head
        # runs once on just the rows the sampler needs.
        logits: Tensor | None = None
        offsets: Tensor | None = None
        last: Tensor
        if self.return_logits == ReturnLogits.ALL:
            logits = self._logits(h)
            last = F.gather(
                logits, input_row_offsets[1:] - 1, axis=0
            ).local_shards[0]
            logits = logits.local_shards[0]
            offsets = row_offsets
        elif self.return_logits == ReturnLogits.VARIABLE:
            assert count is not None, "variable logits need a host count"
            distance = F.range(
                count,
                0,
                -1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=mesh,
            )
            indices = (input_row_offsets[1:].unsqueeze(-1) - distance).reshape(
                (-1,)
            )
            logits = self._logits(F.gather(h, indices, axis=0)).local_shards[0]
            # The last token of each request is the final row of its window
            # (`distance` ends at 1), so the head need not run on it again.
            # Only `verify` reaches this branch and it reads only `logits`, so
            # `next_token_logits` aliases the whole window: its shape here is
            # `[B * count, V]`, not `[B, V]`. A reader that needs the last rows
            # takes rows `count - 1, 2 * count - 1, ...` of `logits`.
            last = logits
            offsets = F.range(
                0,
                indices.shape[0] + count,
                count,
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=mesh,
            ).local_shards[0]
        else:
            last = self._logits(
                F.gather(h, input_row_offsets[1:] - 1, axis=0)
            ).local_shards[0]

        hidden = None
        if self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS:
            hidden = F.concat(captures, axis=-1)
        elif self.return_hidden_states == ReturnHiddenStates.ALL:
            hidden = h
        elif self.return_hidden_states == ReturnHiddenStates.ALL_NORMALIZED:
            hidden = self.backbone.norm_f(h)
        elif self.return_hidden_states == ReturnHiddenStates.LAST:
            hidden = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        return (
            ModuleV3Outputs(
                next_token_logits=last,
                logits=logits,
                logit_offsets=offsets,
                hidden_states=hidden.local_shards[0]
                if hidden is not None
                else None,
            ),
            hidden,
            replays,
        )
