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
"""Block spec-decode adapters for a Nemotron-H target and its DSpark draft.

The target side is the Mamba rollback. A KV cache is rewound by moving a
length pointer, but each token a Mamba layer reads is already folded into a
fixed-size state, so a verify that may reject its drafts must not advance the
live state. Each step therefore runs

1. **verify**: run the target over the anchor and its drafts on the live state
   pools with their ``update_state`` flag clear, so the live rows keep their
   pre-verify value, capturing the rows each Mamba layer's two state updates
   consumed;
2. **accept**: the driver compares the drafts to the target's samples;
3. **replay**: advance the live rows by exactly the accepted rows, from the
   captured inputs, with the verify's arithmetic. A step with no drafts has
   nothing to reject, so its verify sets the flag and advances the live rows
   itself, and there is nothing to replay (see
   :mod:`..nemotron_h_modulev3.state_rollback`).

Step 3 belongs to the proposer's ``materialize``: the first phase after the
accepted count settles, and the last before anything reads the live state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.kernels import argmax_with_bias
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.pipelines.speculative._tensor_compat import as_graph_values
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import broadcast_per_device

from ..nemotron_h_modulev3.layers.mamba2 import (
    MambaReplayCapture,
    MambaReplayInputs,
    NemotronHMamba2Mixer,
)
from ..nemotron_h_modulev3.model_config import LayerKind
from ..nemotron_h_modulev3.nemotron_h import NemotronH, _distributed_state
from ..nemotron_h_modulev3.state_rollback import (
    replay_plan,
    replay_state_pools,
    verify_update_state,
    verify_window,
)
from .dspark_qwen3 import DSparkQwen3, markov_bias

__all__ = [
    "STATE_EXTRA",
    "DSparkNemotronHProposer",
    "NemotronHBlockTarget",
]

STATE_EXTRA = "nemotron_state"
"""Key the per-device Mamba state leaves ride under in ``BlockBatch.extra``.

Each device's leaves are the live conv and SSM pools, in the order
``NemotronHConfig.construct_kv_params`` declares them."""


def _replicate(value: Tensor, mesh: DeviceMesh) -> Tensor:
    """Places a single-device tensor on every device of ``mesh``."""
    if mesh.num_devices > 1:
        return F.distributed_broadcast(value, mesh)
    return value.to(mesh)


def _from_shards(values: Sequence[TensorValue], mesh: DeviceMesh) -> Tensor:
    """One replicated tensor from each device's copy."""
    return Tensor.from_shard_values(
        list(values), DeviceMapping.replicated(mesh)
    )


def _shards(value: Tensor) -> list[TensorValue]:
    """Each device's shard of ``value``, as graph values."""
    return [TensorValue(shard) for shard in value.local_shards]


@dataclass
class _Verified:
    """What a verify leaves for the replay that follows it in the same step."""

    replays: tuple[MambaReplayCapture, ...]
    window_offsets: TensorValue
    state: Sequence[RecurrentStateInputsPerDevice[TensorValue, Any]]


class NemotronHBlockTarget:
    """Nemotron-H verifies on read-only live state, then replays the accepted prefix."""

    def __init__(self, target: NemotronH) -> None:
        self.target = target
        self.mesh: DeviceMesh = target.backbone.norm_f.weight.mesh
        self._verified: _Verified | None = None
        self._mambas = [
            block.mixer
            for block, kind in zip(
                target.backbone.layers, target.backbone.layer_kinds, strict=True
            )
            if kind is LayerKind.MAMBA
        ]

    def verify(self, batch: BlockBatch) -> Verified[list[TensorValue]]:
        state = batch.extra[STATE_EXTRA]
        assert len(state) == batch.n_devs
        for device_state in state:
            if len(device_state.leaves) != 2:
                raise ValueError(
                    "Speculative Nemotron-H needs the live conv and SSM state"
                    f" leaves; the KV cache declares {len(device_state.leaves)}"
                    " state leaves."
                )
        # Drafts are verified without advancing the live rows, which keep
        # their pre-verify value for the replay.
        update_state = _from_shards(
            [
                verify_update_state(
                    batch.num_draft_tokens,
                    device_state.leaves[0].live_row_ids.shape[1],
                    device_state.leaves[0].live_row_ids.device,
                )
                for device_state in state
            ],
            self.mesh,
        )
        live_state = _distributed_state(
            list(state), DeviceMapping.replicated(self.mesh)
        )
        window_rows, window_offsets = verify_window(
            batch.merged_offsets, batch.num_draft_tokens, batch.device0
        )
        outputs, hidden, replays = self.target.verify(
            tokens=Tensor.from_graph_value(batch.merged_tokens),
            return_n_logits=Tensor.from_graph_value(batch.return_n_logits),
            input_row_offsets=Tensor.from_graph_value(batch.merged_offsets),
            kv_collection=PagedCacheValues.from_upstream(
                batch.kv_collections, DeviceMapping.replicated(self.mesh)
            ),
            state=live_state,
            capture_rows=_replicate(
                Tensor.from_graph_value(window_rows), self.mesh
            ),
            update_state=update_state,
        )
        self._verified = _Verified(
            replays=replays,
            window_offsets=window_offsets,
            state=state,
        )
        assert outputs.logits is not None
        # VARIABLE logits, then the taps concatenated; one copy per device. The
        # driver runs on graph values, so hand it those.
        return Verified(
            logits=as_graph_values(outputs.logits),
            hidden=as_graph_values(list(hidden.local_shards)),
        )

    def roll_forward(self, batch: BlockBatch) -> None:
        """Replays the verify's accepted prefix onto the live state.

        Must run after the accepted count is known and before anything reads
        the live pools.
        """
        verified, self._verified = self._verified, None
        assert verified is not None, "roll_forward needs the verify's captures"
        assert batch.num_accepted is not None
        device0 = batch.device0
        k = batch.num_draft_tokens
        replay_bounds = replay_plan(
            verified.window_offsets, batch.num_accepted, k, device0
        )
        bounds_per_device = broadcast_per_device(
            replay_bounds, batch.signal_buffers, batch.n_devs
        )
        for device in range(batch.n_devs):
            live_conv, live_ssm = verified.state[device].leaves
            replay_state_pools(
                self._captures(verified.replays, device),
                live_conv.pool,
                live_ssm.pool,
                live_conv.live_row_ids,
                live_ssm.live_row_ids,
                bounds_per_device[device],
            )

    def _captures(
        self,
        replays: tuple[MambaReplayCapture, ...],
        device: int,
    ) -> list[MambaReplayInputs[TensorValue]]:
        """One device's replay inputs of every Mamba layer, in layer order."""
        assert len(replays) == len(self._mambas)
        captures = []
        for mixer, capture in zip(self._mambas, replays, strict=True):
            assert isinstance(mixer, NemotronHMamba2Mixer)
            A, dt_bias = (
                TensorValue(t.local_shards[device])
                for t in mixer.replay_statics(DType.bfloat16)
            )
            captures.append(
                MambaReplayInputs(
                    projected=TensorValue(
                        capture.projected.local_shards[device]
                    ),
                    convolved=TensorValue(
                        capture.convolved.local_shards[device]
                    ),
                    A=A,
                    dt_bias=dt_bias,
                )
            )
        return captures

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DSparkNemotronHProposer:
    """The DSpark block draft: the target's embedding and head, its own blocks.

    The block is ``[anchor, mask x (block_size - 1)]`` and its first slot
    predicts nothing: ``samples_from_anchor`` is ``False``, as the draft's
    ``sample_from_anchor: false`` says, and a block of 8 carries 7 drafts.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(
        self,
        draft: DSparkQwen3,
        target: NemotronHBlockTarget,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
    ) -> None:
        self.draft = draft
        self.target = target
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.mesh = target.mesh

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[Any],
    ) -> None:
        # The first phase after the accepted count settles, and the last
        # before anything reads the live state pools.
        self.target.roll_forward(batch)

        # Every merged row is written, including the rows of rejected drafts:
        # the draft's cache lengths advance by the committed rows only, so the
        # rest is overwritten by the next step's context.
        taps = _from_shards(target_hidden, self.mesh)
        self.draft.materialize(
            self.draft.project_context(taps),
            PagedCacheValues.from_upstream(
                ctx_kv, DeviceMapping.replicated(self.mesh)
            ),
            _from_shards(batch.merged_offsets_per_dev, self.mesh),
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        ids = _replicate(Tensor.from_graph_value(block_ids), self.mesh)
        embedded = self.target.target.backbone.embeddings(ids)
        # The anchor embeds as the target embeds it; the draft's own trained
        # row replaces the target's untrained one at every mask slot.
        is_mask = (ids == self.mask_token_id).unsqueeze(-1)
        embedded = F.where(
            is_mask,
            self.draft.mask_embedding.to(embedded.mapping),
            embedded,
        )
        return _shards(embedded)

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[Any],
    ) -> list[TensorValue]:
        hidden = self.draft.forward_block(
            _from_shards(embeds, self.mesh),
            PagedCacheValues.from_upstream(
                block_kv_with_dispatch(block_kv, self.block_size),
                DeviceMapping.replicated(self.mesh),
            ),
            _from_shards(offsets, self.mesh),
        )
        return _shards(hidden)

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        k = self.block_size
        drafts = k - 1
        hidden = _from_shards(block_hs, self.mesh)
        # Drop the anchor slot: it carries the committed token, not a draft.
        drafted = hidden.reshape([-1, k, self.hidden_size])[:, 1:, :]
        # The draft ships no head; the target's reads the draft's own final
        # norm's output, not the target's.
        logits = self.target.target.lm_head(
            drafted.reshape([-1, self.hidden_size])
        )
        logits = F.allgather(logits, tensor_axis=-1)
        # A greedy draft takes each position's argmax of base + bias in one
        # fused pass that reads the logits in place and adds in float32 in
        # registers, the same index as an argmax of the float32 sum. The
        # sampled draft keeps the float32 logits its sampler draws from.
        fused_argmax = (
            isinstance(sampler, ArgmaxDraftSampler)
            and TensorValue(logits.local_shards[0]).device != DeviceRef.CPU()
        )
        if not fused_argmax:
            logits = logits.cast(DType.float32)
        vocab = logits.shape[-1]
        base = TensorValue(logits.local_shards[0]).reshape(
            ("batch_size", drafts, vocab)
        )
        # The Markov correction is sequential: each position's bias reads the
        # token sampled before it, and the anchor for the first.
        head = self.draft.markov_head
        previous = accepted.next_tokens
        sampled: list[TensorValue] = []
        for step in range(drafts):
            bias = TensorValue(
                markov_bias(head, Tensor.from_graph_value(previous))
            )
            if fused_argmax:
                previous = argmax_with_bias(base, bias, position=step)
            else:
                previous = sampler.sample_next(
                    base[:, step, :] + bias.cast(DType.float32), step
                )
            sampled.append(ops.unsqueeze(previous, axis=1))
        return ops.concat(sampled, axis=1)
