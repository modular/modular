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
"""Inkling's call on ModuleV3, and the chained MTP depths its checkpoint ships.

The conv state is a ring of inputs indexed by position, sized for the
speculative width, so a verify that writes past the accepted prefix needs no
rollback: the next chunk overwrites the rejected positions. The depths embed
and project through the target's embedding and head, and each depth runs one
full decoder block. Both the verify and the proposer run on ``Tensor`` values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max import functional as F
from max.graph import DeviceRef, DimLike
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import RecurrentStateInputsPerDevice
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    TensorAdapter,
)
from max.pipelines.speculative.ragged_token_merger import shape_to_scalar
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from ..inkling.model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    STATE_CACHE_KEY,
)
from .inkling import Inkling, _token_positions
from .modules import InklingDecoder
from .mtp import InklingMultiTokenPredictor

if TYPE_CHECKING:
    from max.pipelines.speculative.config import SpeculativeConfig
    from max.pipelines.speculative.driver import SequentialProposer
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ADAPTER", "InklingMTPProposer"]

PRIMARY_KV = GLOBAL_ATTENTION

_DRAFT = "draft_"
"""Prefixes the draft's leaves beside its primary one in ``passthrough_kv``."""

_Batch = SequentialBatch[Tensor, Tensor, Tensor]

_shape_to_scalar = F.functional(shape_to_scalar)


def _replicate(t: Tensor, mesh: DeviceMesh) -> Tensor:
    """Replicates a device-zero input across the mesh, as the target does."""
    if mesh.num_devices > 1:
        return F.distributed_broadcast(t, mesh)
    return t.to(mesh)


def _conv_state(batch: _Batch) -> list[RecurrentStateInputsPerDevice[Any, Any]]:
    """Returns the per-device conv-state leaves the batch carries."""
    state: list[RecurrentStateInputsPerDevice[Any, Any]] = []
    for device in batch.passthrough_kv[STATE_CACHE_KEY]:
        assert isinstance(device, RecurrentStateInputsPerDevice)
        state.append(device)
    return state


def _merged_positions(batch: _Batch) -> Tensor:
    """Returns the device-zero positions of the verify window's rows."""
    return _token_positions(
        batch.merged_offsets,
        batch.kv_collections[0].cache_lengths,
        batch.merged_tokens.shape[0],
    )


def _accepted_positions(batch: _Batch) -> Tensor:
    """Returns each request's position at its last accepted token."""
    assert batch.num_accepted is not None
    last = F.rebind(batch.merged_offsets[1:] - 1, ["batch_size"])
    drafts = _shape_to_scalar(
        batch.num_draft_tokens, DeviceRef.from_device(last.device), last.dtype
    )
    accepted = last - drafts + batch.num_accepted.cast(last.dtype)
    return F.gather(_merged_positions(batch), accepted, axis=0)


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 Inkling model.

    Raises:
        ValueError: If the target wasn't built to return its last
            normalized layer, which the MTP depths read.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "An Inkling target verifies only for drafts that read its last"
            f" normalized layer. Got {hidden_states.value}."
        )
    assert isinstance(target, InklingDecoder)
    return _Verify(target.model)


class _Verify:
    """Inkling's verify pass over the merged batch."""

    def __init__(self, model: Inkling) -> None:
        self.model = model

    def __call__(self, batch: SequentialBatch | BlockBatch) -> Verified[Any]:
        """Verifies the batch's drafts against the target."""
        assert isinstance(batch, SequentialBatch)
        verified: Verified[Any] = as_graph_values(
            self._verify(as_tensors(batch))
        )
        return verified

    def _verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
        model = self.model
        mesh = model.norm.weight.mesh
        replicated = DeviceMapping.replicated(mesh)
        offsets = Tensor.from_shard_values(
            batch.merged_offsets_per_dev, replicated
        )
        h = model.hidden_states(
            _replicate(batch.merged_tokens, mesh),
            offsets,
            Tensor.from_shard_values(batch.vision_embeddings, replicated),
            Tensor.from_shard_values(batch.vision_scatter_indices, replicated),
            {
                PRIMARY_KV: PagedCacheValues.from_upstream(
                    batch.kv_collections, replicated
                ),
                LOCAL_ATTENTION: PagedCacheValues.from_upstream(
                    batch.passthrough_kv[LOCAL_ATTENTION], replicated
                ),
            },
            _conv_state(batch),
        )
        # The verify reads only the logits over its window, so it skips the
        # last-token projection.
        logits, _ = model.variable_logits(h, offsets, batch.return_n_logits)
        return Verified(logits=logits, hidden=list(model.norm(h).local_shards))


def sequential_proposer(
    verify: VerifyFn,
    draft: Any,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> SequentialProposer[Any, Any, Any, Any]:
    """Returns the proposer of the MTP depths, on the verify's target."""
    del draft_config, speculative
    assert isinstance(verify, _Verify)
    assert isinstance(draft, InklingMultiTokenPredictor)
    return InklingMTPProposer(draft, verify.model)


class InklingMTPProposer(TensorAdapter):
    """The chained MTP depths: one full decoder block per speculative step."""

    reuse: ReuseSpec | None = None
    decode_swaps: tuple[DecodeKVSwap, ...] = ()
    draft_cache = DraftCache.PER_DEPTH
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    # A depth returns each device's hidden states, never allgathered.
    step_hidden_mode = ReturnHiddenStates.ALL
    uses_thinking_phase = True
    supports_zero_draft_rows = False

    def __init__(
        self, draft: InklingMultiTokenPredictor, model: Inkling
    ) -> None:
        self.draft = draft
        self.model = model
        mesh = model.norm.weight.mesh
        self.mesh = mesh
        self.hidden_dim: DimLike = model.text_config.hidden_size
        flavors = list(draft.kv_params.children)
        # The driver drives the depths' leaf of the target's primary flavor,
        # or their first when the depths have none of that flavor.
        self._primary = PRIMARY_KV if PRIMARY_KV in flavors else flavors[0]
        self.passthrough_decode_swaps: tuple[str, ...] = tuple(
            f"{_DRAFT}{flavor}" for flavor in flavors if flavor != self._primary
        )

    def prefill(
        self, batch: _Batch, tokens: Tensor, target_hidden: list[Tensor]
    ) -> Proposed[Tensor]:
        """Drafts the first token from the verified window."""
        offsets = self._replicated(batch.query_offsets_per_dev)
        h = self.draft.forward_depth(
            0,
            self.model.embed_tokens(_replicate(tokens, self.mesh)),
            self._replicated(target_hidden),
            self._kv_by_flavor(batch),
            offsets,
            _replicate(_merged_positions(batch), self.mesh),
        )
        # Every row's hidden state, and the logits of each request's last
        # ``return_n_logits`` rows, which the driver gathers at the accepted
        # position.
        logits, _ = self.model.variable_logits(
            h, offsets, batch.return_n_logits
        )
        return Proposed(logits=logits, hidden=list(h.local_shards))

    def step(
        self, batch: _Batch, draft_input: DraftStepInput[Tensor], index: int
    ) -> Proposed[Tensor]:
        """Drafts the next token from the previous step's."""
        rows = f"{self.carry_dim_names.prefix}{index}_batch"
        positions = _accepted_positions(batch)
        offsets = self._replicated(batch.query_offsets_per_dev)
        h = self.draft.forward_depth(
            index,
            self.model.embed_tokens(_replicate(draft_input.tokens, self.mesh)),
            self._replicated(draft_input.hidden),
            self._kv_by_flavor(batch),
            offsets,
            F.rebind(_replicate(positions + index, self.mesh), [rows]),
        )
        logits = self.model.project_logits(F.gather(h, offsets[1:] - 1, axis=0))
        return Proposed(logits=logits, hidden=list(h.local_shards))

    def _replicated(self, values: list[Tensor]) -> Tensor:
        """Joins one single-device value per device into a replicated one."""
        return Tensor.from_shard_values(
            values, DeviceMapping.replicated(self.mesh)
        )

    def _kv_by_flavor(self, batch: _Batch) -> dict[str, PagedCacheValues]:
        """Returns the depths' attention leaves by flavor."""
        leaves = {
            self._primary: batch.draft_kv_collections,
            **{
                name.removeprefix(_DRAFT): value
                for name, value in batch.passthrough_kv.items()
                if name.startswith(_DRAFT)
            },
        }
        replicated = DeviceMapping.replicated(self.mesh)
        return {
            name: PagedCacheValues.from_upstream(value, replicated)
            for name, value in leaves.items()
        }


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    sequential_proposer=sequential_proposer,
)
