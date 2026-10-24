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
"""MiMo-V2.6-Flash + DFlash on ModuleV3: the fused speculative graph on the
block driver.

The graph stages under the graph-API build's name
(:data:`GRAPH_NAME`), which Mach's block-spec executor binds by constant, so
one engine serves either export. A multi-device trace inserts its signal
buffers at the same input position the canonical signature declares them
(:data:`SIGNAL_BUFFER_INDEX`, or :data:`DP_SIGNAL_BUFFER_INDEX` once the
data-parallel batch-split pair precedes them); a single-device one declares
its own there, since the engine binds one at every device count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType
from max.graph.weights import WeightData
from max.nn.comm.ep import EPBatchManager
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.lib import SpeculativeConfig
from max.pipelines.speculative._tensor_compat import as_tensors
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..dflash_mimo_v2 import DFlashMiMoV2Config
from ..dflash_mimo_v2_modulev3 import DFlashMiMoV2
from ..mimo_v2_modulev3.model_config import FULL, SLIDING, MiMoV2Config
from .model_config import DRAFT
from .spec_adapters import (
    DFlashMiMoV2Proposer,
    MiMoV2BlockTarget,
    MiMoV2SpecTarget,
    verify_fn,
)

__all__ = [
    "DP_SIGNAL_BUFFER_INDEX",
    "GRAPH_NAME",
    "SIGNAL_BUFFER_INDEX",
    "UnifiedDflashMiMoV2ModuleV3",
    "UnifiedDflashMiMoV2Spec",
]

GRAPH_NAME = "unified_dflash_mimo_v2"
"""The fused graph's name: the graph-API build's, so Mach's block-spec
executor (``BLOCK_SPEC_MODEL_NAME``) selects this MEF unchanged."""

SIGNAL_BUFFER_INDEX = 3
"""Where a multi-device trace inserts the mesh's signal buffers: after
``tokens``, ``input_row_offsets`` and ``return_n_logits``, before the KV
tree -- where the engine's block adapter binds them at every device count."""

DP_SIGNAL_BUFFER_INDEX = SIGNAL_BUFFER_INDEX + 2
"""The same slot under data-parallel attention: the canonical distributed
signature declares the host row offsets and the per-replica batch splits
between the batch inputs and the signals, so the trace inserts the mesh's
buffers two slots further in."""


@dataclass(kw_only=True)
class UnifiedDflashMiMoV2Spec:
    """What the fused graph is built from."""

    target: MiMoV2Config
    draft: DFlashMiMoV2Config
    speculative_config: SpeculativeConfig
    num_speculative_tokens: int
    """How many of the block's proposals a step verifies."""
    sampleable_vocab_size: int
    """The tokenizer's size. ``lm_head`` rows from here up are padding with
    live weights, so the graph masks them out of every verify."""

    def __post_init__(self) -> None:
        if not 0 < self.sampleable_vocab_size <= self.target.vocab_size:
            raise ValueError(
                f"MiMo-V2 DFlash: sampleable_vocab_size"
                f" {self.sampleable_vocab_size} is outside the"
                f" {self.target.vocab_size}-row lm_head."
            )


class UnifiedDflashMiMoV2ModuleV3(Module[..., tuple[Tensor, ...]]):
    """Verify ``[anchor, d1..dK]`` on the target, write the drafter's context
    for the verified rows, then draft the next block.

    The block is the drafter's trained 8 wide whatever K is, so the next
    step's block always overwrites every rejected row's context. The phases
    live in :class:`BlockDriver`; this module contributes the two
    :class:`~max.pipelines.speculative.driver.TensorAdapter` adapters in
    :mod:`.spec_adapters`.
    """

    GRAPH_NAME = GRAPH_NAME
    """The graph this module's trace names itself; the export-side contract
    checks the name Mach's executor binds against this class, as the GLM
    ModuleV3 build does its module class's qualname."""

    def __init__(
        self,
        spec: UnifiedDflashMiMoV2Spec,
        mask_embedding: WeightData,
        *,
        kv_params: KVCacheParamInterface,
        enable_structured_output: bool = False,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        """Builds the fused graph's module.

        Args:
            spec: The target, drafter and verify width.
            mask_embedding: The drafter's trained mask slot embedding, baked
                into the graph as a constant.
            kv_params: The ``{target, draft}`` KV tree.
            enable_structured_output: Whether acceptance takes the grammar
                bitmask.
            ep_batch_manager: The EP buffers manager when the target serves
                expert parallelism, whose input tail the signature then
                declares past the KV tree. Required beside data-parallel
                attention, whose one-device replicas the P2P dispatch needs.
        """
        self.spec = spec
        self.target = MiMoV2SpecTarget(spec.target, ep_batch_manager)
        # The drafter's degree comes from the spec alone -- the pipeline
        # model sets it beside the draft config -- so a spec whose degree
        # mismatches trips the drafter's own guard instead of being
        # silently repaired here.
        self.draft = DFlashMiMoV2(spec.draft, mask_embedding=mask_embedding)
        self.kv_params = kv_params
        self.enable_structured_output = enable_structured_output

    def driver(self) -> BlockDriver[list[Tensor], list[Tensor]]:
        """Builds the driver over this module's target and draft.

        Built per trace rather than held, because compiling traces a copy of
        this module whose parameters are the graph's weights, and the driver
        must call that copy's layers.
        """
        spec = self.spec
        n_devices = len(spec.target.devices)
        dp_degree = spec.target.data_parallel_degree
        block_target = MiMoV2BlockTarget(
            self.target,
            vocab_size=spec.target.vocab_size,
            sampleable_vocab_size=spec.sampleable_vocab_size,
        )
        return BlockDriver(
            # The driver takes the verify as a callable; the conversion
            # pair it applies to the proposer's calls lives in verify_fn.
            verify_fn(block_target),
            DFlashMiMoV2Proposer(
                self.draft,
                self.target,
                num_speculative_tokens=spec.num_speculative_tokens,
                sampleable_vocab_size=spec.sampleable_vocab_size,
            ),
            # The driver only holds these; the adapters run them.
            target_model=self.target,
            draft_model=self.draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=spec.target.devices,
                # The DP pair, the per-device batch context lengths and the
                # EP tail appear only beside data parallelism; the TP
                # export's ABI is pinned by Mach's sidecar.
                distributed=dp_degree > 1,
                data_parallel_degree=dp_degree,
                # A single-device graph declares its signal buffer in tree
                # order, as the engine's block adapter binds it; a
                # multi-device one takes the mesh's, which the trace
                # inserts at the same position -- under data parallelism two
                # slots further in, past the batch-split pair.
                include_signal_buffers=n_devices == 1,
                trace_owns_signal_buffers=n_devices > 1,
                enable_structured_output=self.enable_structured_output,
            ),
            speculative_config=spec.speculative_config,
            enable_structured_output=self.enable_structured_output,
            # The drafter's own lengths place its context, as they do in an
            # engine that sizes each group's claim separately.
            ctx_at_draft_cache_length=True,
            num_speculative_tokens=spec.num_speculative_tokens,
            use_greedy_acceptance=(
                spec.speculative_config.use_greedy_acceptance
            ),
            ep_input_types=block_target.ep_input_types(),
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """The types :meth:`forward` takes, one per positional input."""
        return self.driver().input_types(self.kv_params)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        driver = self.driver()
        named = driver.decode_inputs(inputs, self.kv_params)
        outputs: tuple[Any, ...] = driver(
            tokens=named.tokens,
            input_row_offsets=named.input_row_offsets,
            draft_tokens=named.draft_tokens,
            kv_collections=named.kv("target", FULL),
            draft_kv_collections=named.kv(DRAFT),
            passthrough_kv={SLIDING: named.kv("target", SLIDING)},
            return_n_logits=named.return_n_logits,
            seed=named.seed,
            temperature=named.temperature,
            top_k=named.top_k,
            max_k=named.max_k,
            top_p=named.top_p,
            min_top_p=named.min_top_p,
            signal_buffers=list(named.signal_buffers)
            if named.signal_buffers
            else None,
            # None at data_parallel_degree <= 1, where the signature
            # declares none of the distributed group.
            host_input_row_offsets=named.host_input_row_offsets,
            data_parallel_splits=named.data_parallel_splits,
            batch_context_lengths=named.batch_context_lengths,
            ep_inputs=named.ep_inputs or None,
            pinned_bitmask=named.pinned_bitmask,
            wait_payload=named.wait_payload,
            device_bitmask_scratch=named.device_bitmask_scratch,
        )
        result: tuple[Tensor, ...] = as_tensors(outputs)
        return result
