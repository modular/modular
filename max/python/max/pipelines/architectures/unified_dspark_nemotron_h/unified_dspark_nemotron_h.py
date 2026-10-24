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
"""DSpark on Nemotron-H: the hybrid target and the block draft, one graph."""

from __future__ import annotations

from typing import Any, cast

from max.experimental.nn import Module
from max.experimental.realization_context import current_realization_context
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.speculative._tensor_compat import (
    as_graph_values,
    as_tensors,
)
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..nemotron_h_modulev3.model_config import ATTN_CACHE_KEY, STATE_CACHE_KEY
from ..nemotron_h_modulev3.nemotron_h import NemotronH
from .dspark_qwen3 import DSparkQwen3
from .model_config import UnifiedDSparkNemotronHConfig
from .spec_adapters import (
    STATE_EXTRA,
    DSparkNemotronHProposer,
    NemotronHBlockTarget,
)

__all__ = ["UnifiedDSparkNemotronH", "UnifiedDSparkNemotronHModule"]


def check_speculative_config(speculative: SpeculativeConfig) -> None:
    """Rejects the speculative settings this architecture cannot serve.

    ``draft_proposal="sampled"`` is supported; the driver then requires
    stochastic acceptance.

    Raises:
        ValueError: If synthetic acceptance is set.
    """
    if speculative.synthetic_acceptance_rate is not None:
        raise ValueError(
            "Synthetic acceptance would bypass the Mamba state rollback's"
            " accepted-length plan; it is not supported on Nemotron-H."
        )


class UnifiedDSparkNemotronH(BlockDriver[list[TensorValue], list[TensorValue]]):
    """Merge, verify on scratch Mamba state, accept, commit, then block draft.

    The phases live in :class:`BlockDriver`. This architecture's own are the
    two adapters in :mod:`.spec_adapters`: the target's snapshot, verify and
    replay of the Mamba state, and the draft's use of the target's embedding
    and head.
    """

    def __init__(
        self,
        config: UnifiedDSparkNemotronHConfig,
        target: NemotronH,
        draft: DSparkQwen3,
        *,
        enable_structured_output: bool = False,
    ) -> None:
        speculative = config.speculative_config
        check_speculative_config(speculative)
        target_adapter = NemotronHBlockTarget(target)
        super().__init__(
            target_adapter.verify,
            DSparkNemotronHProposer(
                draft,
                target_adapter,
                block_size=config.block_size,
                mask_token_id=config.mask_token_id,
                hidden_size=config.target.hidden_size,
            ),
            # The driver only holds these; the adapters run them.
            target_model=cast(Any, target),
            draft_model=cast(Any, draft),
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=False,
                # The traced graph appends the signal buffers of its mesh.
                trace_owns_signal_buffers=True,
            ),
            speculative_config=speculative,
            enable_structured_output=enable_structured_output,
            vocab_size=config.target.vocab_size,
        )
        self.config = config


class UnifiedDSparkNemotronHModule(Module[..., tuple[Tensor, ...]]):
    """The model one compiled graph runs: target, draft and the driver.

    The weights are ``target.*`` and ``draft.*``. Takes the unified
    spec-decode signature's inputs, as :meth:`input_types` declares them, and
    returns the driver's outputs.
    """

    def __init__(
        self,
        config: UnifiedDSparkNemotronHConfig,
        target: NemotronH,
        draft: DSparkQwen3,
        *,
        kv_params: KVCacheParamInterface,
        speculative_config: SpeculativeConfig,
        enable_structured_output: bool,
    ) -> None:
        super().__init__()
        self.target = target
        self.draft = draft
        self.config = config
        self.kv_params = kv_params
        self.speculative_config = speculative_config
        self.enable_structured_output = enable_structured_output

    def driver(self) -> UnifiedDSparkNemotronH:
        """Builds the driver over this module's target and draft.

        Built per trace rather than held, because compiling traces a copy of
        this module whose parameters are the graph's weights, and the driver
        must call that copy's layers.
        """
        return UnifiedDSparkNemotronH(
            self.config,
            self.target,
            self.draft,
            enable_structured_output=self.enable_structured_output,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """The types :meth:`forward` takes, one per positional input."""
        return self.driver().input_types(self.kv_params)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        driver = self.driver()
        named = driver.decode_inputs(inputs, self.kv_params)
        # The driver runs on graph values; this module's boundary is Tensors.
        # A ModuleV3 trace keeps the signal buffers of its mesh in the
        # realization context, where the collective ops read them.
        signals = getattr(
            current_realization_context(None), "signal_buffers", None
        )
        kv_tree = named.kv_tree
        assert isinstance(kv_tree, dict)
        target_tree = kv_tree["target"]
        assert isinstance(target_tree, dict)
        outputs = driver(
            **as_graph_values(
                dict(
                    tokens=named.tokens,
                    input_row_offsets=named.input_row_offsets,
                    draft_tokens=named.draft_tokens,
                    kv_collections=named.kv("target", ATTN_CACHE_KEY),
                    draft_kv_collections=named.kv("draft"),
                    return_n_logits=named.return_n_logits,
                    seed=named.seed,
                    temperature=named.temperature,
                    top_k=named.top_k,
                    max_k=named.max_k,
                    top_p=named.top_p,
                    min_top_p=named.min_top_p,
                    draft_probs_full=named.draft_probs_full,
                    signal_buffers=list(signals) if signals else None,
                    pinned_bitmask=named.pinned_bitmask,
                    wait_payload=named.wait_payload,
                    device_bitmask_scratch=named.device_bitmask_scratch,
                    extra={STATE_EXTRA: target_tree[STATE_CACHE_KEY]},
                )
            )
        )
        result: tuple[Tensor, ...] = as_tensors(outputs)
        return result
