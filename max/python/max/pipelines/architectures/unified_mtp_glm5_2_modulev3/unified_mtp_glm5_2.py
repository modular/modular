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
"""GLM-5.x with MTP speculative decoding, as one ModuleV3 module."""

from __future__ import annotations

from typing import Any

from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    SequentialDriver,
    tensor_compat_verify,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3_2_modulev3.deepseekV3_2 import DeepseekV3_2
from ..deepseekV3_2_modulev3.nextn import DeepseekV3_2NextN
from ..unified_mtp_glm5_2.spec_adapters import (
    INDEXER_DRAFT_KV,
    INDEXER_TARGET_KV,
)
from .spec_adapters import Glm5_2ModuleV3MTPProposer, Glm5_2ModuleV3Target


class UnifiedMTPGlm5_2(Module[..., tuple[Tensor, ...]]):
    """Verifies the previous drafts with the target and proposes the next.

    Takes the unified spec-decode signature's inputs, as
    :meth:`input_types` declares them, and returns the driver's outputs.
    """

    def __init__(
        self,
        target: DeepseekV3_2,
        draft: DeepseekV3_2NextN,
        *,
        kv_params: KVCacheParamInterface,
        speculative_config: SpeculativeConfig,
        enable_structured_output: bool,
    ) -> None:
        super().__init__()
        self.target = target
        self.draft = draft
        self.kv_params = kv_params
        self.speculative_config = speculative_config
        self.enable_structured_output = enable_structured_output

    def driver(self) -> SequentialDriver[Any]:
        """Builds the driver over this module's target and draft.

        Built per trace rather than held, because compiling traces a copy of
        this module whose parameters are the graph's weights, and the driver
        must call that copy's layers.
        """
        config = self.target.config
        target_adapter = Glm5_2ModuleV3Target(self.target)
        return SequentialDriver(
            tensor_compat_verify(target_adapter.verify),
            Glm5_2ModuleV3MTPProposer(self.draft, self.target),
            target_model=self.target,
            draft_model=self.draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                distributed=True,
                data_parallel_degree=config.data_parallel_degree,
                trace_owns_signal_buffers=True,
            ),
            speculative_config=self.speculative_config,
            enable_structured_output=self.enable_structured_output,
            vocab_size=config.vocab_size,
            ep_input_types=target_adapter.ep_input_types(),
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """The types :meth:`forward` takes, one per positional input."""
        return self.driver().input_types(self.kv_params)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        driver = self.driver()
        named = driver.decode_inputs(inputs, self.kv_params)
        return driver(
            tokens=named.tokens,
            input_row_offsets=named.input_row_offsets,
            draft_tokens=named.draft_tokens,
            kv_collections=named.kv("target", "mla"),
            draft_kv_collections=named.kv("draft", "mla"),
            passthrough_kv={
                INDEXER_TARGET_KV: named.kv("target", "indexer"),
                INDEXER_DRAFT_KV: named.kv("draft", "indexer"),
            },
            return_n_logits=named.return_n_logits,
            host_input_row_offsets=named.host_offsets,
            data_parallel_splits=named.dp_splits,
            batch_context_lengths=named.batch_context_lengths,
            seed=named.seed,
            temperature=named.temperature,
            top_k=named.top_k,
            max_k=named.max_k,
            top_p=named.top_p,
            min_top_p=named.min_top_p,
            in_thinking_phase=named.thinking_phase,
            ep_inputs=named.ep_inputs or None,
            pinned_bitmask=named.pinned_bitmask,
            wait_payload=named.wait_payload,
            device_bitmask_scratch=named.device_bitmask_scratch,
            draft_probs_full=named.draft_probs_full,
        )
