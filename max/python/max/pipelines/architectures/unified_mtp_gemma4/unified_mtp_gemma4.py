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
"""Gemma4 with MTP: the assistant draft wired to the spec-decode driver."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max.graph import Graph, TensorValue, Type, Value
from max.graph import Module as GraphModule
from max.nn.kv_cache import KVCacheParamInterface
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..gemma4.gemma4 import Gemma4TextModel
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .spec_adapters import GLOBAL_KV, Gemma4MTPProposer, Gemma4Target

_SLIDING_KV = "sliding_attention"


def gemma4_mtp_input_spec(
    config: Gemma4ForConditionalGenerationConfig,
    *,
    enable_structured_output: bool,
) -> SpecDecodeInputTypeSpec:
    """The signature this arch declares: distributed, with vision inputs.

    The per-row ``in_thinking_phase`` flag is not named here -- the driver adds
    it from :attr:`Gemma4MTPProposer.uses_thinking_phase`. Split out of the
    constructor so a test can assert the declaration without building the
    target and draft modules.
    """
    return SpecDecodeInputTypeSpec(
        devices=config.devices,
        distributed=True,
        data_parallel_degree=1,
        enable_vision=True,
        vision_hidden_size=config.text_config.hidden_size,
        enable_structured_output=enable_structured_output,
    )


class UnifiedMTPGemma4(SequentialDriver[list[TensorValue]]):
    """Gemma4 with MTP: the assistant draft wired to the spec-decode driver.

    The assistant cross-attends into the target's paired sliding/full caches
    rather than keeping one of its own; the loop itself lives in
    :class:`SequentialDriver`.
    """

    target: Gemma4TextModel
    draft: Gemma4Assistant

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        draft_config: Gemma4AssistantConfig,
        draft: Gemma4Assistant,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        use_greedy_acceptance: bool = False,
        kv_params: KVCacheParamInterface | None = None,
    ) -> None:
        target = Gemma4TextModel(config)
        super().__init__(
            Gemma4Target(target).verify,
            Gemma4MTPProposer(draft, draft_config.backbone_hidden_size),
            target_model=target,
            draft_model=draft,
            input_spec=gemma4_mtp_input_spec(
                config, enable_structured_output=enable_structured_output
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            use_greedy_acceptance=use_greedy_acceptance,
        )
        self.config = config
        self._draft_config = draft_config
        self.use_greedy_acceptance = use_greedy_acceptance
        self._kv_params = kv_params

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface | None:
        """The target's paired ``{sliding, full}`` caches, which the draft
        cross-attends into."""
        return self._kv_params

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces one spec-decode step, whose inputs are :meth:`input_types`.

        Decodes the flat graph inputs by :meth:`decode_inputs` and routes the
        KV cache: the target and the draft both read the sliding cache, and
        the full-attention cache passes through to both.

        Args:
            *input_types: The graph's input types, :meth:`input_types`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """

        def forward(*inputs: Value[Any]) -> tuple[TensorValue, ...]:
            graph_inputs = self.decode_inputs(inputs)
            return self(
                tokens=graph_inputs.tokens,
                input_row_offsets=graph_inputs.input_row_offsets,
                vision_embeddings=graph_inputs.vision_embeddings,
                vision_scatter_indices=graph_inputs.vision_scatter_indices,
                draft_tokens=graph_inputs.draft_tokens,
                signal_buffers=graph_inputs.signal_buffers,
                kv_collections=graph_inputs.kv(_SLIDING_KV),
                draft_kv_collections=graph_inputs.kv(_SLIDING_KV),
                passthrough_kv={GLOBAL_KV: graph_inputs.kv(GLOBAL_KV)},
                return_n_logits=graph_inputs.return_n_logits,
                host_input_row_offsets=graph_inputs.host_offsets,
                data_parallel_splits=graph_inputs.dp_splits,
                batch_context_lengths=graph_inputs.batch_context_lengths,
                seed=graph_inputs.seed,
                temperature=graph_inputs.temperature,
                top_k=graph_inputs.top_k,
                max_k=graph_inputs.max_k,
                top_p=graph_inputs.top_p,
                min_top_p=graph_inputs.min_top_p,
                in_thinking_phase=graph_inputs.thinking_phase,
                pinned_bitmask=graph_inputs.pinned_bitmask,
                wait_payload=graph_inputs.wait_payload,
                device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
