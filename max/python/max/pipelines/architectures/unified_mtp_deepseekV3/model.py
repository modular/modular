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
"""DeepseekV3 with MTP PipelineModel: target + draft in one graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.speculative import (
    DraftAliases,
    validate_draft_state_dict,
)
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig
from .batch_processor import UnifiedMTPDeepseekV3BatchProcessor
from .unified_mtp_deepseekV3 import UnifiedMTPDeepseekV3

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedMTPDeepseekV3Inputs(UnifiedSpecDecodeInputs, DeepseekV3Inputs):
    """Inputs for the UnifiedMTPDeepseekV3 model.

    Target-prefix fields come from :class:`DeepseekV3Inputs`; the spec-decode
    fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`. The MTP graph binds the per-row
    ``in_thinking_phase`` flag (consumed by relaxed acceptance).
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return super().buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedMTPDeepseekV3Model(_UnifiedSpecDecodeModelMixin, DeepseekV3Model):
    """DeepseekV3 with MTP: merge + target + rejection + shift in one graph."""

    batch_processor_cls: ClassVar[type[UnifiedMTPDeepseekV3BatchProcessor]] = (
        UnifiedMTPDeepseekV3BatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _draft_config: DeepseekV3NextNConfig

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if self.adapter:
            raw_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            raw_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        self._draft_state_dict = {
            k[len("draft.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("draft.")
        }
        # Some checkpoints share shared_head_norm with the base model's final
        # norm and don't emit it as a draft weight.
        if (
            "shared_head_norm.weight" not in self._draft_state_dict
            and "target.norm.weight" in raw_state_dict
        ):
            self._draft_state_dict["shared_head_norm.weight"] = raw_state_dict[
                "target.norm.weight"
            ]

        return {
            k[len("target.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("target.")
        }

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV3Config:
        config = DeepseekV3Model._create_model_config(self, state_dict)

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        draft_config = self._create_draft_config(self._draft_state_dict)
        if draft_config.ep_config is not None and config.ep_config is not None:
            draft_config.ep_config.node_id = config.ep_config.node_id

        # TODO: don't hard code number of layers
        assert isinstance(self.kv_params, KVCacheParams)
        self._draft_kv_params = replace(self.kv_params, num_layers=1)
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": self.kv_params, "draft": self._draft_kv_params}
        )

        draft_config.return_hidden_states = ReturnHiddenStates.LAST
        self._draft_config = draft_config
        return config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        assert isinstance(model_config, DeepseekV3Config)
        self.ep_comm_initializer = None
        self.draft_ep_comm_initializer = None
        if model_config.ep_config is None or is_virtual_device_mode():
            return

        # Allocate EP buffers with BF16 dispatch dtype (the larger dtype)
        # so both target (FP4) and draft (BF16) can share the same buffers.
        ep_cfg = replace(
            model_config.ep_config,
            dispatch_dtype=DType.bfloat16,
            dispatch_quant_config=None,
        )
        self.ep_comm_initializer = EPCommInitializer(ep_cfg)
        self.ep_comm_initializer.ep_init(session)
        model_config.ep_config.node_id = ep_cfg.node_id
        if model_config.ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP "
                "initialization is successful."
            )
        self.draft_ep_comm_initializer = self.ep_comm_initializer

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, DeepseekV3Config)
        assert self.pipeline_config.speculative is not None

        nn_model = UnifiedMTPDeepseekV3(
            model_config,
            self._draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        # Share embed_tokens and lm_head BEFORE loading so state_dict()
        # deduplicates them — the adapter only emits target.* copies.
        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens
        nn_model.draft.lm_head = nn_model.target.lm_head

        nn_model.target.load_state_dict(
            state_dict, weight_alignment=1, strict=True
        )
        # strict=False because shared weights (embed_tokens, lm_head) are
        # aliased to target's and won't have keys in draft_state_dict.
        nn_model.draft.load_state_dict(
            self._draft_state_dict, weight_alignment=1, strict=False
        )

        # The NextN head ships in the target checkpoint, so the draft
        # inherits both.
        validate_draft_state_dict(
            nn_model.draft.raw_state_dict().keys(),
            self._draft_state_dict.keys(),
            DraftAliases(always=("embed_tokens.", "lm_head.")),
        )

        weights_registry = {
            **nn_model.draft.state_dict(),
            **nn_model.target.state_dict(),
        }

        with Graph(
            "deepseekV3_with_mtp_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            graph_inputs = nn_model.decode_inputs(graph.inputs, self.kv_params)

            outputs = nn_model(
                tokens=graph_inputs.tokens,
                input_row_offsets=graph_inputs.input_row_offsets,
                draft_tokens=graph_inputs.draft_tokens,
                signal_buffers=graph_inputs.signal_buffers,
                kv_collections=graph_inputs.kv("target"),
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
                ep_inputs=graph_inputs.ep_inputs or None,
                draft_kv_collections=graph_inputs.kv("draft"),
                pinned_bitmask=graph_inputs.pinned_bitmask,
                wait_payload=graph_inputs.wait_payload,
                device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            )

            graph.output(*outputs)

        return graph, weights_registry

    def _create_draft_config(
        self, draft_state_dict: dict[str, WeightData]
    ) -> DeepseekV3NextNConfig:
        """Create NextN model config for the draft model."""
        nextn_key = "decoder_layer.self_attn.kv_a_layernorm.weight"
        base_key = "layers.0.self_attn.kv_a_layernorm.weight"

        if nextn_key not in draft_state_dict:
            raise KeyError(
                f"Expected NextN norm key '{nextn_key}' not found in "
                f"draft state_dict. Available keys: "
                f"{list(draft_state_dict.keys())[:10]}..."
            )

        draft_state_dict[base_key] = draft_state_dict[nextn_key]
        base_config = DeepseekV3Model._create_model_config(
            self, draft_state_dict
        )
        if base_key in draft_state_dict and nextn_key in draft_state_dict:
            del draft_state_dict[base_key]

        if (
            base_config.quant_config is not None
            and base_config.quant_config.is_nvfp4
            and not any("weight_scale_2" in key for key in draft_state_dict)
        ):
            logger.info(
                "NextN weights are BF16 (no weight_scale_2 found); "
                "disabling NVFP4 config for draft."
            )
            base_config.quant_config = None
            base_config.dtype = DType.bfloat16
            if base_config.ep_config is not None:
                base_config.ep_config.dispatch_dtype = DType.bfloat16
                base_config.ep_config.dispatch_quant_config = None

        draft_config = DeepseekV3NextNConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(base_config)
            }
        )
        return draft_config
