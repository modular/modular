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
"""Gemma4 with MTP PipelineModel: target + draft in one graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces import (
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .batch_processor import UnifiedMTPGemma4BatchProcessor
from .unified_mtp_gemma4 import UnifiedMTPGemma4
from .weight_adapters import convert_unified_safetensor_state_dict


@dataclass
class UnifiedMTPGemma4Inputs(UnifiedSpecDecodeInputs):
    """Inputs for the UnifiedMTPGemma4 model.

    The spec-decode fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`; the fields below plus the KV cache form
    this distributed MTP graph's prefix. The graph binds the per-row
    ``in_thinking_phase`` flag and, when structured output is enabled, the
    constrained-decoding bitmask triple.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.batch_context_lengths,
        )
        return prefix + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedMTPGemma4Model(
    _UnifiedSpecDecodeModelMixin,
    AlwaysSignalBuffersMixin,
    PipelineModelWithKVCache[TextContext],
):
    """Gemma4 with MTP: merge + target + rejection + shift in one graph."""

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[UnifiedMTPGemma4BatchProcessor]] = (
        UnifiedMTPGemma4BatchProcessor
    )

    model: Model

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        self._max_batch_size = max_batch_size
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        )

        # Force signal buffer initialization.
        _ = self.signal_buffers

        self.model = self.load_model(session)

    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self._max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        with CompilationTimer("unified_mtp_gemma4_model") as timer:
            # -- 1. Load target weights --
            target_state_dict = parse_state_dict_from_weights(
                self.pipeline_config, self.weights, self.adapter
            )

            # -- 2. Load draft weights from draft_model checkpoint --
            assert self.pipeline_config.draft_model is not None
            draft_model_config = self.pipeline_config.draft_model
            draft_weight_paths = draft_model_config.resolved_weight_paths()
            draft_weights = load_weights(draft_weight_paths)

            draft_state_dict = self._convert_draft_weights(
                dict(draft_weights.items())
            )

            # -- 3. Create target config --
            config = Gemma4ForConditionalGenerationConfig.initialize(
                self.pipeline_config
            )
            config.finalize(
                huggingface_config=self.huggingface_config,
                state_dict=target_state_dict,
                return_logits=ReturnLogits.VARIABLE,
            )

            # -- 4. Create draft config --
            draft_hf_config = draft_model_config.huggingface_config
            assert draft_hf_config is not None
            draft_config = self._create_draft_config(
                draft_hf_config, config.devices
            )
            # -- 5. Create unified module --
            assert self.pipeline_config.speculative is not None
            nn_model = UnifiedMTPGemma4(
                config,
                draft_config,
                speculative_config=self.pipeline_config.speculative,
                enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            )

            # Set return modes on the target model
            nn_model.target.return_logits = ReturnLogits.VARIABLE
            nn_model.target.return_hidden_states = (
                ReturnHiddenStates.ALL_NORMALIZED
            )

            # -- 6. Create draft model and share embed_tokens/lm_head --
            assert isinstance(self.kv_params, MultiKVCacheParams)
            target_sliding_kv_params = self.kv_params.children[
                "sliding_attention"
            ]
            assert isinstance(target_sliding_kv_params, KVCacheParams)
            target_global_kv_params = self.kv_params.children["full_attention"]
            assert isinstance(target_global_kv_params, KVCacheParams)
            target_layer_types = config.text_config.layer_types

            nn_model.draft = Gemma4Assistant(
                draft_config,
                target_layer_types=target_layer_types,
                target_sliding_kv_params=target_sliding_kv_params,
                target_global_kv_params=target_global_kv_params,
            )
            # Share the target's embed_tokens for the concat(embed, hidden)
            # input step.  The assistant's own 1024-dim draft_embed_tokens
            # and tied lm_head are loaded from the assistant checkpoint.
            nn_model.draft.embed_tokens = nn_model.target.embed_tokens

            # -- 7. Merge with target.*/draft.* prefixes --
            unified_state_dict = convert_unified_safetensor_state_dict(
                target_state_dict, draft_state_dict
            )

            # strict=False: shared weights (embed_tokens, lm_head) are aliased
            # to target's and won't have draft.* copies.
            nn_model.load_state_dict(
                unified_state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,
            )
            self.state_dict = nn_model.state_dict()

            # -- 8. The draft is Q-only cross-attention into the target's KV
            # caches (no K/V projections), so it allocates no cache of its
            # own. None signals SpecDecodeState to skip the draft manager
            # and the graph to declare no draft KV inputs.
            self._draft_kv_params = None

            # -- 9. Build graph and compile --
            with Graph(
                "gemma4_with_mtp_graph",
                input_types=nn_model.input_types(self.kv_params),
            ) as graph:
                (
                    tokens,
                    device_input_row_offsets,
                    host_input_row_offsets,
                    return_n_logits,
                    data_parallel_splits,
                    *variadic_args,
                ) = graph.inputs

                variadic_args_iter = iter(variadic_args)
                signal_buffers = [
                    next(variadic_args_iter).buffer
                    for _ in range(len(self.devices))
                ]

                # Unflatten the hybrid {sliding, global} KV tree.
                sliding_kv_collections, global_kv_collections = (
                    self.kv_params.unflatten_basic_kv_tree(variadic_args_iter)
                )

                batch_context_lengths = [
                    next(variadic_args_iter).tensor
                    for _ in range(len(self.devices))
                ]

                draft_tokens = next(variadic_args_iter).tensor

                seed = next(variadic_args_iter).tensor
                temperature = next(variadic_args_iter).tensor
                top_k = next(variadic_args_iter).tensor
                max_k = next(variadic_args_iter).tensor
                top_p = next(variadic_args_iter).tensor
                min_top_p = next(variadic_args_iter).tensor
                in_thinking_phase = next(variadic_args_iter).tensor

                pinned_bitmask_graph = None
                wait_payload_graph = None
                device_bitmask_scratch_graph = None
                if nn_model.enable_structured_output:
                    pinned_bitmask_graph = next(variadic_args_iter).tensor
                    wait_payload_graph = next(variadic_args_iter).buffer
                    device_bitmask_scratch_graph = next(
                        variadic_args_iter
                    ).buffer

                outputs = nn_model(
                    tokens=tokens.tensor,
                    input_row_offsets=device_input_row_offsets.tensor,
                    draft_tokens=draft_tokens,
                    signal_buffers=signal_buffers,
                    sliding_kv_collections=sliding_kv_collections,
                    global_kv_collections=global_kv_collections,
                    return_n_logits=return_n_logits.tensor,
                    host_input_row_offsets=host_input_row_offsets.tensor,
                    data_parallel_splits=data_parallel_splits.tensor,
                    batch_context_lengths=batch_context_lengths,
                    seed=seed,
                    temperature=temperature,
                    top_k=top_k,
                    max_k=max_k,
                    top_p=top_p,
                    min_top_p=min_top_p,
                    in_thinking_phase=in_thinking_phase,
                    pinned_bitmask=pinned_bitmask_graph,
                    wait_payload=wait_payload_graph,
                    device_bitmask_scratch=device_bitmask_scratch_graph,
                )

                graph.output(*outputs)

            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> UnifiedEagleOutputs:
        """Execute and return all 3 graph outputs for speculative decoding."""
        assert isinstance(model_inputs, UnifiedMTPGemma4Inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert len(model_outputs) == 3, (
            f"Expected 3 outputs, got {len(model_outputs)}"
        )

        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Gemma4ForConditionalGenerationConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    def _convert_draft_weights(
        self,
        draft_weights_dict: dict[str, Weights],
    ) -> dict[str, WeightData]:
        """Convert HuggingFace assistant checkpoint keys to MAX format.

        The HF assistant checkpoint has keys like:
        - ``model.layers.0.self_attn.q_proj.weight`` -> ``layers.0.self_attn.q_proj.weight``
        - ``model.norm.weight`` -> ``norm.weight``
        - ``pre_projection.weight`` -> ``pre_projection.weight`` (at top level)
        - ``post_projection.weight`` -> ``post_projection.weight``
        - ``model.embed_tokens.weight`` -> kept (assistant's own 1024-dim embedding)
        """
        new_state_dict: dict[str, WeightData] = {}

        for name, value in draft_weights_dict.items():
            data = value.data()

            # Strip "model." prefix for keys under model.*
            if name.startswith("model."):
                max_name = name[len("model.") :]
            else:
                # Top-level keys like pre_projection, post_projection
                max_name = name

            new_state_dict[max_name] = data

        return new_state_dict

    def _create_draft_config(
        self,
        draft_hf_config: AutoConfig,
        devices: list[DeviceRef],
    ) -> Gemma4AssistantConfig:
        """Create Gemma4AssistantConfig from the draft HF config."""
        from ..gemma3.model_config import _HIDDEN_ACTIVATION_MAP
        from ..gemma4.layers.rotary_embedding import ProportionalScalingParams

        raw_text_config = draft_hf_config
        if hasattr(draft_hf_config, "text_config"):
            raw_text_config = draft_hf_config.text_config

        # Normalize to dict so we can use .get() uniformly whether
        # the HF shim stored it as a dict or a sub-config object.
        tc: dict[str, Any] = (
            raw_text_config
            if isinstance(raw_text_config, dict)
            else raw_text_config.__dict__
        )

        # Extract global rope scaling if available.
        global_rope_scaling = None
        rope_parameters = tc.get("rope_parameters")
        if rope_parameters is not None and "full_attention" in rope_parameters:
            full_attn_params = rope_parameters["full_attention"]
            partial_rotary_factor = full_attn_params.get(
                "partial_rotary_factor"
            )
            if partial_rotary_factor is not None:
                global_rope_scaling = ProportionalScalingParams(
                    partial_rotary_factor=partial_rotary_factor,
                )

        # Get backbone hidden size from the target HF config.
        target_text_config = self.huggingface_config.text_config
        backbone_hidden_size = target_text_config.hidden_size

        num_hidden_layers = tc["num_hidden_layers"]
        return Gemma4AssistantConfig(
            devices=devices,
            backbone_hidden_size=backbone_hidden_size,
            hidden_size=tc["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            num_global_key_value_heads=tc.get("num_global_key_value_heads", 4),
            head_dim=tc["head_dim"],
            global_head_dim=tc.get("global_head_dim", 512),
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            hidden_activation=_HIDDEN_ACTIVATION_MAP.get(
                tc.get("hidden_activation", "gelu_pytorch_tanh"),
                tc.get("hidden_activation", "gelu_pytorch_tanh"),
            ),
            layer_types=tc.get(
                "layer_types",
                ["sliding_attention"] * (num_hidden_layers - 1)
                + ["full_attention"],
            ),
            sliding_window=tc.get("sliding_window", 1024),
            sliding_window_rope_theta=tc.get(
                "sliding_window_rope_theta", 10000.0
            ),
            global_rope_theta=tc.get("global_rope_theta", 1000000.0),
            global_rope_scaling=global_rope_scaling,
            attention_k_eq_v=tc.get("attention_k_eq_v", True),
            num_kv_shared_layers=tc.get("num_kv_shared_layers", 4),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
        )
