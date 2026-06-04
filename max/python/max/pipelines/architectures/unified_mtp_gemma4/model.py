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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import (
    KVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
    UnifiedEagleOutputs,
)
from max.pipelines.lib.interfaces import PipelineModelWithKVCache
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .unified_mtp_gemma4 import UnifiedMTPGemma4
from .weight_adapters import convert_unified_safetensor_state_dict


@dataclass
class UnifiedMTPGemma4Inputs(ModelInputs):
    """Inputs for the UnifiedMTPGemma4 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]

    draft_tokens: Buffer | None = None
    draft_kv_blocks: list[Buffer] | None = None
    seed: Buffer | None = None
    temperature: Buffer | None = None
    top_k: Buffer | None = None
    max_k: Buffer | None = None
    top_p: Buffer | None = None
    min_top_p: Buffer | None = None

    in_thinking_phase: Buffer | None = None
    """Per-batch ``bool`` flag marking rows currently inside a
    ``<think>...</think>`` block; consumed by relaxed acceptance."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        # Fixed positional ABI: the graph always consumes draft_tokens, seed,
        # and the sampling buffers, so assert they are present and append in
        # graph order (a missing buffer would silently shift the ABI).
        assert self.draft_tokens is not None
        assert self.seed is not None
        assert self.temperature is not None
        assert self.top_k is not None
        assert self.max_k is not None
        assert self.top_p is not None
        assert self.min_top_p is not None
        assert self.in_thinking_phase is not None
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.batch_context_lengths,
            self.draft_tokens,
        )
        if self.draft_kv_blocks is not None:
            buffers += tuple(self.draft_kv_blocks)
        buffers += (
            self.seed,
            self.temperature,
            self.top_k,
            self.max_k,
            self.top_p,
            self.min_top_p,
            self.in_thinking_phase,
        )
        return buffers


class UnifiedMTPGemma4Model(
    AlwaysSignalBuffersMixin, PipelineModelWithKVCache[TextContext]
):
    """Gemma4 with MTP: merge + target + rejection + shift in one graph."""

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig

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
    ) -> None:
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

        assert pipeline_config.runtime.max_batch_size is not None
        max_batch_size = pipeline_config.runtime.max_batch_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(devices[0])
        )
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(devices))
        ]

    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self.pipeline_config.runtime.max_batch_size
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
            draft_config = self._create_draft_config(draft_hf_config)
            # -- 5. Create unified module --
            assert self.pipeline_config.speculative is not None
            nn_model = UnifiedMTPGemma4(
                config,
                draft_config,
                speculative_config=self.pipeline_config.speculative,
            )

            # Set return modes on the target model
            nn_model.target.return_logits = ReturnLogits.VARIABLE
            nn_model.target.return_hidden_states = (
                ReturnHiddenStates.ALL_NORMALIZED
            )

            # -- 6. Create draft model and share embed_tokens/lm_head --
            assert isinstance(self.kv_params, MultiKVCacheParams)
            target_sliding_kv_params = self.kv_params.params[0]
            target_global_kv_params = self.kv_params.params[1]
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
                input_types=nn_model.input_types(
                    self.kv_params, self._draft_kv_params
                ),
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

                # Unflatten target KV cache inputs. MultiKVCacheParams produces
                # [sliding_dev0, ..., global_dev0, ...] in order.
                kv_flat_types = list(
                    self.kv_params.get_symbolic_inputs().flatten()
                )
                all_kv_caches = self._unflatten_kv_inputs(
                    [
                        next(variadic_args_iter)
                        for _ in range(len(kv_flat_types))
                    ]
                )
                # Split into sliding and global cache collections
                half = len(all_kv_caches) // 2
                sliding_kv_collections = list(all_kv_caches[:half])
                global_kv_collections = list(all_kv_caches[half:])

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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        draft_kv_cache_buffers: list[Buffer] | None = None,
        **kwargs: object,
    ) -> UnifiedMTPGemma4Inputs:
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        device0 = self.devices[0]
        pinned = not device0.is_host

        batch_size = len(context_batch)
        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

        buffer_type = DevicePinnedBuffer if pinned else Buffer
        host_tokens = buffer_type(
            dtype=DType.int64, shape=(total_seq_len,), device=device0
        )
        host_row_offsets = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=host_tokens.to_numpy(),
        )
        device_tokens = host_tokens.to(device0)

        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )
        device_row_offsets = host_row_offsets.to(device0)

        host_input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        )

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        data_parallel_splits = Buffer.from_numpy(
            np.array([0, batch_size], dtype=np.int64)
        )

        batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        return UnifiedMTPGemma4Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            batch_context_lengths=batch_context_lengths,
            draft_tokens=draft_tokens,
            draft_kv_blocks=draft_kv_cache_buffers,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> UnifiedMTPGemma4Inputs:
        raise NotImplementedError("MTP does not support Multistep execution")

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
