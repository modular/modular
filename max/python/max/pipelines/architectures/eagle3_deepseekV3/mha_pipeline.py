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
"""Eagle3 MHA-draft + DeepseekV3 (MLA target) PipelineModel.

Sibling of :class:`Eagle3DeepseekV3Model` for the case where the draft is a
Llama-style MHA Eagle3 head (``LlamaForCausalLMEagle3``) over a bare
``DeepseekV3ForCausalLM`` target.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorValue, Value
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces import UnifiedEagleOutputs
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from ..kimik2_5.eagle3_mha_kimi_k25 import Eagle3MHAKimiK25DraftConfig
from ..kimik2_5.unified_eagle_mha_model import Eagle3MHAKimiK25Unified
from ..kimik2_5.unified_eagle_mha_pipeline_model import (
    _build_mha_draft_config,
    _infer_fc_input_multiplier,
)
from ..kimik2_5.weight_adapters import convert_llama_eagle3_draft_state_dict
from .model import extract_eagle_aux_layer_ids

logger = logging.getLogger("max.pipelines")


@dataclass
class Eagle3MHADeepseekV3Inputs(DeepseekV3Inputs):
    """Inputs for the Eagle3 MHA-draft + DeepseekV3 unified model.

    The draft owns its own ``KVCacheInputs`` (separate from the target's
    MLA cache) so MHA dispatch metadata can be plumbed at both prefill
    (q_max_seq_len = ``1 + num_speculative_tokens``) and decode
    (q_max_seq_len = 1) widths.
    """

    draft_tokens: Buffer | None = None
    draft_kv_blocks: list[Buffer] | None = None
    """One persistent ``kv_blocks`` Buffer per device. Other fields
    (cache_lengths, lookup_table, max_lengths,
    attention_dispatch_metadata) are borrowed from the target's
    ``kv_cache_inputs`` at graph build time."""
    seed: Buffer | None = None
    temperature: Buffer | None = None
    top_k: Buffer | None = None
    max_k: Buffer | None = None
    top_p: Buffer | None = None
    min_top_p: Buffer | None = None
    in_thinking_phase: Buffer | None = None
    """Per-batch ``bool`` flag for relaxed-acceptance gating. Not consumed
    by the graph today but required by the ``_UnifiedEagleInputs`` protocol
    used by ``OverlapTextGenerationPipeline``."""
    token_bitmasks: Buffer | None = None
    """Grammar constraint bitmask for structured output (None when off)."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = super().buffers
        if self.draft_tokens is not None:
            buffers += (self.draft_tokens,)
        if self.draft_kv_blocks is not None:
            buffers += tuple(self.draft_kv_blocks)
        assert self.seed is not None
        buffers += (self.seed,)
        if self.draft_tokens is not None:
            assert self.temperature is not None
            assert self.top_k is not None
            assert self.max_k is not None
            assert self.top_p is not None
            assert self.min_top_p is not None
            assert self.in_thinking_phase is not None
            buffers += (
                self.temperature,
                self.top_k,
                self.max_k,
                self.top_p,
                self.min_top_p,
                self.in_thinking_phase,
            )
        if self.token_bitmasks is not None:
            buffers += (self.token_bitmasks,)
        return buffers


class Eagle3MHADeepseekV3Model(DeepseekV3Model):
    """Eagle3 MHA-draft + DeepseekV3: target + draft in one compiled graph."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.VARIABLE,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.EAGLE3,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.devices[0])

    @override
    def load_model(self, session: InferenceSession) -> Model:
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        dp_size = self.pipeline_config.model.data_parallel_degree
        max_batch_size *= dp_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(self.devices[0])
        )
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        if self.adapter:
            target_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            target_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        config = self._create_model_config(target_state_dict)

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        self.ep_comm_initializer = None
        if config.ep_config is not None and not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            config.ep_config.node_id = self.ep_comm_initializer.config.node_id
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP "
                    "initialization is successful."
                )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_hf = draft_model_config.huggingface_config
        assert draft_hf is not None

        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        draft_state_dict = convert_llama_eagle3_draft_state_dict(
            dict(draft_weights.items())
        )

        fc_input_multiplier = _infer_fc_input_multiplier(
            draft_state_dict, hidden_size=int(draft_hf.hidden_size)
        )

        if config.eagle_aux_hidden_state_layer_ids is None:
            ids = extract_eagle_aux_layer_ids(draft_hf)
            if ids is None:
                target_layers = int(config.num_hidden_layers)
                ids = _default_aux_layer_ids(target_layers, fc_input_multiplier)
                logger.warning(
                    "Draft HF config has no "
                    "'eagle_config.eagle_aux_hidden_state_layer_ids'. "
                    "Falling back to evenly-spaced defaults for "
                    f"{fc_input_multiplier}-way fusion over {target_layers} "
                    f"target layers: {ids}. Override by setting "
                    "``eagle_config.eagle_aux_hidden_state_layer_ids`` on "
                    "the draft HF config if the training-time IDs are "
                    "known."
                )
            config.eagle_aux_hidden_state_layer_ids = ids

        n_target_capture = len(config.eagle_aux_hidden_state_layer_ids)
        if n_target_capture != fc_input_multiplier:
            raise ValueError(
                f"Draft fc fuses {fc_input_multiplier} target hidden states "
                f"but the target is configured to capture {n_target_capture} "
                f"(eagle_aux_hidden_state_layer_ids="
                f"{config.eagle_aux_hidden_state_layer_ids})."
            )

        assert isinstance(self.kv_params, KVCacheParams)
        target_kv = self.kv_params
        if target_kv.dtype != DType.bfloat16:
            logger.warning(
                "Draft KV cache dtype forced to bfloat16 (target uses "
                f"{target_kv.dtype}). MHA flash attention requires q/k/v "
                "and KV cache dtypes to match; the draft module emits bf16."
            )
        self._draft_kv_params = self.pipeline_config.model.kv_cache.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=int(draft_hf.num_key_value_heads),
            head_dim=int(draft_hf.head_dim),
            num_layers=1,
            devices=target_kv.devices,
            data_parallel_degree=target_kv.data_parallel_degree,
            is_mla=False,
            num_q_heads=int(draft_hf.num_attention_heads),
            num_eagle_speculative_tokens=target_kv.num_eagle_speculative_tokens,
        )

        draft_config: Eagle3MHAKimiK25DraftConfig = _build_mha_draft_config(
            draft_hf,
            target_config=config,
            devices=config.devices,
            data_parallel_degree=config.data_parallel_degree,
            fc_input_multiplier=fc_input_multiplier,
            kv_params=self._draft_kv_params,
            sliding_window_override=draft_model_config.sliding_window,
        )

        assert self.pipeline_config.speculative is not None
        nn_model = Eagle3MHAKimiK25Unified(
            config,
            draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.sampling.enable_structured_output,
        )

        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens

        nn_model.target.load_state_dict(
            target_state_dict, weight_alignment=1, strict=True
        )
        nn_model.draft.load_state_dict(
            draft_state_dict, weight_alignment=1, strict=False
        )

        draft_expected = set(nn_model.draft.raw_state_dict().keys())
        draft_provided = set(draft_state_dict.keys())
        shared_prefixes = ("embed_tokens.",)
        missing = {
            k
            for k in draft_expected - draft_provided
            if not k.startswith(shared_prefixes)
        }
        extra = draft_provided - draft_expected
        if missing:
            raise ValueError(
                f"Draft model has unloaded non-shared weights: {sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        draft_weights_registry = nn_model.draft.state_dict()
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith("embed_tokens."):
                continue
            weight.name = f"draft.{name}"

        self.state_dict = dict(nn_model.target.state_dict())
        for k, v in draft_weights_registry.items():
            if k.startswith("embed_tokens."):
                continue
            self.state_dict[f"draft.{k}"] = v

        with CompilationTimer("eagle3_mha_deepseekV3_model") as timer:
            with Graph(
                "eagle3_mha_deepseekV3_graph",
                input_types=nn_model.input_types(
                    self.kv_params, self._draft_kv_params
                ),
            ) as graph:
                (
                    tokens,
                    devices_input_row_offsets,
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

                target_symbolic = self.kv_params.get_symbolic_inputs(
                    draft_attention_group=self._draft_kv_params
                )
                fetch_types = target_symbolic.inputs[0].flatten()
                len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
                kv_caches_per_dev = list(
                    target_symbolic.unflatten(
                        iter(
                            [
                                next(variadic_args_iter)
                                for _ in range(len_of_kv_inputs)
                            ]
                        )
                    ).inputs
                )

                batch_context_lengths = [
                    next(variadic_args_iter).tensor
                    for _ in range(len(self.devices))
                ]

                target_ep_inputs: list[Value[Any]] | None = None
                if nn_model.target.ep_manager is not None:
                    n_target_ep = len(nn_model.target.ep_manager.input_types())
                    target_ep_inputs = [
                        next(variadic_args_iter) for _ in range(n_target_ep)
                    ]

                draft_tokens = next(variadic_args_iter).tensor

                draft_kv_collections: list[PagedCacheValues] = []
                for dev_idx in range(len(self.devices)):
                    draft_kv_blocks = next(variadic_args_iter).buffer
                    target_kv_dev = kv_caches_per_dev[dev_idx]
                    draft_kv_collections.append(
                        PagedCacheValues(
                            kv_blocks=draft_kv_blocks,
                            cache_lengths=target_kv_dev.cache_lengths,
                            lookup_table=target_kv_dev.lookup_table,
                            max_lengths=target_kv_dev.max_lengths,
                            attention_dispatch_metadata=target_kv_dev.draft_attention_dispatch_metadata,
                            draft_attention_dispatch_metadata=target_kv_dev.draft_attention_dispatch_metadata,
                        )
                    )

                seed = next(variadic_args_iter).tensor
                temperature = next(variadic_args_iter).tensor
                top_k = next(variadic_args_iter).tensor
                max_k = next(variadic_args_iter).tensor
                top_p = next(variadic_args_iter).tensor
                min_top_p = next(variadic_args_iter).tensor
                in_thinking_phase = next(variadic_args_iter).tensor

                token_bitmasks_graph: TensorValue | None = None
                if nn_model.enable_structured_output:
                    token_bitmasks_graph = next(variadic_args_iter).tensor

                outputs = nn_model(
                    tokens=tokens.tensor,
                    input_row_offsets=devices_input_row_offsets.tensor,
                    draft_tokens=draft_tokens.tensor,
                    signal_buffers=signal_buffers,
                    kv_collections=kv_caches_per_dev,
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
                    ep_inputs=target_ep_inputs,
                    draft_kv_collections=draft_kv_collections,
                    token_bitmasks=token_bitmasks_graph,
                )
                graph.output(*outputs)

            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        assert isinstance(model_inputs, Eagle3MHADeepseekV3Inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        if len(model_outputs) != 3:
            raise RuntimeError(
                f"Eagle3MHADeepseekV3 graph returned {len(model_outputs)} "
                "outputs; expected 3 (num_accepted, next_tokens, "
                "next_draft_tokens)."
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
        **kwargs,
    ) -> Eagle3MHADeepseekV3Inputs:
        base = DeepseekV3Model.prepare_initial_token_inputs(
            self, replica_batches, kv_cache_inputs, return_n_logits
        )
        return Eagle3MHADeepseekV3Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            draft_tokens=draft_tokens,
            draft_kv_blocks=draft_kv_cache_buffers,
            seed=self._next_seed(),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Eagle3MHADeepseekV3Inputs:
        raise NotImplementedError("Eagle does not support Multistep execution")


def _default_aux_layer_ids(
    num_target_layers: int, fc_input_multiplier: int
) -> list[int]:
    """Picks ``fc_input_multiplier`` aux capture layer IDs spread across
    a target with ``num_target_layers`` layers.

    Fallback used when the draft HF config doesn't declare its
    training-time aux IDs. Shifted one layer below vLLM's
    ``SupportsEagle3.get_eagle3_default_aux_hidden_state_layers`` default
    (``(1, num_layers // 2 - 1, num_layers - 4)`` for 3-way) — empirically
    a better match for the ``modularai/kimi-k2.5-eagle3`` checkpoint.
    """
    if num_target_layers <= 4:
        raise ValueError(
            f"Default aux layer IDs require >4 target layers, got "
            f"{num_target_layers}."
        )
    early = 1
    late = num_target_layers - 4
    if fc_input_multiplier == 2:
        return [early, late]
    if fc_input_multiplier == 3:
        return [early, num_target_layers // 2 - 1, late]
    raise ValueError(
        f"Unsupported fc_input_multiplier={fc_input_multiplier}; "
        "expected 2 or 3."
    )
