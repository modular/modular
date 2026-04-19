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
"""Eagle3 + Kimi K2.5 PipelineModel: target + draft in one graph."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, Value
from max.graph.weights import WeightData, load_weights
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.lib import CompilationTimer, ModelInputs
from max.pipelines.lib.interfaces import UnifiedEagleOutputs
from typing_extensions import override

from ..deepseekV3.model_config import DeepseekV3Config
from .model import KimiK2_5Model, KimiK2_5ModelInputs
from .model_config import (
    KimiK2_5Config,
    KimiK2_5TextConfig,
    _extract_eagle_aux_layer_ids,
)
from .unified_eagle_model import Eagle3KimiK25Unified
from .weight_adapters import convert_eagle3_draft_state_dict

logger = logging.getLogger("max.pipelines")


@dataclass
class Eagle3KimiK25Inputs(KimiK2_5ModelInputs):
    """Inputs for the Eagle3 + Kimi K2.5 model.

    Same as ``KimiK2_5ModelInputs`` but skips the vision related inputs.
    """

    draft_tokens: Buffer | None = None
    draft_kv_blocks: list[Buffer] | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )
        if self.draft_tokens is not None:
            buffers += (self.draft_tokens,)
        if self.draft_kv_blocks is not None:
            buffers += tuple(self.draft_kv_blocks)
        return buffers


class Eagle3KimiK25Model(KimiK2_5Model):
    """Eagle3 + Kimi K2.5: target + draft in one compiled graph.

    Loads target weights from the main Kimi K2.5 checkpoint and draft weights
    from a separate Eagle3 checkpoint (``pipeline_config.draft_model``).

    The Eagle3 language model graph is text-only — the raw ``DeepseekV3``
    target is used, not ``KimiK2_5MoEDecoder``. Vision is handled by the
    base pipeline during initial prefill.
    """

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.EAGLE3
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
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

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in target_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith("language_model.") or key.startswith(
                "language_"
            ):
                llm_state_dict[key] = value

        config = self._create_model_config(target_state_dict)

        # The target HF config doesn't carry eagle_config; propagate from draft.
        if config.eagle_aux_hidden_state_layer_ids is None:
            assert self.pipeline_config.draft_model is not None
            draft_hf = self.pipeline_config.draft_model.huggingface_config
            ids = _extract_eagle_aux_layer_ids(draft_hf)
            if ids is None:
                raise ValueError(
                    "eagle_aux_hidden_state_layer_ids must be present in the "
                    "draft model's eagle_config for EAGLE3 hidden-state "
                    "capture, but was not found in the draft HF config."
                )
            config.eagle_aux_hidden_state_layer_ids = ids

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
        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)

        draft_state_dict = convert_eagle3_draft_state_dict(
            dict(draft_weights.items()),
        )

        draft_config = self._create_draft_config(config, draft_state_dict)
        if draft_config.ep_config is not None and config.ep_config is not None:
            draft_config.ep_config.node_id = config.ep_config.node_id

        assert isinstance(self.kv_params, KVCacheParams)
        self._draft_kv_params = replace(self.kv_params, num_layers=1)

        draft_config.return_hidden_states = ReturnHiddenStates.LAST

        assert self.pipeline_config.speculative is not None
        num_draft_steps = (
            self.pipeline_config.speculative.num_speculative_tokens
        )
        nn_model = Eagle3KimiK25Unified(
            config, draft_config, num_draft_steps=num_draft_steps
        )

        # Share embed_tokens before loading so the graph sees a single
        # Weight object for the shared embedding.  norm and lm_head are
        # loaded independently from the draft checkpoint.
        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens

        target_llm_sd = {
            k[len("language_model.") :]: v
            for k, v in llm_state_dict.items()
            if k.startswith("language_model.")
        }
        nn_model.target.load_state_dict(
            target_llm_sd, weight_alignment=1, strict=True
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

        # Capture concrete draft weights before renaming; ``state_dict()``
        # resets weight.name back to the module-path key.
        draft_weights_registry = nn_model.draft.state_dict()

        # Rename non-shared draft Weights so graph-level names are unique
        # (e.g. "draft.norm.weight" vs "norm.weight" from target).
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith("embed_tokens."):
                continue
            weight.name = f"draft.{name}"

        from .kimik2_5 import KimiK2_5

        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=config,
        )
        self.model_config = kimik2_5_config
        self.nn_model = KimiK2_5(kimik2_5_config)
        self.nn_model.load_state_dict(
            target_state_dict, weight_alignment=1, strict=False
        )
        # The vision graph loads from the regular Kimi registry
        # (``vision_encoder.*`` / ``language_model.*``), while the unified
        # Eagle graph also needs target-only keys plus ``draft.*`` weights.
        self.state_dict = dict(self.nn_model.state_dict())
        self.state_dict.update(nn_model.target.state_dict())
        for k, v in draft_weights_registry.items():
            if k.startswith("embed_tokens."):
                continue
            self.state_dict[f"draft.{k}"] = v

        # TODO(SERVOPT-1304): Support kimi spec decode with vision model
        # with CompilationTimer("vision model") as timer:
        #     vision_graph = self._build_vision_graph(
        #         kimik2_5_config, vision_state_dict
        #     )
        #     timer.mark_build_complete()
        #     vision_model = session.load(
        #         vision_graph, weights_registry=self.state_dict
        #     )
        vision_model = MagicMock(spec=Model)
        logger.warning(
            "Skipping vision model compilation. Vision support is not yet implemented for Kimi Eagle."
        )

        with CompilationTimer("eagle3_language_model") as timer:
            with Graph(
                "eagle3_kimik25_graph",
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

                fetch_types = (
                    self.kv_params.get_symbolic_inputs().inputs[0].flatten()
                )
                len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
                kv_caches_per_dev = self._unflatten_kv_inputs(
                    [next(variadic_args_iter) for _ in range(len_of_kv_inputs)]
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

                # Draft KV: only kv_blocks per device; cache_lengths reused
                # from target (same token count, just fewer layers).
                draft_kv_collections: list[PagedCacheValues] = []
                for dev_idx in range(len(self.devices)):
                    draft_kv_blocks = next(variadic_args_iter).buffer
                    draft_kv_collections.append(
                        PagedCacheValues(
                            kv_blocks=draft_kv_blocks,
                            cache_lengths=kv_caches_per_dev[
                                dev_idx
                            ].cache_lengths,
                            lookup_table=kv_caches_per_dev[
                                dev_idx
                            ].lookup_table,
                            max_lengths=kv_caches_per_dev[dev_idx].max_lengths,
                            attention_dispatch_metadata=kv_caches_per_dev[
                                dev_idx
                            ].attention_dispatch_metadata,
                        )
                    )

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
                    ep_inputs=target_ep_inputs,
                    draft_kv_collections=draft_kv_collections,
                )
                graph.output(*outputs)

            timer.mark_build_complete()
            language_model = session.load(
                graph, weights_registry=self.state_dict
            )

        return vision_model, language_model

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        """Execute and return all graph outputs for speculative decoding."""
        model_outputs = self.language_model.execute(*model_inputs.buffers)
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
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        draft_kv_cache_buffers: list[Buffer] | None = None,
        **kwargs,
    ) -> Eagle3KimiK25Inputs:
        base = KimiK2_5Model.prepare_initial_token_inputs(
            self,
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        return Eagle3KimiK25Inputs(
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
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> KimiK2_5ModelInputs:
        raise NotImplementedError("Eagle does not support Multistep execution")

    def _create_draft_config(
        self,
        target_config: KimiK2_5TextConfig,
        draft_state_dict: dict[str, WeightData],
    ) -> DeepseekV3Config:
        """Create config for the Eagle3 draft model.

        Uses the target config as base but overrides rope_scaling from the
        draft's HF config and dtype/quant based on the draft checkpoint.
        """
        draft_config = DeepseekV3Config(
            **{
                f.name: getattr(target_config, f.name)
                for f in fields(target_config)
                if f.name in {ff.name for ff in fields(DeepseekV3Config)}
            }
        )

        # The draft may use different YarnRoPE parameters (e.g.
        # beta_fast=1.0 vs target's 32.0).
        assert self.pipeline_config.draft_model is not None
        draft_hf_config = self.pipeline_config.draft_model.huggingface_config
        if draft_hf_config is not None:
            draft_rope = getattr(draft_hf_config, "rope_scaling", None)
            if draft_rope is not None:
                draft_config.rope_scaling = draft_rope

        # Avoid mutating the target's ep_config (shallow-copied from target).
        if draft_config.ep_config is not None:
            draft_config.ep_config = replace(draft_config.ep_config)

        # Eagle3 draft has BF16 dense MLP (not quantized, not MoE)
        if (
            draft_config.quant_config is not None
            and draft_config.quant_config.is_nvfp4
            and not any("weight_scale_2" in key for key in draft_state_dict)
        ):
            logger.info(
                "Eagle3 draft weights are BF16 (no weight_scale_2 found); "
                "disabling NVFP4 config for draft."
            )
            draft_config.quant_config = None
            draft_config.dtype = DType.bfloat16
            if draft_config.ep_config is not None:
                draft_config.ep_config.dispatch_dtype = DType.bfloat16
                draft_config.ep_config.dispatch_quant_config = None

        return draft_config
