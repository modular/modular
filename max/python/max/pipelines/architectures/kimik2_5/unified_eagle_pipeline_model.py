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

from max import tree
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, Module
from max.graph.weights import WeightData, load_weights
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.lib import CompilationTimer
from max.pipelines.lib.interfaces import (
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict
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
class Eagle3KimiK25Inputs(UnifiedSpecDecodeInputs, KimiK2_5ModelInputs):
    """Inputs for the Eagle3 + Kimi K2.5 model.

    Inherits all of ``KimiK2_5ModelInputs`` so the per-device vision-merge
    inputs (base ``vision_embeddings`` / ``vision_scatter_indices``) flow
    through to the unified Eagle graph, which scatters them into the merged
    token embedding before the target forward. The spec-decode fields and
    trailing buffer packing come from :class:`UnifiedSpecDecodeInputs`; the
    graph binds the per-row ``in_thinking_phase`` flag.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        # Ordering must match ``Eagle3KimiK25Unified.input_types``: tokens,
        # then per-device image_embeddings, per-device image_token_indices,
        # then the rest of the inputs.
        assert len(self.vision_embeddings) == len(
            self.vision_scatter_indices
        ), (
            "vision_embeddings and vision_scatter_indices must have the "
            "same length"
        )
        buffers = (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class Eagle3KimiK25Model(_UnifiedSpecDecodeModelMixin, KimiK2_5Model):
    """Eagle3 + Kimi K2.5: target + draft in one compiled graph.

    Loads target weights from the main Kimi K2.5 checkpoint and draft weights
    from a separate Eagle3 checkpoint (``pipeline_config.draft_model``).

    The Eagle3 language model graph is text-only — the raw ``DeepseekV3``
    target is used, not ``KimiK2_5MoEDecoder``. Vision is handled by the
    base pipeline during initial prefill.
    """

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.SELECTED_LAYERS
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
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

        # ``_create_model_config`` may mutate the state dictionary.
        config = self._create_model_config(target_state_dict)

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in target_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith(("language_model.", "language_")):
                llm_state_dict[key] = value

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
        target_kv_params = self.kv_params
        self._draft_kv_params = replace(target_kv_params, num_layers=1)
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv_params, "draft": self._draft_kv_params}
        )

        draft_config.return_hidden_states = ReturnHiddenStates.LAST

        assert self.pipeline_config.speculative is not None
        nn_model = Eagle3KimiK25Unified(
            config,
            draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            enable_vision=True,
        )

        # Share embed_tokens before loading so the graph sees a single
        # Weight object for the shared embedding.  norm is loaded
        # independently from the draft checkpoint; lm_head is shared from
        # the target when absent from the draft checkpoint (e.g.
        # nvidia/Kimi-K2.6-Eagle3 omits lm_head.weight).
        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens
        if "lm_head.weight" not in draft_state_dict:
            nn_model.draft.lm_head = nn_model.target.lm_head

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

        # A draft checkpoint with its own ``lm_head`` loads it; one without
        # inherits the target's.
        aliased = validate_draft_state_dict(
            nn_model.draft.raw_state_dict().keys(),
            draft_state_dict.keys(),
            DraftAliases(always=("embed_tokens.",), when_absent=("lm_head.",)),
        )

        # Capture concrete draft weights before renaming; ``state_dict()``
        # resets weight.name back to the module-path key.
        draft_weights_registry = nn_model.draft.state_dict()

        # Rename non-shared draft Weights so graph-level names are unique
        # (e.g. "draft.norm.weight" vs "norm.weight" from target).
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith(aliased):
                continue
            weight.name = f"draft.{name}"

        from .kimik2_5 import KimiK2_5

        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=config,
            max_seq_len=self.max_seq_len,
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
            if k.startswith(aliased):
                continue
            self.state_dict[f"draft.{k}"] = v

        with CompilationTimer("vision + eagle3 language model") as timer:
            graph_module = Module()
            assert self.model_config is not None
            vision_graph, _ = self._build_vision_graph(
                self.model_config, vision_state_dict, module=graph_module
            )
            with Graph(
                "eagle3_kimik25_graph",
                input_types=nn_model.input_types(self.kv_params),
                module=graph_module,
            ) as graph:
                graph_inputs = nn_model.decode_inputs(
                    graph.inputs, self.kv_params
                )

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
                    vision_embeddings=graph_inputs.vision_embeddings,
                    vision_scatter_indices=graph_inputs.vision_scatter_indices,
                    ep_inputs=graph_inputs.ep_inputs or None,
                    draft_kv_collections=graph_inputs.kv("draft"),
                    pinned_bitmask=graph_inputs.pinned_bitmask,
                    wait_payload=graph_inputs.wait_payload,
                    device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
                )
                graph.output(*outputs)

            timer.mark_build_complete()
            models = session.load_all(
                graph_module, weights_registry=self.state_dict
            )
            vision_model = models[vision_graph.name]
            language_model = models[graph.name]

        return vision_model, language_model

    @property
    def _spec_decode_model(self) -> Model:
        return self.language_model

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        **kwargs,
    ) -> Eagle3KimiK25Inputs:
        base = KimiK2_5Model.prepare_initial_token_inputs(
            self,
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        # The overlap pipeline assigns ``seed`` and the rest of the
        # per-batch sampling buffers (temperature / top_k / top_p / max_k
        # / min_top_p) on the returned inputs *after* this call returns —
        # see ``OverlapTextGenerationPipeline._run_forward``.
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
            # Vision inputs computed by the base call's host-side encoder
            # run (or empty placeholders when no images are present).
            image_token_indices=base.image_token_indices,
            precomputed_image_embeddings=base.precomputed_image_embeddings,
            pixel_values=base.pixel_values,
            grid_thws=base.grid_thws,
            cu_seqlens=base.cu_seqlens,
            max_seqlen=base.max_seqlen,
            vision_position_ids=base.vision_position_ids,
            draft_tokens=draft_tokens,
            structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

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
            and draft_config.quant_config.is_fp4
            and not any("weight_scale" in key for key in draft_state_dict)
        ):
            logger.info(
                "Eagle3 draft weights are BF16 (no weight_scale found); "
                "disabling FP4 config for draft."
            )
            draft_config.quant_config = None
            draft_config.dtype = DType.bfloat16
            if draft_config.ep_config is not None:
                draft_config.ep_config.dispatch_dtype = DType.bfloat16
                draft_config.ep_config.dispatch_quant_config = None

        return draft_config
