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
"""GLM-5.2 (DeepSeek-V3.2 sparse) with MTP PipelineModel: target + draft graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import BufferValue, Graph, TensorValue, Value
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import CompilationTimer, UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs
from ..deepseekV3_2.model import DeepseekV3_2Model
from ..deepseekV3_2_nextn.model_config import DeepseekV3_2NextNConfig
from ..glm5_1.model import Glm5_1Model
from .batch_processor import UnifiedMTPGlm5_2BatchProcessor
from .unified_mtp_glm5_2 import UnifiedMTPGlm5_2

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedMTPGlm5_2Inputs(UnifiedSpecDecodeInputs, DeepseekV3Inputs):
    """Inputs for the UnifiedMTPGlm5_2 model.

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


class UnifiedMTPGlm5_2Model(_UnifiedSpecDecodeModelMixin, Glm5_1Model):
    """GLM-5.2 with MTP: merge + V3.2 target + rejection + sparse draft."""

    batch_processor_cls: ClassVar[type[UnifiedMTPGlm5_2BatchProcessor]] = (
        UnifiedMTPGlm5_2BatchProcessor
    )

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("glm5_2_with_mtp_model") as timer:
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=self.huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }

            # Target config from target-only keys (strip "target." prefix).
            target_state_dict = {
                k[len("target.") :]: v
                for k, v in state_dict.items()
                if k.startswith("target.")
            }
            config = self._create_model_config(target_state_dict)

            n_devices = len(self.devices)
            if (
                n_devices > 1
                and self.pipeline_config.runtime.ep_size != n_devices
            ):
                raise ValueError("Only the EP strategy is supported.")

            self.ep_comm_initializer: EPCommInitializer | None = None
            self.draft_ep_comm_initializer: EPCommInitializer | None = None
            if config.ep_config is not None and not is_virtual_device_mode():
                self.ep_comm_initializer = EPCommInitializer(config.ep_config)
                self.ep_comm_initializer.ep_init(session)
                config.ep_config.node_id = (
                    self.ep_comm_initializer.config.node_id
                )
                if config.ep_config.node_id == -1:
                    raise ValueError(
                        "EP node ID is not set. Please check if the EP "
                        "initialization is successful."
                    )
                # Target and draft both run FP8 and execute sequentially in the
                # MTP graph, so they share the same EP communication buffers.
                self.draft_ep_comm_initializer = self.ep_comm_initializer

            # Draft config from draft-only keys (strip "draft." prefix).
            draft_state_dict = {
                k[len("draft.") :]: v
                for k, v in state_dict.items()
                if k.startswith("draft.")
            }
            # Some checkpoints share shared_head_norm with the base model's
            # final norm and don't emit it as a draft weight; copy from
            # target.norm.weight so load_state_dict finds it.
            if (
                "shared_head_norm.weight" not in draft_state_dict
                and "target.norm.weight" in state_dict
            ):
                draft_state_dict["shared_head_norm.weight"] = state_dict[
                    "target.norm.weight"
                ]

            # Build the nested {target: {mla, indexer}, draft: {mla, indexer}}
            # KV tree. The draft caches store a single layer.
            assert isinstance(self.kv_params, MultiKVCacheParams)
            target_kv = self.kv_params
            target_mla_params = target_kv.children["mla"]
            target_indexer_params = target_kv.children["indexer"]
            assert isinstance(target_mla_params, KVCacheParams)
            assert isinstance(target_indexer_params, KVCacheParams)
            draft_kv = MultiKVCacheParams.from_params(
                {
                    "mla": replace(target_mla_params, num_layers=1),
                    "indexer": replace(target_indexer_params, num_layers=1),
                }
            )

            draft_config = self._create_draft_config(draft_state_dict, draft_kv)

            if (
                draft_config.ep_config is not None
                and config.ep_config is not None
            ):
                draft_config.ep_config.node_id = config.ep_config.node_id

            self.kv_params = MultiKVCacheParams.from_params(
                {"target": target_kv, "draft": draft_kv}
            )

            draft_config.return_hidden_states = ReturnHiddenStates.LAST

            assert self.pipeline_config.speculative is not None

            nn_model = UnifiedMTPGlm5_2(
                config,
                draft_config,
                speculative_config=self.pipeline_config.speculative,
                enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            )

            # Share embed_tokens and lm_head BEFORE loading so state_dict()
            # deduplicates them — the adapter only emits target.* copies.
            assert nn_model.draft is not None
            nn_model.draft.embed_tokens = nn_model.target.embed_tokens
            nn_model.draft.lm_head = nn_model.target.lm_head

            nn_model.target.load_state_dict(
                target_state_dict, weight_alignment=1, strict=True
            )
            # strict=False because shared weights (embed_tokens, lm_head) are
            # aliased to target's and won't have keys in draft_state_dict.
            nn_model.draft.load_state_dict(
                draft_state_dict, weight_alignment=1, strict=False
            )

            draft_expected = set(nn_model.draft.raw_state_dict().keys())
            draft_provided = set(draft_state_dict.keys())
            shared_prefixes = ("embed_tokens.", "lm_head.")
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
                logger.warning(
                    f"Draft state_dict has unused keys: {sorted(extra)}"
                )

            self.state_dict = {
                **nn_model.draft.state_dict(),
                **nn_model.target.state_dict(),
            }

            with Graph(
                "glm5_2_with_mtp_graph",
                input_types=nn_model.input_types(self.kv_params),
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

                kv_tree = self.kv_params.unflatten_kv_inputs(variadic_args_iter)
                target_tree = kv_tree.children["target"]
                draft_tree = kv_tree.children["draft"]
                assert isinstance(target_tree, MultiKVCacheInputs)
                assert isinstance(draft_tree, MultiKVCacheInputs)
                target_mla = target_tree.children["mla"]
                target_indexer = target_tree.children["indexer"]
                draft_mla = draft_tree.children["mla"]
                draft_indexer = draft_tree.children["indexer"]
                assert isinstance(target_mla, KVCacheInputs)
                assert isinstance(target_indexer, KVCacheInputs)
                assert isinstance(draft_mla, KVCacheInputs)
                assert isinstance(draft_indexer, KVCacheInputs)
                target_mla_kv = list(target_mla.inputs)
                target_indexer_kv = list(target_indexer.inputs)
                draft_mla_kv = list(draft_mla.inputs)
                draft_indexer_kv = list(draft_indexer.inputs)

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

                seed = next(variadic_args_iter).tensor
                temperature = next(variadic_args_iter).tensor
                top_k = next(variadic_args_iter).tensor
                max_k = next(variadic_args_iter).tensor
                top_p = next(variadic_args_iter).tensor
                min_top_p = next(variadic_args_iter).tensor
                in_thinking_phase = next(variadic_args_iter).tensor

                pinned_bitmask_graph: TensorValue | None = None
                wait_payload_graph: BufferValue | None = None
                device_bitmask_scratch_graph: BufferValue | None = None
                if nn_model.enable_structured_output:
                    pinned_bitmask_graph = next(variadic_args_iter).tensor
                    wait_payload_graph = next(variadic_args_iter).buffer
                    device_bitmask_scratch_graph = next(
                        variadic_args_iter
                    ).buffer

                outputs = nn_model(
                    tokens=tokens.tensor,
                    input_row_offsets=devices_input_row_offsets.tensor,
                    draft_tokens=draft_tokens,
                    signal_buffers=signal_buffers,
                    target_mla_kv=target_mla_kv,
                    target_indexer_kv=target_indexer_kv,
                    draft_mla_kv=draft_mla_kv,
                    draft_indexer_kv=draft_indexer_kv,
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
                    pinned_bitmask=pinned_bitmask_graph,
                    wait_payload=wait_payload_graph,
                    device_bitmask_scratch=device_bitmask_scratch_graph,
                )

                graph.output(*outputs)

            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        if self._batch_processor is not None:
            bind_ep = getattr(
                self._batch_processor, "bind_ep_comm_initializer", None
            )
            if bind_ep is not None:
                bind_ep(self.ep_comm_initializer)

        return model

    def _create_draft_config(
        self,
        draft_state_dict: dict[str, WeightData],
        draft_kv: MultiKVCacheParams,
    ) -> DeepseekV3_2NextNConfig:
        """Create the NextN draft config for the GLM-5.2 MTP layer."""
        nextn_key = "decoder_layer.self_attn.kv_a_layernorm.weight"
        base_key = "layers.0.self_attn.kv_a_layernorm.weight"

        if nextn_key not in draft_state_dict:
            raise KeyError(
                f"Expected NextN norm key '{nextn_key}' not found in "
                f"draft state_dict. Available keys: "
                f"{list(draft_state_dict.keys())[:10]}..."
            )

        draft_state_dict[base_key] = draft_state_dict[nextn_key]
        base_config = DeepseekV3_2Model._create_model_config(
            self, draft_state_dict
        )
        if base_key in draft_state_dict and nextn_key in draft_state_dict:
            del draft_state_dict[base_key]

        draft_config = DeepseekV3_2NextNConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(base_config)
            }
        )
        # The single MTP layer is a ``full`` indexer layer that computes its own
        # top-k at step 0; an empty schedule keeps it full (skip_topk=False) and
        # avoids indexing the 78-element target schedule at the MTP layer index.
        draft_config.indexer_types = []
        # The draft owns a single-layer {mla, indexer} cache.
        draft_config.kv_params = draft_kv

        # ``mlp_quantized_layers`` / ``attn_quantized_layers`` are derived from
        # ``range(num_hidden_layers)`` and therefore exclude the MTP layer's
        # index (== num_hidden_layers). The GLM-5.2 MTP layer ships FP8, so mark
        # its index quantized to match the checkpoint weights.
        if draft_config.quant_config is not None:
            nextn_layer_idx = max(
                draft_config.num_hidden_layers,
                draft_config.first_k_dense_replace,
            )
            draft_config.quant_config.mlp_quantized_layers.add(nextn_layer_idx)
            draft_config.quant_config.attn_quantized_layers.add(nextn_layer_idx)
        return draft_config
