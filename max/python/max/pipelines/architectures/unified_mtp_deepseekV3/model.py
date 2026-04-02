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
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, Value
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import CompilationTimer, ModelInputs
from max.pipelines.lib.speculative_decoding.unified_eagle import (
    UnifiedEagleOutputs,
)
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig
from .unified_mtp_deepseekV3 import UnifiedMTPDeepseekV3

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedMTPDeepseekV3ModelInputs(DeepseekV3Inputs):
    """Inputs for the with-MTP model, adding draft_tokens and draft KV."""

    draft_tokens: Buffer = field(
        default_factory=lambda: Buffer.from_numpy(
            np.zeros((0, 0), dtype=np.int64)
        )
    )

    draft_kv_cache_buffers: list[Buffer] = field(default_factory=list)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.draft_tokens,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(self.kv_cache_inputs or ()),
            *self.draft_kv_cache_buffers,
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class UnifiedMTPDeepseekV3Model(DeepseekV3Model):
    """DeepseekV3 with MTP: merge + target + rejection + shift in one graph."""

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

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

        with CompilationTimer("with_mtp_model") as timer:
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

            # Create target config from target-only keys (strip "target." prefix).
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

            self.ep_comm_initializer = None
            self.draft_ep_comm_initializer = None
            if config.ep_config is not None and not is_virtual_device_mode():
                # Allocate EP buffers with BF16 dispatch dtype (the larger dtype)
                # so both target (FP4) and draft (BF16) can share the same buffers.
                ep_cfg = replace(
                    config.ep_config,
                    dispatch_dtype=DType.bfloat16,
                    dispatch_quant_config=None,
                )
                self.ep_comm_initializer = EPCommInitializer(ep_cfg)
                self.ep_comm_initializer.ep_init(session)
                config.ep_config.node_id = (
                    self.ep_comm_initializer.config.node_id
                )
                if config.ep_config.node_id == -1:
                    raise ValueError(
                        "EP node ID is not set. Please check if the EP "
                        "initialization is successful."
                    )
                self.draft_ep_comm_initializer = self.ep_comm_initializer

            # Create draft config from draft-only keys (strip "draft." prefix).
            draft_state_dict = {
                k[len("draft.") :]: v
                for k, v in state_dict.items()
                if k.startswith("draft.")
            }
            draft_config = self._create_draft_config(draft_state_dict)

            if (
                draft_config.ep_config is not None
                and config.ep_config is not None
            ):
                draft_config.ep_config.node_id = config.ep_config.node_id

            # TODO: don't hard code number of layers
            assert isinstance(self.kv_params, KVCacheParams)
            self._draft_kv_params = replace(self.kv_params, num_layers=1)

            draft_config.return_hidden_states = ReturnHiddenStates.LAST

            logger.info(
                f"[DEBUG] load_model: target config devices={config.devices}, "
                f"draft config devices={draft_config.devices}"
            )
            nn_model = UnifiedMTPDeepseekV3(config, draft_config)

            # Share embed_tokens and lm_head BEFORE loading so state_dict()
            # deduplicates them — the adapter only emits target.* copies.
            assert nn_model.draft is not None
            nn_model.draft.embed_tokens = nn_model.target.embed_tokens
            nn_model.draft.lm_head = nn_model.target.lm_head

            target_sd = {
                k[len("target.") :]: v
                for k, v in state_dict.items()
                if k.startswith("target.")
            }
            nn_model.target.load_state_dict(
                target_sd, weight_alignment=1, strict=True
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
                "deepseekV3_with_mtp_graph",
                input_types=nn_model.input_types(
                    self.kv_params, self._draft_kv_params
                ),
            ) as graph:
                (
                    tokens,
                    devices_input_row_offsets,
                    host_input_row_offsets,
                    draft_tokens,
                    return_n_logits,
                    data_parallel_splits,
                    *variadic_args,
                ) = graph.inputs

                variadic_args_iter = iter(variadic_args)
                signal_buffers = [
                    next(variadic_args_iter).buffer
                    for _ in range(len(self.devices))
                ]

                fetch_types = self.kv_params.get_symbolic_inputs()[0]
                len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
                kv_caches_per_dev = self._unflatten_kv_inputs(
                    [next(variadic_args_iter) for _ in range(len_of_kv_inputs)]
                )

                logger.info(
                    f"[DEBUG] load_model graph build: "
                    f"n_devices={len(self.devices)}, "
                    f"n_signal_buffers={len(signal_buffers)}, "
                    f"n_kv_caches_per_dev={len(kv_caches_per_dev)}, "
                    f"target_kv_params.n_devices={len(self.kv_params.get_symbolic_inputs())}, "
                    f"draft_kv_params.n_devices={len(self._draft_kv_params.get_symbolic_inputs())}"
                )

                # Draft KV: only kv_blocks per device; cache_lengths, lookup_table,
                # max_lengths, and dispatch_metadata are shared from target.
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
                            dispatch_metadata=kv_caches_per_dev[
                                dev_idx
                            ].dispatch_metadata,
                        )
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

                outputs = nn_model(
                    tokens.tensor,
                    devices_input_row_offsets.tensor,
                    draft_tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    host_input_row_offsets.tensor,
                    data_parallel_splits.tensor,
                    batch_context_lengths,
                    ep_inputs=target_ep_inputs,
                    draft_kv_collections=draft_kv_collections,
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
        assert isinstance(model_inputs, UnifiedMTPDeepseekV3ModelInputs)

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
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        draft_kv_cache_buffers: list[Buffer] | None = None,
    ) -> UnifiedMTPDeepseekV3ModelInputs:
        base = super().prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        if draft_tokens is None:
            batch_size = sum(len(b) for b in replica_batches)
            draft_tokens = Buffer.from_numpy(
                np.zeros((batch_size, 0), dtype=np.int64)
            ).to(self.devices[0])

        return UnifiedMTPDeepseekV3ModelInputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            draft_tokens=draft_tokens,
            draft_kv_cache_buffers=draft_kv_cache_buffers or [],
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> UnifiedMTPDeepseekV3ModelInputs:
        assert isinstance(prev_model_inputs, UnifiedMTPDeepseekV3ModelInputs)
        base = super().prepare_next_token_inputs(next_tokens, prev_model_inputs)

        return UnifiedMTPDeepseekV3ModelInputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            draft_tokens=prev_model_inputs.draft_tokens,
            draft_kv_cache_buffers=prev_model_inputs.draft_kv_cache_buffers,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
        )

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
        logger.info(
            f"[DEBUG] _create_draft_config: "
            f"devices={draft_config.devices}, "
            f"dp={draft_config.data_parallel_degree}, "
            f"ep_config={draft_config.ep_config is not None}, "
            f"num_hidden_layers={draft_config.num_hidden_layers}, "
            f"dtype={draft_config.dtype}"
        )
        return draft_config
