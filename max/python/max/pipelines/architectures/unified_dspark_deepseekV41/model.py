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
"""Pipeline model running DeepSeek-V4.1-Flash and its DSpark drafter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import ModelInputs
from max.pipelines.lib.interfaces.batch_processor import BatchProcessor
from max.pipelines.lib.interfaces.pipeline_model import UnifiedEagleOutputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeGraphInputs
from typing_extensions import override

from ..deepseekV41 import kv_leaves
from ..deepseekV41.dspark_drafter import DeepseekV41WithDSpark
from ..deepseekV41.kv_leaves import V41KVLeafSpec
from ..deepseekV41.model import (
    DeepseekV41Model,
    _make_routed_experts_device_resident,
    _stream_routed_experts,
)
from ..deepseekV41.model_config import DeepseekV41Config
from .batch_processor import (
    MERGED_ROWS_DIM,
    UnifiedDSparkDeepseekV41BatchProcessor,
    UnifiedDSparkDeepseekV41Inputs,
)
from .spec_adapters import ENGRAM_LOOKBACK, WINDOW_LEAF
from .unified_dspark_deepseekV41 import UnifiedDSparkDeepseekV41

__all__ = ["UnifiedDSparkDeepseekV41Model", "driver_outputs"]


def driver_outputs(
    driver: UnifiedDSparkDeepseekV41,
    inputs: SpecDecodeGraphInputs,
    specs: Sequence[V41KVLeafSpec],
    *,
    has_engram: bool,
) -> tuple[TensorValue, ...]:
    """Runs one speculative decoding step over the decoded graph inputs.

    The sliding-window KV cache serves as both the target model's and the
    drafter's cache; every other KV cache passes through unchanged. With
    ``has_engram``, the first model-specific graph input is the Engram
    lookback window, which is handed to the target model through ``extra``.
    """
    window = inputs.kv(WINDOW_LEAF)
    return driver(
        tokens=inputs.tokens,
        input_row_offsets=inputs.input_row_offsets,
        draft_tokens=inputs.draft_tokens,
        kv_collections=window,
        draft_kv_collections=window,
        passthrough_kv={
            spec.key: inputs.kv(spec.key)
            for spec in specs
            if spec.key != WINDOW_LEAF
        },
        return_n_logits=inputs.return_n_logits,
        seed=inputs.seed,
        temperature=inputs.temperature,
        top_k=inputs.top_k,
        max_k=inputs.max_k,
        top_p=inputs.top_p,
        min_top_p=inputs.min_top_p,
        draft_probs_full=inputs.draft_probs_full,
        signal_buffers=inputs.signal_buffers,
        pinned_bitmask=inputs.pinned_bitmask,
        wait_payload=inputs.wait_payload,
        device_bitmask_scratch=inputs.device_bitmask_scratch,
        host_input_row_offsets=inputs.host_input_row_offsets,
        data_parallel_splits=inputs.data_parallel_splits,
        batch_context_lengths=inputs.batch_context_lengths,
        ep_inputs=inputs.ep_inputs or None,
        vision_embeddings=inputs.vision_embeddings,
        vision_scatter_indices=inputs.vision_scatter_indices,
        extra=(
            {ENGRAM_LOOKBACK: inputs.trailing[0].tensor} if has_engram else {}
        ),
    )


class UnifiedDSparkDeepseekV41Model(
    _UnifiedSpecDecodeModelMixin, DeepseekV41Model
):
    """DeepSeek-V4.1 that verifies and drafts tokens in one graph per step.

    The drafter's stages store their context KV in the target model's
    sliding-window cache, as layers numbered from ``num_hidden_layers`` up,
    so there is no separate draft cache to size or feed.
    """

    # Declared here as well as in the arch's ``batching=``: the serve model
    # worker reads it from the model class.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        UnifiedDSparkDeepseekV41BatchProcessor
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Acceptance reads the target's logits at every prompt and draft
        # position.
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        super().__init__(*args, **kwargs)

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV41Config:
        model_config = super()._create_model_config(state_dict)
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        if speculative.num_speculative_tokens is None:
            raise ValueError(
                "DeepSeek-V4.1 DSpark needs num_speculative_tokens resolved "
                f"(to dspark_block_size={model_config.dspark_block_size})"
            )
        model_config.validate_num_speculative_tokens(
            speculative.num_speculative_tokens
        )
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: DeepseekV41Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        assert isinstance(self.kv_params, MultiKVCacheParams)
        engram_layer_ids = model_config.engram_layer_ids
        engram_input_types, engram_token_map = self._prepare_engram(
            state_dict, model_config
        )
        lookback_types = (
            [
                TensorType(
                    DType.int64,
                    shape=[
                        "engram_lookback_batch",
                        model_config.engram_max_ngram_size - 1,
                    ],
                    device=self.device_refs[0],
                )
            ]
            if engram_layer_ids
            else []
        )
        # Last: a host tensor whose length declares the prompt-plus-draft
        # row count (`S + B * K`) as a graph-input dim; buffers shaped by
        # that dim fail to compile unless it is an input dim.
        trailing_types = [
            *lookback_types,
            *engram_input_types,
            TensorType(DType.int8, [MERGED_ROWS_DIM], device=DeviceRef.CPU()),
        ]
        specs = kv_leaves.kv_leaf_specs(
            DeepseekV41Config.get_text_config(self.huggingface_config)
        )
        structured = self.pipeline_config.needs_bitmask_constraints
        signature = UnifiedDSparkDeepseekV41(
            model_config,
            speculative,
            specs,
            model=None,
            trailing_input_types=trailing_types,
            enable_structured_output=structured,
        )
        input_types = signature.input_types(self.kv_params)

        with Graph(
            "unified_dspark_deepseekV41", input_types=input_types
        ) as graph:
            trailing = signature.decode_inputs(
                graph.inputs, self.kv_params
            ).trailing
            per_device_tables = self._per_device_engram_tables(
                trailing[len(lookback_types) :], engram_layer_ids
            )
            nn_model = DeepseekV41WithDSpark(
                model_config,
                engram_host_tables=per_device_tables[0],
                engram_token_map=engram_token_map,
            )
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )
            weights_registry = nn_model.state_dict(auto_initialize=False)

            driver = UnifiedDSparkDeepseekV41(
                model_config,
                speculative,
                specs,
                model=nn_model,
                trailing_input_types=trailing_types,
                enable_structured_output=structured,
            )
            if tuple(driver.input_types(self.kv_params)) != tuple(input_types):
                raise RuntimeError(
                    "the DSpark driver's signature drifted from the one the "
                    "graph was declared with"
                )
            # The trunk's routed experts stream straight to each serving
            # copy's device; the drafter's stage experts are host-staged.
            serving = (
                driver.shard(self.device_refs, per_device_tables)
                if len(self.device_refs) > 1
                else [nn_model]
            )
            expert_uploads = _make_routed_experts_device_resident(
                nn_model,
                serving,
                state_dict,
                weights_registry,
                self.devices,
                self.device_refs,
            )

            outputs = driver_outputs(
                driver,
                driver.decode_inputs(graph.inputs, self.kv_params),
                specs,
                has_engram=bool(engram_layer_ids),
            )
            graph.output(*outputs)
        if expert_uploads:
            self._defer_weight_load(
                lambda: weights_registry.update(
                    _stream_routed_experts(expert_uploads)
                ),
                overlaps_compile=False,
            )
        return graph, weights_registry

    @override
    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        """Runs one step, binding the pinned Engram tables as graph inputs.

        The tables follow the batch's own inputs, in the graph's input order.
        """
        assert isinstance(model_inputs, UnifiedDSparkDeepseekV41Inputs)
        model_inputs.engram_host_buffers = self._engram_host_buffers
        model_outputs = self.model.execute(*model_inputs.buffers)
        if len(model_outputs) not in (3, 4):
            raise RuntimeError(
                f"{type(self).__name__} graph returned {len(model_outputs)} "
                "outputs; expected 3 (num_accepted_draft_tokens, next_tokens, "
                "next_draft_tokens) or 4 (+ next_draft_probs_full)."
            )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
            next_draft_probs_full=(
                model_outputs[3] if len(model_outputs) == 4 else None
            ),
        )
