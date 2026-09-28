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
"""Unified DSpark DeepSeek-V4 PipelineModel: trunk + DSpark in one graph."""

from __future__ import annotations

from typing import Any, ClassVar

from max.engine import InferenceSession
from max.graph import Graph
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.interfaces.batch_processor import BatchProcessor
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from typing_extensions import override

from ..deepseekV4.model import DeepseekV4Model
from ..deepseekV4.model_config import DeepseekV4Config
from .batch_processor import UnifiedDSparkDeepseekV4BatchProcessor
from .spec_adapters import VERIFY_WINDOWS, WINDOW_LEAF
from .unified_dspark_deepseekV4 import UnifiedDSparkDeepseekV4

__all__ = ["UnifiedDSparkDeepseekV4Model"]


class UnifiedDSparkDeepseekV4Model(
    _UnifiedSpecDecodeModelMixin, DeepseekV4Model
):
    """DeepSeek-V4 with DSpark: merge + verify + accept + draft in one graph.

    The KV tree is the trunk's own: the stages' windows are already layers of
    its window leaf, so there is no separate draft cache to size or feed.
    """

    # Declared rather than left to the arch's ``batching=`` binding: under
    # ``max serve`` the model worker ran the inherited base batch processor.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        UnifiedDSparkDeepseekV4BatchProcessor
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The accept reads the target's logits at every merged position.
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        super().__init__(*args, **kwargs)

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV4Config:
        model_config = super()._create_model_config(state_dict)
        model_config.dspark_stages = True
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: DeepseekV4Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        nn_model = UnifiedDSparkDeepseekV4(
            model_config,
            speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )
        nn_model.target.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        weights_registry = nn_model.target.state_dict(auto_initialize=False)
        if len(self.device_refs) > 1:
            nn_model.shard(self.device_refs)

        assert isinstance(self.kv_params, MultiKVCacheParams)
        with Graph(
            "unified_dspark_deepseekV4",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            inputs = nn_model.decode_inputs(graph.inputs, self.kv_params)
            window = inputs.kv(WINDOW_LEAF)
            outputs = nn_model(
                tokens=inputs.tokens,
                input_row_offsets=inputs.input_row_offsets,
                draft_tokens=inputs.draft_tokens,
                kv_collections=window,
                draft_kv_collections=window,
                passthrough_kv={
                    spec.key: inputs.kv(spec.key)
                    for spec in model_config.kv_leaf_specs()
                    if spec.key != WINDOW_LEAF
                },
                return_n_logits=inputs.return_n_logits,
                seed=inputs.seed,
                temperature=inputs.temperature,
                top_k=inputs.top_k,
                max_k=inputs.max_k,
                top_p=inputs.top_p,
                min_top_p=inputs.min_top_p,
                signal_buffers=inputs.signal_buffers,
                pinned_bitmask=inputs.pinned_bitmask,
                wait_payload=inputs.wait_payload,
                device_bitmask_scratch=inputs.device_bitmask_scratch,
                extra={
                    VERIFY_WINDOWS: nn_model.verify_windows(inputs.trailing)
                },
            )
            graph.output(*outputs)
        return graph, weights_registry
