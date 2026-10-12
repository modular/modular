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
"""The fused MiMo-V2.6-Flash + DFlash pipeline model: one graph per step."""

from __future__ import annotations

from typing import Any, ClassVar

from max.driver import Device
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    BatchProcessor,
    KVCacheConfig,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)

from ..mimo_v2.model import MiMoV2Model
from ..mimo_v2.model_config import FULL, SLIDING, MiMoV2Config
from .batch_processor import UnifiedDflashMiMoV2BatchProcessor
from .drafter import DrafterExport, drafter_export, load_drafter
from .model_config import (
    DRAFT,
    UnifiedDflashMiMoV2Config,
    drafter_config,
    read_dflash_config,
    repo_file,
    sampleable_vocab_size,
)
from .unified_dflash_mimo_v2 import UnifiedDflashMiMoV2, UnifiedDflashMiMoV2Spec


class UnifiedDflashMiMoV2Model(_UnifiedSpecDecodeModelMixin, MiMoV2Model):
    """Target and drafter in one compiled graph per speculative step."""

    model_config_cls: ClassVar[type[Any]] = UnifiedDflashMiMoV2Config
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        UnifiedDflashMiMoV2BatchProcessor
    )

    drafter_export: DrafterExport
    _spec: UnifiedDflashMiMoV2Spec
    _draft_state: dict[str, WeightData]
    _mask_embedding: WeightData

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        del return_logits
        assert pipeline_config.speculative is not None
        self.resolved_num_speculative_tokens = (
            pipeline_config.speculative.draft_width
        )
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            memory_plan=memory_plan,
            adapter=adapter,
            return_logits=ReturnLogits.VARIABLE,
            max_batch_size=max_batch_size,
        )

    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        assert isinstance(self.kv_params, MultiKVCacheParams)
        target_kv = self.kv_params.children["target"]
        draft_kv = self.kv_params.children[DRAFT]
        assert isinstance(target_kv, MultiKVCacheParams)
        assert isinstance(draft_kv, KVCacheParams)
        # Verify rows and the block reach past the last committed position.
        target = MiMoV2Config.initialize(
            self.pipeline_config,
            max_seq_len=self.max_seq_len
            + spec_decode_cache_slack(self.kv_params),
        )
        target.kv_params = target_kv
        draft_model = self.pipeline_config.draft_model
        assert draft_model is not None
        repo = draft_model.huggingface_weight_repo
        draft = drafter_config(
            read_dflash_config(repo_file(repo, "config.json")),
            draft_kv,
            target.devices,
            self.max_seq_len,
        )
        self._draft_state, self._mask_embedding, directory = load_drafter(
            repo, draft
        )
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        self._spec = UnifiedDflashMiMoV2Spec(
            target=target,
            draft=draft,
            speculative_config=speculative,
            num_speculative_tokens=speculative.draft_width,
            sampleable_vocab_size=sampleable_vocab_size(self.pipeline_config),
        )
        self.drafter_export = drafter_export(
            draft,
            directory,
            num_speculative_tokens=speculative.draft_width,
        )
        return target

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: MiMoV2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session, model_config
        nn_model = self._fused_module()
        nn_model.load_state_dict(
            {
                **{f"target.{k}": v for k, v in state_dict.items()},
                **{f"{DRAFT}.{k}": v for k, v in self._draft_state.items()},
            },
            weight_alignment=1,
            strict=True,
        )
        weights_registry = nn_model.state_dict(auto_initialize=False)
        assert isinstance(self.kv_params, MultiKVCacheParams)
        return fused_graph(nn_model, self.kv_params), weights_registry

    def _fused_module(self) -> UnifiedDflashMiMoV2:
        """Builds the module the fused graph runs, before its weights load."""
        return UnifiedDflashMiMoV2(
            self._spec,
            self._mask_embedding,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )


def fused_graph(
    nn_model: UnifiedDflashMiMoV2, kv_params: MultiKVCacheParams
) -> Graph:
    """Builds the fused graph over the ``{target, draft}`` KV tree."""
    with Graph(
        "unified_dflash_mimo_v2", input_types=nn_model.input_types(kv_params)
    ) as graph:
        inputs = nn_model.decode_inputs(graph.inputs, kv_params)
        outputs = nn_model(
            tokens=inputs.tokens,
            input_row_offsets=inputs.input_row_offsets,
            draft_tokens=inputs.draft_tokens,
            kv_collections=inputs.kv("target", FULL),
            draft_kv_collections=inputs.kv(DRAFT),
            return_n_logits=inputs.return_n_logits,
            seed=inputs.seed,
            temperature=inputs.temperature,
            top_k=inputs.top_k,
            max_k=inputs.max_k,
            top_p=inputs.top_p,
            min_top_p=inputs.min_top_p,
            signal_buffers=inputs.signal_buffers,
            passthrough_kv={SLIDING: inputs.kv("target", SLIDING)},
            pinned_bitmask=inputs.pinned_bitmask,
            wait_payload=inputs.wait_payload,
            device_bitmask_scratch=inputs.device_bitmask_scratch,
        )
        graph.output(*outputs)
    return graph
